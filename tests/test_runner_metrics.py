from pathlib import Path
from stateful_safety.dataset import load_records
from stateful_safety.judge import MockJudge
from stateful_safety.runner import run_sequences
from stateful_safety.metrics import calibrate_threshold,summarize,first_intervention

FIXTURE=Path(__file__).resolve().parents[1]/"fixtures/synthetic_fixture.jsonl"

def test_full_history_changes_mock_decision():
    records=load_records(FIXTURE,index_base=0)
    current,_=run_sequences(records,MockJudge(),'current')
    full,_=run_sequences(records,MockJudge(),'full')
    assert full['syn_train_A'][1]==0.9
    assert current['syn_train_A'][1]==0.45
    assert full['syn_train_B'][1]==0.45

def test_calibration_no_test_peeking():
    records=load_records(FIXTURE,index_base=0)
    scores,_=run_sequences(records,MockJudge(),'full')
    threshold=calibrate_threshold(records,scores,fpr_budget=0.0)
    assert threshold==0.9
    summary=summarize(records,scores,threshold)
    assert summary['sequence_fpr']==0
    assert summary['timely_recall']==1.0

def test_first_detection_and_empty():
    assert first_intervention([0.2,0.8,0.9],0.8)==1
    assert first_intervention([0.2,0.4],0.8) is None
