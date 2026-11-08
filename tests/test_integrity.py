from pathlib import Path
from stateful_safety.dataset import load_records
from stateful_safety.integrity import compare_splits
from stateful_safety.schema import Sequence

ROOT=Path(__file__).resolve().parents[1]

def test_split_integrity_fixture():
    val=load_records(ROOT/'fixtures/synthetic_validation_fixture.jsonl',index_base=0)
    test=load_records(ROOT/'fixtures/synthetic_test_fixture.jsonl',index_base=0)
    result=compare_splits(val,test)
    assert result['valid_for_initial_evaluation']
    assert result['family_overlap_unknown']

def test_duplicate_text_detected_despite_new_id():
    first=Sequence('a','validation','synthetic','benign',('some text','other text'),None)
    second=Sequence('b','test','synthetic','benign',('some text','other text'),None)
    result=compare_splits([first],[second])
    assert result['overlap_exact_step_sequences']==1
    assert not result['valid_for_initial_evaluation']
