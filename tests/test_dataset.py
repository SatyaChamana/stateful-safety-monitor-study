import pytest
from pathlib import Path
from stateful_safety.dataset import load_records,parse_row,audit

FIXTURE=Path(__file__).resolve().parents[1]/"fixtures/synthetic_fixture.jsonl"

def test_fixture_schema():
    recs=load_records(FIXTURE,index_base=0)
    assert len(recs)==4
    assert audit(recs)['record_count']==4
    assert all(r.harmful_index is None for r in recs if r.label=='benign')

def test_explicit_index_base_is_required():
    row={'id':'a','label':'harmful','steps':['x','y'],'harmful_index':1}
    with pytest.raises(ValueError,match='Declare --index-base'):
        parse_row(row)
    assert parse_row(row,index_base=1).harmful_index==0
    assert parse_row(row,index_base=0).harmful_index==1

def test_no_harm_index_on_benign():
    with pytest.raises(ValueError):
        parse_row({'id':'z','label':'benign','steps':['x'],'harmful_index':0},index_base=0)

def test_bad_steps_rejected():
    with pytest.raises(ValueError):
        parse_row({'id':'z','label':'harmful','steps':'not a list','harmful_index':0},index_base=0)
