import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_eval_command(tmp_path):
    dest=tmp_path/'result.json'
    proc=subprocess.run([sys.executable,'-m','stateful_safety','evaluate',
                         '--validation',str(ROOT/'fixtures/synthetic_validation_fixture.jsonl'),
                         '--test',str(ROOT/'fixtures/synthetic_test_fixture.jsonl'),
                         '--index-base','0','--mode','full','--backend','mock','--out',str(dest)],
                        capture_output=True,text=True,cwd=ROOT,
                        env={**os.environ,"PYTHONPATH":str(ROOT/"src")})
    assert proc.returncode==0,proc.stderr
    data=json.loads(dest.read_text())
    assert data['threshold_fitted_on']=='VALIDATION_ONLY'
    assert data['test_summary']['n_harmful']==1
    assert data['test_summary']['timely_recall']==1.0
    assert data['status']=='SMOKE_TEST_ONLY'
