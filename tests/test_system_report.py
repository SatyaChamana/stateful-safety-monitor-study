import importlib.util
from pathlib import Path


def test_report_no_secrets_or_paths():
    file=Path(__file__).resolve().parents[1]/'scripts/system_report.py'
    spec=importlib.util.spec_from_file_location('system_report',file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report=module.report()
    assert 'python' in report and 'gpu' in report
    assert not any('token' in key.lower() or 'username' in key.lower() for key in report.keys())
    assert isinstance(report['gpu']['cuda_available'],bool)
