#!/usr/bin/env python3
"""Freeze Phase 3 thresholds from VALIDATION results, plus the user-approved margin.

Run once, after every validation run and before any test run. Checks that all runs share the
judge, model revision, dtype and FPR budget the plan names, and that every pilot condition is
present exactly once. Refuses to overwrite. Output: IDs-free JSON of thresholds and hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


READOUT_A = "A_continuous_fpr_calibrated"


def build(plan_path: Path, results: list[Path], margin: float, primary_k: int,
          amendment_path: Path = Path("configs/phase3_amendment_1.json"),
          conditions_key: str = "pilot_conditions") -> dict:
    plan = json.loads(plan_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    if amendment["amends_plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError("amendment does not refer to this plan file")
    if not 0 < margin < 1:
        raise ValueError("margin must be between 0 and 1 (timely-recall points as a fraction)")
    judge = plan["judge"]
    out = {"status": "FROZEN_BEFORE_TEST", "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
           "amendment_sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
           "protocol_version": amendment["protocol_version"], "primary_readout": READOUT_A,
           "fpr_budget": plan["protocol"]["fpr_budget"],
           "equivalence_margin_timely_recall": margin, "primary_k": primary_k, "conditions": {}}
    for path in results:
        data = json.loads(path.read_text())
        if data.get("stage") != "validation":
            raise ValueError(f"{path.name}: not a validation-stage result")
        if data.get("protocol_version") != amendment["protocol_version"]:
            raise ValueError(f"{path.name}: protocol version is not {amendment['protocol_version']} (superseded run?)")
        if data["status"] == "SMOKE_TEST_ONLY":
            raise ValueError(f"{path.name}: mock result cannot be frozen")
        if (data["paper_prompt"] != judge["paper_prompt"]
                or data["model"]["model_revision"] != judge["model_revision"]
                or judge["dtype"] not in str(data["model"]["dtype"])):
            raise ValueError(f"{path.name}: judge does not match the plan")
        if data["fpr_budget"] != out["fpr_budget"]:
            raise ValueError(f"{path.name}: FPR budget differs from the plan")
        if data["mode"] in ("dense", "hybrid") and \
                data["dense_encoder"]["revision"] != plan["dense_encoder"]["revision"]:
            raise ValueError(f"{path.name}: dense encoder revision differs from the plan")
        name = data["condition"]
        if name in out["conditions"]:
            raise ValueError(f"{name}: more than one validation result")
        a = data["readouts"][READOUT_A]
        out["conditions"][name] = {
            "threshold": a["threshold"],
            "validation_sequence_fpr": a["summary"]["sequence_fpr"],
            "validation_timely_recall": a["summary"]["timely_recall"],
            "source_file": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    missing = sorted(set(plan[conditions_key]) - set(out["conditions"]))
    if missing:
        raise ValueError(f"missing validation results for {missing}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("configs/phase3_plan.json"))
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--primary-k", type=int, required=True)
    p.add_argument("--amendment", type=Path, default=Path("configs/phase3_amendment_1.json"))
    p.add_argument("--out", type=Path, default=Path("configs/phase3_thresholds.json"))
    p.add_argument("results", type=Path, nargs="+")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; thresholds are frozen. Do not regenerate.")
    frozen = build(a.plan, a.results, a.margin, a.primary_k, a.amendment)
    a.out.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["threshold"] for k, v in frozen["conditions"].items()}, indent=1))
