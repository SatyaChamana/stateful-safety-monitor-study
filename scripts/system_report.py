#!/usr/bin/env python3
"""Write a reproducible environment report without secrets, paths, or credentials."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ("torch", "transformers", "sentence-transformers", "faiss-cpu",
            "numpy", "pandas", "scikit-learn", "pytest", "datasets")


def report() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"cuda_available": False, "device_count": 0, "device_names": [], "cuda_runtime": None}
    try:
        import torch
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["device_count"] = torch.cuda.device_count() if gpu["cuda_available"] else 0
        gpu["cuda_runtime"] = torch.version.cuda
        if gpu["cuda_available"]:
            gpu["device_names"] = [torch.cuda.get_device_name(i) for i in range(gpu["device_count"])]
    except ImportError:
        pass
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine_arch": platform.machine(),
        "packages": packages,
        "gpu": gpu,
        "note": "This report does not query or export tokens, usernames, file paths or environment secrets.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/system_report.json")
    args = parser.parse_args()
    info = report()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
