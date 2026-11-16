#!/usr/bin/env python3
"""Run Gemini extraction on documents in samples/ and save final JSON in results/."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc"}
RESULT_KEYS = {
    "name of work",
    "tender no",
    "ato_criteria",
    "similar_work_definition",
    "documents_to_be_submitted",
    "tech_evaluation_mode",
}


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
        help="Gemini model ID (defaults to GEMINI_MODEL in .env)",
    )
    parser.add_argument("--workers", type=int, default=1, help="documents processed at once")
    args = parser.parse_args()

    sample_dir = ROOT / "samples"
    result_dir = ROOT / "results"
    if not sample_dir.is_dir():
        parser.error(f"sample folder not found: {sample_dir}")

    files = sorted(
        (path for path in sample_dir.iterdir()
         if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not files:
        parser.error(f"no PDF, DOCX, or DOC files found in {sample_dir}")

    stems = [path.stem for path in files]
    if len(stems) != len(set(stems)):
        parser.error("sample files must have unique names (same filename stem).")

    result_dir.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    command = [
        sys.executable,
        str(ROOT / "extract_tender.py"),
        *(str(path) for path in files),
        "--model", args.model,
        "--workers", str(args.workers),
        "--out", str(result_dir),
        "--artifacts-out", str(ROOT / "output"),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)

    failures = []
    if completed.returncode:
        failures.append(f"extractor exited with status {completed.returncode}")

    for path in files:
        result_path = result_dir / f"{path.stem}.json"
        if not result_path.is_file() or result_path.stat().st_mtime_ns < started_ns:
            failures.append(f"{path.name}: no new final result at {result_path}")
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{path.name}: invalid result JSON ({error})")
            continue
        if set(result) != RESULT_KEYS:
            failures.append(f"{path.name}: result has an unexpected JSON schema")
            continue
        print(f"{path.name} -> {result_path.relative_to(ROOT)}")

    if failures:
        print("Pipeline finished with errors:", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print(f"Pipeline complete: {len(files)} results saved in {result_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
