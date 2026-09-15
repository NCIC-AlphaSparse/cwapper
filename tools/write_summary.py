#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Emit summary.json in the SAME schema FlagSparse's Python runner emits.

WHY IDENTICAL AND NOT MERELY SIMILAR. The two repos test the same kernels through
two front ends, and the whole point of running both is to compare them. A schema
that differs by a key name or a status spelling forces whoever reads them to
write a converter, and converters silently drop what they do not recognise. So
this matches run_flagsparse_pytest.py's `_flaggems_summary` exactly:

    {"timestamp", "env", "result"}          -- three top-level keys, in this order
    env:      architecture, os_name, os_release, python, torch, flagtree,
              triton, flag_gems
    result:   {operator: {customized, accuracy, performance, labels}}
    accuracy: total, skipped, failed, passed, details, status, duration,
              exit_code, data_file
    perf:     duration, exit_code, data, status, data_file, test_case

ONE DIFFERENCE FROM THE PYTHON SIDE, deliberate: there, accuracy and performance
are two independent pytest runs and each names its own data_file. Here both come
from one benchmark run -- a row is timed only after its result has been checked
against the host fp64 oracle -- so both phases point at the same
<op>_benchmark.json. The numbers are real; what does not exist is a separate
accuracy artifact, and this says so rather than naming one.
    data:     {dtype: {result, details: {shape: {base, gems, speedup}}, speedup}}

Status strings come from that runner's STATUS_TO_FLAGGEMS ("Passed"/"Failed"/
"Skipped"/...), NOT from the C tests' own lowercase vocabulary.

THE MAPPING THAT MATTERS. FlagGems' perf schema is keyed dtype -> shape. Here a
"shape" is a corpus matrix (plus the dense width where the operator sweeps one),
and `base`/`gems` are the vendor and our median in ms -- which is what the Python
side puts there too, so the two files' numbers are comparable cell by cell.

Usage:
    python3 tools/write_summary.py --bench-dir ./bench --out ./bench
"""

import argparse
import datetime as _dt
import json
import pathlib
import platform
import subprocess
import sys

# Verbatim from run_flagsparse_pytest.py -- if that table changes this must too.
STATUS_TO_FLAGGEMS = {
    "PASS": "Passed", "FAIL": "Failed", "SKIP": "Skipped", "TIMEOUT": "Timeout",
    "NO_TESTS": "NotFound", "CRASH": "Error", "NOT_CONFIGURED": "NotFound",
    "Passed": "Passed", "Failed": "Failed", "Skipped": "Skipped",
    "Timeout": "Timeout", "NotFound": "NotFound", "Error": "Error",
}

# The C tests tag dtypes by component width (c32 = complex<float>); FlagGems
# spells them fp32/fp64 and has no complex alias, so complex keeps the C spelling
# rather than being mangled into a real dtype it is not.
DTYPE_ALIASES = {"f32": "fp32", "f64": "fp64", "f16": "fp16", "bf16": "bf16"}


def flaggems_status(s):
    return STATUS_TO_FLAGGEMS.get(str(s or ""), str(s or "Unknown"))


def flag_gems_dtype(d):
    return DTYPE_ALIASES.get(str(d), str(d))


def env_info(rows):
    """The strict FlagGems env block, filled from this machine and the JSONs."""
    backend = rows[0].get("_backend", "") if rows else ""
    arch = rows[0].get("_arch", "") if rows else ""
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}

    # The C API has no torch; device facts come from the adaptor, which is what
    # the `env` block of every benchmark JSON already carries.
    device_name = ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            device_name = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    return {
        "architecture": platform.machine(),
        "os_name": str(os_release.get("ID") or platform.system()).lower(),
        "os_release": str(os_release.get("VERSION_ID") or platform.release()),
        "python": platform.python_version(),
        "torch": {
            # Deliberately blank: this front end does not go through torch. A
            # fabricated version here would make the two summaries look like they
            # ran the same stack when they did not.
            "version": "",
            "cuda_available": backend == "cuda",
            "device_name": device_name,
            "device_count": 1 if device_name else 0,
        },
        "flagtree": None,
        "triton": {"version": "", "has_config": False},
        "flag_gems": {
            "version": "",
            "vendor": backend or "cpu",
            "device": backend or "cpu",
        },
    }


def shape_key(row):
    """The `shape` a metrics cell is filed under: the matrix, plus the dense
    width when the operator sweeps one, so n=8 and n=128 do not collide."""
    parts = [str(row.get("matrix", "?"))]
    for k in ("n", "k"):
        if row.get(k):
            parts.append(f"{k}={int(row[k])}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    files = sorted(args.bench_dir.glob("*_benchmark.json"))
    if not files:
        sys.exit(f"no *_benchmark.json under {args.bench_dir}")

    all_rows, result = [], {}
    for path in files:
        doc = json.loads(path.read_text())
        op = doc.get("operator", path.stem.replace("_benchmark", ""))
        env = doc.get("env", {})
        rows = doc.get("result", [])
        for r in rows:
            r["_backend"] = env.get("backend", "")
            r["_arch"] = str(env.get("arch", ""))
        all_rows.extend(rows)

        # ---- performance: dtype -> shape -> {base, gems, speedup}
        data = {}
        for r in rows:
            if r.get("status") != "ok":
                continue
            dt = flag_gems_dtype(r.get("dtype", "?"))
            entry = data.setdefault(dt, {"result": "Unknown", "details": {},
                                         "speedup": 0.0, "_sp": []})
            entry["details"][shape_key(r)] = {
                "base": float(r.get("baseline_ms") or 0.0),
                "gems": float(r.get("median_ms") or 0.0),
                "speedup": float(r.get("speedup") or 0.0),
            }
            if r.get("speedup"):
                entry["_sp"].append(float(r["speedup"]))
        for dt, entry in data.items():
            sp = entry.pop("_sp")
            # Arithmetic mean, matching what the Python runner reports per dtype.
            entry["speedup"] = sum(sp) / len(sp) if sp else 0.0
            entry["result"] = "Passed" if sp else "Unknown"

        # ---- accuracy: from the operator's own accuracy artifact when the
        # sweep wrote one, so the phase names a file that really holds the
        # ratios. Falls back to the benchmark rows (same numbers, same run) for
        # a JSON produced before that artifact existed.
        acc_path = path.with_name(path.name.replace("_benchmark", "_accuracy"))
        acc_rows, acc_file = rows, path.name
        if acc_path.exists():
            acc_rows = json.loads(acc_path.read_text()).get("result", [])
            acc_file = acc_path.name
        checked = [r for r in acc_rows if r.get("accuracy") in ("pass", "fail")]
        passed = [r for r in acc_rows if r.get("accuracy") == "pass"]
        failed = [r for r in acc_rows if r.get("accuracy") == "fail"]
        skipped = [r for r in acc_rows if r.get("status", "").startswith("skipped")
                   or r.get("status") == "not_supported"]
        details = {}
        if failed:
            details["failed"] = [
                {"name": r.get("name"), "error_ratio": r.get("error_ratio"),
                 "detail": r.get("detail")} for r in failed]
        unsupported = [r for r in acc_rows if r.get("status") == "not_supported"]
        if unsupported:
            details["not_supported"] = [
                {"name": r.get("name"), "detail": r.get("detail")}
                for r in unsupported]

        acc_status = "Failed" if failed else ("Passed" if passed else "Skipped")
        perf_status = "Passed" if any(r.get("speedup") for r in rows) else "Skipped"

        result[op] = {
            "customized": True,
            "accuracy": {
                "total": len(acc_rows),
                "skipped": len(skipped),
                "failed": len(failed),
                "passed": len(passed),
                "details": details,
                "status": flaggems_status(acc_status),
                "exit_code": 1 if failed else 0,
                "duration": 0.0,
                # The file the accuracy verdict actually came from. The Python
                # runner writes a separate <op>/accuracy_result.json because its
                # two phases are two pytest runs; here both phases are read off
                # ONE benchmark run -- every row checks its result against the
                # host fp64 oracle before timing it. Pointing at a conventional
                # path that this repo never writes would be a dangling reference
                # in a file whose whole purpose is to be machine-read.
                "data_file": acc_file,
            },
            "performance": {
                "duration": 0.0,
                "exit_code": 0,
                "data_file": path.name,
                "data": data,
                "status": flaggems_status(perf_status),
                "test_case": "matrix",
            },
            "labels": ["flagsparse", "c_api"],
        }

    summary = {
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": env_info(all_rows),
        "result": dict(sorted(result.items())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(result)} operators, {len(all_rows)} rows)")


if __name__ == "__main__":
    main()
