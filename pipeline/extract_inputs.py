#!/usr/bin/env python3
"""Extract exact-two production pairs and exact seed/member sidecars."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def expanded(path: str) -> Path:
    import os

    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def load_rows(config: dict) -> list[dict[str, str]]:
    path = expanded(config["paths"]["segment_manifest"])
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    bad = [row for row in rows if row.get("status", "ready") != "ready"]
    if bad:
        raise RuntimeError(f"segment manifest contains {len(bad)} non-ready rows")
    return rows


def selected_rows(
    rows: list[dict[str, str]], run: int | None, segment: int | None
) -> list[dict[str, str]]:
    if segment is not None and run is None:
        raise ValueError("--segment requires --run")
    selected = [
        row
        for row in rows
        if (run is None or int(row["run"]) == run)
        and (segment is None or int(row["segment"]) == segment)
    ]
    if not selected:
        suffix = "all rows" if run is None else f"run={run} segment={segment}"
        raise ValueError(f"segment manifest has no match for {suffix}")
    return selected


def root_call(macro: Path, expression: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["root", "-l", "-b", "-q", f"{macro}({expression})"],
        check=False,
        text=True,
        capture_output=True,
    )


def extract_pair(
    row: dict[str, str], config: dict, overwrite: bool, max_events: int
) -> dict[str, object]:
    run, segment = int(row["run"]), int(row["segment"])
    output_dir = expanded(config["paths"]["production_pair_dir"])
    output = output_dir / f"run{run}_seg{segment}_production_exact2_pairs.tsv"
    if output.is_file() and output.stat().st_size and not overwrite:
        return {"kind": "pair", "run": run, "segment": segment, "status": "existing", "output": output}
    output.parent.mkdir(parents=True, exist_ok=True)
    selection = config["selection"]
    values = [
        str(run), str(segment), f'"{Path(row["hao_production_root"]).resolve()}"', f'"{output}"',
        str(selection["min_photon_energy_gev"]), str(selection["max_photon_energy_gev"]),
        str(selection["timing_half_window_ns"]), str(selection["timing_center_ns"]), str(max_events),
    ]
    result = root_call(HERE / "root" / "extract_production_exact2_pairs.C", ",".join(values))
    ok = result.returncode == 0 and output.is_file() and output.stat().st_size > 0
    return {
        "kind": "pair", "run": run, "segment": segment, "status": "ok" if ok else "failed",
        "output": output, "detail": (result.stderr or result.stdout)[-1000:] if not ok else "",
    }


def extract_member(
    row: dict[str, str], config: dict, overwrite: bool, max_events: int
) -> dict[str, object]:
    run, segment = int(row["run"]), int(row["segment"])
    output_dir = expanded(config["paths"]["member_dir"])
    output = output_dir / f"nps_production_{run}_{segment}_wf_calib.production_members.tsv"
    if output.is_file() and output.stat().st_size and not overwrite:
        return {"kind": "member", "run": run, "segment": segment, "status": "existing", "output": output}
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = expanded(config["paths"]["db_snapshot_dir"]) / f"run_{run}"
    raw_root = Path(row["auxiliary_raw_wf_root"]).resolve()
    with tempfile.TemporaryDirectory(prefix=f"pi0_member_{run}_{segment}_") as temp_text:
        temp = Path(temp_text)
        macro = temp / "prodTree_sidecar.C"
        shutil.copy2(HERE / "root" / "prodTree_sidecar.C", macro)
        (temp / "root").mkdir()
        script = (
            f'.L {macro}+\n'
            f'prodTree_sidecar({run},{segment},"{raw_root.parent}","{temp / "root"}",'
            f'"{output_dir}",{max_events},"snapshot","{snapshot}",false);\n.q\n'
        )
        env = os.environ.copy()
        nps_soft = Path(env.get("NPS_SOFT", ""))
        if not nps_soft.is_dir():
            return {
                "kind": "member", "run": run, "segment": segment, "status": "failed",
                "output": output,
                "detail": "NPS_SOFT is unset or invalid; initialize the NPS replay environment",
            }
        include_parts = [str(nps_soft)]
        if env.get("ROOT_INCLUDE_PATH"):
            include_parts.append(env["ROOT_INCLUDE_PATH"])
        env["ROOT_INCLUDE_PATH"] = os.pathsep.join(include_parts)
        library_parts = [str(nps_soft)]
        if env.get("LD_LIBRARY_PATH"):
            library_parts.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_parts)
        env.setdefault("NPS_DVCS_LIB", str(nps_soft / "libDVCS.so"))
        result = subprocess.run(
            ["root", "-l", "-b"], input=script, check=False, text=True,
            capture_output=True, cwd=temp, env=env,
        )
    ok = result.returncode == 0 and output.is_file() and output.stat().st_size > 0
    return {
        "kind": "member", "run": run, "segment": segment, "status": "ok" if ok else "failed",
        "output": output, "detail": (result.stderr or result.stdout)[-1000:] if not ok else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--kind", choices=("pairs", "members", "both"), default="both")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--run", type=int, help="Process only this run (batch use).")
    parser.add_argument("--segment", type=int, help="Process only this segment; requires --run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hao-root-override", type=Path)
    parser.add_argument("--raw-wf-root-override", type=Path)
    parser.add_argument(
        "--output-dir-override", type=Path,
        help="Write pair/member products here (single-segment batch jobs).",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = selected_rows(load_rows(config), args.run, args.segment)
    if any((args.hao_root_override, args.raw_wf_root_override, args.output_dir_override)):
        if len(rows) != 1:
            parser.error("input/output overrides require exactly one selected segment")
        if args.hao_root_override:
            rows[0]["hao_production_root"] = str(args.hao_root_override.resolve())
        if args.raw_wf_root_override:
            rows[0]["auxiliary_raw_wf_root"] = str(args.raw_wf_root_override.resolve())
        if args.output_dir_override:
            override = str(args.output_dir_override.resolve())
            config["paths"]["production_pair_dir"] = override
            config["paths"]["member_dir"] = override
    jobs = []
    for row in rows:
        if args.kind in {"pairs", "both"}:
            jobs.append((extract_pair, row))
        if args.kind in {"members", "both"}:
            jobs.append((extract_member, row))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(function, row, config, args.overwrite, args.max_events)
            for function, row in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['kind']} {result['run']}/{result['segment']} {result['status']}", flush=True)
    if args.report:
        report = args.report
    elif args.run is not None and args.segment is not None:
        report = expanded(config["paths"]["work_dir"]) / (
            f"extraction_report_run{args.run}_seg{args.segment}.tsv"
        )
    else:
        report = expanded(config["paths"]["work_dir"]) / "extraction_report.tsv"
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "run", "segment", "status", "output", "detail"]
    with report.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: (row["kind"], row["run"], row["segment"])))
    failures = [row for row in results if row["status"] == "failed"]
    print(f"jobs={len(results)} failures={len(failures)} report={report}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
