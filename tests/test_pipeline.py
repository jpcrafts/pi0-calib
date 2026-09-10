from __future__ import annotations

import sys
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "derive"))

import build_segment_manifest  # noqa: E402
import extract_inputs  # noqa: E402
import promote_package  # noqa: E402
import run_pipeline  # noqa: E402


class PipelineContractTest(unittest.TestCase):
    def test_promotion_copies_and_approves_without_mutating_candidate(self) -> None:
        source = Path(__file__).resolve().parents[1] / "packages" / "x60_4_lh2_v5_smooth_lowe"
        with tempfile.TemporaryDirectory() as text:
            base = Path(text)
            candidate, release = base / "candidate", base / "release"
            shutil.copytree(source, candidate)
            rows = promote_package.read_metadata(candidate / "package_metadata.tsv")
            promote_package.set_metadata(rows, "approval_status", "diagnostic_unreviewed")
            promote_package.write_metadata(candidate / "package_metadata.tsv", rows)
            promote_package.write_manifest(candidate)
            result = subprocess.run(
                [
                    sys.executable, str(PIPELINE / "promote_package.py"),
                    "--candidate", str(candidate), "--output", str(release),
                    "--approved-by", "unit-test", "--note", "reviewed test candidate",
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            candidate_meta = dict(
                (row["key"], row["value"])
                for row in promote_package.read_metadata(candidate / "package_metadata.tsv")
            )
            release_meta = dict(
                (row["key"], row["value"])
                for row in promote_package.read_metadata(release / "package_metadata.tsv")
            )
            self.assertEqual(candidate_meta["approval_status"], "diagnostic_unreviewed")
            self.assertEqual(release_meta["approval_status"], "validated")

    def test_segment_selector_is_exact(self) -> None:
        rows = [
            {"run": "1001", "segment": "0"},
            {"run": "1001", "segment": "1"},
            {"run": "1002", "segment": "0"},
        ]
        self.assertEqual(extract_inputs.selected_rows(rows, 1001, 1), [rows[1]])
        with self.assertRaises(ValueError):
            extract_inputs.selected_rows(rows, None, 1)
        with self.assertRaises(ValueError):
            extract_inputs.selected_rows(rows, 9999, None)

    def test_swif_extract_json_preserves_outputs_and_maps_cache_to_mss(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            base = Path(text)
            manifest = base / "segments.tsv"
            manifest.write_text(
                "run\tsegment\thao_production_root\tauxiliary_raw_wf_root\tstatus\n"
                "1001\t0\t/volatile/example/nps_production_1001_0_wf_calib.root\t"
                "/cache/example/nps_production_1001_0_wf.root\tready\n",
                encoding="ascii",
            )
            config = base / "config.yaml"
            config.write_text(
                "paths:\n"
                f"  segment_manifest: {manifest}\n"
                "  production_pair_dir: /work/example/pairs\n"
                "  member_dir: /work/example/members\n"
                "  work_dir: /work/example/calibration\n"
                "  corrected_root_dir: /volatile/example/corrected\n",
                encoding="ascii",
            )
            output = base / "jobs.json"
            result = subprocess.run(
                [
                    sys.executable, str(PIPELINE / "build_swif_json.py"),
                    "--config", str(config), "--mode", "extract",
                    "--workflow-name", "test_extract", "--environment-script", "/env.sh",
                    "--log-dir", "/logs", "--output", str(output),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            job = json.loads(output.read_text(encoding="ascii"))["jobs"][0]
            self.assertEqual(
                job["inputs"][1]["remote"],
                "/mss/example/nps_production_1001_0_wf.root",
            )
            self.assertEqual(job["outputs"][0]["remote"], "/work/example/pairs/run1001_seg0_production_exact2_pairs.tsv")

    def test_manifest_matches_processed_and_raw_segments(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            base = Path(text)
            hao, raw = base / "hao", base / "raw"
            hao.mkdir()
            raw.mkdir()
            (hao / "nps_production_1001_0_wf_calib.root").touch()
            (raw / "nps_production_1001_0_wf.root").touch()
            runs = {1001}
            self.assertEqual(
                set(build_segment_manifest.discover(hao, build_segment_manifest.HAO_RE, runs)),
                {(1001, 0)},
            )
            self.assertEqual(
                set(build_segment_manifest.discover(raw, build_segment_manifest.RAW_RE, runs)),
                {(1001, 0)},
            )

    def test_full_pipeline_audit_accepts_complete_target_pure_config(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            base = Path(text)
            hao = base / "nps_production_1001_0_wf_calib.root"
            raw = base / "nps_production_1001_0_wf.root"
            hao.touch()
            raw.touch()
            runs = base / "runs.txt"
            runs.write_text("1001\n", encoding="ascii")
            segments = base / "segments.tsv"
            segments.write_text(
                "run\tsegment\thao_production_root\tauxiliary_raw_wf_root\tstatus\n"
                f"1001\t0\t{hao}\t{raw}\tready\n",
                encoding="ascii",
            )
            member = base / "members"
            config = {
                "schema_version": 1,
                "kinematic": {
                    "name": "test", "target": "LH2", "baseline": "hao_pi0_db",
                    "input_mode": "hao_production",
                },
                "paths": {
                    "run_manifest": str(runs), "segment_manifest": str(segments),
                    "db_snapshot_dir": str(base / "db"), "member_dir": str(member),
                    "member_source_dirs": [str(member)], "production_pair_dir": str(base / "pairs"),
                    "work_dir": str(base / "work"), "corrected_root_dir": str(base / "corrected"),
                },
                "periods": [{"name": "p1", "runs": [1001]}],
                "selection": {
                    "timing_center_ns": 0.0, "timing_half_window_ns": 1.0,
                    "min_photon_energy_gev": 0.6, "max_photon_energy_gev": 2.5,
                },
                "low_energy": {
                    "scope": "period", "range_gev": [0.6, 0.9], "profile": "smoothstep",
                    "full_until_gev": 0.8, "unity_at_gev": 0.9,
                },
            }
            rows, summary = run_pipeline.audit(config)
            self.assertEqual(len(rows), 1)
            self.assertIn("status: ready", summary.read_text(encoding="ascii"))

    def test_snapshot_command_uses_configured_database_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            base = Path(text)
            config_path = base / "config.yaml"
            config_path.touch()
            config = {
                "paths": {
                    "run_manifest": str(base / "runs.txt"),
                    "db_snapshot_dir": str(base / "snapshots"),
                    "member_dir": str(base / "members"),
                    "production_pair_dir": str(base / "pairs"),
                    "work_dir": str(base / "work"),
                    "corrected_root_dir": str(base / "corrected"),
                },
                "database": {"host": "db.example", "name": "detector", "user": "reader"},
            }
            (base / "work").mkdir()
            command = run_pipeline.commands(config_path, config, 1, -1, False)["snapshots"]
            self.assertEqual(command[command.index("--host") + 1], "db.example")
            self.assertEqual(command[command.index("--database") + 1], "detector")
            self.assertEqual(command[command.index("--user") + 1], "reader")


if __name__ == "__main__":
    unittest.main()
