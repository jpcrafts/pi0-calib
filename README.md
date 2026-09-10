# NPS Pi0 Cluster-Energy Correction

Top-to-bottom workflow for deriving residual NPS Pi0 calibration constants
from processed waveform runs, freezing an audited package, and writing
corrected cluster-energy branches into production ROOT files. The frozen
package application layer has no Python dependencies outside the standard
library.

## Scope

Two paths are included:

- `pipeline/`: derive a new kinematic-specific package from matched processed
  and raw-WF inputs, then apply it to every cluster in every segment.
- `src/nps_pi0_calibration/` and `root/`: apply an existing frozen package.

A package remains valid only for its named kinematic, target, baseline energy
definition, and explicit run list.

Current included package:

- `packages/x60_4_lh2_v5_smooth_lowe`: validated x60_4 LH2 correction starting
  from Hao's `CALO_calib_Pi0Coef` cluster energies.

The correction acts on every cluster, independent of whether an event is later
identified as a Pi0:

```text
E_corrected = E_input * C_period(E_input)
                      * C_run
                      * C_seed
                      * C_lowE(E_input)
```

The Pi0 sample derives and validates the factors; it is not an application
gate.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Install derivation dependencies when fitting a new package:

```bash
python3 -m pip install -e '.[derive]'
```

ROOT and NPS replay libraries remain experiment-environment dependencies.

## Derive And Apply A New Package

See the [method definition](docs/METHOD.md) and
[full pipeline guide](pipeline/README.md). Database credentials and connection
testing are documented in [NPS database access](docs/DB_ACCESS.md). Minimal
sequence:

```bash
export PI0_WORK=/path/to/new_kinematic_work
export HAO_ROOT_DIR=/path/to/processed/wf_calib/files
export RAW_WF_ROOT_DIR=/path/to/matching/raw/wf/files

python3 pipeline/build_segment_manifest.py \
  --runs "$PI0_WORK/runs.txt" \
  --hao-dir "$HAO_ROOT_DIR" --raw-wf-dir "$RAW_WF_ROOT_DIR" \
  --output "$PI0_WORK/segments.tsv"

cp pipeline/config/example.yaml "$PI0_WORK/config.yaml"
# Edit periods, selection, fit policy, and production validation runs.

python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml"
python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml" \
  --stages snapshots,extract,derive --workers 4 --execute
```

Default run is dry-run audit. `--max-events 10000` provides smoke processing;
omit it or use `-1` for all events. Review and promote the diagnostic package
as described in the pipeline guide, set `paths.package_dir` to that immutable
release, then run the `apply` stage.

## Apply To TSV Or CSV

Input requires one row per cluster and fields for run, input energy in GeV,
and seed block:

```bash
nps-pi0-correct \
  --package packages/x60_4_lh2_v5_smooth_lowe \
  --input examples/clusters.tsv \
  --output corrected_clusters.tsv
```

Output preserves input columns and adds period, each scale component, total
scale, and corrected energy. Field names are configurable:

```bash
nps-pi0-correct --package PACKAGE --input INPUT --output OUTPUT \
  --run-field run --energy-field energy_gev --seed-field seed_block
```

Existing output files are rejected unless `--overwrite` is supplied. Input
and output paths may never be the same.

Unsupported runs fail. Missing or unsupported seed blocks default to unity for
the spatial layer while all non-spatial factors still apply. Use
`--missing-seed error` when exact seed coverage is required.

## Python API

```python
from nps_pi0_calibration import FrozenCalibration

calibration = FrozenCalibration.load("packages/x60_4_lh2_v5_smooth_lowe")
result = calibration.correct(run=4253, energy_gev=0.75, seed_block=523)
print(result.corrected_energy_gev)
print(result.total_scale)
```

## Apply To NPS ROOT

The optional ROOT adapter clones `t_prod`, preserves existing branches, and
adds corrected cluster vectors:

```bash
python3 root/apply_root.py \
  --input-root nps_production_4253_0_wf_calib.root \
  --output-root corrected_4253_0.root \
  --package packages/x60_4_lh2_v5_smooth_lowe \
  --run 4253 --segment 0 \
  --sidecar nps_production_4253_0_wf_calib.production_members.tsv
```

Existing output ROOT files are rejected unless `--overwrite` is supplied.
Inputs already containing correction branches are rejected to prevent applying
the package twice. By default, all other top-level trees and histograms are
copied from the input file; only `t_prod` is replaced with its cloned tree plus
new branches. `--tree-only` is available for small smoke tests, not production.

Required input tree and branches:

- tree `t_prod`
- scalar `g.runnum`
- vectors `NPS.prod.clusE`, `NPS.prod.clusX`, `NPS.prod.clusY`

Added vectors:

- `NPS_haoFinal_clusE`
- `NPS_haoFinal_scale`
- `NPS_haoFinal_curve_scale`
- `NPS_haoFinal_run_scale`
- `NPS_haoFinal_seed_scale`
- `NPS_haoFinal_lowe_scale`
- `NPS_haoFinal_seed_block`
- `NPS_haoFinal_seed_source`

Exact seed sidecar fields are `t_entry`, `prod_cluster_slot`, `seed_block`, and
`member_index`. Only `member_index == 0` is read. The ROOT adapter also writes
`hao_final_metadata` with coverage counters and deletes output if strict
coverage fails.

The included sidecar extractor depends on Hall C replay libraries, run DB
snapshots, timing offsets, masks, geometry, and raw-WF files. Run the full
`pipeline/` workflow in that experiment environment to produce exact sidecars.
For application-only use without sidecars, explicitly choose
`--missing-seed centroid` or `--missing-seed identity` and treat the result as
approximate.

## Validate

```bash
python3 -m unittest discover -s tests -v
python3 -m nps_pi0_calibration.cli \
  --package packages/x60_4_lh2_v5_smooth_lowe \
  --input examples/clusters.tsv --output /tmp/corrected.tsv
```

Package hashes are checked automatically. See [package format](docs/PACKAGE_FORMAT.md).

## Calibration Requirements

Do not reuse included x60_4 numbers for another setting. New calibration needs:

1. Target-filtered run manifest and explicit chronological periods.
2. Hao-calibrated production cluster energies as baseline.
3. Exact seed/member sidecars from matching raw-WF replay.
4. Pi0 exact-two selection and canonical Gaussian-plus-linear-background fits.
5. Held-out derivation of period energy curve, run scalar, neutralized seed
   residual, and low-energy closure layer.
6. Independent production transfer validation before freezing a package.

Derivation outputs include fit tables and PDFs. Every production fit still
requires visual inspection before package promotion. Numerical fit status
alone is insufficient.

## License

This project is distributed under the [BSD 3-Clause License](LICENSE).
