# Portable Pi0 Calibration Agent Guide

This file applies to the entire portable calibration repository. It is the
authoritative agent guide if this directory is split into a standalone
repository. Keep kinematic-specific progress notes outside this file; this
file contains durable workflow rules and invariants.

## Purpose

This repository provides a top-to-bottom workflow to:

1. Extract Pi0 calibration samples and exact seed/member metadata from matched
   processed and raw-WF ROOT files.
2. Derive and validate a new kinematic-specific residual calibration.
3. Freeze and approve an immutable calibration package.
4. Apply the approved package to every cluster in every event and write new
   ROOT branches without changing the original branches.

Pi0 selection is a derivation and validation tool. It is never an application
gate.

## Input Contract

Every run/segment requires a matched pair:

```text
nps_production_<run>_<segment>_wf_calib.root
nps_production_<run>_<segment>_wf.root
```

- The `*_wf_calib.root` file is the energy baseline and must contain Hao's
  run-dependent `CALO_calib_Pi0Coef` calibration.
- The matching raw `*_wf.root` file supplies exact seed/member metadata using
  a frozen run DB snapshot. It is not a substitute energy baseline.
- Treat all input ROOT files and collaboration DB/configuration files as
  read-only.
- Never silently substitute intermediate WF files, elastic-calibrated files,
  or products from another kinematic.
- Run lists must be target-pure. Never mix LH2 and LD2 in one calibration.
- Every configured run must belong to exactly one explicit chronological
  period.

## Environment And Secrets

Derivation and ROOT extraction require the Hall C/NPS environment, ROOT,
replay headers, `libDVCS.so`, and the MySQL client. The initialized environment
must define `NPS_SOFT`. Set `NPS_DVCS_LIB` explicitly if the library is not on
`LD_LIBRARY_PATH`.

DB credentials must come from `MYSQL_PWD` or `DB_PASS` in the environment.
Never write credentials into this repository, generated configs, logs, JSON,
scratch directories, or calibration packages.

Use `docs/DB_ACCESS.md` for the maintained endpoint, required tables, private
environment setup, and connection test. Never replace that process by
committing the collaboration password.

Do not commit generated ROOT files, temporary extraction products, Python
build artifacts, or credentials.

## Standard Workflow

Read `README.md`, `pipeline/README.md`, and `docs/METHOD.md` before changing
the calibration logic. The normal sequence is:

```bash
python3 pipeline/build_segment_manifest.py ...
python3 pipeline/run_pipeline.py --config CONFIG
python3 pipeline/run_pipeline.py --config CONFIG \
  --stages snapshots,extract --max-events 10000 --execute
python3 pipeline/run_pipeline.py --config CONFIG \
  --stages extract,derive --max-events -1 --execute
python3 pipeline/promote_package.py ...
python3 pipeline/run_pipeline.py --config CONFIG \
  --stages apply --max-events -1 --execute
```

The first `run_pipeline.py` invocation is a dry-run audit. Use a small smoke
sample before full statistics. Never derive production constants from smoke
outputs.

Large extraction and application campaigns use one segment per batch job.
Generate SWIF2 JSON with `pipeline/build_swif_json.py`, inspect paths and two
representative jobs, then let the user import it. Do not submit or import a
workflow without explicit user instruction.

## Calibration Model

The frozen correction currently has the form:

```text
E_corrected = E_Hao * C_period(E_Hao)
                    * C_run
                    * C_seed
                    * C_lowE(E_Hao)
```

- `C_period(E)` is the period-dependent photon-energy response curve.
- `C_run` is the regularized chronological run scalar.
- `C_seed` is the neutralized seed-block residual correction.
- `C_lowE(E)` is the validated smooth low-energy closure layer.

Do not move factors between layers merely to improve a plot. Preserve the
held-out derivation structure and identify which layer produces an effect.
New functional forms require direct comparison against the existing model.

Exact member deposits may be used to derive or validate corrections, but the
production package must define behavior for every cluster. Unsupported seed
blocks use the package's documented fallback policy; they must not disappear
from output.

## Fit And Validation Rules

- Calibration peak means and sigmas must come from the canonical
  Gaussian-plus-linear-background fit. Arithmetic means or RMS values are QA
  statistics, not substitutes for fitted peak parameters.
- Every fit used for calibration or promotion requires visual inspection.
- Require successful fit status, acceptable covariance/EDM, no important
  parameter pinned at a boundary, and a physically reasonable peak and pull.
- Low-stat runs may use documented chronological support groups. Build groups
  from the nearest valid chronological neighbors and do not bridge large run
  gaps or target/kinematic boundaries.
- Validate with held-out folds and inspect run trends by photon-energy bin.
- Explicitly inspect the `0.6-0.8 GeV` region, detector bulk, and columns
  `0-2` separately.
- Do not promote a method merely because its mean residual map is unusually
  flat. Width, fold stability, energy closure, and independent transfer must
  also improve or remain acceptable.
- Independent production-transfer validation is required before promotion.

Newly derived packages must remain `diagnostic_unreviewed`. Production
application must reject them. Promote by copying with
`pipeline/promote_package.py`; never edit approval metadata in place and never
overwrite an existing release.

## ROOT Application Contract

The application stage must process every event and every cluster, regardless
of whether the event passes Pi0 cuts. It must:

- preserve all original `t_prod` branches unchanged;
- preserve the latest versions of other top-level trees and histograms;
- reject an input that already contains the correction branches;
- reject missing exact seed coverage when strict mode is selected;
- remove a failed or incomplete output rather than leaving a plausible stale
  file;
- write only to a new, descriptive, versioned output directory.

The intended added vectors are:

```text
NPS_haoFinal_clusE
NPS_haoFinal_scale
NPS_haoFinal_curve_scale
NPS_haoFinal_run_scale
NPS_haoFinal_seed_scale
NPS_haoFinal_lowe_scale
NPS_haoFinal_seed_block
NPS_haoFinal_seed_source
```

Do not add event-level Pi0 quantities such as `mgg`, missing mass, selected
cluster indices, run, or segment merely for calibration provenance. Run and
segment already exist in the file/name, and physics quantities belong in
downstream analysis.

## Output Safety

- Never overwrite collaboration inputs or completed ROOT outputs.
- Keep extraction, derivation, release, application, and validation products
  in separate versioned directories.
- Scratch is temporary and may be wiped. Final packages, reports, and ROOT
  products belong on documented persistent storage.
- Preserve absolute input provenance in audit products, but do not bake
  site-specific paths into reusable source defaults.
- Package manifests and hashes must validate whenever a package is loaded.

## Required Checks

Before accepting a code change:

1. Run the unit tests:

   ```bash
   PYTHONPATH=src python3 -m unittest discover -s tests -v
   ```

2. Compile-check modified Python files.
3. Run `git diff --check`.
4. For extraction changes, run matched pair/member smoke extraction and check
   closure and join-loss reporting.
5. For derivation changes, run held-out validation and visually inspect all
   changed fits and run/energy trends.
6. For writer changes, process a representative ROOT sample and verify tree
   entries, preserved objects, exactly the intended new branches, finite
   scales, and strict seed coverage.
7. For batch changes, inspect generated input/output paths and run two jobs
   before broader submission.

## Source Map

- `pipeline/run_pipeline.py`: stage orchestration and dry-run audit.
- `pipeline/build_segment_manifest.py`: exact processed/raw segment matching.
- `pipeline/build_db_snapshots.py`: immutable run DB snapshots.
- `pipeline/extract_inputs.py`: pair and exact member/seed extraction.
- `pipeline/derive/run_kinematic_calibration.py`: central derivation driver.
- `pipeline/promote_package.py`: reviewed immutable release creation.
- `pipeline/apply_dataset.py`: segment-parallel ROOT application.
- `pipeline/build_swif_json.py`: SWIF2 extraction/application manifests.
- `src/nps_pi0_calibration/`: frozen-package loader and correction API.
- `root/apply_root.py`: ROOT application wrapper.
- `root/write_nps_corrected_cluster_tree.C`: all-cluster ROOT writer.
- `docs/PACKAGE_FORMAT.md`: frozen-package schema and integrity rules.
- `docs/DB_ACCESS.md`: safe collaboration DB connection and snapshot setup.

When adding a script or changing a stage contract, update this map and the
corresponding user-facing documentation in the same change.
