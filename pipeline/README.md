# Full Calibration Pipeline

## Data Contract

Exact workflow needs matched files for every run and segment:

```text
nps_production_<run>_<segment>_wf_calib.root
nps_production_<run>_<segment>_wf.root
```

The first file supplies cluster energies already reconstructed with the run's
Hao `CALO_calib_Pi0Coef`. The raw-WF file supplies exact seed and member
metadata using a frozen run DB snapshot. Event entry and production cluster
slot join the two products.

Do not mix LH2 and LD2. Do not use elastic-baseline energy in place of Hao
production energy. Do not infer exact seed blocks from corrected centroids.

## Environment

Required:

- Python 3.9+
- packages in `requirements.txt`
- ROOT 6
- NPS replay headers and `libDVCS.so`
- `mysql` client for DB snapshot stage
- `MYSQL_PWD` or `DB_PASS` in environment

See `docs/DB_ACCESS.md` for the endpoint, required tables, private credential
setup, and a connection test. The credential itself is intentionally not
stored in this repository.

Initialize collaboration replay environment before extraction. If `libDVCS.so`
is not discoverable through `LD_LIBRARY_PATH`, set:

```bash
export NPS_DVCS_LIB=/absolute/path/to/libDVCS.so
```

The initialized environment must define `NPS_SOFT`. The extractor adds that
directory to both `ROOT_INCLUDE_PATH` and `LD_LIBRARY_PATH`; this is required
to compile `TCaloEvent.h` and load `libDVCS.so` reliably.

## 1. Select Runs

Create target-pure `runs.txt`, one run per line. Run selection is authoritative;
directory discovery never decides target membership.

## 2. Match Segments

```bash
python3 pipeline/build_segment_manifest.py \
  --runs "$PI0_WORK/runs.txt" \
  --hao-dir "$HAO_ROOT_DIR" \
  --raw-wf-dir "$RAW_WF_ROOT_DIR" \
  --output "$PI0_WORK/segments.tsv"
```

Command fails on duplicate or missing matched inputs.

## 3. Configure Kinematic

Copy `pipeline/config/example.yaml`. Set:

- target and kinematic name
- all paths
- chronological run periods
- measured timing center/window
- photon-energy selection
- canonical fit quality thresholds
- run support/fallback policy
- seed support/shrinkage and identity edge columns
- low-energy profile
- independent production validation runs

Every configured run must map to exactly one period.
`execution.max_pair_member_join_loss_fraction` is an aggregate hard gate; the
default `0.001` permits at most 0.1% unmatched selected pairs and reports the
exact loss. Tighten it only after confirming production/raw entry alignment.

## 4. Audit And Plan

```bash
python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml"
```

This validates files, target/run coverage, period assignment, and output
contracts. It writes `pipeline_input_audit.txt` and `pipeline_commands.sh`.
Nothing executes without `--execute`.

## 5. Smoke Test

```bash
python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml" \
  --stages snapshots,extract --workers 2 --max-events 10000 --execute
```

Inspect extraction report and exact pair/member closure. Smoke products must
not be used to derive final constants. Remove/version them, then rerun full
statistics with separate output paths.

## 6. Full Derivation

```bash
python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml" \
  --stages extract,derive --workers 4 --max-events -1 --execute
```

Derivation order:

1. Convert matched exact-two pairs plus member sidecars into compact samples.
2. Fit period-dependent photon-energy curves.
3. Fit Gaussian-core run scalars with chronological support fallback.
4. Derive neutralized seed-block residual scales.
5. Derive low-energy closure scale and configured smooth transition.
6. Export immutable `frozen_package/` with source config and run LUT.

New exports are always `diagnostic_unreviewed` and cannot be applied by the
production driver. After all required review succeeds, promote by copying into
a new versioned package. The candidate remains unchanged:

```bash
python3 pipeline/promote_package.py \
  --candidate "$PI0_WORK/calibration/frozen_package" \
  --output "$PI0_WORK/releases/my_kinematic_v1" \
  --approved-by analyst_name \
  --note "Reviewed held-out fits, low-E closure, and production transfer."
```

Set `paths.package_dir` in the application config to that release directory.
Promotion records UTC time and candidate provenance, then rebuilds and checks
the package manifest. Existing release directories are never overwritten.

Fit peak means and sigmas come from canonical Gaussian-plus-linear-background
fits. Arithmetic widths are not calibration decisions.

## 7. Required Review

Before application:

- inspect every flagged or accepted peak fit
- inspect run trends by photon-energy bin
- inspect both held-out folds
- verify low-energy `0.6-0.8 GeV` closure
- inspect bulk and columns `0-2` separately
- reject suspiciously flat spatial maps without width improvement
- verify independent production-transfer sample

Config values are starting policy, not universal detector constants.

## 8. Write Corrected ROOT Files

```bash
python3 pipeline/run_pipeline.py --config "$PI0_WORK/config.yaml" \
  --stages apply --workers 4 --max-events -1 --execute
```

Every event and every cluster is processed. Pi0 selection is used only during
derivation. Original branches remain unchanged; new `NPS_haoFinal_*` vectors
carry corrected energy and factor provenance. Strict exact-seed coverage is
the default.

## Restart And Overwrite Policy

Existing extraction/application products are reused. Pass `--overwrite` only
for a deliberate versioned rerun. Never point corrected output directory at
the input directory.

## Batch Processing

`pipeline_commands.sh` contains stable stage commands. Large extraction and
application should map one segment per batch job. Both segment-parallel stages
accept explicit selectors:

```bash
python3 pipeline/extract_inputs.py --config "$PI0_WORK/config.yaml" \
  --kind both --run 1001 --segment 0 --max-events -1

python3 pipeline/apply_dataset.py --config "$PI0_WORK/config.yaml" \
  --run 1001 --segment 0 --max-events -1
```

The central derivation remains one job after all extraction products land.
Batch-site import format is separate from physics logic; validate two segment
jobs before full submission. Segment-selected commands write uniquely named
reports, so independently scheduled jobs do not overwrite each other's QA.

Observed safe starting requests on the JLab farm are 10-12 GB RAM and one CPU
for member extraction/application, and at least 4 GB RAM for central fitting.
Measure a representative full segment before reducing those requests.

For SWIF2 sites, generate import JSON directly from the audited manifest:

```bash
python3 pipeline/build_swif_json.py --config "$PI0_WORK/config.yaml" \
  --mode extract --workflow-name my_kin_extract \
  --environment-script /path/to/replay_environment.sh \
  --log-dir /path/to/farm/logs --output "$PI0_WORK/extract.json"

swif2 import -file "$PI0_WORK/extract.json"
```

After derivation, review, promotion, and setting `paths.package_dir`, generate
the application workflow with `--mode apply`. The default raw input mapping is
`/cache/` to `/mss/` so SWIF prestages raw-WF files; both prefixes are
configurable. Inspect generated input/output paths before import.
