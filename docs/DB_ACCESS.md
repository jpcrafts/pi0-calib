# NPS Database Access

The snapshot stage reads the run-dependent NPS replay constants needed to
reconstruct exact seed/member metadata. It does not modify the database.

## Connection

The current collaboration endpoint is:

```text
host:     jmysql.jlab.org
database: nps
user:     dvcs
```

These defaults are recorded in `pipeline/config/example.yaml`. They can be
changed there for a new deployment or supplied directly to
`pipeline/build_db_snapshots.py` with `--host`, `--database`, and `--user`.

The password is deliberately not distributed in source control. Obtain the
current read-only `dvcs` credential through the NPS calibration/replay
maintainer. A person without collaboration DB authorization must not bypass
that access process.

## Private Environment Setup

Copy the supplied template outside the repository and restrict its
permissions:

```bash
mkdir -p ~/.config/nps_pi0_calibration
cp pipeline/config/db_environment.example \
  ~/.config/nps_pi0_calibration/env.sh
chmod 600 ~/.config/nps_pi0_calibration/env.sh
```

Edit the private copy, set the supplied password, and source it before running
the snapshot stage:

```bash
source ~/.config/nps_pi0_calibration/env.sh
```

The code accepts either `MYSQL_PWD` or `DB_PASS`. `MYSQL_PWD` is passed only to
the `mysql` subprocess environment; it is not written to snapshot files,
commands, logs, packages, or SWIF JSON.

## Verify Access

After initializing the JLab environment and sourcing the private credential:

```bash
mysql --protocol=TCP \
  -h "${NPS_DB_HOST:-jmysql.jlab.org}" \
  -u "${NPS_DB_USER:-dvcs}" \
  -D "${NPS_DB_NAME:-nps}" \
  -N -B -e 'SELECT 1'
```

Expected output is `1`. A network error generally means the machine cannot
reach the JLab database. An access-denied error means the credential or DB
authorization must be resolved before calibration.

## Frozen Tables

For every requested run, `pipeline/build_db_snapshots.py` freezes the matching
records from:

```text
BEAM_param_Energy
SIMU_param_HMSmomentum
SIMU_param_HMSangle
TARGET_param_Amu
CALO_geom_Dist
CALO_geom_Yaw
CALO_calib_TimeOffset
CALO_calib_Pi0Coef
CALO_calib_ElasCoef
CALO_flag_MaskBlock
```

Snapshots include values and non-secret row metadata for provenance. Complete
existing snapshots are reused. Partial snapshots fail rather than silently
mixing DB versions; rebuild them explicitly with `--overwrite`.

## Run The Snapshot Stage

```bash
python3 pipeline/run_pipeline.py --config /path/to/config.yaml \
  --stages snapshots --execute
```

Inspect `db_snapshots/snapshot_summary.tsv` and retain the complete snapshot
directory with the calibration derivation record.
