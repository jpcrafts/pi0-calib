# Frozen Package Format

All files are tab-separated text. Numerical application uses the original
input cluster energy for both the period curve lookup and low-energy profile.

## Required Files

### `package_metadata.tsv`

Key/value package identity and policy. Required keys:

- `schema_version`
- `kinematic`
- `target`
- `baseline`; currently only `hao_pi0_db`
- `lowe_damping`
- `approval_status`; production application requires `validated`

Optional policy keys default to legacy behavior:

- `lowe_scope`: `global` or `period`
- `lowe_profile`: `hard_window` or `smoothstep`
- `lowe_full_until`
- `lowe_unity_at`

### `run_period_lut.tsv`

Maps every supported `run` to one `period`. Unlisted runs are rejected.

### `period_photon_curve.tsv`

Energy knots per period. Required columns are `period`, `energy_gev`, and
`log_energy_scale`. Application linearly interpolates log scale and exponentiates.
Values outside the knot range use the nearest endpoint.

### `run_scalar_lut.tsv`

Required columns are `run`, `fit_ok`, and `applied_energy_scale`. Every listed
run must have exactly one valid scale.

### `seed_block_scale.tsv`

Required columns are `seed_block`, `support_pass`, and `applied_energy_scale`.
Unsupported blocks use unity only under the identity missing-seed policy.

### `lowe_photon_scale.tsv`

Rows are selected where `fold == full_sample` and `damping` equals metadata
`lowe_damping`. Required columns include `emin`, `emax`, and
`applied_photon_energy_scale`. Period scope additionally requires `period`.

## Optional Files

### `lowe_shape_tilt.tsv`

Adds a continuous log-scale shape correction. When present, it must contain
one `full_sample` row with coefficient and energy-domain columns.

### `package_manifest.tsv`

Lists `file`, `bytes`, and `sha256`. The Python loader verifies listed files at
load time. The manifest does not list itself.

## Baseline Contract

`hao_pi0_db` means input cluster energy already uses the run's
`CALO_calib_Pi0Coef`. Applying this package to elastic-calibrated or raw
amplitudes is invalid. Package factors are residual corrections, not absolute
ADC-to-energy coefficients.
