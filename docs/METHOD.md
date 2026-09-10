# Calibration Method

## Baseline

The supported derivation starts from processed production cluster energies
already reconstructed with Hao's run-dependent `CALO_calib_Pi0Coef` values.
These are residual corrections to that baseline. They are not replacement
ADC-to-GeV constants and must not be applied to elastic-baseline or raw ADC
values.

Exact seed/member metadata comes from the matching raw-WF file and a frozen DB
snapshot. Production entry number plus flattened production-cluster slot is the
join key. The join-loss fraction is measured and gated.

## Calibration Sample

Calibration uses exact-two-cluster Pi0 candidates with the configured timing
and photon-energy cuts. Target identity, run list, chronological periods, and
known bad calibration runs are explicit configuration inputs. Directory
discovery never decides target membership.

Peak locations and widths used for decisions come from the canonical
Gaussian-plus-linear-background fit. Arithmetic means and RMS values may be QA
statistics but do not define calibration constants.

## Applied Layers

For each cluster with baseline energy `E`:

```text
E_corrected = E * C_period(period(run), E)
                * C_run(run)
                * C_seed(seed_block)
                * C_lowE(period(run), E)
```

### Period photon-energy curve

`C_period` is fitted from photon-energy cells within each configured
chronological period. The frozen package stores energy knots in log-scale
space. Application linearly interpolates log scale, then exponentiates it.
Endpoint values are held constant outside the knot range.

### Run scalar

`C_run` removes remaining run-wide peak displacement after the period curve.
It uses accepted run fits where statistics permit. Weak runs use constrained
chronological support from nearby runs, subject to configured gap, distance,
and support-size limits. Updates are damped and clipped.

### Seed-block residual

`C_seed` is derived after the period and run layers. Per-block updates require
minimum support, shrink toward unity at low support, and are clipped. Configured
edge columns remain identity/QA regions. This layer is spatially centered so it
cannot absorb the global energy or run scale.

### Low-energy closure

`C_lowE` addresses residual closure in the configured low-energy range. It may
be global or period-specific. The supported smoothstep profile is fully active
through `full_until_gev`, transitions continuously, and is unity at
`unity_at_gev`.

## Validation Contract

Each fit stage emits held-out segment-parity products and full-sample products.
Promotion requires visual fit review plus numerical checks:

- Pi0 peak closure and fitted sigma, pooled and by run
- run trends in photon-energy bins, especially `0.6-0.8 GeV`
- bulk block residuals and edge columns reported separately
- support-group and fallback provenance for weak runs
- update clipping and support coverage
- independent production-transfer validation

A flatter mean residual map alone is not evidence of improvement. Width,
held-out closure, and stability must also improve or remain acceptable.

## Application Contract

Pi0 selection is used only to derive constants. The frozen correction is
applied to every cluster in every event of a covered run. Existing branches are
unchanged. The writer adds corrected cluster-energy and factor-provenance
vectors, preserves other top-level ROOT objects, and rejects duplicate
application or incomplete exact-seed coverage.

## Deliberate Limits

- Packages are valid only for their named target, kinematic, baseline, and run
  LUT.
- Period definitions and fit acceptance still require analyst review.
- The standard pipeline exports period curve, run scalar, seed residual, and
  smooth low-energy layers.
- Member/hybrid residual corrections are diagnostic and are not promoted by
  this pipeline.
- A specialized low-energy shape term must pass its own held-out and physics
  validation before being added as a versioned package extension.
