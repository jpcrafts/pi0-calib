# x60_4 LH2 v5 Smooth-LowE Package

Validated correction package for NPS `x60_4` LH2 runs listed in
`run_period_lut.tsv`.

Input contract:

- Input energy is `NPS.prod.clusE` reconstructed using Hao's
  `CALO_calib_Pi0Coef` baseline.
- Run must appear in `run_period_lut.tsv` and `run_scalar_lut.tsv`.
- Exact seed block is preferred. An unsupported or absent seed may use unity
  only when the caller explicitly accepts that policy.
- Do not apply this package to LD2, another kinematic, elastic-baseline energy,
  or a run absent from the LUT.

Per-cluster application:

```text
E_corrected = E_Hao * curve(run_period, E_Hao)
                    * run_scalar(run)
                    * seed_scale(seed_block)
                    * smooth_lowE(E_Hao)
```

The low-energy scale is fully active through `0.8 GeV`, then transitions
smoothly to unity at `0.9 GeV`.
