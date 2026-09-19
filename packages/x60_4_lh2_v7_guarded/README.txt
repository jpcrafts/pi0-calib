Frozen Hao-start x60_4_lh2 production calibration package

Baseline:
  Input cluster energies already use Hao CALO_calib_Pi0Coef.

Application per cluster:
  E1 = E_Hao * period_photon_curve(period(run), E_Hao)
  E2 = E1 * run_scalar(run)
  E3 = E2 * seed_block_scale(seed_block)
  E_final = E3 * lowe_photon_scale(global) when E_Hao is in 0.6-1 GeV
  Low-E profile: smoothstep; full through 0.8 GeV; unity at 1 GeV.

Policy:
  Target: LH2; covered runs: 124.
  Run-to-period assignments come only from run_period_lut.tsv.
  Run scalar uses own-run Gaussian fit when accepted; weak runs use existing chronological support policy.
  Seed columns 0-2 remain identity/QA when unsupported.
  Member/hybrid residual layer is not included.
  The entire correction stack tapers from full strength at 2.0 GeV to identity at 2.5 GeV.
  Original Hao cluster energies at or above 2.5 GeV are copied unchanged.
  All mass peak means and sigmas used in derivation/QA come from canonical Gaussian+linear-background fits.

Run LUT rows: 124
Approval status: diagnostic_unreviewed
Run support modes: chronological_group_one_sided_shrunk=5, chronological_group_two_sided=11, run_local=108
Supported seed rows: 931/972
Selected low-E photon factors (d=1.00, scope=global): global=1.006912980

This package is frozen. Do not tune tables in place; create a versioned successor.
