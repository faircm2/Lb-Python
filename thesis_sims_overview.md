# Simulations for the thesis: definitive scripts and parameter-study sims

State 25 Oct 2026. "Definitive" means: the script that runs only its own model, has run to the end, and gives the expected result.
How each script was judged: (1) which model functions are actually called in its main loop (call-graph check of the code), (2) the result folders and images, (3) the parameter-study tables.
Model names: both families are multiphase lattice Boltzmann models. Inamuro et al. 2003: Cahn-Hilliard phase field + Shan-Chen. Zhang et al. 2020: conservative Allen-Cahn phase field + free-energy approach.
Common to all runs: channel width D = 1.0 mm (1.2 mm for the experimental model), nu = 2.9e-6 m^2/s (1.0e-6 for the experimental model), rho_0 = 1000 kg/m^3.

## 1. The six definitive simulations

Tab. 1 LB simulations categorised by phase-field and force model

| Simulation | Phase-field and force model | Script file(s) | Key parameters |
|---|---|---|---|
| **Inamuro 2D, square lattice** | Cahn-Hilliard phase field + Shan-Chen | `...D2Q9_08.py` (step test) | Step interface 200 x 50, tau_f 1.5, tau_g 1.0, K_f 0.008, a 1, b 6.7, T 3.5e-2 |
| **Inamuro 3D, cubic lattice** | Cahn-Hilliard phase field + Shan-Chen | `...D3Q15_1.py` | Step interface 200 x 50 x 50, tau_f 2.5, tau_g 1.0, K_f 0.05, K_g 1e-6 n_dx^2, a 1, b 6.7, T 3.5e-2 |
| **Zhang 2D, square lattice** | Conservative Allen-Cahn + free energy | `...D2Q9_08_Zhang_Cap02.py` | N 300, alpha 0, sigma 0.01, W 4, tau_f 1.25, tau_g 0.65, theta 120 (60 and 90 also run) |
| **Zhang 3D, cubic lattice** | Conservative Allen-Cahn + free energy | `...D3Q15_08_Zhang_Cap02_MPI_bwUniCluster.py` + `.sh` | N 200 and 384, alpha 0, sigma 0.01, W 4, tau_f 1.25, tau_g 0.65, theta 120 |
| **Zhang 3D, channel model** | Conservative Allen-Cahn + free energy | `...Cap07.py` + `.sh` | 768 x 384 x 384, rho_L 10, alpha 90, periodic x, W 4, tau_g 0.65, theta 60; defaults sigma 0.01, tau_f 1.0 |
| **Zhang 3D, experimental model** | Conservative Allen-Cahn + free energy | `Cap07` + `parm_study_v28` | nu 1e-6, D 1.2 mm, tau_f 0.55, tau_g 0.65, sigma 0.625, theta 65, W 10, fill 51 cells |

### Evidence and superfluous elements per definitive script

| Simulation | Evidence | Foreign or superfluous elements |
|---|---|---|
| **Inamuro 2D, square lattice** | Calls only Inamuro functions, no Zhang function. Ran to iteration 384,032 with a density ratio of 50 and a sharp step interface. | None active. Dead code: `psi`, `tau_h`. |
| **Inamuro 3D, cubic lattice** | Calls only Inamuro functions. The only 3D Inamuro run that went beyond iteration 428 (the D3Q27 versions stopped at 428): it reached iteration 15,495 of 48,004. The last snapshot shows a smooth, bulged interface and not a clean step, so this one is only partly verified. | None active. Dead code: `psi`, `tau_h`. |
| **Zhang 2D, square lattice** | Calls only Zhang functions. In the 70 runs of the parameter study the measured surface tension is 1.017 to 1.0185 times the input value, and the meniscus rise changes sign at theta = 90 (+6.7 at 50 degrees, -13.7 at 130 degrees). Clean meniscus at theta = 120. | Inamuro-era leftovers, not used by the physics: config fields Kf, Kg, a, b, T, xi, U_nd; functions `init_step_phi`, `sufficient_stability_condition`, `optimal_stability_condition`. f boundary still in the old (2/3, 1/3) form. |
| **Zhang 3D, cubic lattice (capillary test)** | Calls only Zhang functions. Cluster runs with N = 200 and N = 384 at theta = 120 exist (folders `run_200_results`, `run_384_results`); I did not inspect their images. | Same Inamuro-era leftovers as the 2D script (Kf, Kg, a, b, T, xi, U_nd, `init_step_phi`, stability functions, `BENCHMARK_STEP`). The file is now set to the inclined-channel case; the capillary case needs `ACTIVE_CASE = "proof_capillary"`. |
| **Zhang 3D, channel model** | Calls only Zhang functions, no Inamuro leftovers left. Reproduces the Cap06 results to the last digit (check run S0p5_C07). Floor stays clean at sigma 0.05 for 24,002 iterations with the corrected f boundary; at sigma 0.5 it is stable from W = 7 on. | None. (Cap06 has the same physics, with the output code and the Inamuro-era fields still in.) |
| **Zhang 3D, experimental model** | Not run yet (submitted, waiting in the queue). Flat-floor rectangular channel, not the herringbone geometry. | None. |

## 2. Simulations used for the parameter studies

Tab. 2 Parameter studies of LB simulations

| Study | Phase-field and force model | Simulation |
|---|---|---|
| sigma | Conservative Allen-Cahn + free energy | 3D channel model (`Cap07`, `Cap06`): sigma 0.05 and 0.5 done (W 4); sigma 0.39, 0.5, 0.65, 0.8, 1.0 at W 10 queued |
| tau_f | Conservative Allen-Cahn + free energy | 2D (`Zhang_Cap02` + `param_study.py`, phase 1): tau_g 0.55-1.0 x tau_f 1.0-2.0 (60 runs). 3D channel model (`Cap06 study/`): tau_f 0.6, 0.8, 1.0, 1.25 |
| theta | Conservative Allen-Cahn + free energy | 2D (`Zhang_Cap02` + `param_study.py`, phase 2): theta 50-130 (10 runs). 3D channel model (`Cap06 study/`): theta 60, 75, 90, 105, 120 |
| alpha | Conservative Allen-Cahn + free energy | 3D channel model (`Cap04`): alpha 45, 65, 75, 85, 90 |
| W | Conservative Allen-Cahn + free energy | 3D channel model (`Cap06`, `Cap07`): W 4-13 done; W 14-17 and reruns of W 7 and 8 queued |

### Status and caveats per study

| Study | Status and caveat |
|---|---|
| sigma | The earlier sigma runs (Cap05: 0.01, 0.02, 0.05, 0.1; Cap06 `study/`: 0.001, 0.01, 0.1, 1.0) used the old f boundary and the high-sigma runs collapsed. They are superseded and need a rerun with Cap07. |
| tau_f | The 2D runs are complete and consistent (sigma ratio 1.017). The 3D runs used the old f boundary; rerun with Cap07 for the thesis. |
| theta | 2D complete. The 3D runs used the old f boundary; rerun with Cap07. |
| alpha | Only study on alpha; early script, old f boundary. A rerun with Cap07 would be needed to quote it with the corrected model. |
| W | Corrected f boundary. Results in `stability_vs_W.png` and `.csv`. |

## 3. Left out of the table (and why)

- **Inamuro scripts with Zhang elements:** `D2Q9_08_Cap110`, `D2Q9_08_Inamuro_Cap6.1`, `D2Q9_08_Inamuro_Cap6.2`. They run the Inamuro model with Zhang's wetting, chemical potential and boundary functions added, to get a meniscus. The final snapshot of `Inamuro_Cap6.1` (theta 60) shows a flat interface, no meniscus. The 2D Inamuro parameter studies on Kf, tau_f, theta, W and sigma (`results/`, `lb_results/`, from `Cap110`) belong to this group.
- **Inamuro wetting attempts without Zhang models:** `D2Q9_08_Cap0` to `Cap7`, `Cap101`, `Cap102`, `Cap_Inamuro_g`. They use only `compContactAngle` from Zhang. The `Cap_Inamuro_g` snapshot shows a noisy, checkerboard interface.
- **Older versions:** `D2Q9_01` to `_06`, `D2Q9_07`, `free_surface_SchanChen_nondim06/07`, `D3Q27_1` to `_3`, Zhang `Cap01`, `Cap01_numba`, `Cap02`, `Cap02_MPI`, `Cap03` to `Cap06`.
- **Error analysis (instability):** the `A_*`, `B_periodic`, `T90` runs, the closed-x runs, `base45_long`, `big_run_60k`, `cap06_s01_*`, `a45_*`, `cap05_ctrl`, `test2d/`.

## 4. Points still open

1. The 2D Zhang scripts (`Zhang_Cap02`) still have the old (2/3, 1/3) f boundary. The 2D test model showed the collapse only after about 29,000 iterations at sigma 0.05, so the 12,001-iteration 2D studies at sigma 0.01 are probably not affected, but this has not been checked.
2. I judged "works" from the code path, the result folders and a few final images. The 3D Zhang capillary results (N = 200 and 384) and the Inamuro 3D run were not examined in detail.
3. The 3D studies on sigma, tau_f, theta and alpha need a rerun with Cap07 before they can be quoted.
