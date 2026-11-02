# Cap102 (2D Inamuro, D2Q9): boundary-condition sequence and probe findings

Script: `free_surface_SchanChen_nondim_D2Q9_08_Cap102.py`, 100 x 100 fluid cells (arrays 102 x 102 with ghost rows y = 0, Yn+1 and ghost columns x = 0, Xn+1).
Case of the probe: theta = 72 deg, `phi_wall_bc_method = "Neumann"`, crashes (NaN in pressure) at iteration 8826 to 8850.
Neither the Inamuro papers/book (section 4.4, Eqs. 4.26/4.27) nor Zhang et al. (Phys. Fluids 32, 103301, Eqs. 10, 11, 31) describe corner cells. The corner treatment is our own choice.

## Order of operations in one iteration (read from the code)

All arrays include the ghost rows and columns. Every `np.roll` wraps around the padded grid.

1. `fi_c`, `gi_c`: equilibria from the current phi and u, all nodes.
2. `fi()`, `gi()`: BGK collision and streaming (`np.roll`) in place, all nodes.
3. `bounceBackTopBottom_conservation`: writes fluid rows y = 1 and y = Yn (x = 1..Xn) from the ghost rows y = 0 and Yn+1.
4. `bounceBackLeftRight_conservation`: writes fluid columns x = 1 and x = Xn (y = 1..Yn) from the ghost columns. The fluid corner cells get their diagonal population from step 3 and again from step 4 (same ghost-corner source value, by reading the code).
5. `phi = sum_i f_i` at all nodes (ghost nodes hold a mix of streamed-in and stale populations).
6. If `ENFORCE_MASS_CONSERVATION`: top ghost row phi = phi_G, bottom ghost row phi = phi_L, full width, corners included (phi only, f_i untouched).
7. `phi_wall_dispatcher` sets phi in the ghost columns x = 0 and Xn+1 (Neumann: `interior + cos(theta)|dphi/dy|`, clipped, corner entries pinned to phi_L at the bottom and phi_G at the top). Then `sync_ghost_fi_with_phi_wall` resets f_i in the whole ghost columns (corners included) to the equilibrium of that phi.
8. Capillary force (`csf_fc`) from the derivatives of phi (`np.roll` stencils, ghost values included; zero at the corner cells in the probe).
9. `apply_mass_conservation`: one global factor on interior phi and f_i (the factor was 1 within 1e-13 in the probe).
10. Pressure: SOR on `ph()`, streaming wraps top to bottom; only pressure BC is a copy of the first fluid column into the ghost column left/right; div u is set to 0 in the ghost rows; `gradient_p` sets dp/dy = 0 in the ghost rows.
11. u = u* - grad p, next iteration.

## Probe (iterations 8400 to the crash, values after steps 2, 4, 5, 6, 7, 9, force, p, u at the 4 fluid corners, 4 ghost corners, 3 wall cells, 1 bulk cell)

- The failure is a slow drift over about 360 iterations (8400 to 8760), followed by a blow-up within 10 iterations (8826 to 8836).
- It sits in the gas region at the TOP corners, 50 cells above the interface (y about 50), not at the contact line.
- Drift at the fluid corner cells comes entirely from collision + streaming + bounce-back (change of phi over 8400..8760: top-left -0.0157, top-right +0.0531, bottom-left +0.0064, bottom-right +0.0168). The wall condition step and the mass-conservation step contributed 0.0000 at these cells.
- Top-right corner cell: density rises 1.5 -> 1.8 -> 2.4 -> 3.4 -> 6.2 -> 12.9 -> 44.6 -> 49.8 (gas 1, liquid about 50): a liquid-like blob forms in the gas corner.
- Top-left corner cell: phi falls below the gas value to -0.0004, density clamps at 1.0, pressure and velocity grow (u*y: -2e-4 -> -2.3e-2).
- The growth is antisymmetric left/right (top-left depletes, top-right fills); the capillary force at the corner cells is exactly zero.
- Along both side walls near the top (y = 96..100) phi sits at 0.017 to 0.019 from iteration 4000 on, i.e. above the gas value 0.0155, before the instability starts.
- Ghost-corner phi stays at its forced value until the last ~10 iterations.
- Same pattern in the other crashed runs (extremes of phi in the last dump before the NaN lie at the top or bottom corners).
- Crash time falls with theta: Neumann theta = 45: ~1500-2000; 63: ~6500-7000; 72: ~8830; 81, 90, 108: no crash in 20,000 iterations. The current Dirichlet condition crashes at 81 (~4500), 72 (~9000), 63 (~2500).
- Two corner edits tested at theta = 63 (Neumann ghost corners copy the neighbour; top/bottom overwrite skips the corner columns): both still crash at the same time.

## Not established

- Which operation inside collision / streaming / bounce-back amplifies the corner mode (collision equilibrium with gradients of phi that include ghost values, the bounce-back, or the pressure/velocity coupling).
- Whether the liquid collecting at the top-right corner is a corner-wetting effect created by the ghost values (for a 90 deg wedge a wetting liquid should only fill the corner for theta below about 45 deg) or a purely numerical gas-phase instability.
