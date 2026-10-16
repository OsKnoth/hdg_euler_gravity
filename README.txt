HDGEulerGravity.jl: split-explicit solvers for the Euler equations with gravity (1D, 2D x-z),
two spatial discretizations (DGSEM with HDG/LMARS fast operator, WRF-style finite volumes)
sharing one time-stepping layer.

Layout (Julia package; standard library only, VTKHDF/Makie output as package extensions)
  Project.toml
  src/HDGEulerGravity.jl     top module
  src/Common/                gas, pressure, reference atmospheres; model API implemented by both
                             discretizations: project2d, total2d, output_fields, field_section,
                             sound_speed_max, with_sponge, open_vtkhdf, write_vtkhdf!, plot_fields
  src/TimeStepping/          independent of the discretization:
                               interface.jl  functions a discretization implements
                                             (rhs_nonlinear!, fast_operator, fast_apply!, and per
                                             fast integrator the split parts, see below)
                               tableaus.jl   WSRK3, MIS2, MIS3C, MIS4, MIS4a
                               mis.jl        run_mis, mis_step!, run_rk4 (reference)
                               fast.jl       fast_solve!(...; method) with the fast integrators
                             Slow part S = N - L_n with the fast operator L_n re-frozen at q^n;
                             nsub fast substeps per large step (stage i: round(d_i nsub));
                             keep nsub <= 24 (WRF: 6 or 12).
                             Fast integrators and what they need from the discretization:
                               :imex    explicit RK2 of the IMEX tableau     fast_apply!
                               :hevi    horizontal explicit, vertical        fast_rhs_horizontal!,
                                        implicit (IMEX tableau)              vertical_apply!,
                                                                             vertical_factor,
                                                                             vertical_solve!
                               :verlet  Stoermer-Verlet (Xu, van der Vegt,   fast_rhs_thermo!,
                                        Bokhove 2008, eq. 4.6, f = 0)        fast_rhs_momentum!,
                               :fb      forward-backward                     interface_update!
                               :wrf     WRF acoustic step: horizontal        fast_rhs_hmomentum!,
                                        momentum forward, then (rho, Theta,  fast_rhs_hthermo!,
                                        w) vertically implicit, off-centred  vertical_apply!,
                                        Crank-Nicolson (beta = 0.55; first  vertical_factor,
                                        order in dtau), divergence damping  vertical_solve!
                                        p + 0.1 (p - p_old) as in WRF
                                        (needed: without it WSRK3 + wrf
                                        grows in the long channel)
  src/DGSEM/                 discontinuous Galerkin spectral element discretizations
    HDGRelaxTheta/           v2: 1D nodal DGSEM fast operator in (rho, rho w, rho theta), HDG with
                             LMARS and relaxed traces, energy-stable split form (form=:split),
                             conservative, well balanced; form=:standard has a spurious O(h) growth.
    SplitExplicitEuler/      v3: 1D nonlinear DGSEM N with flux differencing (flux_tec of Artiano,
                             Knoth, Spichtinger, Ranocha 2025) and gravity as non-conservative
                             two-point term (Waruszewski et al. log mean), LMARS, walls; fast part =
                             v2 operator re-frozen at q^n.  A pointwise gravity source -g rho_i gives
                             an O(h) linear instability about stratified states.
    Euler2D/                 2D x-z slice (periodic in x, walls) with terrain:
                               grid           Cartesian background mesh, LGL nodes, Gal-Chen map
                               metric         collocation metrics from the nodes
                               model, rhs     nonlinear N, hydrostatic residual subtracted, sponge
                               fast_operator  HDG operator frozen at the full 2D state, traces
                                              eliminated, matrix free
                               fast_split     thermodynamic / momentum split (verlet, fb):
                                              interface variable = contravariant LMARS momentum
                               hevi           column solver: element-local elimination of rho,
                                              Theta, U = Ja^eta.m, tridiagonal system for the face
                                              momentum (cf. CGDycore.jl DGSEM/JacHDGJNeu.jl)
                               diagnostics    energy
                             Fast integrators: verlet (recommended), fb, imex, hevi.
                             Stable CFL (dtau = CFL h/((2p+1)c)), p = 2/3/4/5:
                               verlet 1.05/0.79/0.63/0.52, fb ~0.7 verlet,
                               imex 1.07/0.24/0.088/0.050, hevi (horizontal) 0.98/0.28/0.075/0.060
                             (the explicit RK2 part of imex/hevi is stabilized only by LMARS).
  src/FV/                    FV2D: WRF-style finite volumes (Klemp, Skamarock, Dudhia 2007), flat:
                               grid           Arakawa C-grid, q = [rho; U; W (interior z-faces); Theta]
                               model          reference state, sponge, initialization, output
                               advection, rhs flux form, 3rd-order upwind (WRF flux3), p' and rho'
                                              relative to the hydrostatic reference (rest exact)
                               fast_operator  acoustic-gravity part linearized at q^n (energy
                                              conserving, no dissipation), split by row masks
                               vertical       tridiagonal system for W per column (as in WRF)
                             Fast integrators: wrf (default, c dtau/dx <= 1.0), verlet and fb
                             (c dtau/min(dx, dz) <= 0.92); imex and hevi are unstable on the C-grid.
  src/Legacy/                v1: modal DG, linearized perturbation equations
  ext/                       HDF5 extension (VTKHDF: DG Lagrange quads with point data, FV quads
                             with cell data), Makie extension (plot_fields)
  test/runtests.jl           unit tests:   julia --project -e 'using Pkg; Pkg.test()'
  examples/                  own environment (uses the package via [sources], Julia >= 1.11):
                               julia --project=examples -e 'using Pkg; Pkg.instantiate()'
                               julia --project=examples examples/v1_perturbation_report.jl
                               julia --project=examples examples/v2_theta_report.jl
                               julia --project=examples examples/v2_single_run.jl
                               julia --project=examples examples/v3_mis_report.jl
                               julia --project=examples examples/euler2d_gravity_wave.jl [dg|fv] [fast]
                                 (Skamarock-Klemp 1994 gravity waves, 300 km channel)
                               julia --project=examples examples/euler2d_gravity_wave_long.jl [dg|fv] [fast]
                                 (same in a 6000 km channel, hydrostatic regime)
                               julia --project=examples examples/euler2d_hills.jl a1|a10|a100 [T|-] [fast]
                                 (Gallus & Klemp 2000 bell-shaped hills, Na/U = 1, 10, 100, DG
                                  only; periodic channel of 40 a, Rayleigh layer above 9 km;
                                  env NEX (default 10), CREST, H0)
                               defaults: dg: MIS4, nsub 12, hevi;  fv: WSRK3, nsub 6, wrf
                             examples/euler2d_common.jl holds the shared setup (build_model,
                             run_with_output, fast_step with the CFL tables, gravity_wave).
                             The 2D examples write VTKHDF time series for ParaView (>= 5.13) and
                             Makie PNGs; the scripts write CSV files into the current directory.
                             Plots of v1/v2 need Plots.jl:
                               julia --project=examples -e 'using Pkg; Pkg.add("Plots")'
  docs/v1_perturbation/      LaTeX source + PDF of v1
  docs/v2_theta_dgsem/       LaTeX source + PDF of v2

Use in code:
  using HDGEulerGravity.Euler2D       (or .FV2D, .SplitExplicitEuler, .HDGRelaxTheta, .HDGRelaxEuler)
  using HDGEulerGravity.TimeStepping  run_mis(model, q0, dt, nsteps; tab, nsub, fast)

Adding a discretization (e.g. in Breeze.jl): implement the Common model API and, for the fast
integrators wanted, the functions listed in src/TimeStepping/interface.jl; the fast operator is a
subtype of AbstractFastOperator with fields fb (frozen forcing), Mdiag and work (or overload
fast_forcing, fast_mass, fast_workspace).

The earlier Python implementations of v1 and v2 were removed; they are in the git history
(commit 1af09b7, directories v1_perturbation/python and v2_theta_dgsem/python).
