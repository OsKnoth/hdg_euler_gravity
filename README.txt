HDGEulerGravity.jl: HDG with LMARS flux and relaxed trace conditions, Euler with gravity (1D, 2D x-z)

Layout (Julia package; standard library only, VTKHDF/Makie output as package extensions)
  Project.toml
  src/HDGEulerGravity.jl     top module, includes the four submodules:
    HDGRelaxEuler.jl         v1: modal DG, linearized about the hydrostatic state,
                             perturbations (rho', rho0 w', p')
    HDGRelaxTheta.jl         v2: nodal DGSEM (Gauss-Lobatto collocation, SBP), fast linear operator
                             in (rho, rho w, rho theta): no momentum advection, theta frozen in the
                             rho-theta flux, p = (p/(rho theta)) * rho theta with p/(rho theta) frozen.
                             Default form=:split: provably energy stable, conservative in rho and
                             rho theta, well balanced.  form=:standard (plain collocation) kept for
                             comparison: it has a spurious O(h) growth rate.  Frozen coefficients
                             from functions of z or from nodal arrays (made continuous by DSS).
                             Time integration: explicit IMEX (relaxed traces implicit for eps > 0);
                             fast_solve! is the allocation-free core used by the split-explicit
                             schemes.  (DIRK was removed.)
    SplitExplicitEuler.jl    v3: split-explicit (MIS) solver for the NONLINEAR equations:
                             nonlinear DGSEM N with flux differencing (default flux_tec of Artiano,
                             Knoth, Spichtinger, Ranocha 2025) and gravity as non-conservative
                             two-point term (default Waruszewski et al. log mean), LMARS, walls;
                             fast part = v2 operator re-frozen at q^n (theta = Theta/rho,
                             a = gamma p/Theta, b = (1-gamma) p); slow part N - L_n.
                             Tables WSRK3, MIS2, MIS3C, MIS4, MIS4a; nsub fast substeps per large
                             step (stage i: round(d_i nsub)); keep nsub <= 24 (WRF: 6 or 12).
                             MIS4a/MIS3C amplify pure acoustic modes for omega*dt > 28.8 (15.8).
                             Note: a pointwise gravity source -g rho_i gives an O(h) linear
                             instability about stratified states; the two-point form does not.
    Euler2D.jl               2D x-z slice (periodic in x, walls), terrain-following mesh (Gal-Chen),
                             collocation metrics.  Nonlinear DGSEM N as in v3, residual of the
                             hydrostatic reference subtracted (rest state exact over terrain),
                             optional Rayleigh sponge (top and lateral).  Fast HDG operator frozen
                             at the full 2D state, traces eliminated, applied MATRIX-FREE; energy
                             stable for a horizontally homogeneous frozen state (also with terrain).
                             The operator splits exactly into a thermodynamic part (rho, Theta;
                             interface variable = contravariant LMARS momentum) and a momentum part
                             (LMARS interface pressure).  Fast integrators:
                               :verlet  Stoermer-Verlet (Xu, van der Vegt, Bokhove 2008, eq. 4.6,
                                        f = 0), recommended; tau_scale = 0 gives the central
                                        (Hamiltonian) interface, exactly symplectic
                               :fb      forward-backward (m explicit, interface momentum implicit)
                               :imex    explicit IMEX tableau
                               :hevi    horizontal explicit, vertical implicit (column-local matrix
                                        on a fixed pattern, values and LU updated in place)
                             Stable CFL (dt = CFL h/((2p+1)c)), p = 2/3/4/5:
                               verlet 1.05/0.79/0.63/0.52 (~3.15/(p+1)), fb ~0.7 verlet,
                               imex 1.07/0.24/0.088/0.050, hevi (horizontal) 0.98/0.28/0.075/0.060;
                               the explicit part of imex/hevi is RK2 (unstable on the imaginary
                               axis, stabilized only by the LMARS dissipation).
                             N, the fast kernels, the fast loops and the MIS step do not allocate.
  ext/                       HDF5 extension (open_vtkhdf, write_vtkhdf!), Makie extension (plot_fields)
  test/runtests.jl           unit tests:   julia --project -e 'using Pkg; Pkg.test()'
  examples/                  own environment (uses the package via [sources], Julia >= 1.11):
                               julia --project=examples -e 'using Pkg; Pkg.instantiate()'
                               julia --project=examples examples/v1_perturbation_report.jl
                               julia --project=examples examples/v2_theta_report.jl
                               julia --project=examples examples/v2_single_run.jl
                               julia --project=examples examples/v3_mis_report.jl
                               julia --project=examples examples/euler2d_gravity_wave.jl [fast]
                                 (Skamarock-Klemp 1994 gravity waves, 300 km channel)
                               julia --project=examples examples/euler2d_gravity_wave_long.jl [fast]
                                 (same in a 6000 km channel, hydrostatic regime)
                               julia --project=examples examples/euler2d_hills.jl a1|a10|a100 [T|-] [fast]
                                 (Gallus & Klemp 2000 bell-shaped hills, Na/U = 1, 10, 100;
                                  periodic channel of 40 a, Rayleigh layer above 9 km)
                               fast = hevi (default), verlet, fb, imex
                             The 2D examples write VTKHDF time series for ParaView (>= 5.13) and
                             Makie PNGs (HDF5.jl and CairoMakie are in the examples environment;
                             package extensions of HDGEulerGravity).
                             The scripts write CSV files into the current directory.  Plots
                             (v1_plot_results.jl, PNGs of v2_single_run.jl) need Plots.jl:
                               julia --project=examples -e 'using Pkg; Pkg.add("Plots")'
  docs/v1_perturbation/      LaTeX source + PDF of v1 (figures next to the .tex)
  docs/v2_theta_dgsem/       LaTeX source + PDF of v2

Use in code:
  using HDGEulerGravity.SplitExplicitEuler     (or .Euler2D, .HDGRelaxTheta, .HDGRelaxEuler)

The earlier Python implementations of v1 and v2 were removed; they are in the git history
(commit 1af09b7, directories v1_perturbation/python and v2_theta_dgsem/python).
