HDGEulerGravity.jl: HDG with LMARS flux and relaxed trace conditions, 1D Euler with gravity

Layout (Julia package; standard library only)
  Project.toml
  src/HDGEulerGravity.jl     top module, includes the three submodules:
    HDGRelaxEuler.jl         v1: modal DG, linearized about the hydrostatic state,
                             perturbations (rho', rho0 w', p')
    HDGRelaxTheta.jl         v2: nodal DGSEM (Gauss-Lobatto collocation, SBP), fast linear operator
                             in (rho, rho w, rho theta): no momentum advection, theta frozen in the
                             rho-theta flux, p = (p/(rho theta)) * rho theta with p/(rho theta) frozen.
                             Default form=:split: provably energy stable, conservative in rho and
                             rho theta, well balanced.  form=:standard (plain collocation) kept for
                             comparison: it has a spurious O(h) growth rate.  Frozen coefficients
                             from functions of z or from nodal arrays (made continuous by DSS).
    SplitExplicitEuler.jl    v3: split-explicit (MIS) solver for the NONLINEAR equations:
                             nonlinear DGSEM (Kennedy-Gruber flux differencing, LMARS, walls) as
                             full operator N; fast part = v2 operator re-frozen at q^n every large
                             step (theta = Theta/rho, a = gamma p/Theta, b = (1-gamma) p); slow part
                             N - L_n.  Tables WSRK3, MIS2, MIS3C, MIS4, MIS4a; fast integrator :imex
                             (explicit substeps) or :dirk.
                             Note: MIS4a (and MIS3C) amplify pure acoustic modes for
                             omega*dt > 28.8 (15.8); WSRK3, MIS2, MIS4 do not.
    Euler2D.jl               2D x-z slice (periodic in x, walls), terrain-following mesh (Gal-Chen),
                             collocation metrics.  Nonlinear DGSEM N (Kennedy-Gruber, LMARS) with
                             analytic hydrostatic reference subtracted (p - p_ref, rho - rho_ref):
                             rest state exact over terrain.  Fast HDG operator frozen at the full 2D
                             state q^n (DSS-continuous coefficients, phi integrated along columns):
                             energy stable for a horizontally homogeneous frozen state (also with
                             terrain), otherwise bounded zero-order growth from horizontal gradients
                             of theta and c^2.  MIS via SplitExplicitEuler.run_mis; fast part
                             :imex (explicit) or :hevi (horizontal explicit, vertical implicit).
  test/runtests.jl           unit tests:   julia --project -e 'using Pkg; Pkg.test()'
  examples/                  own environment (uses the package via [sources], Julia >= 1.11):
                               julia --project=examples -e 'using Pkg; Pkg.instantiate()'
                               julia --project=examples examples/v1_perturbation_report.jl
                               julia --project=examples examples/v2_theta_report.jl
                               julia --project=examples examples/v2_single_run.jl
                               julia --project=examples examples/v3_mis_report.jl
                               julia --project=examples examples/euler2d_gravity_wave.jl
                                 (Skamarock-Klemp 1994 gravity waves, MIS4 + HEVI, ~70 s)
                             The scripts write CSV files into the current directory.  Plots
                             (v1_plot_results.jl, PNGs of v2_single_run.jl) need Plots.jl:
                               julia --project=examples -e 'using Pkg; Pkg.add("Plots")'
  docs/v1_perturbation/      LaTeX source + PDF of v1 (figures next to the .tex)
  docs/v2_theta_dgsem/       LaTeX source + PDF of v2

Use in code:
  using HDGEulerGravity.SplitExplicitEuler     (or .Euler2D, .HDGRelaxTheta, .HDGRelaxEuler)

The earlier Python implementations of v1 and v2 were removed; they are in the git history
(commit 1af09b7, directories v1_perturbation/python and v2_theta_dgsem/python).
