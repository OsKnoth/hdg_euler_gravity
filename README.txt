HDG with LMARS flux and relaxed trace conditions, 1D Euler with gravity

v1_perturbation/   modal DG, linearized about the hydrostatic state,
                   perturbations (rho', rho0 w', p')
v2_theta_dgsem/    nodal DGSEM (Gauss-Lobatto collocation, SBP), fast linear operator
                   in (rho, rho w, rho theta) for a split-explicit method:
                   no momentum advection, theta frozen in the rho-theta flux,
                   p = (p/(rho theta)) * rho theta with p/(rho theta) frozen.
                   Default form="split": provably energy stable, conservative in rho and
                   rho theta, well balanced.  form="standard" (plain collocation) kept for
                   comparison: it has a spurious O(h) growth rate.

v2_theta_dgsem also has a single-run driver (p=4, 5 GLL nodes, Ne=40, IMEX, walls,
pressure pulse + warm bubble) with initial/final plots of rho, rho w, rho theta:
  python3 run_single_theta.py      julia run_single_theta.jl  (plots need Plots.jl)
Parameters are at the top of the file.

Each version:
  python/  solver + tests (python3 run_tests*.py), reference output, plots
  julia/   module + tests (julia run_tests*.jl), standard library only
  doc/     LaTeX source + compiled PDF (figures next to the .tex)
