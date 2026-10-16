"""
    HDGEulerGravity

Split-explicit time integration of the compressible Euler equations with gravity in
potential-temperature form, with two spatial discretizations sharing the time steppers.

Submodules (load with e.g. `using HDGEulerGravity.Euler2D`):
  Common              gas, pressure, reference atmospheres
  TimeStepping        MIS / WS-RK3 split-explicit stepper, fast integrators (:imex, :hevi,
                      :verlet, :fb, :wrf), RK4 reference; interface for the discretizations
  DGSEM:
    HDGRelaxTheta     1D nodal DGSEM fast operator (HDG, LMARS, relaxed traces, energy-stable split form)
    SplitExplicitEuler 1D nonlinear DGSEM with the fast operator of HDGRelaxTheta
    Euler2D           2D x–z DGSEM with terrain, matrix-free fast operator, HEVI column solver
  FV:
    FV2D              WRF-style finite volumes on an Arakawa C-grid (flat)
  Legacy:
    HDGRelaxEuler     v1: modal DG, linearized perturbation equations
"""
module HDGEulerGravity

include("Common/Common.jl")
include("TimeStepping/TimeStepping.jl")
include("Legacy/HDGRelaxEuler.jl")
include("DGSEM/HDGRelaxTheta/HDGRelaxTheta.jl")
include("DGSEM/SplitExplicitEuler/SplitExplicitEuler.jl")
include("DGSEM/Euler2D/Euler2D.jl")
include("FV/FV2D.jl")

export Common, TimeStepping, HDGRelaxEuler, HDGRelaxTheta, SplitExplicitEuler, Euler2D, FV2D

end # module
