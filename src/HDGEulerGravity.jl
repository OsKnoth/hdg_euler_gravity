"""
    HDGEulerGravity

Hybridized DG with LMARS flux and relaxed trace conditions for the Euler equations with gravity.

Submodules (each with its own exports; load with e.g. `using HDGEulerGravity.SplitExplicitEuler`):
  HDGRelaxEuler       v1: modal DG, linearized about a hydrostatic state, perturbations (ρ', ρ₀w', p')
  HDGRelaxTheta       v2: nodal DGSEM fast linear operator in (ρ, ρw, ρθ), energy-stable split form
  SplitExplicitEuler  v3: MIS split-explicit solver for the nonlinear equations, fast part = v2
"""
module HDGEulerGravity

include("HDGRelaxEuler.jl")
include("HDGRelaxTheta.jl")
include("SplitExplicitEuler.jl")

export HDGRelaxEuler, HDGRelaxTheta, SplitExplicitEuler

end # module
