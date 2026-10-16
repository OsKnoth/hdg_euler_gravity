"""
    HDGRelaxTheta

1D hybridized DGSEM (nodal, Gauss–Lobatto collocation) with LMARS interface flux for the
LINEARIZED (fast) part of the compressible Euler equations in potential-temperature form,
with relaxed trace conditions integrated in time WITHOUT elimination.

Linearized system (frozen coefficients; momentum advection removed, θ frozen in the Θ
flux, p = (p/Θ) Θ with p/Θ frozen), q = (ρ, m = ρw, Θ = ρθ):

    ρ_t + m_z = 0,   m_t + (a Θ + b)_z = -g ρ,   Θ_t + (θ̄ m)_z = 0
    a = pfac p̄/Θ̄,  b = (1 - pfac) p̄        (pfac = 1: c² = R T̄;  pfac = γ: c² = γ R T̄)

Energy (p = aΘ, χ = Θ - θ̄ρ, c² = aθ̄, φ = θ̄ exp(∫ g/c² dz)):
    E = ½ ∫ φ [ m² + p²/c² + g χ²/(θ̄ θ̄_z) ] dz

DGSEM, form = :split (default, energy stable), per element with SBP matrix D, J = h/2:
    ρ_i' = -(D m)_i / J
    Θ_i' = -(θ̄_i (D m)_i + m_i (D θ̄)_i) / J                       (product form)
    m_i' = -[(D(φ a Θ))_i / φ_i - a_i Θ_i (D θ̄)_i / θ̄_i + (D b)_i] / J + g Θ_i/θ̄_i - g ρ_i
plus SAT / HDG coupling at the end nodes:  J w_b q_b' += (A_f n - T_f)(q_b - q̂_f),
    A_f = [0 1 0; 0 0 a; 0 θ̄ 0],  T_f = |A_f| = [0 0 a/c; 0 c 0; 0 0 c]   (LMARS).
form = :standard: plain collocation weak form (only O(h)-stable for variable θ̄).

Trace conditions: interior q̂ = mean(q), wall q̂ = diag(1,0,1) q_int; relaxed as
ε dq̂/dt = H q - q̂.  Semi-discrete system, U = (q, q̂):
    [M 0; 0 εI] U_t = [K G; H -I] U + [f_b + M F_slow; 0],   M = diag(J w_i)

Standard library only (LinearAlgebra, SparseArrays). Variables r = 1 (ρ), 2 (m), 3 (Θ);
node i = 1 is the bottom (ξ = -1), node p+1 the top (ξ = +1) of each element.
"""
module HDGRelaxTheta

using LinearAlgebra, SparseArrays
using ..Common: Atmosphere, atm_T, atm_p, atm_rho, atm_theta, atm_Theta, atm_pi
using ..TimeStepping: AbstractFastOperator, fast_solve!, IMEX_GM, IMEX_DL, IMEX_AE, IMEX_AI
import ..TimeStepping: fast_apply!, fast_workspace

export AbstractHDGOperator, fast_apply!, fast_solve!, fast_workspace, Atmosphere, FrozenState, frozen_from_atmosphere, HDGRelaxTh,
       gauss_legendre, gll, lagrange_basis, iq, it, blk, nodes,
       hydrostatic_rho, balanced_state, forcing_vector, tendency, energy,
       project, evaluate, l2_error, total, initial_state, trace_residual, run!,
       atm_T, atm_p, atm_rho, atm_theta, atm_Theta, atm_pi, coef_a, coef_b, dss!

const NV = 3

include("atmosphere.jl")
include("quadrature.jl")
include("operator.jl")
include("helpers.jl")
include("timestepping.jl")

end # module
