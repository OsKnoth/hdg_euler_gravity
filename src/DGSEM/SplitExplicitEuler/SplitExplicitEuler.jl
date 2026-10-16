"""
    SplitExplicitEuler

Split-explicit (MIS) time integration of the NONLINEAR 1D compressible Euler equations with
gravity in (ρ, m = ρw, Θ = ρθ), using the hybridized DGSEM fast operator of HDGRelaxTheta (v2)
as the linear (fast) part.

    ρ_t + m_z = 0,   m_t + (m w + p)_z = -g ρ,   Θ_t + (θ m)_z = 0,   p = p00 (R Θ/p00)^γ

Full operator N (nonlinear DGSEM, same GLL nodes and data layout as v2):
  volume   flux differencing with Kennedy–Gruber two-point fluxes
           F_ρ = {ρ}{w},  F_m = {ρ}{w}{w} + {p},  F_Θ = {ρ}{w}{θ}      (conservative in ρ, Θ)
  faces    LMARS: w* = {w} - [p]/(2 ρ̄c̄),  p* = {p} - ρ̄c̄ [w]/2,  F* = w* q_upwind + p* e_m
  walls    mirror state (w_ghost = -w)
  gravity  -g ρ_i at the nodes.
A discrete hydrostatic state (m = 0, nodal Θ) has ρ_i = -(D p(Θ))_i / (g J) exactly.

Fast operator L_n (re-frozen at the start of every large step, from qⁿ):
  θ̄ = Θ/ρ,  a = pfac p/Θ,  b = (1 - pfac) p   at the nodes, made continuous by DSS
  (see HDGRelaxTh nodal constructor). pfac = γ (default) puts the full acoustic speed into L;
  with pfac = 1 the slow part keeps a fraction (γ-1) of the acoustic pressure.

MIS step (Wensch, Knoth, Galant 2009; Knoth, Wensch 2014), slow part S = N - L_n:
  Y_1 = qⁿ
  Z_i(0) = qⁿ + Σ_{j<i} α_ij (Y_j - qⁿ)
  Z_i'   = L_n(Z_i) + Σ_{j<i} [ γ_ij (Y_j - qⁿ)/(d_i Δt) + β_ij S(Y_j)/d_i ],   τ ∈ [0, d_i Δt]
  Y_i = Z_i(d_i Δt),   qⁿ⁺¹ = Y_{s+1},   d_i = Σ_j β_ij.
The fast ODE of each stage is integrated by HDGRelaxTheta.fast_solve! (:imex; for the 2D operator
also :verlet, :fb, :hevi) with n_i = round(d_i nsub) substeps, nsub = fast/slow ratio
(WRF: 6 or 12 with RK3; keep ≤ 24), and the constant forcing F_slow. Since F_slow = N(Y) - L_n(Y),
steady states of N stay steady. mis_step!/run_mis work on preallocated MISWork buffers and the
in-place rhs_nonlinear!, so a large step does not allocate.
"""
module SplitExplicitEuler

using LinearAlgebra, SparseArrays

using ..HDGRelaxTheta
using ..Common: Gas, pressure
import ..Common: sound_speed_max
using ..TimeStepping: MISTableau, mis_tableau, nstages, MISWork, stage_substeps, mis_step!, mis_step,
                      run_mis, run_rk4, rhs_nonlinear
import ..TimeStepping: rhs_nonlinear!, fast_operator

export HDGRelaxTheta, Gas, pressure, Euler1D, rhs_nonlinear, rhs_nonlinear!, MISWork, mis_step!, stage_substeps, hydrostatic_state, fast_operator,
       MISTableau, mis_tableau, mis_step, run_mis, run_rk4, total, evaluate, l2_error, project,
       blk, iq, nodes, sound_speed_max, Atmosphere, atm_Theta, atm_rho, atm_theta, atm_p

const NV = 3

include("grid.jl")
include("fluxes.jl")
include("rhs.jl")
include("fast_operator.jl")
include("helpers.jl")

end # module
