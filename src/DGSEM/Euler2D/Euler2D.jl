"""
    Euler2D

Split-explicit (MIS) solver for the nonlinear compressible Euler equations with gravity in an
x–z slice with terrain, q = (ρ, m_x = ρu, m_z = ρw, Θ = ρθ):

    ρ_t + ∇·m = 0,   m_t + ∇·(m⊗u) + ∇p = -ρ g e_z,   Θ_t + ∇·(θ m) = 0,   p = p00 (RΘ/p00)^γ

Mesh: periodic in x, walls at the terrain z_s(x) and at z = H; uniform elements in (x, ζ),
terrain-following z = z_s(x) + ζ (H - z_s(x))/H (Gal-Chen), tensor GLL nodes, collocation metrics
Ja^ξ = (z_η, -x_η), Ja^η = (-z_ξ, x_ξ) (discrete metric identities hold exactly in 2D).

Full operator N (DGSEM, strong form):
  volume  flux differencing with averaged metrics; two-point fluxes :tec (default), :ec, :etec
          (Artiano, Knoth, Spichtinger, Ranocha 2025) or :kg (Kennedy–Gruber)
  gravity non-conservative two-point term ρ_avg (φ_k - φ_i), φ = g z, ρ_avg logarithmic
          (:waruszewski, default), Stolarsky (:artiano) or arithmetic (:souza) mean, as in
          TrixiAtmo; the residual of the reference state is subtracted (exact balance)
  faces   LMARS along the face normal; walls by mirror state

Fast operator L (frozen at the full 2D state qⁿ, made continuous by DSS):
  θ̄ = Θ/ρ,  a = pfac p/Θ,  b = (1 - pfac) p,  c² = a θ̄,  φ = θ̄ exp(χ),
  χ = ∫ g/c² dz along each column (η-line) from the surface, χ_s = g z_s / c²(z_s).
  Split form with the non-conservative gradient ∇_h u = (Ja^ξ D_ξ u + Ja^η D_η u)/J and the
  conservative divergence ∇_h·v = (D_ξ(Ja^ξ·v) + D_η(Ja^η·v))/J (an SBP pair):
    ρ_t = -∇_h·m
    Θ_t = -(θ̄ ∇_h·m + m·∇_h θ̄)
    m_t = -[∇_h(φ a Θ)/φ - a Θ ∇_h θ̄/θ̄ + ∇_h b] + g (Θ/θ̄ - ρ) e_z
  plus the HDG/SAT coupling (A_ñ - T_ñ)(q_b - q̂) with T = |A_ñ| (LMARS) at every face node.
  The discrete energy identity of the 1D operator holds for any φ > 0 up to the terms
  m_x ∂_x θ̄ (buoyancy) — exact for a horizontally homogeneous frozen state on a flat mesh,
  otherwise bounded zero-order terms. φ only enters consistency; inconsistencies of L are
  absorbed by the slow part S = N - L.

The traces are eliminated (ε = 0: q̂ = average at interior faces, wall projection at walls), so
the fast operator is applied MATRIX-FREE from the frozen coefficient arrays. It also splits
exactly into a thermodynamic part (ρ, Θ; face terms through the interface variable M̂ = ñ·m̂,
the contravariant LMARS momentum) and a momentum part (face terms through the LMARS pressure P̂):
`fast_rhs_thermo!`, `fast_rhs_momentum!`, `interface_momentum!`. Fast integrators:
  :verlet  Störmer–Verlet (Xu, van der Vegt, Bokhove 2008, eq. 4.6 with f = 0): ρ, Θ half step,
           m full step, ρ, Θ half step; with tau_scale = 0 (central interface) exactly symplectic
  :fb      forward-backward: m explicit, M̂ implicit (relaxation time eps_if), then ρ, Θ
  :imex    explicit IMEX tableau (2-stage RK2: only the LMARS dissipation stabilizes it)
  :hevi    horizontal explicit matrix-free, vertical implicit: with the face momentum as unknown
           the vertical system reduces per node column to element-local dense blocks in
           (ρ, Θ, U = Ja^η·m) and a tridiagonal system for the face momentum (cf. CGDycore.jl,
           DGSEM/JacHDGJNeu.jl); factors are recomputed at every re-freeze. The buoyancy is split
           exactly, g J e_z = Ja^ξ D_ξ(gZ) + Ja^η D_η(gZ), into the horizontal and vertical part.
"""
module Euler2D

using LinearAlgebra
using ..Common: Gas, pressure, stratified_reference
import ..Common: project2d, total2d, output_fields, field_section, sound_speed_max, with_sponge
using ..Common: open_vtkhdf, write_vtkhdf!, plot_fields
using ..TimeStepping: AbstractFastOperator, rhs_nonlinear, IMEX_GM
import ..TimeStepping: rhs_nonlinear!, fast_operator, fast_apply!, fast_rhs_thermo!, fast_rhs_momentum!,
                       interface_update!, fast_rhs_horizontal!, vertical_apply!, vertical_factor, vertical_solve!
using ..HDGRelaxTheta: gll, lagrange_basis, gauss_legendre
import ..HDGRelaxTheta: initial_state, tendency
using ..SplitExplicitEuler: mass_theta_flux, gravity_mean

export Mesh2D, Euler2DModel, HDGFast2D, with_sponge, output_fields, field_section, open_vtkhdf, write_vtkhdf!, plot_fields,
       rhs_nonlinear, rhs_nonlinear!, fast_operator, fast_matrix, sound_speed_max, dss,
       interface_momentum!, fast_rhs_thermo!, fast_rhs_momentum!,
       project2d, total2d, field, stratified_reference, energy2d, node_iq

const NV = 4

include("metric.jl")
include("grid.jl")
include("model.jl")
include("rhs.jl")
include("fast_operator.jl")
include("fast_split.jl")
include("hevi.jl")
include("diagnostics.jl")

end # module
