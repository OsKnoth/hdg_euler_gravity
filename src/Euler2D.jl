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
  :hevi    horizontal explicit matrix-free, vertical implicit (column-local vertical operator on a
           fixed sparsity pattern, values and LU factors updated in place at every re-freeze)
"""
module Euler2D

using LinearAlgebra, SparseArrays
using ..HDGRelaxTheta: AbstractHDGOperator, gll, lagrange_basis, gauss_legendre, IMEX_AE, IMEX_AI, IMEX_GM
import ..HDGRelaxTheta: initial_state, tendency, fast_apply!, fast_workspace, hevi_loop!, verlet_loop!, fb_loop!
using ..SplitExplicitEuler: Gas, pressure, mass_theta_flux, gravity_mean
import ..SplitExplicitEuler: rhs_nonlinear, rhs_nonlinear!, fast_operator, sound_speed_max

export Mesh2D, Euler2DModel, HDGFast2D, with_sponge, output_fields, open_vtkhdf, write_vtkhdf!, plot_fields,
       rhs_nonlinear, rhs_nonlinear!, fast_operator, fast_matrix, sound_speed_max, dss,
       interface_momentum!, fast_rhs_thermo!, fast_rhs_momentum!,
       project2d, total2d, field, stratified_reference, energy2d, node_iq

const NV = 4

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
struct Mesh2D
    Nex::Int; Nez::Int; p::Int; np::Int; Ne::Int
    x0::Float64; Lx::Float64; H::Float64
    xg::Vector{Float64}; wg::Vector{Float64}; D::Matrix{Float64}
    X::Array{Float64,3}; Z::Array{Float64,3}; J::Array{Float64,3}
    Jξx::Array{Float64,3}; Jξz::Array{Float64,3}; Jηx::Array{Float64,3}; Jηz::Array{Float64,3}
    gid::Array{Int,3}; mult::Vector{Float64}
end

elem(m::Mesh2D, ex, ez) = (ez - 1) * m.Nex + ex
node_iq(np, e, r, i, j) = ((e - 1) * NV + (r - 1)) * np^2 + (j - 1) * np + i

"""
    Mesh2D(Nex, Nez, p, Lx, H; x0=0, zs = x -> 0)

`zs` must be periodic with period Lx and smaller than H.
"""
function Mesh2D(Nex::Int, Nez::Int, p::Int, Lx::Real, H::Real; x0::Real = 0.0, zs = x -> 0.0)
    np = p + 1; Ne = Nex * Nez
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    hx, hz = Lx / Nex, H / Nez
    X = zeros(np, np, Ne); Z = zeros(np, np, Ne)
    gid = zeros(Int, np, np, Ne)
    ngx = Nex * p
    for ez in 1:Nez, ex in 1:Nex
        e = (ez - 1) * Nex + ex
        for j in 1:np, i in 1:np
            x = x0 + (ex - 1 + (xg[i] + 1) / 2) * hx
            ζ = (ez - 1 + (xg[j] + 1) / 2) * hz
            X[i, j, e] = x
            Z[i, j, e] = zs(x) + ζ * (H - zs(x)) / H
            gid[i, j, e] = ((ez - 1) * p + j - 1) * ngx + mod((ex - 1) * p + i - 1, ngx) + 1
        end
    end
    Jξx = similar(X); Jξz = similar(X); Jηx = similar(X); Jηz = similar(X); Jac = similar(X)
    for e in 1:Ne
        Xξ = D * X[:, :, e]; Xη = X[:, :, e] * D'
        Zξ = D * Z[:, :, e]; Zη = Z[:, :, e] * D'
        Jξx[:, :, e] = Zη; Jξz[:, :, e] = -Xη
        Jηx[:, :, e] = -Zξ; Jηz[:, :, e] = Xξ
        Jac[:, :, e] = Xξ .* Zη .- Xη .* Zξ
    end
    mult = zeros(maximum(gid))
    for g in gid
        mult[g] += 1
    end
    return Mesh2D(Nex, Nez, p, np, Ne, Float64(x0), Float64(Lx), Float64(H), xg, w, D, X, Z, Jac,
                  Jξx, Jξz, Jηx, Jηz, gid, mult)
end

"""Direct stiffness summation: average the values at shared nodes (continuous field)."""
dss(m::Mesh2D, F::AbstractArray{<:Real,3}) = dss!(Array{Float64}(F), m, zeros(length(m.mult)))

"""In-place DSS of F (np × np × Ne) with a work vector of length = number of global nodes."""
function dss!(F::AbstractArray{Float64,3}, m::Mesh2D, buf::Vector{Float64})
    fill!(buf, 0.0)
    @inbounds for k in eachindex(F)
        buf[m.gid[k]] += F[k]
    end
    @inbounds for g in eachindex(buf)
        buf[g] /= m.mult[g]
    end
    @inbounds for k in eachindex(F)
        F[k] = buf[m.gid[k]]
    end
    return F
end

# ---------------------------------------------------------------------------
# Model and reference state
# ---------------------------------------------------------------------------
"""Work arrays of the nonlinear operator (velocities and pressure at the nodes)."""
struct NCache
    U::Array{Float64,3}; W::Array{Float64,3}; P::Array{Float64,3}
end
NCache(np, Ne) = NCache(zeros(np, np, Ne), zeros(np, np, Ne), zeros(np, np, Ne))

struct Euler2DModel
    mesh::Mesh2D; gas::Gas; nq::Int
    ρref::Array{Float64,3}; Θref::Array{Float64,3}; pref::Array{Float64,3}
    damp::Vector{Float64}       # Rayleigh damping rate [1/s] per unknown (sponge), in N
    qtarget::Vector{Float64}    # state the sponge relaxes to
    vflux::Symbol; gflux::Symbol
    nres::Vector{Float64}       # N of the reference state, subtracted (exact balance)
    cache::NCache
    fastref::Base.RefValue{Any} # fast operator, reused (re-frozen in place)
end

"""
    Euler2DModel(mesh, gas, ρref, Θref; volume_flux=:tec, gravity=:waruszewski, subtract_reference=true)

Hydrostatic reference given by functions ρref(z), Θref(z). Gravity enters as the
non-conservative two-point term ρ_avg (φ_k - φ_i), φ = g z (SBP-compatible with flux
differencing; a pointwise source -g ρ_i makes the linearization about stratified states
unstable with a growth rate O(h)). With subtract_reference the residual of the nodal reference
state is subtracted, so it is steady to round-off for any background and over terrain (this
does not change the Jacobian). Fluxes: see `SplitExplicitEuler.Euler1D`.
"""
function Euler2DModel(mesh::Mesh2D, gas::Gas, ρref, Θref; volume_flux::Symbol = :tec,
                      gravity::Symbol = :waruszewski, subtract_reference::Bool = true)
    volume_flux in (:kg, :tec, :ec, :etec) || error("unknown volume_flux $volume_flux")
    gravity in (:souza, :waruszewski, :artiano) || error("unknown gravity $gravity")
    R = ρref.(mesh.Z); T = Θref.(mesh.Z)
    nq = mesh.Ne * NV * mesh.np^2
    m = Euler2DModel(mesh, gas, nq, R, T, pressure.(Ref(gas), T), zeros(nq), zeros(nq), volume_flux, gravity,
                     zeros(nq), NCache(mesh.np, mesh.Ne), Ref{Any}(nothing))
    subtract_reference || return m
    qr = zeros(nq); Q = fields(m, qr)
    Q[:, :, 1, :] .= R; Q[:, :, 4, :] .= T
    r = rhs_nonlinear!(zeros(nq), m, qr)        # (m.nres is still zero here; do not write into it directly)
    m.nres .= r
    return m
end

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Rayleigh damping -α(x,z) (q - qtarget) added to N (slow part):
top layer α = αtop sin²(π/2 (z - zD)/(H - zD)) for z > zD (Klemp & Lilly 1978 type), and for
xwidth > 0 lateral layers of that width at both ends of the periodic channel,
α = αside sin²(π/2 d/xwidth), d = distance into the layer. Stable for α Δt ≲ 1.
"""
function with_sponge(m::Euler2DModel, qtarget::AbstractVector; zD::Real, αtop::Real = 0.05,
                     xwidth::Real = 0.0, αside::Real = αtop)
    M = m.mesh
    α = zeros(M.np, M.np, M.Ne)
    x0, x1 = M.x0, M.x0 + M.Lx
    for k in eachindex(α)
        z, x = M.Z[k], M.X[k]
        a = z > zD ? αtop * sin(π / 2 * (z - zD) / (M.H - zD))^2 : 0.0
        if xwidth > 0
            d = max(x0 + xwidth - x, x - (x1 - xwidth))
            d > 0 && (a = max(a, αside * sin(π / 2 * min(d, xwidth) / xwidth)^2))
        end
        α[k] = a
    end
    damp = vec(reshape(repeat(reshape(α, M.np, M.np, 1, M.Ne), 1, 1, NV, 1), :))
    return Euler2DModel(M, m.gas, m.nq, m.ρref, m.Θref, m.pref, damp, Vector{Float64}(qtarget), m.vflux, m.gflux,
                        m.nres, m.cache, m.fastref)
end

"""Constant-N reference: θ̄ = θ0 exp(N² z/g), hydrostatic Exner; returns (ρ(z), Θ(z), θ(z))."""
function stratified_reference(gas::Gas; θ0 = 300.0, N = 0.01)
    g, R, γ, p00 = gas.g, gas.R, gas.γ, gas.p00
    cp = R * γ / (γ - 1)
    θ(z) = N == 0 ? θ0 : θ0 * exp(N^2 * z / g)
    π(z) = N == 0 ? 1 - g * z / (cp * θ0) : 1 + g^2 / (cp * θ0 * N^2) * (exp(-N^2 * z / g) - 1)
    ρ(z) = p00 * π(z)^(cp / R) / (R * π(z) * θ(z))
    return ρ, z -> ρ(z) * θ(z), θ
end

fields(m::Euler2DModel, q) = reshape(q, m.mesh.np, m.mesh.np, NV, m.mesh.Ne)
"""Nodal values of variable r, (np × np × Ne)."""
field(m::Euler2DModel, q, r) = fields(m, q)[:, :, r, :]

"""Nodal initialization: `fun(x, z)` returns (ρ, m_x, m_z, Θ)."""
function project2d(m::Euler2DModel, fun)
    q = zeros(m.nq); Q = fields(m, q)
    for k in CartesianIndices(m.mesh.X)
        v = fun(m.mesh.X[k], m.mesh.Z[k])
        for r in 1:NV
            Q[k[1], k[2], r, k[3]] = v[r]
        end
    end
    return q
end

"""Discrete integral Σ J w_i w_j q of variable r."""
function total2d(m::Euler2DModel, q, r)
    Q = fields(m, q); M = m.mesh
    return sum(M.J[i, j, e] * M.wg[i] * M.wg[j] * Q[i, j, r, e] for i in 1:M.np, j in 1:M.np, e in 1:M.Ne)
end

"""Output fields at the nodes as name => (np × np × Ne) array: rho, u, w, theta, and the
deviations from the reference rho_prime, theta_prime, p_prime."""
function output_fields(m::Euler2DModel, q)
    Q = fields(m, q)
    ρ = Q[:, :, 1, :]; Θ = Q[:, :, 4, :]
    return ["rho" => ρ, "u" => Q[:, :, 2, :] ./ ρ, "w" => Q[:, :, 3, :] ./ ρ, "theta" => Θ ./ ρ,
            "rho_prime" => ρ .- m.ρref, "theta_prime" => Θ ./ ρ .- m.Θref ./ m.ρref,
            "p_prime" => pressure.(Ref(m.gas), Θ) .- m.pref]
end

"""
    vtk = open_vtkhdf(filename, model)
    write_vtkhdf!(vtk, model, q, t)
    close(vtk)

Time series in the VTKHDF format (UnstructuredGrid, one VTK_LAGRANGE_QUADRILATERAL per element,
solution resampled to equispaced nodes) for ParaView ≥ 5.13. Needs `using HDF5` (package
extension).
"""
function open_vtkhdf end
function write_vtkhdf! end

"""
    fig = plot_fields(model, q; names=["theta_prime", "w"], title="", zoom=nothing)

Makie figure of nodal output fields on the (curvilinear) mesh; zoom = ((x0, x1), (z0, z1)).
Needs a Makie backend, e.g. `using CairoMakie` (package extension).
"""
function plot_fields end

sound_speed_max(m::Euler2DModel, q) =
    maximum(sqrt.(m.gas.γ .* pressure.(Ref(m.gas), field(m, q, 4)) ./ field(m, q, 1)))

# ---------------------------------------------------------------------------
# Nonlinear operator
# ---------------------------------------------------------------------------
"""LMARS flux along the unit normal (nx, nz) from inside L to outside R."""
@inline function lmars2(γ, ρL, uL, wL, θL, pL, ρR, uR, wR, θR, pR, nx, nz)
    unL = uL * nx + wL * nz; unR = uR * nx + wR * nz
    ρm = 0.5 * (ρL + ρR)
    cm = 0.5 * (sqrt(γ * pL / ρL) + sqrt(γ * pR / ρR))
    ws = 0.5 * (unL + unR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (pL + pR) - 0.5 * ρm * cm * (unR - unL)
    if ws >= 0
        mf = ws * ρL; return mf, mf * uL + ps * nx, mf * wL + ps * nz, mf * θL
    else
        mf = ws * ρR; return mf, mf * uR + ps * nx, mf * wR + ps * nz, mf * θR
    end
end

"""Two-point volume flux (contravariant, metric (ax, az)) and non-conservative gravity term."""
@inline function vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, ub, wb, pb, zb, ax, az)
    um = 0.5 * (ua + ub); wm = 0.5 * (wa + wb); pm = 0.5 * (pa + pb)
    f1, f4 = mass_theta_flux(vf, ρa, Θa, ρb, Θb, ax * um + az * wm, γ)
    gm = gravity_mean(gf, ρa, ρb, γ) * g * (zb - za)
    return f1, f1 * um + ax * pm, f1 * wm + az * pm, f4, ax * gm, az * gm
end

"""Strong-form surface update of one face side: q_b' -= (F̂·ñ - f(q_b)·ñ)/(J w_normal)."""
@inline function side_update!(out, m, q, i, j, e, nx, nz, we, F1, F2, F3, F4)
    M = m.mesh; np = M.np; c = m.cache
    @inbounds begin
        un = c.U[i, j, e] * nx + c.W[i, j, e] * nz
        p = c.P[i, j, e]
        k1, k2, k3, k4 = node_iq(np, e, 1, i, j), node_iq(np, e, 2, i, j), node_iq(np, e, 3, i, j), node_iq(np, e, 4, i, j)
        s = 1 / (M.J[i, j, e] * we)
        out[k1] -= s * (F1 - q[k1] * un)
        out[k2] -= s * (F2 - q[k2] * un - p * nx)
        out[k3] -= s * (F3 - q[k3] * un - p * nz)
        out[k4] -= s * (F4 - q[k4] * un)
    end
    return nothing
end

@inline function face_nl!(out, m, q, iL, jL, eL, iR, jR, eR, nx, nz, we)
    np = m.mesh.np; c = m.cache
    @inbounds begin
        s = hypot(nx, nz); n1, n2 = nx / s, nz / s
        ρL = q[node_iq(np, eL, 1, iL, jL)]; ρR = q[node_iq(np, eR, 1, iR, jR)]
        F1, F2, F3, F4 = lmars2(m.gas.γ, ρL, c.U[iL, jL, eL], c.W[iL, jL, eL], q[node_iq(np, eL, 4, iL, jL)] / ρL, c.P[iL, jL, eL],
                                ρR, c.U[iR, jR, eR], c.W[iR, jR, eR], q[node_iq(np, eR, 4, iR, jR)] / ρR, c.P[iR, jR, eR], n1, n2)
        side_update!(out, m, q, iL, jL, eL, nx, nz, we, s * F1, s * F2, s * F3, s * F4)
        side_update!(out, m, q, iR, jR, eR, -nx, -nz, we, -s * F1, -s * F2, -s * F3, -s * F4)
    end
    return nothing
end

@inline function wall_nl!(out, m, q, i, j, e, nx, nz, we)
    np = m.mesh.np; c = m.cache
    @inbounds begin
        s = hypot(nx, nz); n1, n2 = nx / s, nz / s
        ρ = q[node_iq(np, e, 1, i, j)]; θ = q[node_iq(np, e, 4, i, j)] / ρ
        u, w, p = c.U[i, j, e], c.W[i, j, e], c.P[i, j, e]
        un = u * n1 + w * n2
        F1, F2, F3, F4 = lmars2(m.gas.γ, ρ, u, w, θ, p, ρ, u - 2un * n1, w - 2un * n2, θ, p, n1, n2)
        side_update!(out, m, q, i, j, e, nx, nz, we, s * F1, s * F2, s * F3, s * F4)
    end
    return nothing
end

"""dq/dt of the nonlinear operator N(q)."""
rhs_nonlinear(m::Euler2DModel, q::AbstractVector) = rhs_nonlinear!(zeros(m.nq), m, q)

"""In-place N(q); allocation-free (uses the model's NCache)."""
function rhs_nonlinear!(out, m::Euler2DModel, q::AbstractVector)
    M = m.mesh; np, Ne, D, w, g, γ = M.np, M.Ne, M.D, M.wg, m.gas.g, m.gas.γ
    U, W, P = m.cache.U, m.cache.W, m.cache.P
    vf, gf = m.vflux, m.gflux
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        ρ = q[node_iq(np, e, 1, i, j)]
        U[i, j, e] = q[node_iq(np, e, 2, i, j)] / ρ
        W[i, j, e] = q[node_iq(np, e, 3, i, j)] / ρ
        P[i, j, e] = pressure(m.gas, q[node_iq(np, e, 4, i, j)])
    end
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        s1 = s2 = s3 = s4 = gx = gz = 0.0
        ρa = q[node_iq(np, e, 1, i, j)]; Θa = q[node_iq(np, e, 4, i, j)]
        ua, wa, pa, za = U[i, j, e], W[i, j, e], P[i, j, e], M.Z[i, j, e]
        for k in 1:np
            ρb = q[node_iq(np, e, 1, k, j)]; Θb = q[node_iq(np, e, 4, k, j)]
            f1, f2, f3, f4, g2, g3 = vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, U[k, j, e], W[k, j, e],
                                              P[k, j, e], M.Z[k, j, e], 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]),
                                              0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e]))
            d = 2 * D[i, k]
            s1 += d * f1; s2 += d * f2; s3 += d * f3; s4 += d * f4; gx += D[i, k] * g2; gz += D[i, k] * g3
            ρb = q[node_iq(np, e, 1, i, k)]; Θb = q[node_iq(np, e, 4, i, k)]
            f1, f2, f3, f4, g2, g3 = vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, U[i, k, e], W[i, k, e],
                                              P[i, k, e], M.Z[i, k, e], 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e]),
                                              0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e]))
            d = 2 * D[j, k]
            s1 += d * f1; s2 += d * f2; s3 += d * f3; s4 += d * f4; gx += D[j, k] * g2; gz += D[j, k] * g3
        end
        Jinv = 1 / M.J[i, j, e]
        out[node_iq(np, e, 1, i, j)] = -s1 * Jinv
        out[node_iq(np, e, 2, i, j)] = -(s2 + gx) * Jinv
        out[node_iq(np, e, 3, i, j)] = -(s3 + gz) * Jinv
        out[node_iq(np, e, 4, i, j)] = -s4 * Jinv
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np                                         # ξ-face to the right (periodic)
            face_nl!(out, m, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[np])
        end
        for i in 1:np                                         # η-faces
            if ez < M.Nez
                face_nl!(out, m, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            else
                wall_nl!(out, m, q, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            end
            ez == 1 && wall_nl!(out, m, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[1])
        end
    end
    @inbounds for k in eachindex(out)
        out[k] -= m.nres[k] + m.damp[k] * (q[k] - m.qtarget[k])
    end
    return out
end

# ---------------------------------------------------------------------------
# Fast operator (matrix-free, traces eliminated)
# ---------------------------------------------------------------------------
mutable struct HDGFast2D <: AbstractHDGOperator
    mesh::Mesh2D; gas::Gas; nq::Int; nt::Int; eps::Float64; tau::Float64
    Mdiag::Vector{Float64}; Minv::Vector{Float64}; fb::Vector{Float64}
    # frozen, continuous coefficients at the nodes (np × np × Ne)
    θ::Array{Float64,3}; a::Array{Float64,3}; φ::Array{Float64,3}; φa::Array{Float64,3}; b::Array{Float64,3}
    gξx::Array{Float64,3}; gξz::Array{Float64,3}; gηx::Array{Float64,3}; gηz::Array{Float64,3}  # Ja ∂θ̄
    cmax::Float64
    # HEVI: vertical operator A_v on a fixed pattern, LU per substep size
    hevi::Bool
    vI::Vector{Int}; vJ::Vector{Int}; vV::Vector{Float64}; vmap::Vector{Int}; dmap::Vector{Int}
    Av::SparseMatrixCSC{Float64,Int}
    lucache::Dict{Float64,Any}
    # interface variable: contravariant momentum ñ·m̂ at the face nodes (Verlet / forward-backward)
    Mhx::Matrix{Float64}            # ξ-faces: (np, Nex Nez), face to the right of element e
    Mhz::Matrix{Float64}            # η-faces: (np, Nex (Nez+1)), index (kz-1) Nex + ex
    eps_if::Float64                 # relaxation time of the interface variable (:fb)
    # work arrays
    work::Vector{Vector{Float64}}
    Fξ::Matrix{Float64}; Fη::Matrix{Float64}; ψ::Matrix{Float64}
    Iint::Matrix{Float64}; dssbuf::Vector{Float64}
end

function HDGFast2D(m::Euler2DModel; tau_scale = 1.0)
    M = m.mesh; np, Ne, nq = M.np, M.Ne, m.nq
    arr() = zeros(np, np, Ne)
    Mdiag = zeros(nq)
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np, r in 1:NV
        Mdiag[node_iq(np, e, r, i, j)] = M.J[i, j, e] * M.wg[i] * M.wg[j]
    end
    xq, wq = gauss_legendre(np)
    Iint = zeros(np, np)                               # Iint[j, k] = ∫_{-1}^{η_j} ℓ_k
    for j in 1:np
        V, _ = lagrange_basis(M.xg, -1.0 .+ 0.5 * (M.xg[j] + 1) .* (xq .+ 1))
        Iint[j, :] = 0.5 * (M.xg[j] + 1) .* (V' * wq)
    end
    return HDGFast2D(M, m.gas, nq, 0, 0.0, Float64(tau_scale), Mdiag, 1 ./ Mdiag, zeros(nq),
                     arr(), arr(), arr(), arr(), arr(), arr(), arr(), arr(), arr(), 0.0,
                     false, Int[], Int[], Float64[], Int[], Int[], spzeros(nq, nq), Dict{Float64,Any}(),
                     zeros(np, Ne), zeros(np, M.Nex * (M.Nez + 1)), 0.0,
                     [zeros(nq) for _ in 1:7], zeros(np, np), zeros(np, np), zeros(np, np), Iint,
                     zeros(length(M.mult)))
end

"""
    fast_operator(model, q; pfac=γ, eps=0.0, tau_scale=1.0, fast=:imex, reuse=true)

Fast operator with coefficients frozen at the full 2D state q (matrix-free). With reuse = true
the operator stored in the model is re-frozen in place (no reallocation); fast = :hevi also
(re)builds the vertical implicit operator.
"""
function fast_operator(m::Euler2DModel, q; pfac = m.gas.γ, eps = 0.0, tau_scale = 1.0, fast::Symbol = :imex,
                       reuse::Bool = true)
    eps == 0 || error("the 2D fast operator eliminates the traces (eps = 0)")
    s = reuse && m.fastref[] isa HDGFast2D ? m.fastref[]::HDGFast2D : HDGFast2D(m; tau_scale = tau_scale)
    s.hevi |= fast == :hevi
    refreeze!(s, m, q, pfac)
    reuse && (m.fastref[] = s)
    return s
end

"""Re-freeze the coefficients of s at the state q (in place)."""
function refreeze!(s::HDGFast2D, m::Euler2DModel, q, pfac)
    M = m.mesh; np, Ne, D, w, g = M.np, M.Ne, M.D, M.wg, m.gas.g
    θ, a, b, φ = s.θ, s.a, s.b, s.φ
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        ρ = q[node_iq(np, e, 1, i, j)]; Θ = q[node_iq(np, e, 4, i, j)]
        p = pressure(m.gas, Θ)
        θ[i, j, e] = Θ / ρ; a[i, j, e] = pfac * p / Θ; b[i, j, e] = (1 - pfac) * p
    end
    dss!(θ, M, s.dssbuf); dss!(a, M, s.dssbuf); dss!(b, M, s.dssbuf)
    # χ = ∫ g/c² dz along the η-lines from χ_s = g z_s / c²(z_s); φ = θ̄ exp(χ)
    @inbounds for ex in 1:M.Nex, i in 1:np
        e1 = elem(M, ex, 1)
        base = g * M.Z[i, 1, e1] / (a[i, 1, e1] * θ[i, 1, e1])
        for ez in 1:M.Nez
            e = elem(M, ex, ez)
            for j in 1:np
                I = base
                for k in 1:np
                    zη = 0.0
                    for l in 1:np
                        zη += D[k, l] * M.Z[i, l, e]
                    end
                    I += s.Iint[j, k] * g / (a[i, k, e] * θ[i, k, e]) * zη
                end
                φ[i, j, e] = I
            end
            base = φ[i, np, e]
        end
    end
    s.cmax = 0.0
    @inbounds for k in eachindex(φ)
        φ[k] = θ[k] * exp(φ[k])
        s.φa[k] = φ[k] * a[k]
        s.cmax = max(s.cmax, sqrt(a[k] * θ[k]))
    end
    fill!(s.fb, 0.0)
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        dξθ = dηθ = dξb = dηb = 0.0
        for k in 1:np
            dξθ += D[i, k] * θ[k, j, e]; dηθ += D[j, k] * θ[i, k, e]
            dξb += D[i, k] * b[k, j, e]; dηb += D[j, k] * b[i, k, e]
        end
        s.gξx[i, j, e] = M.Jξx[i, j, e] * dξθ; s.gξz[i, j, e] = M.Jξz[i, j, e] * dξθ
        s.gηx[i, j, e] = M.Jηx[i, j, e] * dηθ; s.gηz[i, j, e] = M.Jηz[i, j, e] * dηθ
        W = w[i] * w[j]
        s.fb[node_iq(np, e, 2, i, j)] = -W * (M.Jξx[i, j, e] * dξb + M.Jηx[i, j, e] * dηb)
        s.fb[node_iq(np, e, 3, i, j)] = -W * (M.Jξz[i, j, e] * dξb + M.Jηz[i, j, e] * dηb)
    end
    s.hevi && update_vertical!(s)
    return s
end

"""(A_ñ d, T_ñ d) of the SAT at a face node: ñ = scaled outward normal, d = (d1, d2, d3, d4)."""
@inline function sat_parts(a, θ, nx, nz, d2, d3, d4, tau)
    c = sqrt(a * θ); s = hypot(nx, nz); n1, n2 = nx / s, nz / s
    mn = nx * d2 + nz * d3
    dn = n1 * d2 + n2 * d3
    st = tau * s
    return (mn, a * nx * d4, a * nz * d4, θ * mn), (st * a / c * d4, st * c * n1 * dn, st * c * n2 * dn, st * c * d4)
end

@inline function add4!(out, np, e, i, j, f, v1, v2, v3, v4)
    @inbounds begin
        out[node_iq(np, e, 1, i, j)] += f * v1; out[node_iq(np, e, 2, i, j)] += f * v2
        out[node_iq(np, e, 3, i, j)] += f * v3; out[node_iq(np, e, 4, i, j)] += f * v4
    end
    return nothing
end

"""SAT contribution of an interior face node pair L (outward ñ) / R, traces = averages."""
@inline function sat_face!(out, s, q, iL, jL, eL, iR, jR, eR, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        d2 = 0.5 * (q[node_iq(np, eL, 2, iL, jL)] - q[node_iq(np, eR, 2, iR, jR)])
        d3 = 0.5 * (q[node_iq(np, eL, 3, iL, jL)] - q[node_iq(np, eR, 3, iR, jR)])
        d4 = 0.5 * (q[node_iq(np, eL, 4, iL, jL)] - q[node_iq(np, eR, 4, iR, jR)])
        Ad, Td = sat_parts(s.a[iL, jL, eL], s.θ[iL, jL, eL], nx, nz, d2, d3, d4, s.tau)
        # L: (A_ñ - T) d;  R: (A_{-ñ} - T)(-d) = (A_ñ + T) d
        add4!(out, np, eL, iL, jL, we, Ad[1] - Td[1], Ad[2] - Td[2], Ad[3] - Td[3], Ad[4] - Td[4])
        add4!(out, np, eR, iR, jR, we, Ad[1] + Td[1], Ad[2] + Td[2], Ad[3] + Td[3], Ad[4] + Td[4])
    end
    return nothing
end

"""SAT at a wall node: q - q̂ = (0, n̂ (n̂·m), 0)."""
@inline function sat_wall!(out, s, q, i, j, e, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        h = hypot(nx, nz); n1, n2 = nx / h, nz / h
        mn = n1 * q[node_iq(np, e, 2, i, j)] + n2 * q[node_iq(np, e, 3, i, j)]
        Ad, Td = sat_parts(s.a[i, j, e], s.θ[i, j, e], nx, nz, n1 * mn, n2 * mn, 0.0, s.tau)
        add4!(out, np, e, i, j, we, Ad[1] - Td[1], Ad[2] - Td[2], Ad[3] - Td[3], Ad[4] - Td[4])
    end
    return nothing
end

"""
out = (K_part q + G_part H q + f) ./ M, matrix-free; HP: horizontal (ξ) part, VP: vertical (η)
part including buoyancy. Allocation-free.
"""
function fast_kernel!(out, s::HDGFast2D, q, f, ::Val{HP}, ::Val{VP}) where {HP,VP}
    M = s.mesh; np, D, w, g = M.np, M.D, M.wg, s.gas.g
    Fξ, Fη, ψ = s.Fξ, s.Fη, s.ψ
    fill!(out, 0.0)
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            mx = q[node_iq(np, e, 2, i, j)]; mz = q[node_iq(np, e, 3, i, j)]
            Fξ[i, j] = M.Jξx[i, j, e] * mx + M.Jξz[i, j, e] * mz
            Fη[i, j] = M.Jηx[i, j, e] * mx + M.Jηz[i, j, e] * mz
            ψ[i, j] = s.φa[i, j, e] * q[node_iq(np, e, 4, i, j)]
        end
        for j in 1:np, i in 1:np
            ρ = q[node_iq(np, e, 1, i, j)]; mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]; Θ = q[node_iq(np, e, 4, i, j)]
            θ = s.θ[i, j, e]; aΘθ = s.a[i, j, e] * Θ / θ; iφ = 1 / s.φ[i, j, e]
            r1 = r2 = r3 = r4 = 0.0
            if HP
                div = dψ = 0.0
                for k in 1:np
                    div += D[i, k] * Fξ[k, j]; dψ += D[i, k] * ψ[k, j]
                end
                gx, gz = s.gξx[i, j, e], s.gξz[i, j, e]
                r1 -= div
                r4 -= θ * div + mx * gx + mz * gz
                r2 -= M.Jξx[i, j, e] * dψ * iφ - aΘθ * gx
                r3 -= M.Jξz[i, j, e] * dψ * iφ - aΘθ * gz
            end
            if VP
                div = dψ = 0.0
                for k in 1:np
                    div += D[j, k] * Fη[i, k]; dψ += D[j, k] * ψ[i, k]
                end
                gx, gz = s.gηx[i, j, e], s.gηz[i, j, e]
                r1 -= div
                r4 -= θ * div + mx * gx + mz * gz
                r2 -= M.Jηx[i, j, e] * dψ * iφ - aΘθ * gx
                r3 -= M.Jηz[i, j, e] * dψ * iφ - aΘθ * gz
                r3 += M.J[i, j, e] * g * (Θ / θ - ρ)
            end
            W = w[i] * w[j]
            out[node_iq(np, e, 1, i, j)] = W * r1; out[node_iq(np, e, 2, i, j)] = W * r2
            out[node_iq(np, e, 3, i, j)] = W * r3; out[node_iq(np, e, 4, i, j)] = W * r4
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        if HP
            eR = elem(M, mod1(ex + 1, M.Nex), ez)
            for j in 1:np
                sat_face!(out, s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
            end
        end
        if VP
            for i in 1:np
                if ez < M.Nez
                    sat_face!(out, s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
                else
                    sat_wall!(out, s, q, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
                end
                ez == 1 && sat_wall!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i])
            end
        end
    end
    @inbounds for k in eachindex(out)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

fast_apply!(out, s::HDGFast2D, q, f) = fast_kernel!(out, s, q, f, Val(true), Val(true))

# ---------------------------------------------------------------------------
# Thermodynamic / momentum split of the fast operator with the interface variable
# M̂ = ñ·m̂ (contravariant momentum at the face nodes):
#   interior  M̂ = ñ·{m} - |ñ| a [Θ]/(2c)     (LMARS mass flux, [Θ] = Θ_R - Θ_L)
#   wall      M̂ = 0
#   ρ_b += w (ñ·m_b - M̂)/(J W),  Θ_b += θ̄ w (ñ·m_b - M̂)/(J W)     (ñ outward)
#   m_b += w ñ (a Θ_b - P̂)/(J W),  P̂ = {aΘ} - c [m_n]/2,  wall P̂ = a Θ_b + c n̂·m_b
# thermo + momentum with M̂ = M̂(q) reproduce fast_apply! exactly. The dissipative parts scale
# with tau_scale; tau_scale = 0 gives the central (alternating, θ = 1/2) interface of Xu, van der
# Vegt, Bokhove (J. Sci. Comput. 35, 2008): the operator is then energy conserving and Verlet
# (their eq. 4.6 with f = 0) is exactly symplectic.
# ---------------------------------------------------------------------------
"""Interface momentum M̂ (LMARS) from the state q; with relax = true it is relaxed towards that
value by one implicit Euler step, M̂ ← (ε M̂ + dt M̂*)/(ε + dt) (ε = s.eps_if)."""
function interface_momentum!(s::HDGFast2D, q; relax::Bool = false, dt = 0.0)
    M = s.mesh; np = M.np
    r = relax && s.eps_if > 0 ? s.eps_if / (s.eps_if + dt) : 0.0
    @inbounds for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            v = mhat_face(s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e])
            s.Mhx[j, e] = r * s.Mhx[j, e] + (1 - r) * v
        end
        for i in 1:np
            ftop = ez * M.Nex + ex
            v = ez < M.Nez ? mhat_face(s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e]) : 0.0
            s.Mhz[i, ftop] = r * s.Mhz[i, ftop] + (1 - r) * v
            ez == 1 && (s.Mhz[i, ex] = 0.0)
        end
    end
    return s
end

@inline function mhat_face(s, q, iL, jL, eL, iR, jR, eR, nx, nz)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        c = sqrt(a * θ)
        mL = nx * q[node_iq(np, eL, 2, iL, jL)] + nz * q[node_iq(np, eL, 3, iL, jL)]
        mR = nx * q[node_iq(np, eR, 2, iR, jR)] + nz * q[node_iq(np, eR, 3, iR, jR)]
        return 0.5 * (mL + mR) - s.tau * hypot(nx, nz) * a / (2c) * (q[node_iq(np, eR, 4, iR, jR)] - q[node_iq(np, eL, 4, iL, jL)])
    end
end

@inline function thermo_side!(out, s, q, i, j, e, nx, nz, we, Mh)
    np = s.mesh.np
    @inbounds begin
        d = we * (nx * q[node_iq(np, e, 2, i, j)] + nz * q[node_iq(np, e, 3, i, j)] - Mh)
        out[node_iq(np, e, 1, i, j)] += d
        out[node_iq(np, e, 4, i, j)] += s.θ[i, j, e] * d
    end
    return nothing
end

"""
    fast_rhs_thermo!(out, s, q, f)

ρ and Θ tendencies of the fast operator (rows 1 and 4 of out; the momentum rows are set to
zero), using the interface momentum stored in s (`interface_momentum!`). f: forcing (length nq).
"""
function fast_rhs_thermo!(out, s::HDGFast2D, q, f)
    M = s.mesh; np, D, w = M.np, M.D, M.wg
    Fξ, Fη = s.Fξ, s.Fη
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            mx = q[node_iq(np, e, 2, i, j)]; mz = q[node_iq(np, e, 3, i, j)]
            Fξ[i, j] = M.Jξx[i, j, e] * mx + M.Jξz[i, j, e] * mz
            Fη[i, j] = M.Jηx[i, j, e] * mx + M.Jηz[i, j, e] * mz
        end
        for j in 1:np, i in 1:np
            div = 0.0
            for k in 1:np
                div += D[i, k] * Fξ[k, j] + D[j, k] * Fη[i, k]
            end
            mx = q[node_iq(np, e, 2, i, j)]; mz = q[node_iq(np, e, 3, i, j)]
            W = w[i] * w[j]
            out[node_iq(np, e, 1, i, j)] = -W * div
            out[node_iq(np, e, 4, i, j)] = -W * (s.θ[i, j, e] * div + mx * (s.gξx[i, j, e] + s.gηx[i, j, e]) +
                                                 mz * (s.gξz[i, j, e] + s.gηz[i, j, e]))
            out[node_iq(np, e, 2, i, j)] = 0.0; out[node_iq(np, e, 3, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            nx, nz = M.Jξx[np, j, e], M.Jξz[np, j, e]; Mh = s.Mhx[j, e]
            thermo_side!(out, s, q, np, j, e, nx, nz, w[j], Mh)
            thermo_side!(out, s, q, 1, j, eR, -nx, -nz, w[j], -Mh)
        end
        for i in 1:np
            nx, nz = M.Jηx[i, np, e], M.Jηz[i, np, e]; Mh = s.Mhz[i, ez * M.Nex + ex]
            thermo_side!(out, s, q, i, np, e, nx, nz, w[i], Mh)
            ez < M.Nez && thermo_side!(out, s, q, i, 1, elem(M, ex, ez + 1), -nx, -nz, w[i], -Mh)
            ez == 1 && thermo_side!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i], -s.Mhz[i, ex])
        end
    end
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np, r in (1, 4)
        k = node_iq(np, e, r, i, j)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

@inline function momentum_face!(out, s, q, iL, jL, eL, iR, jR, eR, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        c = sqrt(a * θ); h = hypot(nx, nz); n1, n2 = nx / h, nz / h
        pL = a * q[node_iq(np, eL, 4, iL, jL)]; pR = a * q[node_iq(np, eR, 4, iR, jR)]
        mnL = n1 * q[node_iq(np, eL, 2, iL, jL)] + n2 * q[node_iq(np, eL, 3, iL, jL)]
        mnR = n1 * q[node_iq(np, eR, 2, iR, jR)] + n2 * q[node_iq(np, eR, 3, iR, jR)]
        Ph = 0.5 * (pL + pR) - 0.5 * s.tau * c * (mnR - mnL)
        out[node_iq(np, eL, 2, iL, jL)] += we * nx * (pL - Ph); out[node_iq(np, eL, 3, iL, jL)] += we * nz * (pL - Ph)
        out[node_iq(np, eR, 2, iR, jR)] -= we * nx * (pR - Ph); out[node_iq(np, eR, 3, iR, jR)] -= we * nz * (pR - Ph)
    end
    return nothing
end

@inline function momentum_wall!(out, s, q, i, j, e, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[i, j, e], s.θ[i, j, e]
        c = sqrt(a * θ); h = hypot(nx, nz)
        mn = (nx * q[node_iq(np, e, 2, i, j)] + nz * q[node_iq(np, e, 3, i, j)]) / h
        d = -s.tau * c * mn                                          # p_b - P̂
        out[node_iq(np, e, 2, i, j)] += we * nx * d; out[node_iq(np, e, 3, i, j)] += we * nz * d
    end
    return nothing
end

"""
    fast_rhs_momentum!(out, s, q, f)

Momentum tendencies of the fast operator (rows 2 and 3 of out; the thermodynamic rows are set to
zero): φ-weighted pressure gradient, buoyancy, b-gradient (in f) and the LMARS interface pressure.
"""
function fast_rhs_momentum!(out, s::HDGFast2D, q, f)
    M = s.mesh; np, D, w, g = M.np, M.D, M.wg, s.gas.g
    ψ = s.ψ
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            ψ[i, j] = s.φa[i, j, e] * q[node_iq(np, e, 4, i, j)]
        end
        for j in 1:np, i in 1:np
            dξ = dη = 0.0
            for k in 1:np
                dξ += D[i, k] * ψ[k, j]; dη += D[j, k] * ψ[i, k]
            end
            ρ = q[node_iq(np, e, 1, i, j)]; Θ = q[node_iq(np, e, 4, i, j)]
            θ = s.θ[i, j, e]; aΘθ = s.a[i, j, e] * Θ / θ; iφ = 1 / s.φ[i, j, e]
            W = w[i] * w[j]
            out[node_iq(np, e, 2, i, j)] = -W * ((M.Jξx[i, j, e] * dξ + M.Jηx[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξx[i, j, e] + s.gηx[i, j, e]))
            out[node_iq(np, e, 3, i, j)] = -W * ((M.Jξz[i, j, e] * dξ + M.Jηz[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξz[i, j, e] + s.gηz[i, j, e])) +
                                           W * M.J[i, j, e] * g * (Θ / θ - ρ)
            out[node_iq(np, e, 1, i, j)] = 0.0; out[node_iq(np, e, 4, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            momentum_face!(out, s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
        end
        for i in 1:np
            if ez < M.Nez
                momentum_face!(out, s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            else
                momentum_wall!(out, s, q, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            end
            ez == 1 && momentum_wall!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i])
        end
    end
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np, r in (2, 3)
        k = node_iq(np, e, r, i, j)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

"""Add dt × the rows `rows` of k to q."""
@inline function axpy_rows!(q, dt, k, np, Ne, rows)
    @inbounds for e in 1:Ne, r in rows
        o = ((e - 1) * NV + (r - 1)) * np^2
        for l in 1:np^2
            q[o+l] += dt * k[o+l]
        end
    end
    return q
end

"""Störmer–Verlet: ρ, Θ half step with M̂(qⁿ); m full step with P̂(Θⁿ⁺¹ᐟ², mⁿ); ρ, Θ half
step with M̂(Θⁿ⁺¹ᐟ², mⁿ⁺¹). Second order, explicit, allocation-free."""
function verlet_loop!(q, s::HDGFast2D, f, dt, nsteps, kT, km)
    np, Ne = s.mesh.np, s.mesh.Ne
    for _ in 1:nsteps
        interface_momentum!(s, q)
        fast_rhs_thermo!(kT, s, q, f);   axpy_rows!(q, dt / 2, kT, np, Ne, (1, 4))
        fast_rhs_momentum!(km, s, q, f); axpy_rows!(q, dt, km, np, Ne, (2, 3))
        interface_momentum!(s, q)
        fast_rhs_thermo!(kT, s, q, f);   axpy_rows!(q, dt / 2, kT, np, Ne, (1, 4))
    end
    return q
end

"""Forward-backward Euler: m explicit with P̂(qⁿ); interface momentum implicit (relaxed with
time scale s.eps_if, = LMARS value of (Θⁿ, mⁿ⁺¹) for eps_if = 0); ρ, Θ with M̂ⁿ⁺¹. First order."""
function fb_loop!(q, s::HDGFast2D, f, dt, nsteps, kT, km)
    np, Ne = s.mesh.np, s.mesh.Ne
    s.eps_if > 0 && interface_momentum!(s, q)                    # start the relaxed variable at equilibrium
    for _ in 1:nsteps
        fast_rhs_momentum!(km, s, q, f); axpy_rows!(q, dt, km, np, Ne, (2, 3))
        interface_momentum!(s, q; relax = true, dt = dt)
        fast_rhs_thermo!(kT, s, q, f);   axpy_rows!(q, dt, kT, np, Ne, (1, 4))
    end
    return q
end
fast_workspace(s::HDGFast2D) = (s.work[1], s.work[2], s.work[3], s.work[4], s.work[5], s.work[6])
tendency(s::HDGFast2D, q; F_slow = nothing) =
    fast_apply!(zeros(s.nq), s, q, F_slow === nothing ? s.fb : s.fb .+ s.Mdiag .* F_slow)
initial_state(s::HDGFast2D, q) = Vector{Float64}(q)

"""Dense matrix of q ↦ (K + G H) q ./ M (testing and spectra; small meshes only)."""
function fast_matrix(s::HDGFast2D)
    A = zeros(s.nq, s.nq); e = zeros(s.nq); z = zeros(s.nq); col = zeros(s.nq)
    for k in 1:s.nq
        e[k] = 1.0
        fast_kernel!(col, s, e, z, Val(true), Val(true))
        A[:, k] .= col
        e[k] = 0.0
    end
    return A
end

# ---------------------------------------------------------------------------
# HEVI: vertical operator (fixed pattern) and IMEX loop
# ---------------------------------------------------------------------------
"""Fill the values (or, with build = true, the pattern) of A_v = vertical part of K + G H
(unscaled, i.e. M dq/dt). Every structural entry is always written, so the order is fixed."""
mutable struct Triplets
    I::Vector{Int}; J::Vector{Int}; V::Vector{Float64}; n::Int; build::Bool
end
@inline function (t::Triplets)(r, c, v)
    t.n += 1
    if t.build
        push!(t.I, r); push!(t.J, c); push!(t.V, v)
    else
        @inbounds t.V[t.n] = v
    end
    return nothing
end

@inline function sat_block(a, θ, nx, nz, tau)
    c = sqrt(a * θ); h = hypot(nx, nz); n1, n2 = nx / h, nz / h; st = tau * h
    A = (0.0, nx, nz, 0.0, 0.0, 0.0, 0.0, a * nx, 0.0, 0.0, 0.0, a * nz, 0.0, θ * nx, θ * nz, 0.0)
    T = (0.0, 0.0, 0.0, st * a / c, 0.0, st * c * n1^2, st * c * n1 * n2, 0.0,
         0.0, st * c * n1 * n2, st * c * n2^2, 0.0, 0.0, 0.0, 0.0, st * c)
    return A, T
end

function vertical_triplets!(s::HDGFast2D, build::Bool)
    M = s.mesh; np, D, w, g = M.np, M.D, M.wg, s.gas.g
    if build
        empty!(s.vI); empty!(s.vJ); empty!(s.vV)
    end
    add = Triplets(s.vI, s.vJ, s.vV, 0, build)
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        W = w[i] * w[j]; θ = s.θ[i, j, e]; a = s.a[i, j, e]; iφ = 1 / s.φ[i, j, e]
        rρ, rx, rz, rΘ = node_iq(np, e, 1, i, j), node_iq(np, e, 2, i, j), node_iq(np, e, 3, i, j), node_iq(np, e, 4, i, j)
        for r in 1:NV
            add(node_iq(np, e, r, i, j), node_iq(np, e, r, i, j), 0.0)      # diagonal placeholders
        end
        for k in 1:np
            ck2, ck3, ck4 = node_iq(np, e, 2, i, k), node_iq(np, e, 3, i, k), node_iq(np, e, 4, i, k)
            add(rρ, ck2, -W * D[j, k] * M.Jηx[i, k, e]); add(rρ, ck3, -W * D[j, k] * M.Jηz[i, k, e])
            add(rΘ, ck2, -W * θ * D[j, k] * M.Jηx[i, k, e]); add(rΘ, ck3, -W * θ * D[j, k] * M.Jηz[i, k, e])
            c = -W * D[j, k] * s.φa[i, k, e] * iφ
            add(rx, ck4, c * M.Jηx[i, j, e]); add(rz, ck4, c * M.Jηz[i, j, e])
        end
        gx, gz = s.gηx[i, j, e], s.gηz[i, j, e]
        add(rΘ, rx, -W * gx); add(rΘ, rz, -W * gz)
        add(rx, rΘ, W * a * gx / θ); add(rz, rΘ, W * a * gz / θ + W * M.J[i, j, e] * g / θ)
        add(rz, rρ, -W * M.J[i, j, e] * g)
    end
    # SAT blocks B = we (A_ñ - T) applied to ½(q_L - q_R) (interior) or n̂ n̂ᵀ m (walls)
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        for i in 1:np
            nx, nz = M.Jηx[i, np, e], M.Jηz[i, np, e]
            A, T = sat_block(s.a[i, np, e], s.θ[i, np, e], nx, nz, s.tau)
            if ez < M.Nez
                eU = elem(M, ex, ez + 1)
                for r in 1:NV, t in 1:NV
                    k = 4(r - 1) + t
                    bl = 0.5 * w[i] * (A[k] - T[k]); br = 0.5 * w[i] * (A[k] + T[k])   # R block: (A_ñ + T)
                    rL, cL = node_iq(np, e, r, i, np), node_iq(np, e, t, i, np)
                    rR, cR = node_iq(np, eU, r, i, 1), node_iq(np, eU, t, i, 1)
                    add(rL, cL, bl); add(rL, cR, -bl); add(rR, cL, br); add(rR, cR, -br)
                end
            else
                wall_block!(add, A, T, np, e, i, np, nx, nz, w[i])
            end
            if ez == 1
                nxb, nzb = -M.Jηx[i, 1, e], -M.Jηz[i, 1, e]
                A, T = sat_block(s.a[i, 1, e], s.θ[i, 1, e], nxb, nzb, s.tau)
                wall_block!(add, A, T, np, e, i, 1, nxb, nzb, w[i])
            end
        end
    end
    return add.n
end

function wall_block!(add, A, T, np, e, i, j, nx, nz, we)
    h = hypot(nx, nz); n = (nx / h, nz / h)
    for r in 1:NV, t in 1:2, u in 1:2               # B (I - P): columns m_u, (I - P)_{t u} = n_t n_u
        k = 4(r - 1) + 1 + t
        add(node_iq(np, e, r, i, j), node_iq(np, e, 1 + u, i, j), we * (A[k] - T[k]) * n[t] * n[u])
    end
end

"""Update A_v (pattern built on the first call) and the cached LU factors in place."""
function update_vertical!(s::HDGFast2D)
    if isempty(s.vmap)
        vertical_triplets!(s, true)
        s.Av = sparse(s.vI, s.vJ, s.vV, s.nq, s.nq)
        s.vmap = [nzindex(s.Av, s.vI[k], s.vJ[k]) for k in eachindex(s.vI)]
        s.dmap = [nzindex(s.Av, r, r) for r in 1:s.nq]
    else
        vertical_triplets!(s, false)
    end
    nz = s.Av.nzval
    fill!(nz, 0.0)
    @inbounds for k in eachindex(s.vV)
        nz[s.vmap[k]] += s.vV[k]
    end
    for (dt, (B, F)) in s.lucache
        shifted!(B, s, dt)
        lu!(F, B)
    end
    return s
end

function nzindex(A::SparseMatrixCSC, r, c)
    rng = A.colptr[c]:A.colptr[c+1]-1
    k = searchsortedfirst(view(A.rowval, rng), r)
    (k <= length(rng) && A.rowval[rng[k]] == r) || error("entry ($r, $c) not in the pattern")
    return rng[k]
end

"""B = M - dt γ A_v on the pattern of A_v (diagonal included)."""
function shifted!(B, s, dt)
    c = dt * IMEX_GM
    @inbounds for k in eachindex(B.nzval)
        B.nzval[k] = -c * s.Av.nzval[k]
    end
    @inbounds for r in 1:s.nq
        B.nzval[s.dmap[r]] += s.Mdiag[r]
    end
    return B
end

function hevi_factor(s::HDGFast2D, dt)
    get!(s.lucache, Float64(dt)) do
        B = copy(s.Av)
        shifted!(B, s, dt)
        (B, lu(B))
    end
end

"""HEVI with the IMEX tableau of HDGRelaxTheta: horizontal part (matrix-free) explicit,
vertical part A_v implicit; allocation-free apart from the sparse triangular solves."""
function hevi_loop!(q, s::HDGFast2D, f, dt, nsteps, Y, k1, k2, tmp)
    s.hevi || error("operator built without the vertical part: use fast_operator(...; fast = :hevi)")
    _, F = hevi_factor(s, dt)
    rhs = s.work[7]
    a21, a31, a32, i32 = IMEX_AE[2, 1], IMEX_AE[3, 1], IMEX_AE[3, 2], IMEX_AI[3, 2]
    Md, Mi = s.Mdiag, s.Minv
    for _ in 1:nsteps
        fast_kernel!(k1, s, q, f, Val(true), Val(false))             # explicit stage 1 (Y1 = q)
        @. rhs = Md * (q + (dt * a21) * k1)
        ldiv!(Y, F, rhs)                                             # implicit stage 2
        fast_kernel!(k2, s, Y, f, Val(true), Val(false))
        mul!(tmp, s.Av, Y)
        @. rhs = Md * (q + dt * (a31 * k1 + a32 * k2 + i32 * Mi * tmp))
        ldiv!(q, F, rhs)                                             # stage 3 = new state
    end
    return q
end

"""Weighted energy of a deviation dq with respect to the frozen coefficients of s (needs ∂_zθ̄ > 0)."""
function energy2d(s::HDGFast2D, dq)
    M = s.mesh; np = M.np
    Q = reshape(dq, np, np, NV, M.Ne)
    E = 0.0
    for e in 1:M.Ne
        θe = s.θ[:, :, e]
        θz = (M.Jξz[:, :, e] .* (M.D * θe) .+ M.Jηz[:, :, e] .* (θe * M.D')) ./ M.J[:, :, e]
        χθ = Q[:, :, 4, e] .- θe .* Q[:, :, 1, e]
        dens = s.φ[:, :, e] .* (Q[:, :, 2, e] .^ 2 .+ Q[:, :, 3, e] .^ 2 .+ s.a[:, :, e] .* Q[:, :, 4, e] .^ 2 ./ θe .+
                                9.81 .* χθ .^ 2 ./ (θe .* θz))
        E += 0.5 * sum(M.J[:, :, e] .* (M.wg * M.wg') .* dens)
    end
    return E
end


end # module
