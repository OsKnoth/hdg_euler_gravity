"""
    Euler2D

Split-explicit (MIS) solver for the nonlinear compressible Euler equations with gravity in an
x–z slice with terrain, q = (ρ, m_x = ρu, m_z = ρw, Θ = ρθ):

    ρ_t + ∇·m = 0,   m_t + ∇·(m⊗u) + ∇p = -ρ g e_z,   Θ_t + ∇·(θ m) = 0,   p = p00 (RΘ/p00)^γ

Mesh: periodic in x, walls at the terrain z_s(x) and at z = H; uniform elements in (x, ζ),
terrain-following z = z_s(x) + ζ (H - z_s(x))/H (Gal-Chen), tensor GLL nodes, collocation metrics
Ja^ξ = (z_η, -x_η), Ja^η = (-z_ξ, x_ξ) (discrete metric identities hold exactly in 2D).

Full operator N (DGSEM, strong form):
  volume  flux differencing, Kennedy–Gruber two-point fluxes with averaged metrics
  faces   LMARS along the face normal; walls by mirror state
  The pressure enters as p' = p - p_ref and gravity as -g (ρ - ρ_ref): the reference
  (hydrostatic, given by functions of z) is subtracted analytically, so the rest state ρ_ref,
  Θ_ref is steady to round-off, also over terrain.

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

Fast integrators: :imex (explicit, traces relaxed/averaged, from HDGRelaxTheta) and :hevi
(horizontal explicit, vertical implicit: K = K_h + K_v, η-face traces eliminated in the implicit
part; same IMEX tableau).
"""
module Euler2D

using LinearAlgebra, SparseArrays
using ..HDGRelaxTheta: AbstractHDGOperator, gll, lagrange_basis, gauss_legendre, forcing_vector
import ..HDGRelaxTheta: run_hevi, initial_state, tendency
using ..SplitExplicitEuler: Gas, pressure
import ..SplitExplicitEuler: rhs_nonlinear, fast_operator, sound_speed_max

export Mesh2D, Euler2DModel, HDGFast2D, rhs_nonlinear, fast_operator, sound_speed_max, dss,
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
function dss(m::Mesh2D, F::AbstractArray{<:Real,3})
    s = zeros(length(m.mult))
    for k in eachindex(F)
        s[m.gid[k]] += F[k]
    end
    s ./= m.mult
    return [s[m.gid[k]] for k in CartesianIndices(F)]
end

# ---------------------------------------------------------------------------
# Model and reference state
# ---------------------------------------------------------------------------
struct Euler2DModel
    mesh::Mesh2D; gas::Gas; nq::Int
    ρref::Array{Float64,3}; Θref::Array{Float64,3}; pref::Array{Float64,3}
end

"""Model with hydrostatic reference given by functions ρref(z), Θref(z)."""
function Euler2DModel(mesh::Mesh2D, gas::Gas, ρref, Θref)
    R = ρref.(mesh.Z); T = Θref.(mesh.Z)
    return Euler2DModel(mesh, gas, mesh.Ne * NV * mesh.np^2, R, T, pressure.(Ref(gas), T))
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

sound_speed_max(m::Euler2DModel, q) =
    maximum(sqrt.(m.gas.γ .* pressure.(Ref(m.gas), field(m, q, 4)) ./ field(m, q, 1)))

# ---------------------------------------------------------------------------
# Nonlinear operator
# ---------------------------------------------------------------------------
@inline function kg2(ρi, ui, wi, θi, pi, ρj, uj, wj, θj, pj, ax, az)
    ρa = 0.5 * (ρi + ρj); ua = 0.5 * (ui + uj); wa = 0.5 * (wi + wj); pa = 0.5 * (pi + pj)
    fρ = ρa * (ax * ua + az * wa)
    return fρ, fρ * ua + ax * pa, fρ * wa + az * pa, fρ * 0.5 * (θi + θj)
end

"""LMARS flux along the unit normal (nx, nz) from inside L to outside R; p = full pressure
(for the jump and the sound speed), pp = p - p_ref (enters the momentum flux)."""
@inline function lmars2(γ, ρL, uL, wL, θL, pL, ppL, ρR, uR, wR, θR, pR, ppR, nx, nz)
    unL = uL * nx + wL * nz; unR = uR * nx + wR * nz
    ρm = 0.5 * (ρL + ρR)
    cm = 0.5 * (sqrt(γ * pL / ρL) + sqrt(γ * pR / ρR))
    ws = 0.5 * (unL + unR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (ppL + ppR) - 0.5 * ρm * cm * (unR - unL)
    if ws >= 0
        mf = ws * ρL; return mf, mf * uL + ps * nx, mf * wL + ps * nz, mf * θL
    else
        mf = ws * ρR; return mf, mf * uR + ps * nx, mf * wR + ps * nz, mf * θR
    end
end

"""dq/dt of the nonlinear operator N(q)."""
function rhs_nonlinear(m::Euler2DModel, q::AbstractVector)
    M = m.mesh; np, Ne, D, w, g, γ = M.np, M.Ne, M.D, M.wg, m.gas.g, m.gas.γ
    Q = fields(m, q)
    out = zeros(m.nq); O = fields(m, out)
    ρ = Q[:, :, 1, :]; U = Q[:, :, 2, :] ./ ρ; W = Q[:, :, 3, :] ./ ρ; θ = Q[:, :, 4, :] ./ ρ
    P = pressure.(Ref(m.gas), Q[:, :, 4, :]); PP = P .- m.pref
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        s1 = s2 = s3 = s4 = 0.0
        for k in 1:np
            ax = 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]); az = 0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e])
            f1, f2, f3, f4 = kg2(ρ[i, j, e], U[i, j, e], W[i, j, e], θ[i, j, e], PP[i, j, e],
                                 ρ[k, j, e], U[k, j, e], W[k, j, e], θ[k, j, e], PP[k, j, e], ax, az)
            d = 2 * D[i, k]
            s1 += d * f1; s2 += d * f2; s3 += d * f3; s4 += d * f4
            ax = 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e]); az = 0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e])
            f1, f2, f3, f4 = kg2(ρ[i, j, e], U[i, j, e], W[i, j, e], θ[i, j, e], PP[i, j, e],
                                 ρ[i, k, e], U[i, k, e], W[i, k, e], θ[i, k, e], PP[i, k, e], ax, az)
            d = 2 * D[j, k]
            s1 += d * f1; s2 += d * f2; s3 += d * f3; s4 += d * f4
        end
        Jinv = 1 / M.J[i, j, e]
        O[i, j, 1, e] = -s1 * Jinv
        O[i, j, 2, e] = -s2 * Jinv
        O[i, j, 3, e] = -s3 * Jinv - g * (ρ[i, j, e] - m.ρref[i, j, e])
        O[i, j, 4, e] = -s4 * Jinv
    end
    # one side of a face: node (i,j,e), outward scaled normal (nx,nz), edge weight we, flux F̂·ñ
    @inline function side!(i, j, e, nx, nz, we, F)
        un = U[i, j, e] * nx + W[i, j, e] * nz
        f = (ρ[i, j, e] * un, Q[i, j, 2, e] * un + PP[i, j, e] * nx, Q[i, j, 3, e] * un + PP[i, j, e] * nz,
             Q[i, j, 4, e] * un)
        c = 1 / (M.J[i, j, e] * we)
        for r in 1:NV
            O[i, j, r, e] -= c * (F[r] - f[r])
        end
    end
    @inline function face!(iL, jL, eL, iR, jR, eR, nx, nz, we)      # (nx,nz) outward for L
        s = hypot(nx, nz); n1, n2 = nx / s, nz / s
        F = lmars2(γ, ρ[iL, jL, eL], U[iL, jL, eL], W[iL, jL, eL], θ[iL, jL, eL], P[iL, jL, eL], PP[iL, jL, eL],
                   ρ[iR, jR, eR], U[iR, jR, eR], W[iR, jR, eR], θ[iR, jR, eR], P[iR, jR, eR], PP[iR, jR, eR], n1, n2)
        side!(iL, jL, eL, nx, nz, we, s .* F)
        side!(iR, jR, eR, -nx, -nz, we, (-s) .* F)
    end
    @inline function wall!(i, j, e, nx, nz, we)
        s = hypot(nx, nz); n1, n2 = nx / s, nz / s
        un = U[i, j, e] * n1 + W[i, j, e] * n2
        uR, wR = U[i, j, e] - 2un * n1, W[i, j, e] - 2un * n2
        F = lmars2(γ, ρ[i, j, e], U[i, j, e], W[i, j, e], θ[i, j, e], P[i, j, e], PP[i, j, e],
                   ρ[i, j, e], uR, wR, θ[i, j, e], P[i, j, e], PP[i, j, e], n1, n2)
        side!(i, j, e, nx, nz, we, s .* F)
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np                                         # ξ-face to the right (periodic)
            face!(np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[np])
        end
        for i in 1:np                                         # η-faces
            if ez < M.Nez
                eU = elem(M, ex, ez + 1)
                face!(i, np, e, i, 1, eU, M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            else
                wall!(i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            end
            ez == 1 && wall!(i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[1])
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Fast operator
# ---------------------------------------------------------------------------
struct HDGFast2D <: AbstractHDGOperator
    mesh::Mesh2D; nq::Int; nt::Int; nth::Int; eps::Float64
    K::SparseMatrixCSC{Float64,Int}; G::SparseMatrixCSC{Float64,Int}; H::SparseMatrixCSC{Float64,Int}
    L::SparseMatrixCSC{Float64,Int}; Mbig::SparseMatrixCSC{Float64,Int}
    Mdiag::Vector{Float64}; fb::Vector{Float64}
    Ah::SparseMatrixCSC{Float64,Int}; Av::SparseMatrixCSC{Float64,Int}   # (K_h + G_h H), (K_v + G_v H)
    cmax::Float64
    a::Array{Float64,3}; θ::Array{Float64,3}; φ::Array{Float64,3}
end

"""
    fast_operator(model, q; pfac=γ, eps=0.0, tau_scale=1.0)

Fast operator with coefficients frozen at the full 2D state q.
"""
function fast_operator(m::Euler2DModel, q; pfac = m.gas.γ, eps = 0.0, tau_scale = 1.0)
    M = m.mesh; np, Ne, Nex, Nez, D, w, g = M.np, M.Ne, M.Nex, M.Nez, M.D, M.wg, m.gas.g
    Q = fields(m, q)
    P = pressure.(Ref(m.gas), Q[:, :, 4, :])
    θb = dss(M, Q[:, :, 4, :] ./ Q[:, :, 1, :])
    a = dss(M, pfac .* P ./ Q[:, :, 4, :])
    b = dss(M, (1 - pfac) .* P)
    c2 = a .* θb
    # χ = ∫ g/c² dz along the η-lines, from χ_s = g z_s / c²(z_s) at the surface
    xq, wq = gauss_legendre(np)
    Iint = zeros(np, np)
    for i in 1:np
        V, _ = lagrange_basis(M.xg, -1.0 .+ 0.5 * (M.xg[i] + 1) .* (xq .+ 1))
        Iint[i, :] = 0.5 * (M.xg[i] + 1) .* (V' * wq)
    end
    χ = zeros(np, np, Ne)
    for ex in 1:Nex, i in 1:np
        e1 = elem(M, ex, 1)
        base = g * M.Z[i, 1, e1] / c2[i, 1, e1]
        for ez in 1:Nez
            e = elem(M, ex, ez)
            Zη = M.Z[i, :, e]' * D'                           # z_η along the η-line
            I = base .+ Iint * (g ./ c2[i, :, e] .* vec(Zη))
            χ[i, :, e] = I
            base = I[end]
        end
    end
    φ = θb .* exp.(χ)

    nq = m.nq
    nfx = Nex * Nez
    nth = nfx * np * NV
    nt = nth + Nex * (Nez + 1) * np * NV
    itx(fx, j, r) = ((fx - 1) * np + (j - 1)) * NV + r
    itz(fz, i, r) = nth + ((fz - 1) * np + (i - 1)) * NV + r
    iqn(e, r, i, j) = node_iq(np, e, r, i, j)
    KhI, KhJ, KhV = Int[], Int[], Float64[]; KvI, KvJ, KvV = Int[], Int[], Float64[]
    GhI, GhJ, GhV = Int[], Int[], Float64[]; GvI, GvJ, GvV = Int[], Int[], Float64[]
    addh!(i, j, v) = v == 0 || (push!(KhI, i); push!(KhJ, j); push!(KhV, v))
    addv!(i, j, v) = v == 0 || (push!(KvI, i); push!(KvJ, j); push!(KvV, v))
    fb = zeros(nq)
    Mdiag = zeros(nq)

    for e in 1:Ne
        θe, ae, φe, be = θb[:, :, e], a[:, :, e], φ[:, :, e], b[:, :, e]
        Dξθ = D * θe; Dηθ = θe * D'; Dξb = D * be; Dηb = be * D'
        for j in 1:np, i in 1:np
            W = w[i] * w[j]; Je = M.J[i, j, e]
            ξx, ξz, ηx, ηz = M.Jξx[i, j, e], M.Jξz[i, j, e], M.Jηx[i, j, e], M.Jηz[i, j, e]
            for r in 1:NV
                Mdiag[iqn(e, r, i, j)] = Je * W
            end
            rρ, rx, rz, rΘ = iqn(e, 1, i, j), iqn(e, 2, i, j), iqn(e, 3, i, j), iqn(e, 4, i, j)
            for k in 1:np
                # ξ-direction: node (k, j)
                dk = D[i, k]
                if dk != 0
                    addh!(rρ, iqn(e, 2, k, j), -W * dk * M.Jξx[k, j, e])
                    addh!(rρ, iqn(e, 3, k, j), -W * dk * M.Jξz[k, j, e])
                    addh!(rΘ, iqn(e, 2, k, j), -W * θe[i, j] * dk * M.Jξx[k, j, e])
                    addh!(rΘ, iqn(e, 3, k, j), -W * θe[i, j] * dk * M.Jξz[k, j, e])
                    c = -W * dk * φe[k, j] * ae[k, j] / φe[i, j]
                    addh!(rx, iqn(e, 4, k, j), c * ξx)
                    addh!(rz, iqn(e, 4, k, j), c * ξz)
                end
                # η-direction: node (i, k)
                dk = D[j, k]
                if dk != 0
                    addv!(rρ, iqn(e, 2, i, k), -W * dk * M.Jηx[i, k, e])
                    addv!(rρ, iqn(e, 3, i, k), -W * dk * M.Jηz[i, k, e])
                    addv!(rΘ, iqn(e, 2, i, k), -W * θe[i, j] * dk * M.Jηx[i, k, e])
                    addv!(rΘ, iqn(e, 3, i, k), -W * θe[i, j] * dk * M.Jηz[i, k, e])
                    c = -W * dk * φe[i, k] * ae[i, k] / φe[i, j]
                    addv!(rx, iqn(e, 4, i, k), c * ηx)
                    addv!(rz, iqn(e, 4, i, k), c * ηz)
                end
            end
            # -m·∇θ̄ in Θ, +aΘ ∇θ̄/θ̄ in m
            addh!(rΘ, rx, -W * ξx * Dξθ[i, j]); addh!(rΘ, rz, -W * ξz * Dξθ[i, j])
            addv!(rΘ, rx, -W * ηx * Dηθ[i, j]); addv!(rΘ, rz, -W * ηz * Dηθ[i, j])
            addh!(rx, rΘ, W * ae[i, j] * ξx * Dξθ[i, j] / θe[i, j])
            addh!(rz, rΘ, W * ae[i, j] * ξz * Dξθ[i, j] / θe[i, j])
            addv!(rx, rΘ, W * ae[i, j] * ηx * Dηθ[i, j] / θe[i, j])
            addv!(rz, rΘ, W * ae[i, j] * ηz * Dηθ[i, j] / θe[i, j])
            # buoyancy g (Θ/θ̄ - ρ) e_z (vertical part)
            addv!(rz, rΘ, W * Je * g / θe[i, j])
            addv!(rz, rρ, -W * Je * g)
            fb[rx] -= W * (ξx * Dξb[i, j] + ηx * Dηb[i, j])
            fb[rz] -= W * (ξz * Dξb[i, j] + ηz * Dηb[i, j])
        end
    end

    # SAT at one face node: K += we (A - T), G += -we (A - T) on the trace columns;
    # we = quadrature weight ALONG the face (the normal-direction weight cancels against M)
    function sat!(horizontal, e, i, j, nx, nz, we, tcols)
        ae, θe = a[i, j, e], θb[i, j, e]
        cf = sqrt(ae * θe); s = hypot(nx, nz); n1, n2 = nx / s, nz / s
        A = [0 nx nz 0; 0 0 0 ae*nx; 0 0 0 ae*nz; 0 θe*nx θe*nz 0]
        T = tau_scale * s .* [0 0 0 ae/cf; 0 cf*n1^2 cf*n1*n2 0; 0 cf*n1*n2 cf*n2^2 0; 0 0 0 cf]
        B = we .* (A .- T)
        for r in 1:NV, t in 1:NV
            B[r, t] == 0 && continue
            row = iqn(e, r, i, j)
            horizontal ? addh!(row, iqn(e, t, i, j), B[r, t]) : addv!(row, iqn(e, t, i, j), B[r, t])
            if horizontal
                push!(GhI, row); push!(GhJ, tcols[t]); push!(GhV, -B[r, t])
            else
                push!(GvI, row); push!(GvJ, tcols[t]); push!(GvV, -B[r, t])
            end
        end
    end
    HI, HJ, HV = Int[], Int[], Float64[]
    for ez in 1:Nez, ex in 1:Nex
        e = elem(M, ex, ez); eR = elem(M, mod1(ex + 1, Nex), ez)
        fx = (ez - 1) * Nex + ex
        for j in 1:np
            tc = [itx(fx, j, r) for r in 1:NV]
            nx, nz = M.Jξx[np, j, e], M.Jξz[np, j, e]
            sat!(true, e, np, j, nx, nz, w[j], tc)
            sat!(true, eR, 1, j, -nx, -nz, w[j], tc)
            for r in 1:NV
                append!(HI, [tc[r], tc[r]]); append!(HJ, [iqn(e, r, np, j), iqn(eR, r, 1, j)]); append!(HV, [0.5, 0.5])
            end
        end
        for i in 1:np
            ftop = ez * Nex + ex                               # η-face above element (index kz = ez+1)
            tc = [itz(ftop, i, r) for r in 1:NV]
            nx, nz = M.Jηx[i, np, e], M.Jηz[i, np, e]
            sat!(false, e, i, np, nx, nz, w[i], tc)
            if ez < Nez
                eU = elem(M, ex, ez + 1)
                sat!(false, eU, i, 1, -nx, -nz, w[i], tc)
                for r in 1:NV
                    append!(HI, [tc[r], tc[r]]); append!(HJ, [iqn(e, r, i, np), iqn(eU, r, i, 1)]); append!(HV, [0.5, 0.5])
                end
            else
                wall_trace!(HI, HJ, HV, tc, e, i, np, nx, nz, iqn)
            end
            if ez == 1
                tb = [itz(ex, i, r) for r in 1:NV]
                nxb, nzb = -M.Jηx[i, 1, e], -M.Jηz[i, 1, e]
                sat!(false, e, i, 1, nxb, nzb, w[i], tb)
                wall_trace!(HI, HJ, HV, tb, e, i, 1, nxb, nzb, iqn)
            end
        end
    end

    Kh = sparse(KhI, KhJ, KhV, nq, nq); Kv = sparse(KvI, KvJ, KvV, nq, nq)
    Gh = sparse(GhI, GhJ, GhV, nq, nt); Gv = sparse(GvI, GvJ, GvV, nq, nt)
    H = sparse(HI, HJ, HV, nt, nq)
    K = Kh + Kv; G = Gh + Gv
    L = [K G; H -sparse(1.0I, nt, nt)]
    Mbig = spdiagm(0 => vcat(Mdiag, fill(Float64(eps), nt)))
    return HDGFast2D(M, nq, nt, nth, Float64(eps), K, G, H, L, Mbig, Mdiag, fb,
                     Kh + Gh * H, Kv + Gv * H, sqrt(maximum(c2)), a, θb, φ)
end

"""Wall trace: q̂ = (ρ, m - n̂(n̂·m), Θ) of the interior node."""
function wall_trace!(HI, HJ, HV, tc, e, i, j, nx, nz, iqn)
    s = hypot(nx, nz); n1, n2 = nx / s, nz / s
    P = [1 - n1^2 -n1*n2; -n1*n2 1 - n2^2]
    push!(HI, tc[1]); push!(HJ, iqn(e, 1, i, j)); push!(HV, 1.0)
    push!(HI, tc[4]); push!(HJ, iqn(e, 4, i, j)); push!(HV, 1.0)
    for r in 1:2, t in 1:2
        P[r, t] == 0 && continue
        push!(HI, tc[1+r]); push!(HJ, iqn(e, 1 + t, i, j)); push!(HV, P[r, t])
    end
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

# ---------------------------------------------------------------------------
# HEVI: horizontal explicit, vertical implicit (ε = 0), IMEX tableau of HDGRelaxTheta.run_imex
# ---------------------------------------------------------------------------
function run_hevi(s::HDGFast2D, U0, dt, nsteps, f, monitor, every)
    s.eps == 0 || error("HEVI needs eps = 0")
    gm = 1 - 1 / sqrt(2); dl = 1 - 1 / (2gm)
    AE = [0.0 0.0 0.0; gm 0.0 0.0; dl (1 - dl) 0.0]
    AI = [0.0 0.0 0.0; 0.0 gm 0.0; 0.0 (1 - gm) gm]
    nq = s.nq
    fq = f[1:nq]
    Minv = 1 ./ s.Mdiag
    F = lu(spdiagm(0 => s.Mdiag) - (dt * gm) * s.Av)
    q = U0[1:nq]
    FE = [zeros(nq) for _ in 1:3]; FI = [zeros(nq) for _ in 1:3]
    for n in 1:nsteps
        for i in 1:3
            Ys = copy(q)
            for j in 1:i-1
                Ys .+= dt .* (AE[i, j] .* FE[j] .+ AI[i, j] .* FI[j])
            end
            Y = AI[i, i] == 0 ? Ys : F \ (s.Mdiag .* Ys)
            FE[i] .= (s.Ah * Y .+ fq) .* Minv
            FI[i] .= (s.Av * Y) .* Minv
            i == 3 && (q = Y)
        end
        if monitor !== nothing && n % every == 0
            monitor(n, vcat(q, s.H * q))
        end
    end
    return vcat(q, s.H * q)
end

end # module
