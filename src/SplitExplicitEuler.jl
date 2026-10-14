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
The fast ODE of each stage is integrated by HDGRelaxTheta.run! (method :imex or :dirk) with the
constant forcing passed as F_slow. Since F_slow = N(Y) - L_n(Y), steady states of N stay steady.
"""
module SplitExplicitEuler

using LinearAlgebra, SparseArrays

using ..HDGRelaxTheta

export HDGRelaxTheta, Gas, pressure, Euler1D, rhs_nonlinear, hydrostatic_state, fast_operator,
       MISTableau, mis_tableau, mis_step, run_mis, run_rk4, total, evaluate, l2_error, project,
       blk, iq, nodes, sound_speed_max, Atmosphere, atm_Theta, atm_rho, atm_theta, atm_p

const NV = 3

# ---------------------------------------------------------------------------
# Gas and grid
# ---------------------------------------------------------------------------
Base.@kwdef struct Gas
    R::Float64   = 287.0
    γ::Float64   = 1.4
    p00::Float64 = 1.0e5
    g::Float64   = 9.81
end
pressure(gas::Gas, Θ) = gas.p00 * (gas.R * Θ / gas.p00)^gas.γ

"""
    Euler1D(Ne, p, zmin, zmax, gas)

Nodal DGSEM grid for the nonlinear operator (walls at zmin and zmax). Same layout as v2:
variable r of node i in element e at iq(np, e, r, i).
"""
struct Euler1D
    Ne::Int; p::Int; np::Int
    zmin::Float64; zmax::Float64; h::Float64; J::Float64; xe::Vector{Float64}
    gas::Gas; nq::Int
    xg::Vector{Float64}; wg::Vector{Float64}; Dg::Matrix{Float64}
end
function Euler1D(Ne::Int, p::Int, zmin::Real, zmax::Real, gas::Gas = Gas())
    h = (zmax - zmin) / Ne
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    return Euler1D(Ne, p, p + 1, zmin, zmax, h, h / 2, collect(zmin .+ h .* (0:Ne)), gas,
                   Ne * NV * (p + 1), xg, w, D)
end

HDGRelaxTheta.iq(eu::Euler1D, e, r, i) = iq(eu.np, e, r, i)
HDGRelaxTheta.blk(eu::Euler1D, q, e, r) = view(q, iq(eu.np, e, r, 1):iq(eu.np, e, r, eu.np))
zpts(eu::Euler1D, e, ξ) = 0.5 * (eu.xe[e] + eu.xe[e+1]) .+ 0.5 * eu.h .* ξ
HDGRelaxTheta.nodes(eu::Euler1D) = reduce(hcat, [zpts(eu, e, eu.xg) for e in 1:eu.Ne])
"""Nodal field r as an (np × Ne) matrix."""
field(eu::Euler1D, q, r) = reduce(hcat, [Vector(blk(eu, q, e, r)) for e in 1:eu.Ne])

# ---------------------------------------------------------------------------
# Nonlinear operator
# ---------------------------------------------------------------------------
@inline function kg_flux(ρi, wi, θi, pi, ρj, wj, θj, pj)
    fρ = 0.25 * (ρi + ρj) * (wi + wj)
    return fρ, 0.5 * fρ * (wi + wj) + 0.5 * (pi + pj), 0.5 * fρ * (θi + θj)
end

"""LMARS flux between left and right states (ρ, w, θ, p); returns (F_ρ, F_m, F_Θ)."""
@inline function lmars(gas::Gas, ρL, wL, θL, pL, ρR, wR, θR, pR)
    ρm = 0.5 * (ρL + ρR)
    cm = 0.5 * (sqrt(gas.γ * pL / ρL) + sqrt(gas.γ * pR / ρR))
    ws = 0.5 * (wL + wR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (pL + pR) - 0.5 * ρm * cm * (wR - wL)
    ρu, θu = ws >= 0 ? (ρL, θL) : (ρR, θR)
    return ws * ρu, ws * ρu * (ws >= 0 ? wL : wR) + ps, ws * ρu * θu
end

"""dq/dt of the nonlinear DGSEM operator N(q)."""
function rhs_nonlinear(eu::Euler1D, q::AbstractVector)
    np, Ne, J, D, w, gas = eu.np, eu.Ne, eu.J, eu.Dg, eu.wg, eu.gas
    out = zeros(eu.nq)
    ρ = field(eu, q, 1); m = field(eu, q, 2); Θ = field(eu, q, 3)
    W = m ./ ρ; θ = Θ ./ ρ; P = pressure.(Ref(gas), Θ)
    for e in 1:Ne
        for i in 1:np
            sρ = sm = sΘ = 0.0
            for j in 1:np
                fρ, fm, fΘ = kg_flux(ρ[i, e], W[i, e], θ[i, e], P[i, e], ρ[j, e], W[j, e], θ[j, e], P[j, e])
                sρ += D[i, j] * fρ; sm += D[i, j] * fm; sΘ += D[i, j] * fΘ
            end
            out[iq(np, e, 1, i)] = -2sρ / J
            out[iq(np, e, 2, i)] = -2sm / J - gas.g * ρ[i, e]
            out[iq(np, e, 3, i)] = -2sΘ / J
        end
    end
    # surface terms (strong form): q_b' -= n/(J w_b) (F* - f(q_b))
    for f in 1:Ne+1
        if f == 1
            iR, eR = 1, 1
            st = (ρ[iR, eR], -W[iR, eR], θ[iR, eR], P[iR, eR])
            F = lmars(gas, st..., ρ[iR, eR], W[iR, eR], θ[iR, eR], P[iR, eR])
        elseif f == Ne + 1
            iL, eL = np, Ne
            F = lmars(gas, ρ[iL, eL], W[iL, eL], θ[iL, eL], P[iL, eL], ρ[iL, eL], -W[iL, eL], θ[iL, eL], P[iL, eL])
        else
            F = lmars(gas, ρ[np, f-1], W[np, f-1], θ[np, f-1], P[np, f-1], ρ[1, f], W[1, f], θ[1, f], P[1, f])
        end
        for (e, i, n) in ((f - 1, np, 1.0), (f, 1, -1.0))
            (e < 1 || e > Ne) && continue
            fb = (m[i, e], m[i, e] * W[i, e] + P[i, e], m[i, e] * θ[i, e])
            for r in 1:NV
                out[iq(np, e, r, i)] -= n * (F[r] - fb[r]) / (J * w[i])
            end
        end
    end
    return out
end

"""Discrete hydrostatic state of N for Θ(z): m = 0, nodal Θ, ρ = -(D p(Θ))/(g J)."""
function hydrostatic_state(eu::Euler1D, Θfun)
    q = zeros(eu.nq)
    for e in 1:eu.Ne
        Θe = Θfun.(zpts(eu, e, eu.xg))
        blk(eu, q, e, 3) .= Θe
        blk(eu, q, e, 1) .= -(eu.Dg * pressure.(Ref(eu.gas), Θe)) ./ (eu.gas.g * eu.J)
    end
    return q
end

# ---------------------------------------------------------------------------
# Fast operator frozen at a state
# ---------------------------------------------------------------------------
"""HDG fast operator (v2) with coefficients frozen at the state q: θ̄ = Θ/ρ, a = pfac p/Θ,
b = (1 - pfac) p (continuous after DSS)."""
function fast_operator(eu::Euler1D, q; pfac = eu.gas.γ, eps = 0.0, form = :split, tau_scale = 1.0)
    ρ = field(eu, q, 1); Θ = field(eu, q, 3)
    P = pressure.(Ref(eu.gas), Θ)
    return HDGRelaxTh(eu.Ne, eu.p, eu.zmin, eu.zmax, pfac .* P ./ Θ, Θ ./ ρ, (1 - pfac) .* P;
                      g = eu.gas.g, bc = :wall, eps = eps, form = form, tau_scale = tau_scale)
end

sound_speed_max(eu::Euler1D, q) =
    maximum(sqrt.(eu.gas.γ .* pressure.(Ref(eu.gas), field(eu, q, 3)) ./ field(eu, q, 1)))

# ---------------------------------------------------------------------------
# MIS
# ---------------------------------------------------------------------------
"""MIS tableau in the ClimaTimeSteppers convention: row i is paper stage i+1, column j multiplies
(Y_{j+1} - qⁿ) for α, γ and S(Y_j) for β."""
struct MISTableau
    name::Symbol
    α::Matrix{Float64}; β::Matrix{Float64}; γ::Matrix{Float64}; d::Vector{Float64}
end
MISTableau(name, α, β, γ) = MISTableau(name, α, β, γ, vec(sum(β; dims = 2)))
nstages(t::MISTableau) = length(t.d)

"""Tables: :MIS2, :MIS3C, :MIS4, :MIS4a (Knoth & Wensch 2014, values as in ClimaTimeSteppers.jl,
MIS4a with the authors' corrected β₅₁), :WSRK3 (Wicker–Skamarock RK3: β = diag(1/3, 1/2, 1))."""
function mis_tableau(name::Symbol)
    name == :WSRK3 && return MISTableau(name, zeros(3, 3), [1/3 0 0; 0 1/2 0; 0 0 1.0], zeros(3, 3))
    name == :MIS2 && return MISTableau(name,
        [0 0 0; 0.536946566710 0 0; 0.480892968551 0.500561163566 0],
        [0.126848494553 0 0; -0.784838278826 1.37442675268 0; -0.0456727081749 -0.00875082271190 0.524775788629],
        [0 0 0; 0.652465126004 0 0; -0.0732769849457 0.144902430420 0])
    name == :MIS3C && return MISTableau(name,
        [0 0 0; 0.589557277145 0 0; 0.544036601551 0.565511042564 0],
        [0.397525189225 0 0; -0.227036463644 0.624528794618 0; -0.00295238076840 -0.270971764284 0.671323159437],
        [0 0 0; 0.142798786398 0 0; -0.0428918957402 0.0202720980282 0])
    name == :MIS4 && return MISTableau(name,
        [0 0 0 0; 0.914092810304 0 0 0; 1.14274417397 -0.295211246188 0 0;
         0.112965282231 0.337369411296 0.503747183119 0],
        [0.136296478423 0 0 0; 0.280462398979 -0.0160351333596 0 0;
         0.904713355208 -1.04011183154 0.652337563489 0;
         0.0671969845546 -0.365621862610 -0.154861470835 0.970362444469],
        [0 0 0 0; 0.678951983291 0 0 0; -1.38974164070 0.503864576302 0 0;
         -0.375328608282 0.320925021109 -0.158259688945 0])
    name == :MIS4a && return MISTableau(name,
        [0 0 0 0; 0.52349249922385610 0 0 0; 1.1683374366893629 -0.75762080241712637 0 0;
         -0.036477233846797109 0.56936148730740477 0.47746263002599681 0],
        [0.38758444641450318 0 0 0; -0.025318448354142823 0.38668943087310403 0 0;
         0.20899983523553325 -0.45856648476371231 0.43423187573425748 0;
         -0.10048822195663100 -0.46186171956333327 0.83045062122462809 0.27014914900250392],
        [0 0 0 0; 0.13145089796226542 0 0 0; -0.36855857648747881 0.33159232636600550 0 0;
         -0.065767130537473045 0.040591093109036858 0.064902111640806712 0])
    error("unknown MIS tableau $name")
end

"""
    mis_step(eu, q, Δt, tab; dτ, fast=:imex, pfac=γ, eps=0.0)

One large step. The fast operator is re-frozen at q; each stage integrates the fast ODE over
d_i Δt with ceil(d_i Δt / dτ) substeps.
"""
function mis_step(eu::Euler1D, q::AbstractVector, Δt::Real, tab::MISTableau; dτ::Real,
                  fast::Symbol = :imex, pfac = eu.gas.γ, eps = 0.0, sF = nothing)
    sF === nothing && (sF = fast_operator(eu, q; pfac = pfac, eps = eps))
    ns = nstages(tab)
    ΔY = Vector{Vector{Float64}}(undef, ns)          # Y_{j+1} - qⁿ
    S = Vector{Vector{Float64}}(undef, ns)           # S(Y_j)
    Y = q
    for i in 1:ns
        S[i] = rhs_nonlinear(eu, Y) .- tendency(sF, Y)
        Z0 = copy(q)
        Fs = (tab.β[i, i] / tab.d[i]) .* S[i]
        for j in 1:i-1
            Z0 .+= tab.α[i, j] .* ΔY[j]
            Fs .+= (tab.γ[i, j] / (tab.d[i] * Δt)) .* ΔY[j] .+ (tab.β[i, j] / tab.d[i]) .* S[j]
        end
        T = tab.d[i] * Δt
        nsub = max(1, ceil(Int, T / dτ - 1e-10))
        U = run!(sF, initial_state(sF, Z0), T / nsub, nsub; method = fast, F_slow = Fs)
        Y = U[1:sF.nq]
        ΔY[i] = Y .- q
    end
    return Y
end

"""
    run_mis(eu, q0, Δt, nsteps; tab=mis_tableau(:MIS4a), dτ, fast=:imex, pfac=γ, refreeze=true,
            monitor=nothing, every=1)

refreeze = false keeps the fast operator frozen at q0 for the whole run.
"""
function run_mis(eu::Euler1D, q0, Δt, nsteps; tab::MISTableau = mis_tableau(:MIS4a), dτ::Real,
                 fast::Symbol = :imex, pfac = eu.gas.γ, eps = 0.0, refreeze::Bool = true,
                 monitor = nothing, every::Int = 1)
    q = copy(q0)
    sF = refreeze ? nothing : fast_operator(eu, q0; pfac = pfac, eps = eps)
    for n in 1:nsteps
        q = mis_step(eu, q, Δt, tab; dτ = dτ, fast = fast, pfac = pfac, eps = eps, sF = sF)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end

"""Classical RK4 on the full nonlinear operator (reference solution)."""
function run_rk4(eu::Euler1D, q0, dt, nsteps)
    q = copy(q0)
    for _ in 1:nsteps
        k1 = rhs_nonlinear(eu, q)
        k2 = rhs_nonlinear(eu, q .+ (dt / 2) .* k1)
        k3 = rhs_nonlinear(eu, q .+ (dt / 2) .* k2)
        k4 = rhs_nonlinear(eu, q .+ dt .* k3)
        q .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return q
end

# ---------------------------------------------------------------------------
# Helpers (mirroring v2)
# ---------------------------------------------------------------------------
"""Discrete integral Σ J w_i q_i of variable r."""
HDGRelaxTheta.total(eu::Euler1D, q, r) = eu.J * sum(dot(eu.wg, blk(eu, q, e, r)) for e in 1:eu.Ne)

"""Nodal interpolation; `fun(z::Vector)` returns 3 × length(z)."""
function HDGRelaxTheta.project(eu::Euler1D, fun)
    q = zeros(eu.nq)
    for e in 1:eu.Ne
        F = fun(zpts(eu, e, eu.xg))
        for r in 1:NV
            blk(eu, q, e, r) .= F[r, :]
        end
    end
    return q
end

function HDGRelaxTheta.evaluate(eu::Euler1D, q; npts::Int = 8)
    ξ = collect(range(-1, 1; length = npts))
    V, _ = lagrange_basis(eu.xg, ξ)
    Z = Float64[]; Q = zeros(NV, 0)
    for e in 1:eu.Ne
        append!(Z, zpts(eu, e, ξ))
        Q = hcat(Q, reduce(vcat, [(V * blk(eu, q, e, r))' for r in 1:NV]))
    end
    return Z, Q
end

"""Discrete L2 norm (GLL quadrature) of each variable of dq."""
l2_error(eu::Euler1D, dq) =
    [sqrt(eu.J * sum(dot(eu.wg, blk(eu, dq, e, r) .^ 2) for e in 1:eu.Ne)) for r in 1:NV]

end # module
