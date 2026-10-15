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

export HDGRelaxTheta, Gas, pressure, Euler1D, rhs_nonlinear, rhs_nonlinear!, MISWork, mis_step!, stage_substeps, hydrostatic_state, fast_operator,
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
    vflux::Symbol; gflux::Symbol
    φ::Matrix{Float64}          # geopotential g z at the nodes (np × Ne)
    P::Matrix{Float64}          # pressure cache (np × Ne)
end
"""
    Euler1D(Ne, p, zmin, zmax, gas=Gas(); volume_flux=:tec, gravity=:waruszewski)

volume_flux: :kg (Kennedy–Gruber), :tec, :ec, :etec (total-energy / entropy conservative
fluxes of Artiano, Knoth, Spichtinger, Ranocha 2025). gravity: :pointwise (-g ρ_i) or the
non-conservative two-point term ρ_avg (φ_j - φ_i), φ = g z, with ρ_avg = arithmetic (:souza),
logarithmic (:waruszewski, balanced for isothermal states) or Stolarsky(γ) mean (:artiano,
balanced for constant θ), as in TrixiAtmo. The pointwise source is only kept for comparison:
its linearization about a stratified state has a real positive eigenvalue O(h).
"""
function Euler1D(Ne::Int, p::Int, zmin::Real, zmax::Real, gas::Gas = Gas();
                 volume_flux::Symbol = :tec, gravity::Symbol = :waruszewski)
    volume_flux in (:kg, :tec, :ec, :etec) || error("unknown volume_flux $volume_flux")
    gravity in (:pointwise, :souza, :waruszewski, :artiano) || error("unknown gravity $gravity")
    h = (zmax - zmin) / Ne
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    xe = collect(zmin .+ h .* (0:Ne))
    φ = gas.g .* reduce(hcat, [0.5 * (xe[e] + xe[e+1]) .+ 0.5 * h .* xg for e in 1:Ne])
    return Euler1D(Ne, p, p + 1, zmin, zmax, h, h / 2, xe, gas, Ne * NV * (p + 1), xg, w, D,
                   volume_flux, gravity, φ, zeros(p + 1, Ne))
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
# two-point means (as in Trixi.jl)
@inline function ln_mean(x, y)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    f2 < 1e-4 && return (x + y) / @evalpoly(f2, 2, 2 / 3, 2 / 5, 2 / 7)
    return (y - x) / log(y / x)
end
@inline function inv_ln_mean(x, y)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    f2 < 1e-4 && return @evalpoly(f2, 2, 2 / 3, 2 / 5, 2 / 7) / (x + y)
    return log(y / x) / (y - x)
end
@inline function stolarsky_mean(x, y, γ)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    if f2 < 1e-4
        c1 = (γ - 2) / 3
        c2 = -(γ + 1) * (γ - 3) * c1 / 15
        c3 = -(2γ * (γ - 2) - 9) * c2 / 21
        return 0.5 * (x + y) * @evalpoly(f2, 1, c1, c2, c3)
    end
    yg = exp((γ - 1) * log(y)); xg = exp((γ - 1) * log(x))
    return (γ - 1) * (yg * y - xg * x) / (γ * (yg - xg))
end

"""Mass and Θ two-point fluxes for the averaged normal velocity vn (momentum flux = f_ρ {v} + {p} n)."""
@inline function mass_theta_flux(vf::Symbol, ρi, Θi, ρj, Θj, vn, γ)
    if vf === :kg
        f1 = 0.5 * (ρi + ρj) * vn
        return f1, f1 * 0.5 * (Θi / ρi + Θj / ρj)
    elseif vf === :tec
        return ln_mean(ρi, ρj) * vn, stolarsky_mean(Θi, Θj, γ) * vn
    elseif vf === :ec
        f1 = ln_mean(ρi, ρj) * vn
        return f1, inv_ln_mean(ρi / Θi, ρj / Θj) * f1
    else # :etec
        f3 = stolarsky_mean(Θi, Θj, γ) * vn
        return f3 * ln_mean(ρi / Θi, ρj / Θj), f3
    end
end

"""Density mean of the non-conservative gravity term ρ_avg (φ_j - φ_i)."""
@inline gravity_mean(gf::Symbol, ρi, ρj, γ) =
    gf === :souza ? 0.5 * (ρi + ρj) : gf === :waruszewski ? ln_mean(ρi, ρj) : stolarsky_mean(ρi, ρj, γ)

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
rhs_nonlinear(eu, q::AbstractVector) = rhs_nonlinear!(zeros(length(q)), eu, q)

"""In-place N(q) (allocation-free; uses the pressure cache of `eu`)."""
function rhs_nonlinear!(out, eu::Euler1D, q::AbstractVector)
    np, Ne, J, D, w, gas, P, φ = eu.np, eu.Ne, eu.J, eu.Dg, eu.wg, eu.gas, eu.P, eu.φ
    γ, vf, gf = gas.γ, eu.vflux, eu.gflux
    pointwise = gf === :pointwise
    @inbounds for e in 1:Ne, i in 1:np
        P[i, e] = pressure(gas, q[iq(np, e, 3, i)])
    end
    @inbounds for e in 1:Ne, i in 1:np
        ρi = q[iq(np, e, 1, i)]; wi = q[iq(np, e, 2, i)] / ρi; Θi = q[iq(np, e, 3, i)]
        sρ = sm = sΘ = sg = 0.0
        for j in 1:np
            ρj = q[iq(np, e, 1, j)]; wj = q[iq(np, e, 2, j)] / ρj; Θj = q[iq(np, e, 3, j)]
            vn = 0.5 * (wi + wj)
            fρ, fΘ = mass_theta_flux(vf, ρi, Θi, ρj, Θj, vn, γ)
            fm = fρ * vn + 0.5 * (P[i, e] + P[j, e])
            sρ += D[i, j] * fρ; sm += D[i, j] * fm; sΘ += D[i, j] * fΘ
            pointwise || (sg += D[i, j] * gravity_mean(gf, ρi, ρj, γ) * (φ[j, e] - φ[i, e]))
        end
        out[iq(np, e, 1, i)] = -2sρ / J
        out[iq(np, e, 2, i)] = -2sm / J - (pointwise ? gas.g * ρi : sg / J)
        out[iq(np, e, 3, i)] = -2sΘ / J
    end
    # surface terms (strong form): q_b' -= n/(J w_b) (F* - f(q_b))
    @inbounds for f in 1:Ne+1
        eL, iL = f == 1 ? (1, 1) : (f - 1, np)          # left state (mirror at the bottom wall)
        eR, iR = f == Ne + 1 ? (Ne, np) : (f, 1)        # right state (mirror at the top wall)
        ρL = q[iq(np, eL, 1, iL)]; wL = q[iq(np, eL, 2, iL)] / ρL; ΘL = q[iq(np, eL, 3, iL)]
        ρR = q[iq(np, eR, 1, iR)]; wR = q[iq(np, eR, 2, iR)] / ρR; ΘR = q[iq(np, eR, 3, iR)]
        f == 1 && (wL = -wR)
        f == Ne + 1 && (wR = -wL)
        F = lmars(gas, ρL, wL, ΘL / ρL, P[iL, eL], ρR, wR, ΘR / ρR, P[iR, eR])
        for (e, i, n) in ((f - 1, np, 1.0), (f, 1, -1.0))
            (e < 1 || e > Ne) && continue
            ρb = q[iq(np, e, 1, i)]; mb = q[iq(np, e, 2, i)]; wb = mb / ρb
            fb1 = mb; fb2 = mb * wb + P[i, e]; fb3 = q[iq(np, e, 3, i)] * wb
            c = n / (J * w[i])
            out[iq(np, e, 1, i)] -= c * (F[1] - fb1)
            out[iq(np, e, 2, i)] -= c * (F[2] - fb2)
            out[iq(np, e, 3, i)] -= c * (F[3] - fb3)
        end
    end
    return out
end

"""Discrete hydrostatic state of N for Θ(z): m = 0, nodal Θ, ρ such that the momentum residual
vanishes (explicit for pointwise gravity, ρ = -(D p(Θ))/(g J); Newton iteration otherwise)."""
function hydrostatic_state(eu::Euler1D, Θfun; tol = 1e-13, maxit = 30)
    q = zeros(eu.nq)
    for e in 1:eu.Ne
        Θe = Θfun.(zpts(eu, e, eu.xg))
        blk(eu, q, e, 3) .= Θe
        blk(eu, q, e, 1) .= -(eu.Dg * pressure.(Ref(eu.gas), Θe)) ./ (eu.gas.g * eu.J)
    end
    eu.gflux === :pointwise && return q
    iρ = [iq(eu.np, e, 1, i) for e in 1:eu.Ne for i in 1:eu.np]
    im = [iq(eu.np, e, 2, i) for e in 1:eu.Ne for i in 1:eu.np]
    for _ in 1:maxit
        r = rhs_nonlinear(eu, q)[im]
        maximum(abs, r) < tol * eu.gas.g * maximum(q[iρ]) && break
        Jr = zeros(length(iρ), length(iρ))
        for (k, c) in enumerate(iρ)
            h = 1e-7 * q[c]; qp = copy(q); qp[c] += h
            Jr[:, k] = (rhs_nonlinear(eu, qp)[im] .- r) ./ h
        end
        q[iρ] .-= Jr \ r
    end
    return q
end

# ---------------------------------------------------------------------------
# Fast operator frozen at a state
# ---------------------------------------------------------------------------
"""HDG fast operator (v2) with coefficients frozen at the state q: θ̄ = Θ/ρ, a = pfac p/Θ,
b = (1 - pfac) p (continuous after DSS)."""
function fast_operator(eu::Euler1D, q; pfac = eu.gas.γ, eps = 0.0, form = :split, tau_scale = 1.0, fast = :imex)
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

"""Preallocated work vectors of the MIS step (ΔY_i = Y_{i+1} - qⁿ, S_i = S(Y_i))."""
struct MISWork
    ΔY::Vector{Vector{Float64}}; S::Vector{Vector{Float64}}
    Y::Vector{Float64}; Z::Vector{Float64}; Fs::Vector{Float64}; L::Vector{Float64}
end
MISWork(nq::Int, ns::Int) = MISWork([zeros(nq) for _ in 1:ns], [zeros(nq) for _ in 1:ns],
                                    zeros(nq), zeros(nq), zeros(nq), zeros(nq))

"""Fast substeps per MIS stage: n_i = max(1, round(d_i nsub)); for WSRK3 nsub/3, nsub/2, nsub (WRF)."""
stage_substeps(tab::MISTableau, nsub::Int) = [max(1, round(Int, d * nsub)) for d in tab.d]

"""
    mis_step!(qnew, eu, q, Δt, tab, sF, work; nsub=12, fast=:imex)

One large step with the frozen fast operator sF; stage i integrates the fast system over d_i Δt
with n_i = max(1, round(d_i nsub)) substeps. qnew may alias q. Allocation-free for models with
in-place `rhs_nonlinear!` and operators with a cached fast workspace.
"""
function mis_step!(qnew, eu, q, Δt, tab::MISTableau, sF, w::MISWork; nsub::Int = 12, fast::Symbol = :imex)
    for i in 1:nstages(tab)
        if i == 1
            w.Y .= q
        else
            @. w.Y = q + w.ΔY[i-1]
        end
        S = w.S[i]
        rhs_nonlinear!(S, eu, w.Y)
        fast_apply!(w.L, sF, w.Y, sF.fb)                  # L_n(Y_i)
        S .-= w.L                                          # S(Y_i) = N(Y_i) - L_n(Y_i)
        d = tab.d[i]
        c = tab.β[i, i] / d
        w.Z .= q
        @. w.Fs = c * S
        for j in 1:i-1
            a, g, b = tab.α[i, j], tab.γ[i, j] / (d * Δt), tab.β[i, j] / d
            ΔYj, Sj = w.ΔY[j], w.S[j]
            @. w.Z += a * ΔYj
            @. w.Fs += g * ΔYj + b * Sj
        end
        n = max(1, round(Int, d * nsub))
        fast_solve!(w.Z, sF, w.Z, w.Fs, d * Δt / n, n; method = fast)
        @. w.ΔY[i] = w.Z - q
    end
    qnew .= w.Z
    return qnew
end

"""
    mis_step(eu, q, Δt, tab; nsub=12, fast=:imex, pfac=γ, sF=nothing)

Allocating convenience version of `mis_step!` (fast operator frozen at q unless sF is given).
"""
function mis_step(eu, q::AbstractVector, Δt::Real, tab::MISTableau; nsub::Int = 12, fast::Symbol = :imex,
                  pfac = eu.gas.γ, eps = 0.0, sF = nothing)
    sF === nothing && (sF = fast_operator(eu, q; pfac = pfac, eps = eps, fast = fast))
    return mis_step!(similar(q), eu, q, Δt, tab, sF, MISWork(length(q), nstages(tab)); nsub = nsub, fast = fast)
end

"""
    run_mis(eu, q0, Δt, nsteps; tab=mis_tableau(:MIS4a), nsub=12, fast=:imex, pfac=γ,
            refreeze=true, refreeze_every=1, monitor=nothing, every=1)

Split-explicit integration with `nsub` fast substeps per large step (ratio fast/slow; WRF uses
6 or 12 with RK3; keep it ≤ 24). The fast operator is re-frozen at the current state every
`refreeze_every` large steps; refreeze = false keeps it frozen at q0.
"""
function run_mis(eu, q0, Δt, nsteps; tab::MISTableau = mis_tableau(:MIS4a), nsub::Int = 12,
                 fast::Symbol = :imex, pfac = eu.gas.γ, eps = 0.0, refreeze::Bool = true,
                 refreeze_every::Int = 1, monitor = nothing, every::Int = 1)
    nsub > 24 && @warn "nsub = $nsub fast steps per slow step (ratio > 24)"
    q = Vector{Float64}(q0)
    sF = fast_operator(eu, q; pfac = pfac, eps = eps, fast = fast)
    w = MISWork(length(q), nstages(tab))
    for n in 1:nsteps
        refreeze && n > 1 && (n - 1) % refreeze_every == 0 &&
            (sF = fast_operator(eu, q; pfac = pfac, eps = eps, fast = fast))
        mis_step!(q, eu, q, Δt, tab, sF, w; nsub = nsub, fast = fast)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end

"""Classical RK4 on the full nonlinear operator (reference solution)."""
function run_rk4(eu, q0, dt, nsteps)
    q = Vector{Float64}(q0)
    k1, k2, k3, k4, y = (zeros(length(q)) for _ in 1:5)
    for _ in 1:nsteps
        rhs_nonlinear!(k1, eu, q)
        @. y = q + (dt / 2) * k1; rhs_nonlinear!(k2, eu, y)
        @. y = q + (dt / 2) * k2; rhs_nonlinear!(k3, eu, y)
        @. y = q + dt * k3;       rhs_nonlinear!(k4, eu, y)
        @. q += (dt / 6) * (k1 + 2k2 + 2k3 + k4)
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
