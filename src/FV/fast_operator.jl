# Fast operator (acoustic-gravity part linearized about the frozen state), C-grid

mutable struct FVFast <: AbstractFastOperator
    grid::FVGrid
    gas::Gas
    nq::Int
    a::Matrix{Float64}             # γ p/Θ at the centres (nx, nz)
    b::Matrix{Float64}             # (1 - γ) p at the centres
    θx::Matrix{Float64}            # θ̄ at the x-faces (nx, nz)
    θz::Matrix{Float64}            # θ̄ at the z-faces (nx, nz+1), walls unused
    fb::Vector{Float64}            # constant forcing of the fast operator
    Mdiag::Vector{Float64}         # mass (ones: tendencies per unit volume)
    work::Vector{Vector{Float64}}  # 7 work vectors for the fast integrators
    lucache::Dict{Float64,Any}     # c -> tridiagonal factors of I - c A_v
    cmax::Float64
    colbuf::Vector{Float64}        # column work (nz + 1)
end

function FVFast(m::FVModel)
    g = m.grid
    nq = m.nq
    return FVFast(g, m.gas, nq, zeros(g.nx, g.nz), zeros(g.nx, g.nz), zeros(g.nx, g.nz), zeros(g.nx, g.nz + 1),
                  zeros(nq), ones(nq), [zeros(nq) for _ in 1:7], Dict{Float64,Any}(), 0.0, zeros(g.nz + 1))
end

"""
    fast_operator(model, q; pfac=γ, fast=:imex, reuse=true)

Fast operator frozen at q: a = pfac p/Θ, b = (1 - pfac) p, θ̄ = Θ/ρ averaged to the faces.
With reuse = true the operator stored in the model is re-frozen in place.
"""
function fast_operator(m::FVModel, q; pfac = m.gas.γ, fast::Symbol = :imex, reuse::Bool = true, kwargs...)
    s = reuse && m.fastref[] isa FVFast ? m.fastref[]::FVFast : FVFast(m)
    refreeze!(s, m, q, pfac)
    reuse && (m.fastref[] = s)
    return s
end

function refreeze!(s::FVFast, m::FVModel, q, pfac)
    g = s.grid
    nx = g.nx
    nz = g.nz
    s.cmax = 0.0
    @inbounds for k in 1:nz, i in 1:nx
        Θ = q[iΘ(g, i, k)]
        p = pressure(m.gas, Θ)
        s.a[i, k] = pfac * p / Θ
        s.b[i, k] = (1 - pfac) * p
        s.cmax = max(s.cmax, sqrt(m.gas.γ * p / q[iρ(g, i, k)]))
    end
    @inbounds for k in 1:nz, i in 1:nx
        θl = q[iΘ(g, i - 1, k)] / q[iρ(g, i - 1, k)]
        θr = q[iΘ(g, i, k)] / q[iρ(g, i, k)]
        s.θx[i, k] = 0.5 * (θl + θr)
        if k > 1
            θb = q[iΘ(g, i, k - 1)] / q[iρ(g, i, k - 1)]
            s.θz[i, k] = 0.5 * (θb + θr)
        end
    end
    fill!(s.fb, 0.0)
    grav = m.gas.g
    @inbounds for k in 1:nz, i in 1:nx
        s.fb[iU(g, i, k)] = -(s.b[i, k] - s.b[ip(g, i - 1), k]) / g.dx
        if k > 1
            s.fb[iW(g, i, k)] = -(s.b[i, k] - s.b[i, k-1]) / g.dz + (m.pref[k] - m.pref[k-1]) / g.dz +
                                grav * 0.5 * (m.ρref[k-1] + m.ρref[k])
        end
    end
    for (c, F) in s.lucache
        factor_vertical!(F, s, c)
    end
    return s
end

"""
    fv_fast!(out, s, q, f, hor, ver, rows, frows)

Fast tendency: horizontal (hor) and/or vertical (ver) part of the rows selected by
rows = (ρ, U, W, Θ), plus the forcing f on the rows selected by frows; all other rows are zero.
"""
function fv_fast!(out, s::FVFast, q, f, hor::Bool, ver::Bool, rows::NTuple{4,Bool}, frows::NTuple{4,Bool})
    g = s.grid
    nx = g.nx
    nz = g.nz
    dx = g.dx
    dz = g.dz
    grav = s.gas.g
    rρ, rU, rW, rΘ = rows
    fill!(out, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        if hor
            Ul = q[iU(g, i, k)]
            Ur = q[iU(g, i + 1, k)]
            rρ && (out[iρ(g, i, k)] -= (Ur - Ul) / dx)
            rΘ && (out[iΘ(g, i, k)] -= (s.θx[ip(g, i + 1), k] * Ur - s.θx[i, k] * Ul) / dx)
            rU && (out[iU(g, i, k)] -= (s.a[i, k] * q[iΘ(g, i, k)] - s.a[ip(g, i - 1), k] * q[iΘ(g, i - 1, k)]) / dx)
        end
        if ver
            Wb = Wface(g, q, i, k)
            Wt = Wface(g, q, i, k + 1)
            rρ && (out[iρ(g, i, k)] -= (Wt - Wb) / dz)
            if rΘ
                θt = k < nz ? s.θz[i, k+1] : 0.0
                θb = k > 1 ? s.θz[i, k] : 0.0
                out[iΘ(g, i, k)] -= (θt * Wt - θb * Wb) / dz
            end
            if rW && k > 1
                out[iW(g, i, k)] -= (s.a[i, k] * q[iΘ(g, i, k)] - s.a[i, k-1] * q[iΘ(g, i, k - 1)]) / dz +
                                    grav * 0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)])
            end
        end
    end
    fρ, fU, fW, fΘ = frows
    @inbounds for k in 1:nz, i in 1:nx
        fρ && (out[iρ(g, i, k)] += f[iρ(g, i, k)])
        fU && (out[iU(g, i, k)] += f[iU(g, i, k)])
        fΘ && (out[iΘ(g, i, k)] += f[iΘ(g, i, k)])
        fW && k > 1 && (out[iW(g, i, k)] += f[iW(g, i, k)])
    end
    return out
end

const ALL = (true, true, true, true)
const NONE = (false, false, false, false)
const THERMO = (true, false, false, true)
const MOMENTUM = (false, true, true, false)
const HMOM = (false, true, false, false)
const NOT_HMOM = (true, false, true, true)

# TimeStepping interface
fast_apply!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, ALL, ALL)
fast_rhs_thermo!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, THERMO, THERMO)
fast_rhs_momentum!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, MOMENTUM, MOMENTUM)
fast_rhs_horizontal!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, ALL, ALL)
vertical_apply!(out, s::FVFast, q) = fv_fast!(out, s, q, s.fb, false, true, ALL, NONE)
fast_rhs_hmomentum!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, HMOM, HMOM)
fast_rhs_hthermo!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, THERMO, NOT_HMOM)

"""Dense matrix of the linear part of the fast operator (tests and spectra; small grids only)."""
function fast_matrix(s::FVFast)
    A = zeros(s.nq, s.nq)
    e = zeros(s.nq)
    col = zeros(s.nq)
    z = zeros(s.nq)
    for k in 1:s.nq
        e[k] = 1.0
        fv_fast!(col, s, e, z, true, true, ALL, NONE)
        A[:, k] .= col
        e[k] = 0.0
    end
    return A
end
