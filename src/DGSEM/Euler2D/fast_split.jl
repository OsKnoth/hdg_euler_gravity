# Thermodynamic/momentum split with interface momentum; Verlet and forward-backward

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
    M = s.mesh
    np = M.np
    r = relax && s.eps_if > 0 ? s.eps_if / (s.eps_if + dt) : 0.0
    @inbounds for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
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
    M = s.mesh
    np, D, w = M.np, M.D, M.wg
    Fξ, Fη = s.Fξ, s.Fη
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            Fξ[i, j] = M.Jξx[i, j, e] * mx + M.Jξz[i, j, e] * mz
            Fη[i, j] = M.Jηx[i, j, e] * mx + M.Jηz[i, j, e] * mz
        end
        for j in 1:np, i in 1:np
            div = 0.0
            for k in 1:np
                div += D[i, k] * Fξ[k, j] + D[j, k] * Fη[i, k]
            end
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            W = w[i] * w[j]
            out[node_iq(np, e, 1, i, j)] = -W * div
            out[node_iq(np, e, 4, i, j)] = -W * (s.θ[i, j, e] * div + mx * (s.gξx[i, j, e] + s.gηx[i, j, e]) +
                                                 mz * (s.gξz[i, j, e] + s.gηz[i, j, e]))
            out[node_iq(np, e, 2, i, j)] = 0.0
            out[node_iq(np, e, 3, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            nx, nz = M.Jξx[np, j, e], M.Jξz[np, j, e]
            Mh = s.Mhx[j, e]
            thermo_side!(out, s, q, np, j, e, nx, nz, w[j], Mh)
            thermo_side!(out, s, q, 1, j, eR, -nx, -nz, w[j], -Mh)
        end
        for i in 1:np
            nx, nz = M.Jηx[i, np, e], M.Jηz[i, np, e]
            Mh = s.Mhz[i, ez * M.Nex + ex]
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
        c = sqrt(a * θ)
        h = hypot(nx, nz)
        n1, n2 = nx / h, nz / h
        pL = a * q[node_iq(np, eL, 4, iL, jL)]
        pR = a * q[node_iq(np, eR, 4, iR, jR)]
        mnL = n1 * q[node_iq(np, eL, 2, iL, jL)] + n2 * q[node_iq(np, eL, 3, iL, jL)]
        mnR = n1 * q[node_iq(np, eR, 2, iR, jR)] + n2 * q[node_iq(np, eR, 3, iR, jR)]
        Ph = 0.5 * (pL + pR) - 0.5 * s.tau * c * (mnR - mnL)
        out[node_iq(np, eL, 2, iL, jL)] += we * nx * (pL - Ph)
        out[node_iq(np, eL, 3, iL, jL)] += we * nz * (pL - Ph)
        out[node_iq(np, eR, 2, iR, jR)] -= we * nx * (pR - Ph)
        out[node_iq(np, eR, 3, iR, jR)] -= we * nz * (pR - Ph)
    end
    return nothing
end

@inline function momentum_wall!(out, s, q, i, j, e, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[i, j, e], s.θ[i, j, e]
        c = sqrt(a * θ)
        h = hypot(nx, nz)
        mn = (nx * q[node_iq(np, e, 2, i, j)] + nz * q[node_iq(np, e, 3, i, j)]) / h
        d = -s.tau * c * mn                                          # p_b - P̂
        out[node_iq(np, e, 2, i, j)] += we * nx * d
        out[node_iq(np, e, 3, i, j)] += we * nz * d
    end
    return nothing
end

"""
    fast_rhs_momentum!(out, s, q, f)

Momentum tendencies of the fast operator (rows 2 and 3 of out; the thermodynamic rows are set to
zero): φ-weighted pressure gradient, buoyancy, b-gradient (in f) and the LMARS interface pressure.
"""
function fast_rhs_momentum!(out, s::HDGFast2D, q, f)
    M = s.mesh
    np, D, w, g = M.np, M.D, M.wg, s.gas.g
    ψ = s.ψ
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            ψ[i, j] = s.φa[i, j, e] * q[node_iq(np, e, 4, i, j)]
        end
        for j in 1:np, i in 1:np
            dξ = dη = 0.0
            for k in 1:np
                dξ += D[i, k] * ψ[k, j]
                dη += D[j, k] * ψ[i, k]
            end
            ρ = q[node_iq(np, e, 1, i, j)]
            Θ = q[node_iq(np, e, 4, i, j)]
            θ = s.θ[i, j, e]
            aΘθ = s.a[i, j, e] * Θ / θ
            iφ = 1 / s.φ[i, j, e]
            W = w[i] * w[j]
            out[node_iq(np, e, 2, i, j)] = -W * ((M.Jξx[i, j, e] * dξ + M.Jηx[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξx[i, j, e] + s.gηx[i, j, e]))
            out[node_iq(np, e, 3, i, j)] = -W * ((M.Jξz[i, j, e] * dξ + M.Jηz[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξz[i, j, e] + s.gηz[i, j, e])) +
                                           W * M.J[i, j, e] * g * (Θ / θ - ρ)
            out[node_iq(np, e, 1, i, j)] = 0.0
            out[node_iq(np, e, 4, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
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

"""Interface variable of the thermodynamic/momentum split: the face momentum M̂."""
interface_update!(s::HDGFast2D, q; relax::Bool = false, dt = 0.0) = interface_momentum!(s, q; relax = relax, dt = dt)

tendency(s::HDGFast2D, q; F_slow = nothing) =
    fast_apply!(zeros(s.nq), s, q, F_slow === nothing ? s.fb : s.fb .+ s.Mdiag .* F_slow)
initial_state(s::HDGFast2D, q) = Vector{Float64}(q)

"""Dense matrix of q ↦ (K + G H) q ./ M (testing and spectra; small meshes only)."""
function fast_matrix(s::HDGFast2D)
    A = zeros(s.nq, s.nq)
    e = zeros(s.nq)
    z = zeros(s.nq)
    col = zeros(s.nq)
    for k in 1:s.nq
        e[k] = 1.0
        fast_kernel!(col, s, e, z, Val(true), Val(true))
        A[:, k] .= col
        e[k] = 0.0
    end
    return A
end
