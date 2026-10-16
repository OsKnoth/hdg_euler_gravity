# Nonlinear operator N

"""LMARS flux along the unit normal (nx, nz) from inside L to outside R."""
@inline function lmars2(γ, ρL, uL, wL, θL, pL, ρR, uR, wR, θR, pR, nx, nz)
    unL = uL * nx + wL * nz
    unR = uR * nx + wR * nz
    ρm = 0.5 * (ρL + ρR)
    cm = 0.5 * (sqrt(γ * pL / ρL) + sqrt(γ * pR / ρR))
    ws = 0.5 * (unL + unR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (pL + pR) - 0.5 * ρm * cm * (unR - unL)
    if ws >= 0
        mf = ws * ρL
        return mf, mf * uL + ps * nx, mf * wL + ps * nz, mf * θL
    else
        mf = ws * ρR
        return mf, mf * uR + ps * nx, mf * wR + ps * nz, mf * θR
    end
end

"""Two-point volume flux (contravariant, metric (ax, az)) and non-conservative gravity term."""
@inline function vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, ub, wb, pb, zb, ax, az)
    um = 0.5 * (ua + ub)
    wm = 0.5 * (wa + wb)
    pm = 0.5 * (pa + pb)
    f1, f4 = mass_theta_flux(vf, ρa, Θa, ρb, Θb, ax * um + az * wm, γ)
    gm = gravity_mean(gf, ρa, ρb, γ) * g * (zb - za)
    return f1, f1 * um + ax * pm, f1 * wm + az * pm, f4, ax * gm, az * gm
end

"""Strong-form surface update of one face side: q_b' -= (F̂·ñ - f(q_b)·ñ)/(J w_normal)."""
@inline function side_update!(out, m, q, i, j, e, nx, nz, we, F1, F2, F3, F4)
    M = m.mesh
    np = M.np
    c = m.cache
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
    np = m.mesh.np
    c = m.cache
    @inbounds begin
        s = hypot(nx, nz)
        n1, n2 = nx / s, nz / s
        ρL = q[node_iq(np, eL, 1, iL, jL)]
        ρR = q[node_iq(np, eR, 1, iR, jR)]
        F1, F2, F3, F4 = lmars2(m.gas.γ, ρL, c.U[iL, jL, eL], c.W[iL, jL, eL], q[node_iq(np, eL, 4, iL, jL)] / ρL, c.P[iL, jL, eL],
                                ρR, c.U[iR, jR, eR], c.W[iR, jR, eR], q[node_iq(np, eR, 4, iR, jR)] / ρR, c.P[iR, jR, eR], n1, n2)
        side_update!(out, m, q, iL, jL, eL, nx, nz, we, s * F1, s * F2, s * F3, s * F4)
        side_update!(out, m, q, iR, jR, eR, -nx, -nz, we, -s * F1, -s * F2, -s * F3, -s * F4)
    end
    return nothing
end

@inline function wall_nl!(out, m, q, i, j, e, nx, nz, we)
    np = m.mesh.np
    c = m.cache
    @inbounds begin
        s = hypot(nx, nz)
        n1, n2 = nx / s, nz / s
        ρ = q[node_iq(np, e, 1, i, j)]
        θ = q[node_iq(np, e, 4, i, j)] / ρ
        u, w, p = c.U[i, j, e], c.W[i, j, e], c.P[i, j, e]
        un = u * n1 + w * n2
        F1, F2, F3, F4 = lmars2(m.gas.γ, ρ, u, w, θ, p, ρ, u - 2un * n1, w - 2un * n2, θ, p, n1, n2)
        side_update!(out, m, q, i, j, e, nx, nz, we, s * F1, s * F2, s * F3, s * F4)
    end
    return nothing
end


"""In-place N(q); allocation-free (uses the model's NCache)."""
function rhs_nonlinear!(out, m::Euler2DModel, q::AbstractVector)
    M = m.mesh
    np, Ne, D, w, g, γ = M.np, M.Ne, M.D, M.wg, m.gas.g, m.gas.γ
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
        ρa = q[node_iq(np, e, 1, i, j)]
        Θa = q[node_iq(np, e, 4, i, j)]
        ua, wa, pa, za = U[i, j, e], W[i, j, e], P[i, j, e], M.Z[i, j, e]
        for k in 1:np
            ρb = q[node_iq(np, e, 1, k, j)]
            Θb = q[node_iq(np, e, 4, k, j)]
            f1, f2, f3, f4, g2, g3 = vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, U[k, j, e], W[k, j, e],
                                              P[k, j, e], M.Z[k, j, e], 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]),
                                              0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e]))
            d = 2 * D[i, k]
            s1 += d * f1
            s2 += d * f2
            s3 += d * f3
            s4 += d * f4
            gx += D[i, k] * g2
            gz += D[i, k] * g3
            ρb = q[node_iq(np, e, 1, i, k)]
            Θb = q[node_iq(np, e, 4, i, k)]
            f1, f2, f3, f4, g2, g3 = vol_pair(vf, gf, γ, g, ρa, Θa, ua, wa, pa, za, ρb, Θb, U[i, k, e], W[i, k, e],
                                              P[i, k, e], M.Z[i, k, e], 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e]),
                                              0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e]))
            d = 2 * D[j, k]
            s1 += d * f1
            s2 += d * f2
            s3 += d * f3
            s4 += d * f4
            gx += D[j, k] * g2
            gz += D[j, k] * g3
        end
        Jinv = 1 / M.J[i, j, e]
        out[node_iq(np, e, 1, i, j)] = -s1 * Jinv
        out[node_iq(np, e, 2, i, j)] = -(s2 + gx) * Jinv
        out[node_iq(np, e, 3, i, j)] = -(s3 + gz) * Jinv
        out[node_iq(np, e, 4, i, j)] = -s4 * Jinv
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
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
