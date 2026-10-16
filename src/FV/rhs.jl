# Nonlinear operator N (flux form, C-grid)

"""θ at the z-face k of column i (3rd order where the stencil fits, 2nd order next to the walls)."""
@inline function face_z(F::AbstractMatrix, i, k, nz, v)
    if k >= 3 && k <= nz - 1
        return interp3(F[i, k-2], F[i, k-1], F[i, k], F[i, k+1], v)
    else
        return interp2(F[i, k-1], F[i, k])
    end
end

"""In-place N(q); allocation-free (uses the model's FVCache)."""
function rhs_nonlinear!(out, m::FVModel, q::AbstractVector)
    g = m.grid
    nx = g.nx
    nz = g.nz
    dx = g.dx
    dz = g.dz
    grav = m.gas.g
    c = m.cache
    u = c.u
    w = c.w
    θ = c.θ
    pp = c.pp
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        θ[i, k] = q[iΘ(g, i, k)] / ρ
        pp[i, k] = pressure(m.gas, q[iΘ(g, i, k)]) - m.pref[k]
        u[i, k] = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i - 1, k)] + ρ))
    end
    @inbounds for i in 1:nx
        w[i, 1] = 0.0
        w[i, nz+1] = 0.0
        for k in 2:nz
            w[i, k] = q[iW(g, i, k)] / (0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)]))
        end
    end
    fill!(out, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        # mass and Θ: fluxes through the left x-face (i) and the lower z-face (k) of cell (i, k)
        U = q[iU(g, i, k)]
        θx = interp3(θ[ip(g, i - 2), k], θ[ip(g, i - 1), k], θ[i, k], θ[ip(g, i + 1), k], U)
        out[iρ(g, i, k)] += U / dx
        out[iρ(g, i - 1, k)] -= U / dx
        out[iΘ(g, i, k)] += U * θx / dx
        out[iΘ(g, i - 1, k)] -= U * θx / dx
        if k > 1
            W = q[iW(g, i, k)]
            θz = face_z(θ, i, k, nz, W)
            out[iρ(g, i, k)] += W / dz
            out[iρ(g, i, k - 1)] -= W / dz
            out[iΘ(g, i, k)] += W * θz / dz
            out[iΘ(g, i, k - 1)] -= W * θz / dz
        end
        # U: flux U u at the centre of cell i (between x-faces i and i+1)
        Uc = 0.5 * (q[iU(g, i, k)] + q[iU(g, i + 1, k)])
        uc = interp3(u[ip(g, i - 1), k], u[i, k], u[ip(g, i + 1), k], u[ip(g, i + 2), k], Uc)
        out[iU(g, i + 1, k)] += Uc * uc / dx
        out[iU(g, i, k)] -= Uc * uc / dx
        # U: vertical flux W u at the corner (x-face i, z-face k), k = 2 .. nz
        if k > 1
            Wk = 0.5 * (q[iW(g, i - 1, k)] + q[iW(g, i, k)])
            uz = face_z(u, i, k, nz, Wk)
            out[iU(g, i, k)] += Wk * uz / dz
            out[iU(g, i, k - 1)] -= Wk * uz / dz
            # W: horizontal flux U w at the same corner (between columns i-1 and i, z-face k)
            Uk = 0.5 * (q[iU(g, i, k - 1)] + q[iU(g, i, k)])
            wx = interp3(w[ip(g, i - 2), k], w[ip(g, i - 1), k], w[i, k], w[ip(g, i + 1), k], Uk)
            out[iW(g, i, k)] += Uk * wx / dx
            out[iW(g, i - 1, k)] -= Uk * wx / dx
        end
        # W: vertical flux W w at the centre of cell (i, k), between z-faces k and k+1
        Wc = 0.5 * (Wface(g, q, i, k) + Wface(g, q, i, k + 1))
        if k >= 2 && k <= nz - 1
            wc = interp3(w[i, k-1], w[i, k], w[i, k+1], w[i, k+2], Wc)
        else
            wc = interp2(w[i, k], w[i, k+1])
        end
        k + 1 <= nz && (out[iW(g, i, k + 1)] += Wc * wc / dz)
        k >= 2 && (out[iW(g, i, k)] -= Wc * wc / dz)
        # pressure gradient and gravity (deviations from the reference)
        out[iU(g, i, k)] -= (pp[i, k] - pp[ip(g, i - 1), k]) / dx
        if k > 1
            ρw = 0.5 * (q[iρ(g, i, k - 1)] - m.ρref[k-1] + q[iρ(g, i, k)] - m.ρref[k])
            out[iW(g, i, k)] -= (pp[i, k] - pp[i, k-1]) / dz + grav * ρw
        end
    end
    @inbounds for j in eachindex(out)
        out[j] -= m.damp[j] * (q[j] - m.qtarget[j])
    end
    return out
end
