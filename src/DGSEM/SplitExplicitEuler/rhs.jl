# Nonlinear operator N and discrete hydrostatic state


"""In-place N(q) (allocation-free; uses the pressure cache of `eu`)."""
function rhs_nonlinear!(out, eu::Euler1D, q::AbstractVector)
    np, Ne, J, D, w, gas, P, φ = eu.np, eu.Ne, eu.J, eu.Dg, eu.wg, eu.gas, eu.P, eu.φ
    γ, vf, gf = gas.γ, eu.vflux, eu.gflux
    pointwise = gf === :pointwise
    @inbounds for e in 1:Ne, i in 1:np
        P[i, e] = pressure(gas, q[iq(np, e, 3, i)])
    end
    @inbounds for e in 1:Ne, i in 1:np
        ρi = q[iq(np, e, 1, i)]
        wi = q[iq(np, e, 2, i)] / ρi
        Θi = q[iq(np, e, 3, i)]
        sρ = sm = sΘ = sg = 0.0
        for j in 1:np
            ρj = q[iq(np, e, 1, j)]
            wj = q[iq(np, e, 2, j)] / ρj
            Θj = q[iq(np, e, 3, j)]
            vn = 0.5 * (wi + wj)
            fρ, fΘ = mass_theta_flux(vf, ρi, Θi, ρj, Θj, vn, γ)
            fm = fρ * vn + 0.5 * (P[i, e] + P[j, e])
            sρ += D[i, j] * fρ
            sm += D[i, j] * fm
            sΘ += D[i, j] * fΘ
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
        ρL = q[iq(np, eL, 1, iL)]
        wL = q[iq(np, eL, 2, iL)] / ρL
        ΘL = q[iq(np, eL, 3, iL)]
        ρR = q[iq(np, eR, 1, iR)]
        wR = q[iq(np, eR, 2, iR)] / ρR
        ΘR = q[iq(np, eR, 3, iR)]
        f == 1 && (wL = -wR)
        f == Ne + 1 && (wR = -wL)
        F = lmars(gas, ρL, wL, ΘL / ρL, P[iL, eL], ρR, wR, ΘR / ρR, P[iR, eR])
        for (e, i, n) in ((f - 1, np, 1.0), (f, 1, -1.0))
            (e < 1 || e > Ne) && continue
            ρb = q[iq(np, e, 1, i)]
            mb = q[iq(np, e, 2, i)]
            wb = mb / ρb
            fb1 = mb
            fb2 = mb * wb + P[i, e]
            fb3 = q[iq(np, e, 3, i)] * wb
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
            h = 1e-7 * q[c]
            qp = copy(q)
            qp[c] += h
            Jr[:, k] = (rhs_nonlinear(eu, qp)[im] .- r) ./ h
        end
        q[iρ] .-= Jr \ r
    end
    return q
end
