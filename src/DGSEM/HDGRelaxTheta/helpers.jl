# Forcing, tendency, balanced state, energy, projection, errors

function forcing_vector(s::AbstractHDGOperator, F_slow = nothing)
    f = copy(s.fb)
    F_slow === nothing || (f .+= s.Mdiag .* F_slow)
    return vcat(f, zeros(s.nt))
end

"""dq/dt of the fast operator (HDG limit q̂ = H q unless qh is given)."""
function tendency(s::AbstractHDGOperator, q; qh = nothing, F_slow = nothing)
    qh === nothing && (qh = s.H * q)
    out = (s.K * q .+ s.G * qh .+ s.fb) ./ s.Mdiag
    F_slow === nothing || (out .+= F_slow)
    return out
end

"""Nodal interpolation (DGSEM initialization); `fun(z::Vector)` returns 3 × length(z)."""
function project(s::HDGRelaxTh, fun)
    q = zeros(s.nq)
    for e in 1:s.Ne
        F = fun(zpts(s, e, s.xg))
        for r in 1:NV
            blk(s, q, e, r) .= F[r, :]
        end
    end
    return q
end

"""Discrete hydrostatic ρ (np × Ne) for m = 0 and continuous nodal Θ (np × Ne):
chosen such that the discrete momentum residual vanishes exactly."""
function hydrostatic_rho(s::HDGRelaxTh, Th::AbstractMatrix)
    q = zeros(s.nq)
    for e in 1:s.Ne
        blk(s, q, e, 3) .= Th[:, e]
    end
    r = s.K * q .+ s.G * (s.H * q) .+ s.fb
    return reduce(hcat, [blk(s, r, e, 2) ./ (s.g * s.J .* s.wg) for e in 1:s.Ne])
end

"""Discrete hydrostatic state for Θ(z) (scalar function): m = 0, nodal Θ, balanced ρ."""
function balanced_state(s::HDGRelaxTh, Θfun)
    Th = Θfun.(nodes(s))
    rho = hydrostatic_rho(s, Th)
    q = zeros(s.nq)
    for e in 1:s.Ne
        blk(s, q, e, 1) .= rho[:, e]
        blk(s, q, e, 3) .= Th[:, e]
    end
    return q
end

"""Discrete weighted energy of a deviation dq (non-increasing for form=:split, ε = 0)."""
function energy(s::HDGRelaxTh, dq)
    E = 0.0
    for e in 1:s.Ne
        rho, m, Th = blk(s, dq, e, 1), blk(s, dq, e, 2), blk(s, dq, e, 3)
        a, th, ph = s.Za[:, e], s.Zth[:, e], s.Zphi[:, e]
        thz = (s.Dg * th) ./ s.J
        chi = Th .- th .* rho
        E += 0.5 * s.J * sum(s.wg .* ph .* (m .^ 2 .+ a .* Th .^ 2 ./ th .+ s.g .* chi .^ 2 ./ (th .* thz)))
    end
    return E
end

function evaluate(s::HDGRelaxTh, q; npts::Int = 8)
    ξ = collect(range(-1, 1; length = npts))
    V, _ = lagrange_basis(s.xg, ξ)
    Z = Float64[]
    Q = zeros(NV, 0)
    for e in 1:s.Ne
        append!(Z, zpts(s, e, ξ))
        Q = hcat(Q, reduce(vcat, [(V * blk(s, q, e, r))' for r in 1:NV]))
    end
    return Z, Q
end

function l2_error(s::HDGRelaxTh, q, fun)
    xq, wq = gauss_legendre(s.p + 6)
    V, _ = lagrange_basis(s.xg, xq)
    err = zeros(NV)
    for e in 1:s.Ne
        F = fun(zpts(s, e, xq))
        for r in 1:NV
            d = V * blk(s, q, e, r) .- F[r, :]
            err[r] += s.J * sum(wq .* d .^ 2)
        end
    end
    return sqrt.(err)
end

"""Discrete integral Σ J w_i q_i of variable r (1 mass, 2 momentum, 3 Θ)."""
total(s::HDGRelaxTh, q, r) = s.J * sum(dot(s.wg, blk(s, q, e, r)) for e in 1:s.Ne)
initial_state(s::AbstractHDGOperator, q) = vcat(q, s.H * q)
trace_residual(s::AbstractHDGOperator, U) = maximum(abs.(s.H * U[1:s.nq] .- U[s.nq+1:end]))
