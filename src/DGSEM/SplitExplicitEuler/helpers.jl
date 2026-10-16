# Helpers (mirroring v2)

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
    Z = Float64[]
    Q = zeros(NV, 0)
    for e in 1:eu.Ne
        append!(Z, zpts(eu, e, ξ))
        Q = hcat(Q, reduce(vcat, [(V * blk(eu, q, e, r))' for r in 1:NV]))
    end
    return Z, Q
end

"""Discrete L2 norm (GLL quadrature) of each variable of dq."""
l2_error(eu::Euler1D, dq) =
    [sqrt(eu.J * sum(dot(eu.wg, blk(eu, dq, e, r) .^ 2) for e in 1:eu.Ne)) for r in 1:NV]
