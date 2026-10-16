# Diagnostics

"""Weighted energy of a deviation dq with respect to the frozen coefficients of s (needs ∂_zθ̄ > 0)."""
function energy2d(s::HDGFast2D, dq)
    M = s.mesh
    np = M.np
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
