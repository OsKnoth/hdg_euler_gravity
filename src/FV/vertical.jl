# Vertically implicit part: tridiagonal system for W per column (WRF)
#
# (I - c A_v) Y = Ys with Y = (ρ, Θ, W) of one column. ρ_k and Θ_k are explicit in the W of
# the two faces of cell k; inserting them into the W equation of face k (between cells k-1, k)
#   W_k + (c/Δz)(a_k Θ_k - a_{k-1} Θ_{k-1}) + (c g/2)(ρ_{k-1} + ρ_k) = Ws_k
# gives a tridiagonal system for W_2 .. W_nz (W_1 = W_{nz+1} = 0):
#   W_{k-1} [-(c/Δz)² a_{k-1} θ_{k-1} + c² g/(2Δz)]
#   + W_k [1 + (c/Δz)² θ_k (a_k + a_{k-1})]
#   + W_{k+1} [-(c/Δz)² a_k θ_{k+1} - c² g/(2Δz)]
#   = Ws_k - (c/Δz)(a_k Θs_k - a_{k-1} Θs_{k-1}) - (c g/2)(ρs_{k-1} + ρs_k)
# (θ at z-faces). The horizontal momentum rows are the identity.

struct FVVerticalFactor
    c::Float64               # coefficient of I - c A_v
    sub::Matrix{Float64}     # (nz+1, nx): coefficient of W_{k-1} in row k
    cp::Matrix{Float64}      # Thomas: c'
    iden::Matrix{Float64}    # Thomas: 1/denominator
end

function factor_vertical!(F::FVVerticalFactor, s::FVFast, c)
    g = s.grid
    nz = g.nz
    r = (c / g.dz)^2
    gg = c^2 * s.gas.g / (2 * g.dz)
    @inbounds for i in 1:g.nx
        for k in 2:nz
            sub = k > 2 ? -r * s.a[i, k-1] * s.θz[i, k-1] + gg : 0.0
            diag = 1 + r * s.θz[i, k] * (s.a[i, k] + s.a[i, k-1])
            sup = k < nz ? -r * s.a[i, k] * s.θz[i, k+1] - gg : 0.0
            den = k == 2 ? diag : diag - sub * F.cp[k-1, i]
            F.sub[k, i] = sub
            F.iden[k, i] = 1 / den
            F.cp[k, i] = sup / den
        end
    end
    return F
end

function vertical_factor(s::FVFast, c)
    get!(s.lucache, Float64(c)) do
        g = s.grid
        F = FVVerticalFactor(Float64(c), zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx))
        factor_vertical!(F, s, c)
    end
end

"""Y = (I - c A_v)⁻¹ Ys (Y may alias Ys)."""
function vertical_solve!(Y, s::FVFast, F::FVVerticalFactor, Ys)
    g = s.grid
    nz = g.nz
    dz = g.dz
    grav = s.gas.g
    c = F.c
    W = s.colbuf
    @inbounds for i in 1:g.nx
        # forward sweep (dp stored in W)
        for k in 2:nz
            rhs = Ys[iW(g, i, k)] - (c / dz) * (s.a[i, k] * Ys[iΘ(g, i, k)] - s.a[i, k-1] * Ys[iΘ(g, i, k - 1)]) -
                  (c * grav / 2) * (Ys[iρ(g, i, k - 1)] + Ys[iρ(g, i, k)])
            W[k] = k == 2 ? rhs * F.iden[k, i] : (rhs - F.sub[k, i] * W[k-1]) * F.iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= F.cp[k, i] * W[k+1]
        end
        W[1] = 0.0
        W[nz+1] = 0.0
        # back substitution for ρ and Θ (reads Ys before Y is written)
        for k in 1:nz
            θb = k > 1 ? s.θz[i, k] : 0.0
            θt = k < nz ? s.θz[i, k+1] : 0.0
            ρ = Ys[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
            Θ = Ys[iΘ(g, i, k)] - (c / dz) * (θt * W[k+1] - θb * W[k])
            Y[iρ(g, i, k)] = ρ
            Y[iΘ(g, i, k)] = Θ
            Y[iU(g, i, k)] = Ys[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
        end
    end
    return Y
end
