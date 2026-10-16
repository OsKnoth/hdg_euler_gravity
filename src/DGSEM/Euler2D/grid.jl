# Grid: Cartesian background mesh, LGL nodes, terrain-following map, DSS

struct Mesh2D
    Nex::Int
    Nez::Int
    p::Int
    np::Int
    Ne::Int
    x0::Float64
    Lx::Float64
    H::Float64
    xg::Vector{Float64}
    wg::Vector{Float64}
    D::Matrix{Float64}
    X::Array{Float64,3}
    Z::Array{Float64,3}
    J::Array{Float64,3}
    Jξx::Array{Float64,3}
    Jξz::Array{Float64,3}
    Jηx::Array{Float64,3}
    Jηz::Array{Float64,3}
    gid::Array{Int,3}
    mult::Vector{Float64}
end

elem(m::Mesh2D, ex, ez) = (ez - 1) * m.Nex + ex
node_iq(np, e, r, i, j) = ((e - 1) * NV + (r - 1)) * np^2 + (j - 1) * np + i

"""
    X, ζ, gid = cartesian_nodes(Nex, Nez, p, Lx, H, x0)

Step 1: Cartesian background mesh of Nex × Nez elements on [x0, x0 + Lx] × [0, H]; the LGL
points of the reference element (ξ, η) ∈ [-1, 1]² are mapped affinely to every element.
Neighbouring elements share identical LGL points on common edges and corners. gid numbers the
distinct (continuous) nodes, periodic in x.
"""
function cartesian_nodes(Nex, Nez, p, Lx, H, x0)
    np = p + 1
    Ne = Nex * Nez
    xg, _ = gll(p)
    hx = Lx / Nex
    hz = H / Nez
    X = zeros(np, np, Ne)
    ζ = zeros(np, np, Ne)
    gid = zeros(Int, np, np, Ne)
    ngx = Nex * p
    for ez in 1:Nez, ex in 1:Nex
        e = (ez - 1) * Nex + ex
        for j in 1:np, i in 1:np
            X[i, j, e] = x0 + (ex - 1 + (xg[i] + 1) / 2) * hx
            ζ[i, j, e] = (ez - 1 + (xg[j] + 1) / 2) * hz
            gid[i, j, e] = ((ez - 1) * p + j - 1) * ngx + mod((ex - 1) * p + i - 1, ngx) + 1
        end
    end
    return X, ζ, gid
end

"""
    Z = terrain_following(X, ζ, H, zs)

Step 2: terrain-following map (Gal-Chen) of the background nodes, z = z_s(x) + ζ (H - z_s(x))/H.
It is applied to every LGL node, so shared nodes of neighbouring elements stay identical.
"""
terrain_following(X, ζ, H, zs) = [zs(X[k]) + ζ[k] * (H - zs(X[k])) / H for k in CartesianIndices(X)]

"""
    Mesh2D(Nex, Nez, p, Lx, H; x0=0, zs = x -> 0)

Grid of Nex × Nez elements of polynomial degree p: Cartesian background mesh with LGL nodes
(step 1, `cartesian_nodes`), terrain-following map of the nodes (step 2, `terrain_following`),
metric terms from the physical nodes (step 3, `metric_terms`). `zs` must be periodic with
period Lx and smaller than H.
"""
function Mesh2D(Nex::Int, Nez::Int, p::Int, Lx::Real, H::Real; x0::Real = 0.0, zs = x -> 0.0)
    np = p + 1
    Ne = Nex * Nez
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    X, ζ, gid = cartesian_nodes(Nex, Nez, p, Float64(Lx), Float64(H), Float64(x0))
    Z = terrain_following(X, ζ, H, zs)
    Jac, Jξx, Jξz, Jηx, Jηz = metric_terms(X, Z, D)
    mult = zeros(maximum(gid))
    for g in gid
        mult[g] += 1
    end
    return Mesh2D(Nex, Nez, p, np, Ne, Float64(x0), Float64(Lx), Float64(H), xg, w, D, X, Z, Jac,
                  Jξx, Jξz, Jηx, Jηz, gid, mult)
end

"""Direct stiffness summation: average the values at shared nodes (continuous field)."""
dss(m::Mesh2D, F::AbstractArray{<:Real,3}) = dss!(Array{Float64}(F), m, zeros(length(m.mult)))

"""In-place DSS of F (np × np × Ne) with a work vector of length = number of global nodes."""
function dss!(F::AbstractArray{Float64,3}, m::Mesh2D, buf::Vector{Float64})
    fill!(buf, 0.0)
    @inbounds for k in eachindex(F)
        buf[m.gid[k]] += F[k]
    end
    @inbounds for g in eachindex(buf)
        buf[g] /= m.mult[g]
    end
    @inbounds for k in eachindex(F)
        F[k] = buf[m.gid[k]]
    end
    return F
end
