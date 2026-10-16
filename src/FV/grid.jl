# Grid: Arakawa C-grid, uniform cells, periodic in x, walls at the bottom and top

"""
    FVGrid(nx, nz, Lx, H; x0=0)

Uniform C-grid with nx × nz cells of size Δx × Δz on [x0, x0 + Lx] × [0, H]. Cell centres
(xc, zc), x-faces xf (left face of cell i), z-faces zf (k = 1 .. nz+1, walls at 1 and nz+1).
"""
struct FVGrid
    nx::Int
    nz::Int
    x0::Float64
    Lx::Float64
    H::Float64
    dx::Float64
    dz::Float64
    xc::Vector{Float64}
    zc::Vector{Float64}
    xf::Vector{Float64}
    zf::Vector{Float64}
end

function FVGrid(nx::Int, nz::Int, Lx::Real, H::Real; x0::Real = 0.0)
    dx = Lx / nx
    dz = H / nz
    xc = [x0 + (i - 0.5) * dx for i in 1:nx]
    zc = [(k - 0.5) * dz for k in 1:nz]
    xf = [x0 + (i - 1) * dx for i in 1:nx]
    zf = [(k - 1) * dz for k in 1:nz+1]
    return FVGrid(nx, nz, Float64(x0), Float64(Lx), Float64(H), dx, dz, xc, zc, xf, zf)
end

ncell(g::FVGrid) = g.nx * g.nz
nunknowns(g::FVGrid) = 3 * ncell(g) + g.nx * (g.nz - 1)

# indices into the state vector (i periodic in x; W only at interior faces k = 2 .. nz)
@inline ip(g::FVGrid, i) = mod1(i, g.nx)
@inline iρ(g::FVGrid, i, k) = ip(g, i) + (k - 1) * g.nx
@inline iU(g::FVGrid, i, k) = ncell(g) + ip(g, i) + (k - 1) * g.nx
@inline iW(g::FVGrid, i, k) = 2 * ncell(g) + ip(g, i) + (k - 2) * g.nx
@inline iΘ(g::FVGrid, i, k) = 2 * ncell(g) + g.nx * (g.nz - 1) + ip(g, i) + (k - 1) * g.nx

"""W at z-face k of column i, zero at the walls."""
@inline Wface(g::FVGrid, q, i, k) = (k == 1 || k == g.nz + 1) ? 0.0 : q[iW(g, i, k)]
