"""
Makie plots for Euler2D: each element is resampled (exactly) to an equispaced (2p+1)² grid and
drawn as a triangle mesh with per-vertex colours, so curved terrain-following elements and the
polynomial solution inside each element are shown without an intermediate regular grid.
Works with any Makie backend (CairoMakie for files, GLMakie for screens).
"""
module HDGEulerGravityMakieExt

using Makie
using HDGEulerGravity.Euler2D: Euler2DModel, output_fields
using HDGEulerGravity.HDGRelaxTheta: lagrange_basis
import HDGEulerGravity.Euler2D: plot_fields

const DIVERGING = ("u_prime", "w", "rho_prime", "theta_prime", "p_prime")

function element_triangulation(m::Euler2DModel, nsub)
    M = m.mesh
    V, _ = lagrange_basis(M.xg, collect(range(-1, 1; length = nsub)))
    npe = nsub^2
    pts = zeros(Float32, npe * M.Ne, 2)
    faces = zeros(Int, 2 * (nsub - 1)^2 * M.Ne, 3)
    lin(i, j) = i + (j - 1) * nsub
    k = 0
    for e in 1:M.Ne
        o = (e - 1) * npe
        pts[o+1:o+npe, 1] = vec(V * M.X[:, :, e] * V')
        pts[o+1:o+npe, 2] = vec(V * M.Z[:, :, e] * V')
        for j in 1:nsub-1, i in 1:nsub-1
            a, b, c, d = o + lin(i, j), o + lin(i + 1, j), o + lin(i + 1, j + 1), o + lin(i, j + 1)
            faces[k+1, :] = [a, b, c]; faces[k+2, :] = [a, c, d]; k += 2
        end
    end
    return V, pts, faces
end

function plot_fields(m::Euler2DModel, q; names = ["theta_prime", "w"], title = "", zoom = nothing,
                     nsub = 2 * m.mesh.p + 1, xscale = 1e-3, zscale = 1e-3, size = (1100, 380 * length(names)))
    M = m.mesh
    V, pts, faces = element_triangulation(m, nsub)
    pts[:, 1] .*= xscale; pts[:, 2] .*= zscale
    F = Dict(output_fields(m, q))
    fig = Figure(; size = size)
    for (row, name) in enumerate(names)
        haskey(F, name) || error("unknown field $name; available: $(join(first.(output_fields(m, q)), ", "))")
        vals = reduce(vcat, [vec(V * F[name][:, :, e] * V') for e in 1:M.Ne])
        if name in DIVERGING
            a = maximum(abs, vals); a = a > 0 ? a : 1.0
            cmap, crange = :balance, (-a, a)
        else
            cmap, crange = :viridis, extrema(vals)
        end
        ax = Axis(fig[row, 1]; title = row == 1 && !isempty(title) ? "$title — $name" : name,
                  xlabel = xscale == 1e-3 ? "x [km]" : "x", ylabel = zscale == 1e-3 ? "z [km]" : "z")
        plt = mesh!(ax, pts, faces; color = Float32.(vals), colormap = cmap, colorrange = crange,
                    shading = NoShading)
        if zoom !== nothing
            xlims!(ax, zoom[1] .* xscale); ylims!(ax, zoom[2] .* zscale)
        end
        Colorbar(fig[row, 2], plt)
    end
    return fig
end

end # module
