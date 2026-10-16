# Model, reference state, sponge, initialization, output fields

"""Work arrays of the nonlinear operator (velocities and pressure at the nodes)."""
struct NCache
    U::Array{Float64,3}
    W::Array{Float64,3}
    P::Array{Float64,3}
end
NCache(np, Ne) = NCache(zeros(np, np, Ne), zeros(np, np, Ne), zeros(np, np, Ne))

struct Euler2DModel
    mesh::Mesh2D
    gas::Gas
    nq::Int
    ρref::Array{Float64,3}
    Θref::Array{Float64,3}
    pref::Array{Float64,3}
    damp::Vector{Float64}       # Rayleigh damping rate [1/s] per unknown (sponge), in N
    qtarget::Vector{Float64}    # state the sponge relaxes to
    vflux::Symbol
    gflux::Symbol
    clmars::Float64             # LMARS sound speed: > 0 constant, 0 local mean
    nres::Vector{Float64}       # N of the reference state, subtracted (exact balance)
    cache::NCache
    fastref::Base.RefValue{Any} # fast operator, reused (re-frozen in place)
end

"""
    Euler2DModel(mesh, gas, ρref, Θref; volume_flux=:tec, gravity=:waruszewski, subtract_reference=true,
                 c_lmars=0)

Hydrostatic reference given by functions ρref(z), Θref(z). Gravity enters as the
non-conservative two-point term ρ_avg (φ_k - φ_i), φ = g z (SBP-compatible with flux
differencing; a pointwise source -g ρ_i makes the linearization about stratified states
unstable with a growth rate O(h)). With subtract_reference the residual of the nodal reference
state is subtracted, so it is steady to round-off for any background and over terrain (this
does not change the Jacobian). Fluxes: see `SplitExplicitEuler.Euler1D`; volume_flux=:kg with
gravity=:souza is the Kennedy–Gruber split form with the arithmetic-mean gravity term (CGDycore).
c_lmars > 0: constant sound speed in the LMARS interface flux (0: mean of the local values).
"""
function Euler2DModel(mesh::Mesh2D, gas::Gas, ρref, Θref; volume_flux::Symbol = :tec,
                      gravity::Symbol = :waruszewski, subtract_reference::Bool = true, c_lmars::Real = 0.0)
    volume_flux in (:kg, :tec, :ec, :etec) || error("unknown volume_flux $volume_flux")
    gravity in (:souza, :waruszewski, :artiano) || error("unknown gravity $gravity")
    R = ρref.(mesh.Z)
    T = Θref.(mesh.Z)
    nq = mesh.Ne * NV * mesh.np^2
    m = Euler2DModel(mesh, gas, nq, R, T, pressure.(Ref(gas), T), zeros(nq), zeros(nq), volume_flux, gravity,
                     Float64(c_lmars), zeros(nq), NCache(mesh.np, mesh.Ne), Ref{Any}(nothing))
    subtract_reference || return m
    qr = zeros(nq)
    Q = fields(m, qr)
    Q[:, :, 1, :] .= R
    Q[:, :, 4, :] .= T
    r = rhs_nonlinear!(zeros(nq), m, qr)        # (m.nres is still zero here; do not write into it directly)
    m.nres .= r
    return m
end

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Rayleigh damping -α(x,z) (q - qtarget) added to N (slow part):
top layer α = αtop sin²(π/2 (z - zD)/(H - zD)) for z > zD (Klemp & Lilly 1978 type), and for
xwidth > 0 lateral layers of that width at both ends of the periodic channel,
α = αside sin²(π/2 d/xwidth), d = distance into the layer. Stable for α Δt ≲ 1.
"""
function with_sponge(m::Euler2DModel, qtarget::AbstractVector; zD::Real, αtop::Real = 0.05,
                     xwidth::Real = 0.0, αside::Real = αtop)
    M = m.mesh
    α = zeros(M.np, M.np, M.Ne)
    x0, x1 = M.x0, M.x0 + M.Lx
    for k in eachindex(α)
        z, x = M.Z[k], M.X[k]
        a = z > zD ? αtop * sin(π / 2 * (z - zD) / (M.H - zD))^2 : 0.0
        if xwidth > 0
            d = max(x0 + xwidth - x, x - (x1 - xwidth))
            d > 0 && (a = max(a, αside * sin(π / 2 * min(d, xwidth) / xwidth)^2))
        end
        α[k] = a
    end
    damp = vec(reshape(repeat(reshape(α, M.np, M.np, 1, M.Ne), 1, 1, NV, 1), :))
    return Euler2DModel(M, m.gas, m.nq, m.ρref, m.Θref, m.pref, damp, Vector{Float64}(qtarget), m.vflux, m.gflux, m.clmars,
                        m.nres, m.cache, m.fastref)
end

# (stratified_reference is defined in Common)

fields(m::Euler2DModel, q) = reshape(q, m.mesh.np, m.mesh.np, NV, m.mesh.Ne)
"""Nodal values of variable r, (np × np × Ne)."""
field(m::Euler2DModel, q, r) = fields(m, q)[:, :, r, :]

"""Nodal initialization: `fun(x, z)` returns (ρ, m_x, m_z, Θ)."""
function project2d(m::Euler2DModel, fun)
    q = zeros(m.nq)
    Q = fields(m, q)
    for k in CartesianIndices(m.mesh.X)
        v = fun(m.mesh.X[k], m.mesh.Z[k])
        for r in 1:NV
            Q[k[1], k[2], r, k[3]] = v[r]
        end
    end
    return q
end

"""Discrete integral Σ J w_i w_j q of variable r."""
function total2d(m::Euler2DModel, q, r)
    Q = fields(m, q)
    M = m.mesh
    return sum(M.J[i, j, e] * M.wg[i] * M.wg[j] * Q[i, j, r, e] for i in 1:M.np, j in 1:M.np, e in 1:M.Ne)
end

"""
    xs, values, zlevel = field_section(model, q, name, z)

Output field along the node row nearest to height z (nodes of the first element column define
the levels; meaningful where the terrain is flat).
"""
function field_section(m::Euler2DModel, q, name, z)
    M = m.mesh
    np = M.np
    F = Dict(output_fields(m, q))[name]
    levels = M.Z[1, :, 1:M.Nex:end][:]
    jz = argmin(abs.(levels .- z))
    ez, j = divrem(jz - 1, np) .+ (1, 1)
    xs = Float64[]
    vals = Float64[]
    for ex in 1:M.Nex, i in 1:np
        (i == 1 && ex > 1) && continue
        e = (ez - 1) * M.Nex + ex
        push!(xs, M.X[i, j, e])
        push!(vals, F[i, j, e])
    end
    return xs, vals, levels[jz]
end

"""Output fields at the nodes as name => (np × np × Ne) array: rho, u, w, theta, and the
deviations from the reference rho_prime, theta_prime, p_prime."""
function output_fields(m::Euler2DModel, q)
    Q = fields(m, q)
    ρ = Q[:, :, 1, :]
    Θ = Q[:, :, 4, :]
    return ["rho" => ρ, "u" => Q[:, :, 2, :] ./ ρ, "w" => Q[:, :, 3, :] ./ ρ, "theta" => Θ ./ ρ,
            "rho_prime" => ρ .- m.ρref, "theta_prime" => Θ ./ ρ .- m.Θref ./ m.ρref,
            "p_prime" => pressure.(Ref(m.gas), Θ) .- m.pref]
end


sound_speed_max(m::Euler2DModel, q) =
    maximum(sqrt.(m.gas.γ .* pressure.(Ref(m.gas), field(m, q, 4)) ./ field(m, q, 1)))
