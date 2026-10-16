# Gas and 1D grid

# (Gas and pressure are defined in Common)

"""
    Euler1D(Ne, p, zmin, zmax, gas)

Nodal DGSEM grid for the nonlinear operator (walls at zmin and zmax). Same layout as v2:
variable r of node i in element e at iq(np, e, r, i).
"""
struct Euler1D
    Ne::Int
    p::Int
    np::Int
    zmin::Float64
    zmax::Float64
    h::Float64
    J::Float64
    xe::Vector{Float64}
    gas::Gas
    nq::Int
    xg::Vector{Float64}
    wg::Vector{Float64}
    Dg::Matrix{Float64}
    vflux::Symbol
    gflux::Symbol
    φ::Matrix{Float64}          # geopotential g z at the nodes (np × Ne)
    P::Matrix{Float64}          # pressure cache (np × Ne)
end
"""
    Euler1D(Ne, p, zmin, zmax, gas=Gas(); volume_flux=:tec, gravity=:waruszewski)

volume_flux: :kg (Kennedy–Gruber), :tec, :ec, :etec (total-energy / entropy conservative
fluxes of Artiano, Knoth, Spichtinger, Ranocha 2025). gravity: :pointwise (-g ρ_i) or the
non-conservative two-point term ρ_avg (φ_j - φ_i), φ = g z, with ρ_avg = arithmetic (:souza),
logarithmic (:waruszewski, balanced for isothermal states) or Stolarsky(γ) mean (:artiano,
balanced for constant θ), as in TrixiAtmo. The pointwise source is only kept for comparison:
its linearization about a stratified state has a real positive eigenvalue O(h).
"""
function Euler1D(Ne::Int, p::Int, zmin::Real, zmax::Real, gas::Gas = Gas();
                 volume_flux::Symbol = :tec, gravity::Symbol = :waruszewski)
    volume_flux in (:kg, :tec, :ec, :etec) || error("unknown volume_flux $volume_flux")
    gravity in (:pointwise, :souza, :waruszewski, :artiano) || error("unknown gravity $gravity")
    h = (zmax - zmin) / Ne
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    xe = collect(zmin .+ h .* (0:Ne))
    φ = gas.g .* reduce(hcat, [0.5 * (xe[e] + xe[e+1]) .+ 0.5 * h .* xg for e in 1:Ne])
    return Euler1D(Ne, p, p + 1, zmin, zmax, h, h / 2, xe, gas, Ne * NV * (p + 1), xg, w, D,
                   volume_flux, gravity, φ, zeros(p + 1, Ne))
end

HDGRelaxTheta.iq(eu::Euler1D, e, r, i) = iq(eu.np, e, r, i)
HDGRelaxTheta.blk(eu::Euler1D, q, e, r) = view(q, iq(eu.np, e, r, 1):iq(eu.np, e, r, eu.np))
zpts(eu::Euler1D, e, ξ) = 0.5 * (eu.xe[e] + eu.xe[e+1]) .+ 0.5 * eu.h .* ξ
HDGRelaxTheta.nodes(eu::Euler1D) = reduce(hcat, [zpts(eu, e, eu.xg) for e in 1:eu.Ne])
"""Nodal field r as an (np × Ne) matrix."""
field(eu::Euler1D, q, r) = reduce(hcat, [Vector(blk(eu, q, e, r)) for e in 1:eu.Ne])
