"""
    Common

Physical constants and reference atmospheres shared by all discretizations (DGSEM, FV):
ideal gas in potential-temperature form, p = p00 (R Θ / p00)^γ with Θ = ρθ, and hydrostatic
background states.
"""
module Common

export Gas, pressure, stratified_reference,
       Atmosphere, atm_T, atm_p, atm_rho, atm_theta, atm_Theta, atm_pi,
       project2d, total2d, output_fields, field_section, sound_speed_max, with_sponge,
       open_vtkhdf, write_vtkhdf!, plot_fields

"""Dry ideal gas: R, γ, p00 and the gravitational acceleration g."""
Base.@kwdef struct Gas
    R::Float64   = 287.0
    γ::Float64   = 1.4
    p00::Float64 = 1.0e5
    g::Float64   = 9.81
end

"""Pressure from Θ = ρθ."""
pressure(gas::Gas, Θ) = gas.p00 * (gas.R * Θ / gas.p00)^gas.γ

"""
    ρ, Θ, θ = stratified_reference(gas; θ0=300, N=0.01)

Constant Brunt–Väisälä frequency N: θ̄ = θ0 exp(N² z/g) with the hydrostatic Exner pressure
(N = 0: constant θ). Returns functions of z for ρ̄, Θ̄ = ρ̄θ̄ and θ̄.
"""
function stratified_reference(gas::Gas; θ0 = 300.0, N = 0.01)
    g = gas.g
    R = gas.R
    γ = gas.γ
    p00 = gas.p00
    cp = R * γ / (γ - 1)
    θ(z) = N == 0 ? θ0 : θ0 * exp(N^2 * z / g)
    π(z) = N == 0 ? 1 - g * z / (cp * θ0) : 1 + g^2 / (cp * θ0 * N^2) * (exp(-N^2 * z / g) - 1)
    ρ(z) = p00 * π(z)^(cp / R) / (R * π(z) * θ(z))
    return ρ, z -> ρ(z) * θ(z), θ
end

"""Hydrostatic atmosphere with constant lapse rate Γ (Γ = 0: isothermal)."""
Base.@kwdef struct Atmosphere
    T0::Float64  = 288.0
    Γ::Float64   = 0.0
    R::Float64   = 287.0
    γ::Float64   = 1.4
    g::Float64   = 9.81
    p00::Float64 = 1.0e5
end
atm_T(a::Atmosphere, z) = a.T0 - a.Γ * z
atm_p(a::Atmosphere, z) = a.Γ == 0.0 ? a.p00 * exp(-a.g * z / (a.R * a.T0)) :
                                       a.p00 * (atm_T(a, z) / a.T0)^(a.g / (a.R * a.Γ))
atm_rho(a::Atmosphere, z) = atm_p(a, z) / (a.R * atm_T(a, z))
atm_theta(a::Atmosphere, z) = atm_T(a, z) * (a.p00 / atm_p(a, z))^((a.γ - 1) / a.γ)
atm_Theta(a::Atmosphere, z) = atm_rho(a, z) * atm_theta(a, z)
atm_pi(a::Atmosphere, z) = atm_p(a, z) / atm_Theta(a, z)        # p/(ρθ) = R × Exner

# ---------------------------------------------------------------------------
# Model API shared by the discretizations (methods in DGSEM/Euler2D and FV/FV2D)
# ---------------------------------------------------------------------------
"""
    q = project2d(model, fun)

Initial state from `fun(x, z) = (ρ, m_x, m_z, Θ)` (nodal for DG, cell/face values for FV).
"""
function project2d end

"""Discrete integral of variable r (1 ρ, 2 ρu, 3 ρw, 4 Θ)."""
function total2d end

"""Output fields as name => array: rho, u, w, theta, rho_prime, theta_prime, p_prime."""
function output_fields end

"""
    xs, values, zlevel = field_section(model, q, name, z)

Values of an output field along the node/cell row nearest to height z (flat part of the grid).
"""
function field_section end

"""Maximum sound speed of the state q."""
function sound_speed_max end

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Model with Rayleigh damping -α (q - qtarget) above zD (sin² profile) and optional lateral layers.
"""
function with_sponge end

"""
    vtk = open_vtkhdf(filename, model)
    write_vtkhdf!(vtk, model, q, t)
    close(vtk)

Time series in the VTKHDF format for ParaView ≥ 5.13 (DG: high-order Lagrange quadrilaterals with
point data; FV: quadrilateral cells with cell data). Needs `using HDF5` (package extension).
"""
function open_vtkhdf end
function write_vtkhdf! end

"""
    fig = plot_fields(model, q; names=["theta_prime", "w"], title="", zoom=nothing)

Makie figure of output fields; needs a Makie backend, e.g. `using CairoMakie` (package extension).
"""
function plot_fields end

end # module
