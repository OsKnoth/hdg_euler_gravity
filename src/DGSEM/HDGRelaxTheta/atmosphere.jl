# Frozen coefficients

# (Atmosphere and atm_* are defined in Common)

"""Frozen coefficients as scalar functions of z: pi = p̄/Θ̄, theta = θ̄, pbar = p̄."""
struct FrozenState
    pi::Function
    theta::Function
    pbar::Function
    g::Float64
    pfac::Float64
end
FrozenState(pi, theta; pbar = z -> 0.0, g = 9.81, pfac = 1.0) =
    FrozenState(pi, theta, pbar, Float64(g), Float64(pfac))
frozen_from_atmosphere(atm::Atmosphere; pfac = 1.0) =
    FrozenState(z -> atm_pi(atm, z), z -> atm_theta(atm, z), z -> atm_p(atm, z), atm.g, Float64(pfac))
coef_a(fr::FrozenState, z) = fr.pfac * fr.pi(z)
coef_b(fr::FrozenState, z) = (1.0 - fr.pfac) * fr.pbar(z)
