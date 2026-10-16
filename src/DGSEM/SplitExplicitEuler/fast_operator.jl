# Fast operator frozen at a state

"""HDG fast operator (v2) with coefficients frozen at the state q: θ̄ = Θ/ρ, a = pfac p/Θ,
b = (1 - pfac) p (continuous after DSS)."""
function fast_operator(eu::Euler1D, q; pfac = eu.gas.γ, eps = 0.0, form = :split, tau_scale = 1.0, fast = :imex)
    ρ = field(eu, q, 1)
    Θ = field(eu, q, 3)
    P = pressure.(Ref(eu.gas), Θ)
    return HDGRelaxTh(eu.Ne, eu.p, eu.zmin, eu.zmax, pfac .* P ./ Θ, Θ ./ ρ, (1 - pfac) .* P;
                      g = eu.gas.g, bc = :wall, eps = eps, form = form, tau_scale = tau_scale)
end

sound_speed_max(eu::Euler1D, q) =
    maximum(sqrt.(eu.gas.γ .* pressure.(Ref(eu.gas), field(eu, q, 3)) ./ field(eu, q, 1)))
