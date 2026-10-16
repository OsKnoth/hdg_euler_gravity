# Two-point means and fluxes (TEC/EC/ETEC, non-conservative gravity, LMARS)

# two-point means (as in Trixi.jl)
@inline function ln_mean(x, y)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    f2 < 1e-4 && return (x + y) / @evalpoly(f2, 2, 2 / 3, 2 / 5, 2 / 7)
    return (y - x) / log(y / x)
end
@inline function inv_ln_mean(x, y)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    f2 < 1e-4 && return @evalpoly(f2, 2, 2 / 3, 2 / 5, 2 / 7) / (x + y)
    return log(y / x) / (y - x)
end
@inline function stolarsky_mean(x, y, γ)
    f2 = (x * (x - 2y) + y * y) / (x * (x + 2y) + y * y)
    if f2 < 1e-4
        c1 = (γ - 2) / 3
        c2 = -(γ + 1) * (γ - 3) * c1 / 15
        c3 = -(2γ * (γ - 2) - 9) * c2 / 21
        return 0.5 * (x + y) * @evalpoly(f2, 1, c1, c2, c3)
    end
    yg = exp((γ - 1) * log(y))
    xg = exp((γ - 1) * log(x))
    return (γ - 1) * (yg * y - xg * x) / (γ * (yg - xg))
end

"""Mass and Θ two-point fluxes for the averaged normal velocity vn (momentum flux = f_ρ {v} + {p} n)."""
@inline function mass_theta_flux(vf::Symbol, ρi, Θi, ρj, Θj, vn, γ)
    if vf === :kg
        f1 = 0.5 * (ρi + ρj) * vn
        return f1, f1 * 0.5 * (Θi / ρi + Θj / ρj)
    elseif vf === :tec
        return ln_mean(ρi, ρj) * vn, stolarsky_mean(Θi, Θj, γ) * vn
    elseif vf === :ec
        f1 = ln_mean(ρi, ρj) * vn
        return f1, inv_ln_mean(ρi / Θi, ρj / Θj) * f1
    else # :etec
        f3 = stolarsky_mean(Θi, Θj, γ) * vn
        return f3 * ln_mean(ρi / Θi, ρj / Θj), f3
    end
end

"""Density mean of the non-conservative gravity term ρ_avg (φ_j - φ_i)."""
@inline gravity_mean(gf::Symbol, ρi, ρj, γ) =
    gf === :souza ? 0.5 * (ρi + ρj) : gf === :waruszewski ? ln_mean(ρi, ρj) : stolarsky_mean(ρi, ρj, γ)

"""LMARS flux between left and right states (ρ, w, θ, p); returns (F_ρ, F_m, F_Θ)."""
@inline function lmars(gas::Gas, ρL, wL, θL, pL, ρR, wR, θR, pR)
    ρm = 0.5 * (ρL + ρR)
    cm = 0.5 * (sqrt(gas.γ * pL / ρL) + sqrt(gas.γ * pR / ρR))
    ws = 0.5 * (wL + wR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (pL + pR) - 0.5 * ρm * cm * (wR - wL)
    ρu, θu = ws >= 0 ? (ρL, θL) : (ρR, θR)
    return ws * ρu, ws * ρu * (ws >= 0 ? wL : wR) + ps, ws * ρu * θu
end
