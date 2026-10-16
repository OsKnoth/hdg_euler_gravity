# Fast integrators for q' = L q + f (L frozen, f constant during the fast steps)

# IMEX tableau (explicit part: 2-stage RK2; implicit part: SDIRK, γ = 1 - 1/√2)
const IMEX_GM = 1 - 1 / sqrt(2)
const IMEX_DL = 1 - 1 / (2IMEX_GM)
const IMEX_AE = [0.0 0.0 0.0; IMEX_GM 0.0 0.0; IMEX_DL (1 - IMEX_DL) 0.0]
const IMEX_AI = [0.0 0.0 0.0; 0.0 IMEX_GM 0.0; 0.0 (1 - IMEX_GM) IMEX_GM]

"""Off-centring of the WRF vertically implicit acoustic step, β = (1 + epssm)/2 with epssm = 0.1."""
const WRF_BETA = 0.55

"""WRF divergence damping (smdiv): horizontal pressure gradient from p + γd (p - p_old)."""
const WRF_SMDIV = 0.1

"""
    fast_solve!(qout, s, q0, F_slow, dt, nsteps; method=:imex)

Integrate the fast system q' = L q + f_b/M + F_slow over nsteps steps of size dt, starting from
q0 (qout may alias q0). Methods:
  :imex    explicit 2-stage RK2 of the IMEX tableau (needs `fast_apply!`)
  :hevi    IMEX: horizontal explicit, vertical implicit (horizontal/vertical split)
  :verlet  Störmer–Verlet: (ρ, Θ) half step, momentum full step, (ρ, Θ) half step
  :fb      forward-backward: momentum explicit, interface implicit, then (ρ, Θ)
  :wrf     WRF acoustic step: horizontal momentum forward, then (ρ, Θ, vertical momentum)
           vertically implicit, off-centred Crank–Nicolson (β = WRF_BETA), divergence damping
           by the pressure extrapolation of WRF (γd = WRF_SMDIV)
Allocation-free for operators with a cached `fast_workspace` (7 vectors).
"""
function fast_solve!(qout, s::AbstractFastOperator, q0, F_slow, dt, nsteps; method::Symbol = :imex)
    f, q, Y, k1, k2, tmp, rhs = fast_workspace(s)
    fb = fast_forcing(s)
    if F_slow === nothing
        @. f = fb
    else
        M = fast_mass(s)
        @. f = fb + M * F_slow
    end
    q .= q0
    if method == :imex
        imex_loop!(q, s, f, dt, nsteps, Y, k1, k2)
    elseif method == :hevi
        hevi_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    elseif method == :verlet
        verlet_loop!(q, s, f, dt, nsteps, k1, k2)
    elseif method == :fb
        fb_loop!(q, s, f, dt, nsteps, k1, k2)
    elseif method == :wrf
        wrf_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    else
        error("unknown fast method $method (available: :imex, :hevi, :verlet, :fb, :wrf)")
    end
    qout .= q
    return qout
end

function imex_loop!(q, s, f, dt, nsteps, Y, k1, k2)
    a21 = IMEX_AE[2, 1]
    a31 = IMEX_AE[3, 1]
    a32 = IMEX_AE[3, 2]
    for _ in 1:nsteps
        fast_apply!(k1, s, q, f)
        @. Y = q + (dt * a21) * k1
        fast_apply!(k2, s, Y, f)
        @. q = q + dt * (a31 * k1 + a32 * k2)
    end
    return q
end

"""HEVI with the IMEX tableau: horizontal part explicit, vertical part implicit."""
function hevi_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    F = vertical_factor(s, dt * IMEX_GM)
    a21 = IMEX_AE[2, 1]
    a31 = IMEX_AE[3, 1]
    a32 = IMEX_AE[3, 2]
    i32 = IMEX_AI[3, 2]
    for _ in 1:nsteps
        fast_rhs_horizontal!(k1, s, q, f)                # explicit stage 1 (Y1 = q)
        @. rhs = q + (dt * a21) * k1
        vertical_solve!(Y, s, F, rhs)                    # implicit stage 2
        fast_rhs_horizontal!(k2, s, Y, f)
        vertical_apply!(tmp, s, Y)
        @. rhs = q + dt * (a31 * k1 + a32 * k2 + i32 * tmp)
        vertical_solve!(q, s, F, rhs)                    # stage 3 = new state
    end
    return q
end

"""Störmer–Verlet (Xu, van der Vegt, Bokhove 2008, eq. 4.6 with f = 0)."""
function verlet_loop!(q, s, f, dt, nsteps, kT, km)
    for _ in 1:nsteps
        interface_update!(s, q)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += (dt / 2) * kT
        fast_rhs_momentum!(km, s, q, f)
        @. q += dt * km
        interface_update!(s, q)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += (dt / 2) * kT
    end
    return q
end

"""Forward-backward Euler: momentum explicit, interface variables implicit (relaxed), then (ρ, Θ)."""
function fb_loop!(q, s, f, dt, nsteps, kT, km)
    interface_update!(s, q)
    for _ in 1:nsteps
        fast_rhs_momentum!(km, s, q, f)
        @. q += dt * km
        interface_update!(s, q; relax = true, dt = dt)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += dt * kT
    end
    return q
end

"""
WRF acoustic step (Klemp, Skamarock, Dudhia 2007): horizontal momentum forward with the old
thermodynamic state extrapolated by the divergence damping, qⁿ + γd (qⁿ - qⁿ⁻¹), then (ρ, Θ,
vertical momentum) with the new horizontal fluxes,
vertically implicit with off-centring β:
    (I - β dt A_v) Yⁿ⁺¹ = Yⁿ + dt [(1 - β) A_v Yⁿ + H(qⁿ with new horizontal momentum) + f]
"""
function wrf_loop!(q, s, f, dt, nsteps, qold, kh, kt, tmp, rhs)
    β = WRF_BETA
    γd = WRF_SMDIV
    F = vertical_factor(s, β * dt)
    qold .= q
    for _ in 1:nsteps
        @. tmp = q + γd * (q - qold)                     # divergence damping (pressure extrapolation)
        qold .= q
        fast_rhs_hmomentum!(kh, s, tmp, f)
        @. q += dt * kh                                  # horizontal momentum forward
        fast_rhs_hthermo!(kt, s, q, f)                   # horizontal divergence with the new momentum
        vertical_apply!(tmp, s, q)
        @. rhs = q + dt * (kt + (1 - β) * tmp)
        vertical_solve!(q, s, F, rhs)
    end
    return q
end
