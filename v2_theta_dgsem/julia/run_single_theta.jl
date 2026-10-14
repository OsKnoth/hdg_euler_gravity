# Single run of the fast operator (DGSEM split form, LMARS, relaxed traces) with IMEX.
#
#     julia run_single_theta.jl
#
# Setup: lapse-rate atmosphere, reflecting walls, one resolution with polynomial order 4
# (5 Gauss–Lobatto nodes per element). Initial state = discrete hydrostatic background
# + pressure pulse (in ρθ) + warm bubble (θ' at constant pressure).
# Writes single_run_profiles.csv and single_run_history.csv; if Plots.jl is installed it
# also writes single_run_profiles.png and single_run_diagnostics.png.
#
# Python reference (same parameters): dt = 0.0097 s, 3105 steps,
#   mass drift ~ 4e-12 (total 7.5372e+03), ρθ drift ~ 5e-10 (total 2.2846e+06),
#   E(T)/E(0) - 1 = -3.29e-06

include(joinpath(@__DIR__, "HDGRelaxTheta.jl"))
using .HDGRelaxTheta
using Printf

# ------------------------------------------------------------------ parameters
const P = 4                  # polynomial order (5 GLL nodes)
const NE = 40                # elements -> h = 250 m
const ZTOP = 10000.0         # domain [0, ZTOP], walls at both ends
const T_END = 30.0           # final time [s]
const CFL = 0.1              # dt = CFL h / (c_max (2p+1)); IMEX stable up to ~0.2,
                             # <= 0.1 keeps the RK2 anti-dissipation below the LMARS dissipation
const EPS_FAC = 0.0          # eps = EPS_FAC h / c_max (0: exact HDG traces)
const PFAC = 1.0             # 1: frozen p/(ρθ);  1.4: tangent linear
const PULSE_PA, PULSE_Z, PULSE_W = 100.0, 3000.0, 300.0
const BUBBLE_K, BUBBLE_Z, BUBBLE_W = 0.5, 6000.0, 500.0
const NPLOT = 10

# ------------------------------------------------------------------ setup
atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
fr = frozen_from_atmosphere(atm; pfac = PFAC)
s0 = HDGRelaxTh(NE, P, 0.0, ZTOP, fr; bc = :wall)
eps = EPS_FAC * s0.h / s0.cmax
s = eps > 0 ? HDGRelaxTh(NE, P, 0.0, ZTOP, fr; bc = :wall, eps = eps) : s0

qb = balanced_state(s, z -> atm_Theta(atm, z))

function perturbation(z)
    a = coef_a.(Ref(fr), z)
    pulse = PULSE_PA .* exp.(-((z .- PULSE_Z) ./ PULSE_W) .^ 2) ./ a
    thp = BUBBLE_K .* exp.(-((z .- BUBBLE_Z) ./ BUBBLE_W) .^ 2)
    rho_p = -atm_rho.(Ref(atm), z) .* thp ./ atm_theta.(Ref(atm), z)
    return vcat(permutedims(rho_p), zeros(1, length(z)), permutedims(pulse))
end

dq = project(s, perturbation)
q0 = qb .+ dq
U0 = initial_state(s, q0)

dt0 = CFL * s.h / (s.cmax * (2P + 1))
nsteps = ceil(Int, T_END / dt0); dt = T_END / nsteps
@printf("p = %d (%d nodes), Ne = %d, h = %.1f m, unknowns = %d + %d traces\n", P, P + 1, NE, s.h, s.nq, s.nt)
@printf("c_max = %.1f m/s, dt = %.4f s, steps = %d, eps = %.2e s, pfac = %g\n", s.cmax, dt, nsteps, eps, PFAC)

# ------------------------------------------------------------------ run
E0 = energy(s, dq)
mass0, Th0 = total(s, q0, 1), total(s, q0, 3)
mnodes = [iq(s, e, 2, i) for e in 1:s.Ne for i in 1:s.np]
hist = zeros(0, 6)
monitor = (n, U) -> begin
    q = U[1:s.nq]
    global hist = vcat(hist, [n * dt  total(s, q, 1) - mass0  total(s, q, 3) - Th0  energy(s, q .- qb) / E0  trace_residual(s, U)  maximum(abs.(q[mnodes]))])
end
U = run!(s, U0, dt, nsteps; method = :imex, monitor = monitor, every = max(1, nsteps ÷ 200))
q1 = U[1:s.nq]
@printf("mass drift = %.2e (total %.4e),  ρθ drift = %.2e (total %.4e)\n", hist[end, 2], mass0, hist[end, 3], Th0)
@printf("E(T)/E(0) - 1 = %.2e   (weighted energy of the deviation; <= 0 expected)\n", hist[end, 4] - 1)

# ------------------------------------------------------------------ output
z, Qb = evaluate(s, qb; npts = NPLOT)
_, Q0 = evaluate(s, q0; npts = NPLOT)
_, Q1 = evaluate(s, q1; npts = NPLOT)
open("single_run_profiles.csv", "w") do f
    println(f, "z,rho_b,m_b,Theta_b,rho_0,m_0,Theta_0,rho_T,m_T,Theta_T")
    for k in eachindex(z)
        println(f, join([z[k]; Qb[:, k]; Q0[:, k]; Q1[:, k]], ","))
    end
end
open("single_run_history.csv", "w") do f
    println(f, "t,mass_drift,Theta_drift,E_over_E0,trace_residual,max_abs_m")
    for k in 1:size(hist, 1)
        println(f, join(hist[k, :], ","))
    end
end
println("wrote single_run_profiles.csv, single_run_history.csv")

if Base.find_package("Plots") !== nothing
    @eval using Plots
    zk = z ./ 1000
    dev = ["ρ' [kg/m³]", "ρw [kg/(m² s)]", "(ρθ)' [K kg/m³]"]
    full = ["ρ [kg/m³]", "ρw [kg/(m² s)]", "ρθ [K kg/m³]"]
    pl = []
    for r in 1:3
        push!(pl, plot([Q0[r, :] .- Qb[r, :], Q1[r, :] .- Qb[r, :]], zk; label = ["initial" "t = $(T_END) s"],
                       color = [:gray :steelblue], xlabel = dev[r] * (r == 2 ? "" : " (deviation)"),
                       ylabel = r == 1 ? "z [km]" : "", legend = r == 1 ? :bottomleft : false))
    end
    for r in 1:3
        push!(pl, plot([Q0[r, :], Q1[r, :]], zk; label = ["initial" "t = $(T_END) s"], color = [:gray :steelblue],
                       ls = [:solid :dash], xlabel = full[r] * " (full field)", ylabel = r == 1 ? "z [km]" : "",
                       legend = false))
    end
    savefig(plot(pl...; layout = (2, 3), size = (1500, 900),
                 plot_title = "DGSEM p=$(P), Ne=$(NE), LMARS, IMEX, eps=$(EPS_FAC) h/c, pfac=$(PFAC)"),
            "single_run_profiles.png")
    t = hist[:, 1]
    d1 = plot(t, [hist[:, 2] ./ mass0, hist[:, 3] ./ Th0]; label = ["mass" "ρθ"], xlabel = "t [s]",
              ylabel = "relative drift of totals")
    d2 = plot(t, hist[:, 4] .- 1; legend = false, xlabel = "t [s]", ylabel = "E(t)/E(0) - 1")
    d3 = eps > 0 ? plot(t, max.(hist[:, 5], 1e-18); yscale = :log10, legend = false, xlabel = "t [s]",
                        ylabel = "trace residual") :
                   plot(t, hist[:, 6]; legend = false, xlabel = "t [s]", ylabel = "max |ρw|")
    savefig(plot(d1, d2, d3; layout = (1, 3), size = (1500, 400)), "single_run_diagnostics.png")
    println("wrote single_run_profiles.png, single_run_diagnostics.png")
else
    println("Plots.jl not installed: only CSV output (install with  using Pkg; Pkg.add(\"Plots\"))")
end
