# Tests of the split-explicit (MIS) solver for the nonlinear Euler equations with gravity,
# fast part = HDG/LMARS DGSEM operator of v2 re-frozen at every large step.
#
#     julia --project=examples examples/v3_mis_report.jl
#
# Writes mis_convergence.csv, mis_stability_function.csv, mis_long_run.csv.

using HDGEulerGravity.SplitExplicitEuler, LinearAlgebra, Printf

const ATM = Atmosphere(T0 = 288.0, Γ = 0.0065)
const TABLES = (:WSRK3, :MIS2, :MIS4, :MIS4a)

"""Pressure pulse (Pa) in Θ plus warm bubble (K, at constant pressure) on the background."""
function perturbation(; pulse = 100.0, zp = 3000.0, wp = 300.0, bubble = 0.5, zb = 6000.0, wb = 500.0)
    return z -> begin
        a = 1.4 .* atm_p.(Ref(ATM), z) ./ atm_Theta.(Ref(ATM), z)
        Θp = pulse .* exp.(-((z .- zp) ./ wp) .^ 2) ./ a
        thp = bubble .* exp.(-((z .- zb) ./ wb) .^ 2)
        vcat(permutedims(-atm_rho.(Ref(ATM), z) .* thp ./ atm_theta.(Ref(ATM), z)), zeros(1, length(z)),
             permutedims(Θp))
    end
end

mnodes(eu) = [iq(eu.np, e, 2, i) for e in 1:eu.Ne for i in 1:eu.np]
fast_dt(eu, q; cfl = 0.1) = cfl * eu.h / (sound_speed_max(eu, q) * (2eu.p + 1))

function test_balance()
    println("1) Well-balancing (discrete hydrostatic state of N, lapse-rate atmosphere, walls)")
    eu = Euler1D(20, 4, 0.0, 1e4)
    qb = hydrostatic_state(eu, z -> atm_Theta(ATM, z))
    @printf("  max|N(q_b)| = %.2e\n", maximum(abs.(rhs_nonlinear(eu, qb))))
    Δt = 12fast_dt(eu, qb)
    for name in (:WSRK3, :MIS4a)
        q = run_mis(eu, qb, Δt, 30; tab = mis_tableau(name), nsub = 12)
        @printf("  %-6s nsub=12, Δt=%.3f s, 30 steps: max rel. deviation = %.2e\n", name, Δt,
                maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)))
    end
end

function test_refrozen_stability()
    println("\n2) Fast operator re-frozen at a non-smooth (discontinuous) state: max Re λ of M⁻¹(K + G H)")
    for p in (2, 4), Ne in (10, 20)
        eu = Euler1D(Ne, p, 0.0, 1e4)
        qb = hydrostatic_state(eu, z -> atm_Theta(ATM, z))
        q0 = qb .+ project(eu, perturbation(pulse = 500.0, wp = 200.0))
        q = run_mis(eu, q0, 12fast_dt(eu, q0), 10; tab = mis_tableau(:WSRK3), nsub = 12)   # discontinuous now
        jmp = maximum(abs(blk(eu, q, e, 3)[end] / blk(eu, q, e, 1)[end] - blk(eu, q, e + 1, 3)[1] / blk(eu, q, e + 1, 1)[1])
                      for e in 1:eu.Ne-1)
        s = fast_operator(eu, q)
        λ = eigvals(Matrix(spdiagm_inv(s.Mdiag) * s.A))
        @printf("  p=%d Ne=%2d: max |[θ]| at faces = %.2e K,  max Re λ = %.2e,  max |λ| = %.2e\n",
                p, Ne, jmp, maximum(real.(λ)), maximum(abs.(λ)))
    end
end
spdiagm_inv(d) = Diagonal(1 ./ d)

function test_convergence()
    println("\n3) Temporal convergence against RK4 on N (Ne=20, p=4, T=10 s, fast part: IMEX, nsub=24)")
    println("   (fixed ratio: the 2nd-order fast integrator limits the observed order to 2)")
    eu = Euler1D(20, 4, 0.0, 1e4)
    qb = hydrostatic_state(eu, z -> atm_Theta(ATM, z))
    q0 = qb .+ project(eu, perturbation(wp = 500.0, wb = 800.0))
    T = 10.0
    dtr = 0.2 * fast_dt(eu, q0; cfl = 1.0)
    nref = ceil(Int, T / dtr)
    qref = run_rk4(eu, q0, T / nref, nref)
    @printf("  reference: RK4, %d steps; Δmass = %.1e, ΔΘ = %.1e\n", nref,
            total(eu, qref, 1) - total(eu, q0, 1), total(eu, qref, 3) - total(eu, q0, 3))
    dts = (0.5, 0.25, 0.125, 0.0625)              # dτ ≈ Δt/24 within the explicit limit
    rows = String[]
    for name in TABLES
        errs = Float64[]
        for Δt in dts
            q = run_mis(eu, q0, Δt, round(Int, T / Δt); tab = mis_tableau(name), nsub = 24)
            push!(errs, l2_error(eu, q .- qref)[2])
        end
        rates = [log2(errs[k] / errs[k+1]) for k in 1:length(errs)-1]
        @printf("  %-6s err(m) = %s   rates = %s\n", name, join([@sprintf("%.2e", e) for e in errs], " "),
                join([@sprintf("%.2f", r) for r in rates], " "))
        append!(rows, ["$name,$(dts[k]),$(errs[k])" for k in eachindex(dts)])
    end
    open("mis_convergence.csv", "w") do f
        println(f, "table,dt,err_m"); foreach(r -> println(f, r), rows)
    end
end

"""Stability function for y' = λ y, everything fast (S = 0), exact inner integration, Δt = 1."""
function mis_R(tab, λ)
    ns = length(tab.d); ΔY = zeros(ComplexF64, ns); Y = 1.0 + 0im
    for i in 1:ns
        Z0 = 1.0 + sum(tab.α[i, j] * ΔY[j] for j in 1:i-1; init = 0.0im)
        f = sum(tab.γ[i, j] * ΔY[j] / tab.d[i] for j in 1:i-1; init = 0.0im)
        T = tab.d[i]
        Y = exp(λ * T) * Z0 + (abs(λ) < 1e-14 ? T : (exp(λ * T) - 1) / λ) * f
        ΔY[i] = Y - 1
    end
    return Y
end

function test_stability_function()
    println("\n4) MIS for purely fast oscillations y' = iω y (S = 0, exact inner integration)")
    ω = range(0, 200, length = 20001)
    open("mis_stability_function.csv", "w") do f
        println(f, "omega_dt," * join(TABLES, ","))
        A = [abs.(mis_R.(Ref(mis_tableau(n)), im .* ω)) for n in TABLES]
        for k in eachindex(ω)
            println(f, join([ω[k]; [a[k] for a in A]], ","))
        end
        for (n, a) in zip(TABLES, A)
            k = findfirst(a .> 1 + 1e-10)
            @printf("  %-6s max |R(iωΔt)| (ωΔt ≤ 200) = %.4f,  first ωΔt with |R| > 1: %s\n", n, maximum(a),
                    k === nothing ? "none" : @sprintf("%.2f", ω[k]))
        end
    end
end

function test_long_run()
    println("\n5) Split-explicit runs: Ne=40, p=4, 600 s, fast part IMEX at acoustic CFL 0.2, Δt = nsub dτ")
    eu = Euler1D(40, 4, 0.0, 1e4)
    qb = hydrostatic_state(eu, z -> atm_Theta(ATM, z))
    q0 = qb .+ project(eu, perturbation())
    dτ = fast_dt(eu, q0; cfl = 0.2)
    @printf("  dτ = %.4f s\n", dτ)
    mn = mnodes(eu)
    open("mis_long_run.csv", "w") do f
        println(f, "table,dt,t,max_abs_m")
        for name in TABLES, nsub in (6, 12, 24)
            Δt = 600 / ceil(600 / (nsub * dτ))
            mx = 0.0
            q = run_mis(eu, q0, Δt, round(Int, 600 / Δt); tab = mis_tableau(name), nsub = nsub,
                        monitor = (n, q) -> (m = maximum(abs.(q[mn])); mx = max(mx, m); println(f, "$name,$Δt,$(n * Δt),$m")))
            @printf("  %-6s nsub=%2d Δt=%.3f s: max|m| = %.3e, final max|m| = %.3e, Δmass = %.1e, ΔΘ = %.1e\n",
                    name, nsub, Δt, mx, maximum(abs.(q[mn])), total(eu, q, 1) - total(eu, q0, 1),
                    total(eu, q, 3) - total(eu, q0, 3))
        end
    end
end

test_balance()
test_refrozen_stability()
test_convergence()
test_stability_function()
test_long_run()
println("\nCSV files: mis_convergence.csv, mis_stability_function.csv, mis_long_run.csv")
