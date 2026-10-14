# Verification and demo for HDGRelaxTheta.jl (DGSEM, energy-stable split form).
# Standard library only; writes CSV files.   Run:  julia --project=examples examples/v2_theta_report.jl
# Python reference values (same algorithm) are given in the comments.

using HDGEulerGravity.HDGRelaxTheta
using LinearAlgebra, Printf, Random

fmt(v) = join([@sprintf("%.3e", x) for x in v], "  ")
fmtr(v) = join([@sprintf("%.2f", x) for x in v], "  ")
const R_ = 287.0
const TH0 = 300.0
const G_ = 9.81

# ------------------------------------------------------------- 1) convergence
const_frozen() = FrozenState(z -> R_, z -> TH0; g = G_)

function normal_mode(fr, Lz)
    a, th, g = coef_a(fr, 0.0), fr.theta(0.0), fr.g
    k = 2π / Lz
    A = [0.0 1.0 0.0; 0.0 0.0 a; 0.0 th 0.0]
    S = [0.0 0.0 0.0; -g 0.0 0.0; 0.0 0.0 0.0]
    E = eigen(ComplexF64.(-im * k .* A .+ S))
    i = argmax(imag.(E.values)); λ = E.values[i]
    v = E.vectors[:, i] ./ E.vectors[3, i] .* (100.0 / a)
    return t -> (z -> real.(v .* exp.(im * k .* permutedims(z) .+ λ * t)))
end

function convergence(method, ef; ps = (1, 2, 3), Nes = (8, 16, 32, 64), form = :split)
    fr = const_frozen(); Lz = 2000.0; ex = normal_mode(fr, Lz)
    c = sqrt(coef_a(fr, 0.0) * fr.theta(0.0)); Tend = Lz / c
    out = Dict{Int,Vector{Float64}}()
    for p in ps
        E = zeros(length(Nes), 3)
        for (k, Ne) in enumerate(Nes)
            s = HDGRelaxTh(Ne, p, 0.0, Lz, fr; bc = :periodic, eps = ef * (Lz / Ne) / c, form = form)
            U0 = initial_state(s, project(s, ex(0.0)))
            cfl = (method == :dirk ? 1.0 : 0.2) / (2p + 1)
            n = ceil(Int, Tend / (cfl * s.h / c))
            U = run!(s, U0, Tend / n, n; method = method)
            E[k, :] = l2_error(s, U[1:s.nq], ex(Tend))
        end
        out[p] = E[:, 3]
        @printf("  %-8s %-4s eps=%g*h/c p=%d: err(Θ)= %s | last rates ρ,m,Θ: %s\n",
                form, method, ef, p, fmt(E[:, 3]), fmtr(log2.(E[end-1, :] ./ E[end, :])))
    end
    return collect(Nes), out
end

function eps_sweep(; Ne = 16, p = 3)
    fr = const_frozen(); Lz = 2000.0; ex = normal_mode(fr, Lz)
    c = sqrt(coef_a(fr, 0.0) * fr.theta(0.0)); Tend = Lz / c
    facs = 10.0 .^ (-8:0)
    res = Dict{Symbol,Vector{Float64}}()
    for method in (:dirk, :imex)
        errs = Float64[]
        for fac in vcat(facs, 0.0)
            s = HDGRelaxTh(Ne, p, 0.0, Lz, fr; bc = :periodic, eps = fac * (Lz / Ne) / c)
            U0 = initial_state(s, project(s, ex(0.0)))
            cfl = (method == :dirk ? 1.0 : 0.2) / (2p + 1)
            n = ceil(Int, Tend / (cfl * s.h / c))
            U = run!(s, U0, Tend / n, n; method = method)
            push!(errs, l2_error(s, U[1:s.nq], ex(Tend))[3])
        end
        res[method] = errs
        @printf("  %s: err(Θ) = %s   (eps=0: %.3e)\n", method, fmt(errs[1:end-1]), errs[end])
    end
    @printf("  eps*c/h  = %s\n", fmt(facs))
    return facs, res
end

# ------------------------------------------------------------- 2) stability
function semidiscrete_matrix(s)
    L = Matrix(s.L); nq = s.nq
    return (L[1:nq, 1:nq] .+ L[1:nq, nq+1:end] * L[nq+1:end, 1:nq]) ./ s.Mdiag
end

function stability_table()
    for (name, atm) in [("isothermal", Atmosphere(T0 = 288.0)), ("lapse", Atmosphere(T0 = 288.0, Γ = 0.0065))]
        for pfac in (1.0, 1.4), form in (:split, :standard)
            mr = [maximum(real.(eigvals(semidiscrete_matrix(
                    HDGRelaxTh(Ne, 3, 0.0, 1e4, frozen_from_atmosphere(atm; pfac = pfac); bc = :wall, form = form)))))
                  for Ne in (8, 16, 32)]
            @printf("  %-10s pfac=%.1f %-8s: max Re λ (Ne=8,16,32) = %s\n", name, pfac, form, fmt(mr))
        end
    end
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    for ef in (1e-3, 0.1, 1.0)
        s = HDGRelaxTh(16, 3, 0.0, 1e4, frozen_from_atmosphere(atm); bc = :wall, eps = ef * 625 / 340)
        Mi = vcat(1 ./ s.Mdiag, fill(1 / s.eps, s.nt))
        @printf("  relaxed eps=%g h/c (split, lapse): max Re λ = %.2e\n", ef,
                maximum(real.(eigvals(Mi .* Matrix(s.L)))))
    end
end

function long_run_growth()
    atm = Atmosphere(T0 = 288.0)
    for form in (:split, :standard)
        s = HDGRelaxTh(16, 3, 0.0, 1e4, frozen_from_atmosphere(atm); bc = :wall, form = form)
        qb = balanced_state(s, z -> atm_Theta(atm, z))
        dq = 1e-3 .* randn(MersenneTwister(1), s.nq)
        E0 = energy(s, dq)
        U = run!(s, initial_state(s, qb .+ dq), 20.0, 1080; method = :dirk)
        @printf("  %-8s: random perturbation, 6 h with dt=20 s: E(6h)/E(0) = %.3e\n",
                form, energy(s, U[1:s.nq] .- qb) / E0)
    end
end

# ------------------------------------------------------------- 3) well-balancing
function well_balanced(method, eps, pfac)
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    s = HDGRelaxTh(40, 3, 0.0, 10000.0, frozen_from_atmosphere(atm; pfac = pfac); bc = :wall, eps = eps)
    q = balanced_state(s, z -> atm_Theta(atm, z))
    U0 = initial_state(s, q)
    dt = (method == :dirk ? 1.0 : 0.2 / 7) * s.h / s.cmax
    U = run!(s, U0, dt, 200; method = method)
    @printf("  pfac=%.1f %-4s eps=%g: max rel. deviation after 200 steps = %.2e\n",
            pfac, method, eps, maximum(abs.(U .- U0)) / maximum(abs.(U0)))
end

function background_residual(pfac)
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    out = Float64[]
    for Ne in (10, 20, 40, 80)
        s = HDGRelaxTh(Ne, 3, 0.0, 10000.0, frozen_from_atmosphere(atm; pfac = pfac); bc = :wall)
        q = project(s, z -> vcat(permutedims(atm_rho.(Ref(atm), z)), 0 .* permutedims(z),
                                 permutedims(atm_Theta.(Ref(atm), z))))
        push!(out, maximum(abs.(tendency(s, q))))
    end
    @printf("  interpolated analytic background (pfac=%.1f), p=3, Ne=10..80: max|dq/dt| = %s\n", pfac, fmt(out))
end

# ------------------------------------------------------------- 4) split-explicit hook
function slow_forcing_check()
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    s = HDGRelaxTh(20, 2, 0.0, 10000.0, frozen_from_atmosphere(atm); bc = :wall, eps = 0.0)
    qs = project(s, z -> vcat(permutedims(atm_rho.(Ref(atm), z)), 0 .* permutedims(z),
                              permutedims(atm_Theta.(Ref(atm), z)))) .+ 1e-3 .* randn(MersenneTwister(0), s.nq)
    F = -tendency(s, qs)
    for method in (:dirk, :imex)
        U = run!(s, initial_state(s, qs), 0.05 * s.h / s.cmax, 50; method = method, F_slow = F)
        @printf("  %s: q* with F_slow = -L(q*) stays steady, max|q - q*| = %.2e\n",
                method, maximum(abs.(U[1:s.nq] .- qs)))
    end
end

# ------------------------------------------------------------- 5) wall demo
function wall_demo()
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    Lz, Ne, p = 10000.0, 80, 3
    cases = [("pfac=1 eps=0 DIRK split", 1.0, 0.0, :dirk, :split),
             ("pfac=1 eps=1e-4h/c IMEX split", 1.0, 1e-4, :imex, :split),
             ("pfac=gamma eps=0 DIRK split", atm.γ, 0.0, :dirk, :split),
             ("pfac=1 eps=0 DIRK standard", 1.0, 0.0, :dirk, :standard)]
    open("theta_wall_profile.csv", "w") do fp
        open("theta_wall_history.csv", "w") do fh
            println(fp, "case,z,pprime")
            println(fh, "case,t,mass_drift,Theta_drift,trace_residual,E_over_E0")
            for (label, pfac, ef, method, form) in cases
                fr = frozen_from_atmosphere(atm; pfac = pfac)
                s0 = HDGRelaxTh(Ne, p, 0.0, Lz, fr; bc = :wall, form = form)
                s = HDGRelaxTh(Ne, p, 0.0, Lz, fr; bc = :wall, eps = ef * (Lz / Ne) / s0.cmax, form = form)
                qb = balanced_state(s, z -> atm_Theta(atm, z))
                a(z) = coef_a(fr, z)
                dq = project(s, z -> vcat(0 .* permutedims(z), 0 .* permutedims(z),
                                          permutedims(100.0 .* exp.(-((z .- 3000.0) ./ 300.0) .^ 2) ./ a.(z))))
                U0 = initial_state(s, qb .+ dq)
                m0, T0 = total(s, qb .+ dq, 1), total(s, qb .+ dq, 3)
                E0 = energy(s, dq)
                Tend = 25.0
                cfl = (method == :dirk ? 0.5 : 0.2) / (2p + 1)
                n = ceil(Int, Tend / (cfl * s.h / s.cmax)); dt = Tend / n
                dm = Float64[]; dT = Float64[]; Er = Float64[]
                mon = (k, U) -> begin
                    q = U[1:s.nq]
                    push!(dm, total(s, q, 1) - m0); push!(dT, total(s, q, 3) - T0)
                    push!(Er, energy(s, q .- qb) / E0)
                    println(fh, "\"$label\",", k * dt, ",", dm[end], ",", dT[end], ",",
                            trace_residual(s, U), ",", Er[end])
                end
                U = run!(s, U0, dt, n; method = method, monitor = mon, every = max(1, n ÷ 200))
                z, Q = evaluate(s, U[1:s.nq] .- qb)
                for k in eachindex(z)
                    println(fp, "\"$label\",", z[k], ",", a(z[k]) * Q[3, k])
                end
                @printf("  %-32s: c(0)=%.1f m/s |Δmass|=%.1e |ΔΘ|=%.1e  max energy increase per output: %.1e\n",
                        label, sqrt(a(0.0) * atm_theta(atm, 0.0)), maximum(abs.(dm)), maximum(abs.(dT)),
                        max(maximum(diff(vcat(1.0, Er))), 0.0))
            end
        end
    end
end

function main()
    println("1) Convergence, constant frozen coefficients, periodic normal mode")
    # Python reference (split, dirk, eps=0) err(Θ):
    #   p=1: 5.579e+00 1.818e+00 4.922e-01 1.258e-01  rates 1.97
    #   p=2: 9.206e-02 9.839e-03 1.166e-03 1.437e-04  rates 3.02
    #   p=3: 3.131e-03 1.957e-04 1.224e-05 7.655e-07  rates 4.00
    open("theta_convergence.csv", "w") do f
        println(f, "form,method,eps_fac,p,Ne,err_Theta")
        for (method, ef) in [(:dirk, 0.0), (:dirk, 1e-9), (:imex, 1e-9)]
            Nes, out = convergence(method, ef; ps = method == :dirk ? (1, 2, 3) : (1, 2))
            for (p, errs) in sort(collect(out)), (Ne, e) in zip(Nes, errs)
                println(f, "split,$method,$ef,$p,$Ne,$e")
            end
        end
        convergence(:dirk, 0.0; ps = (3,), form = :standard)   # ref: 3.122e-03 ... 7.618e-07
    end
    println("\n1b) Relaxation error vs eps (Ne=16, p=3)")
    # Python reference dirk: 1.957e-04 1.966e-04 2.523e-04 1.569e-03 1.553e-02 ...
    facs, sw = eps_sweep()
    open("theta_eps_sweep.csv", "w") do f
        println(f, "eps_fac,err_dirk,err_imex")
        for k in eachindex(facs)
            println(f, facs[k], ",", sw[:dirk][k], ",", sw[:imex][k])
        end
    end
    println("\n2) Stability: largest real part of the spectrum (walls, p=3, eps=0)")
    # Python reference: split ~1e-15 everywhere; standard 7.2e-04 3.6e-04 1.8e-04 (isothermal, pfac=1)
    stability_table()
    # Python reference: split E(6h)/E(0) = 7.882e-01, standard 1.213e+06
    long_run_growth()
    println("\n3) Well-balancing (discrete hydrostatic state, lapse-rate atmosphere, walls)")
    # Python reference: ~1e-12 (dirk), ~1e-15 (imex)
    for pfac in (1.0, 1.4), (method, eps) in [(:dirk, 0.0), (:dirk, 1e-3), (:imex, 1e-3)]
        well_balanced(method, eps, pfac)
    end
    # Python reference: pfac=1 ~1e-12 (exact balance); pfac=1.4: 1.618e-05 2.027e-06 2.537e-07 3.174e-08
    background_residual(1.0); background_residual(1.4)
    println("\n4) Split-explicit hook: constant slow forcing")
    slow_forcing_check()
    println("\n5) Wall demo")
    # Python reference: |Δmass| 2.3e-11 / 1.4e-05 / 3.3e-10 / 4.3e-11, no energy increase
    wall_demo()
    println("\nCSV files: theta_convergence.csv, theta_eps_sweep.csv, theta_wall_profile.csv, theta_wall_history.csv")
end

main()
