# Verification and demo for HDGRelaxEuler.jl (stdlib only; writes CSV files).
# Run:  julia run_tests.jl        Plots (optional): julia plot_results.jl
#
# Reference values from the Python implementation (same algorithm) are given in
# the comments of each test; Julia should agree to ~3-4 significant digits.

include(joinpath(@__DIR__, "HDGRelaxEuler.jl"))
using .HDGRelaxEuler
using LinearAlgebra, Printf

fmt(v) = join([@sprintf("%.3e", x) for x in v], "  ")
fmtr(v) = join([@sprintf("%.3f", x) for x in v], "  ")
rates(e) = log2.(e[1:end-1] ./ e[2:end])

# ------------------------------------------------------------- 1) convergence
"""Exact normal mode of q_t + A q_z = S q for constant c₀² (isothermal), periodic."""
function normal_mode(atm, Lz)
    c2v, g, γ = c2(atm, 0.0), atm.g, atm.γ
    k = 2π / Lz
    A = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 c2v 0.0]
    S = [0.0 0.0 0.0; -g 0.0 0.0; 0.0 (-(γ - 1) * g) 0.0]
    E = eigen(ComplexF64.(-im * k .* A .+ S))
    i = argmax(imag.(E.values))                    # upward acoustic branch
    λ = E.values[i]
    v = E.vectors[:, i] ./ E.vectors[3, i] .* 100  # p' amplitude 100 Pa
    return t -> (z -> real.(v .* exp.(im * k .* permutedims(z) .+ λ * t)))
end

function convergence(method, epsfac; ps = (1, 2, 3), Nes = (8, 16, 32, 64), flux = :lmars)
    atm = Atmosphere(T0 = 250.0, Γ = 0.0)
    Lz = 2000.0
    ex = normal_mode(atm, Lz)
    c = sqrt(c2(atm, 0.0))
    Tend = Lz / c
    out = Dict{Int,Vector{Float64}}()
    for p in ps
        errs = Float64[]; erho = Float64[]
        for Ne in Nes
            s = HDGRelax(Ne, p, 0.0, Lz, atm; bc = :periodic, flux = flux,
                         eps = epsfac * (Lz / Ne) / c)
            U0 = initial_state(s, project(s, ex(0.0)))
            cfl = (method == :dirk ? 1.0 : 0.2) / (2p + 1)
            n = ceil(Int, Tend / (cfl * s.h / c)); dt = Tend / n
            U = run!(s, U0, dt, n; method = method)
            e3 = l2_error(s, U[1:s.nq], ex(Tend))
            push!(errs, e3[3]); push!(erho, e3[1] * c^2)
        end
        out[p] = errs
        @printf("  %-5s %-4s eps=%g*h/c p=%d: err(p')= %s | rates %s | rates(rho') %s\n",
                flux, method, epsfac, p, fmt(errs), fmtr(rates(errs)), fmtr(rates(erho)))
    end
    return collect(Nes), out
end

# ------------------------------------------------------------- 1b) eps sweep
function eps_sweep(; Ne = 16, p = 3)
    atm = Atmosphere(T0 = 250.0, Γ = 0.0)
    Lz = 2000.0; ex = normal_mode(atm, Lz); c = sqrt(c2(atm, 0.0)); Tend = Lz / c
    facs = 10.0 .^ (-8:0)
    out = Dict{Symbol,Vector{Float64}}()
    for method in (:dirk, :imex)
        errs = Float64[]
        for fac in facs
            s = HDGRelax(Ne, p, 0.0, Lz, atm; bc = :periodic, eps = fac * (Lz / Ne) / c)
            U0 = initial_state(s, project(s, ex(0.0)))
            cfl = (method == :dirk ? 1.0 : 0.2) / (2p + 1)
            n = ceil(Int, Tend / (cfl * s.h / c)); dt = Tend / n
            U = run!(s, U0, dt, n; method = method)
            push!(errs, l2_error(s, U[1:s.nq], ex(Tend))[3])
        end
        out[method] = errs
        @printf("  %s: err(p') = %s\n", method, fmt(errs))
    end
    s0 = HDGRelax(Ne, p, 0.0, Lz, atm; bc = :periodic, eps = 0.0)
    U0 = initial_state(s0, project(s0, ex(0.0)))
    n = ceil(Int, Tend / (s0.h / c / (2p + 1)))
    U = run!(s0, U0, Tend / n, n; method = :dirk)
    e0 = l2_error(s0, U[1:s0.nq], ex(Tend))[3]
    @printf("  eps*c/h = %s\n  eps=0 reference err(p') = %.3e\n", fmt(facs), e0)
    return facs, out, e0
end

# ------------------------------------------------------------- 2) well-balancing
"""Discrete hydrostatic perturbation in a lapse-rate atmosphere with walls."""
function well_balanced(method, eps; flux = :lmars)
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    Ne, p = 40, 3
    s = HDGRelax(Ne, p, 0.0, 10000.0, atm; bc = :wall, eps = eps, flux = flux)
    pfun(z) = 50.0 * cos(2π * z / 7000.0) + 30.0 * (z / 1e4)^2
    q = zeros(s.nq)
    ξ = reverse(cos.(π .* (0:p) ./ p))               # Chebyshev–Lobatto, incl. ±1
    V, D = legendre_basis(ξ, s.np)
    for e in 1:Ne
        z = 0.5 * (s.xe[e] + s.xe[e+1]) .+ 0.5 * s.h .* ξ
        cp = V \ pfun.(z)                            # continuous interpolant of p'
        dcp = V \ ((D * cp) .* (2 / s.h))            # its derivative (exact)
        q[iq(s, e, 3, 1):iq(s, e, 3, s.np)] .= cp
        q[iq(s, e, 1, 1):iq(s, e, 1, s.np)] .= -dcp ./ atm.g   # ρ' = -p'_z / g
    end
    U0 = initial_state(s, q)
    c = sqrt(c2(atm, 0.0))
    dt = (method == :dirk ? 1.0 : 0.2 / (2p + 1)) * s.h / c
    U = run!(s, U0, dt, 200; method = method)
    dev = maximum(abs.(U .- U0)) / maximum(abs.(U0))
    @printf("  %-5s %-4s eps=%g: max rel. deviation after 200 steps = %.2e\n", flux, method, eps, dev)
    return dev
end

# ------------------------------------------------------------- 3) wall demo
function wall_demo()
    atm = Atmosphere(T0 = 288.0, Γ = 0.0065)
    Lz, Ne, p = 10000.0, 80, 3
    c = sqrt(c2(atm, 0.0))
    pulse(z) = vcat(0 .* permutedims(z), 0 .* permutedims(z),
                    permutedims(100.0 .* exp.(-((z .- 3000.0) ./ 300.0) .^ 2)))
    cases = [("HDG eps=0 (DIRK)", 0.0, :dirk),
             ("relaxed eps=1e-4 h/c (DIRK)", 1e-4, :dirk),
             ("relaxed eps=1e-4 h/c (IMEX)", 1e-4, :imex),
             ("relaxed eps=1e-2 h/c (DIRK)", 1e-2, :dirk)]
    open("wall_demo_profile.csv", "w") do fp
        open("wall_demo_history.csv", "w") do fh
            println(fp, "case,z,rho,m,p")
            println(fh, "case,t,mass_drift,trace_residual")
            for (label, fac, method) in cases
                s = HDGRelax(Ne, p, 0.0, Lz, atm; bc = :wall, eps = fac * (Lz / Ne) / c)
                q0 = project(s, pulse)
                U0 = initial_state(s, q0)
                m0 = total_mass(s, q0)
                Tend = 40.0
                cfl = (method == :dirk ? 0.5 : 0.2) / (2p + 1)
                n = ceil(Int, Tend / (cfl * s.h / c)); dt = Tend / n
                drift = Float64[]; resid = Float64[]
                mon = (k, U) -> begin
                    push!(drift, total_mass(s, U[1:s.nq]) - m0)
                    push!(resid, trace_residual(s, U))
                    println(fh, "\"$label\",", k * dt, ",", drift[end], ",", resid[end])
                end
                U = run!(s, U0, dt, n; method = method, monitor = mon, every = max(1, n ÷ 200))
                z, Q = evaluate(s, U[1:s.nq])
                for k in eachindex(z)
                    println(fp, "\"$label\",", z[k], ",", Q[1, k], ",", Q[2, k], ",", Q[3, k])
                end
                @printf("  %-28s: |mass drift|max = %.2e, trace residual max = %.2e\n",
                        label, maximum(abs.(drift)), maximum(resid))
            end
        end
    end
end

# ------------------------------------------------------------- main
function main()
    println("1) Convergence, periodic isothermal normal mode, one acoustic transit")
    # Python reference (lmars, dirk, eps=0), err(p'):
    #   p=1: 1.891e+02 3.827e+01 8.797e+00 2.146e+00
    #   p=2: 8.594e+00 1.071e+00 1.340e-01 1.700e-02
    #   p=3: 4.093e-01 2.586e-02 1.616e-03 1.010e-04
    open("convergence.csv", "w") do f
        println(f, "flux,method,eps_fac,p,Ne,err_p")
        for (method, ef) in [(:dirk, 0.0), (:dirk, 1e-9), (:imex, 1e-9)]
            Nes, out = convergence(method, ef; ps = method == :dirk ? (1, 2, 3) : (1, 2))
            for (p, errs) in sort(collect(out)), (Ne, e) in zip(Nes, errs)
                println(f, "lmars,$method,$ef,$p,$Ne,$e")
            end
        end
        println("  -- comparison: LLF penalty c*diag(0,1,1) (rho' loses one order) --")
        Nes, out = convergence(:dirk, 0.0; flux = :llf)
        for (p, errs) in sort(collect(out)), (Ne, e) in zip(Nes, errs)
            println(f, "llf,dirk,0.0,$p,$Ne,$e")
        end
    end

    println("\n1b) Relaxation error vs eps (Ne=16, p=3)")
    # Python reference dirk: 2.586e-02 ... 4.557e+00 (1e-4) ... 4.455e+02 (1e-2)
    facs, sweep, e0 = eps_sweep()
    open("eps_sweep.csv", "w") do f
        println(f, "eps_fac,err_dirk,err_imex,err_eps0")
        for k in eachindex(facs)
            println(f, facs[k], ",", sweep[:dirk][k], ",", sweep[:imex][k], ",", e0)
        end
    end

    println("\n2) Well-balancing (discrete hydrostatic perturbation, lapse-rate atmosphere)")
    for (method, eps) in [(:dirk, 0.0), (:dirk, 1e-3), (:imex, 1e-3), (:imex, 0.0)]
        well_balanced(method, eps)
    end
    well_balanced(:dirk, 0.0; flux = :llf)

    println("\n3) Wall demo: pulse, conservation and constraint residual")
    # Python reference: eps=0 drift ~1e-14, eps=1e-4 h/c drift ~1.1e-05, eps=1e-2 ~1.1e-03
    wall_demo()
    println("\nCSV files written: convergence.csv, eps_sweep.csv, wall_demo_profile.csv, wall_demo_history.csv")
end

main()
