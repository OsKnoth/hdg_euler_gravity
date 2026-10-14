# Unit tests (fast). The detailed verification reports are in examples/v*_report.jl.
#
#     julia --project -e 'using Pkg; Pkg.test()'

using HDGEulerGravity
using LinearAlgebra, Random, Test

const V1 = HDGEulerGravity.HDGRelaxEuler
const V2 = HDGEulerGravity.HDGRelaxTheta
const V3 = HDGEulerGravity.SplitExplicitEuler
const E2 = HDGEulerGravity.Euler2D

rate(e1, e2) = log2(e1 / e2)

@testset "HDGEulerGravity" begin

@testset "v1 HDGRelaxEuler: periodic normal mode" begin
    atm = V1.Atmosphere(T0 = 250.0, Γ = 0.0)
    Lz = 2000.0
    c = sqrt(V1.c2(atm, 0.0)); g, γ = atm.g, atm.γ
    k = 2π / Lz
    A = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 c^2 0.0]
    S = [0.0 0.0 0.0; -g 0.0 0.0; 0.0 (-(γ - 1) * g) 0.0]
    E = eigen(ComplexF64.(-im * k .* A .+ S))
    i = argmax(imag.(E.values)); λ = E.values[i]
    v = E.vectors[:, i] ./ E.vectors[3, i] .* 100
    ex(t) = z -> real.(v .* exp.(im * k .* permutedims(z) .+ λ * t))
    Tend = Lz / c
    errs = map((8, 16)) do Ne
        s = V1.HDGRelax(Ne, 2, 0.0, Lz, atm; bc = :periodic)
        n = ceil(Int, Tend / (s.h / c / 5))
        U = V1.run!(s, V1.initial_state(s, V1.project(s, ex(0.0))), Tend / n, n)
        V1.l2_error(s, U[1:s.nq], ex(Tend))[3]
    end
    @test rate(errs...) > 2.5
end

@testset "v2 HDGRelaxTheta: fast operator" begin
    atm = V2.Atmosphere(T0 = 288.0, Γ = 0.0065)
    for pfac in (1.0, 1.4)
        fr = V2.frozen_from_atmosphere(atm; pfac = pfac)
        s = V2.HDGRelaxTh(10, 3, 0.0, 1e4, fr; bc = :wall)
        λ = eigvals(Diagonal(1 ./ s.Mdiag) * Matrix(s.K + s.G * s.H))
        @test maximum(real.(λ)) < 1e-12                                  # energy stable (split form)
        Z = V2.nodes(s)                                                    # nodal constructor ≈ function one
        s2 = V2.HDGRelaxTh(10, 3, 0.0, 1e4, V2.coef_a.(Ref(fr), Z), fr.theta.(Z), V2.coef_b.(Ref(fr), Z); g = fr.g)
        @test norm(s2.K - s.K) / norm(s.K) < 1e-9
        qb = V2.balanced_state(s, z -> V2.atm_Theta(atm, z))                    # well balanced
        U = V2.run!(s, V2.initial_state(s, qb), 1.0, 20)
        @test maximum(abs.(U[1:s.nq] .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10
    end
    s = V2.HDGRelaxTh(8, 2, 0.0, 1e4, V2.frozen_from_atmosphere(atm); bc = :wall)
    s_std = V2.HDGRelaxTh(8, 2, 0.0, 1e4, V2.frozen_from_atmosphere(atm); bc = :wall, form = :standard)
    λstd = eigvals(Diagonal(1 ./ s_std.Mdiag) * Matrix(s_std.K + s_std.G * s_std.H))
    @test maximum(real.(λstd)) > 1e-6                                     # standard form is not
    q = randn(MersenneTwister(1), s.nq)                                                       # split-explicit hook
    U = V2.run!(s, V2.initial_state(s, q), 0.5, 10; method = :imex, F_slow = -V2.tendency(s, q))
    @test maximum(abs.(U[1:s.nq] .- q)) < 1e-10
end

@testset "v3 SplitExplicitEuler: MIS for nonlinear Euler" begin
    atm = V3.Atmosphere(T0 = 288.0, Γ = 0.0065)
    eu = V3.Euler1D(10, 3, 0.0, 1e4)
    qb = V3.hydrostatic_state(eu, z -> V3.atm_Theta(atm, z))
    @test maximum(abs.(V3.rhs_nonlinear(eu, qb))) < 1e-10
    dτ = 0.1 * eu.h / (V3.sound_speed_max(eu, qb) * (2eu.p + 1))
    q = V3.run_mis(eu, qb, 10.0, 5; tab = V3.mis_tableau(:MIS4a), dτ = dτ)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-11

    pert(z) = begin
        a = 1.4 .* V3.atm_p.(Ref(atm), z) ./ V3.atm_Theta.(Ref(atm), z)
        thp = 0.5 .* exp.(-((z .- 6000) ./ 1000) .^ 2)
        vcat(permutedims(-V3.atm_rho.(Ref(atm), z) .* thp ./ V3.atm_theta.(Ref(atm), z)), zeros(1, length(z)),
             permutedims(100.0 .* exp.(-((z .- 3000) ./ 1000) .^ 2) ./ a))
    end
    q0 = qb .+ V3.project(eu, pert)
    Tend = 4.0
    dtr = 0.2 * eu.h / (V3.sound_speed_max(eu, q0) * (2eu.p + 1))
    nref = ceil(Int, Tend / dtr)
    qref = V3.run_rk4(eu, q0, Tend / nref, nref)
    for (name, order) in ((:MIS2, 2), (:MIS4, 3), (:WSRK3, 2))
        e = [V3.l2_error(eu, V3.run_mis(eu, q0, Δt, round(Int, Tend / Δt); tab = V3.mis_tableau(name), dτ = 0.02,
                                  fast = :dirk) .- qref)[2] for Δt in (1.0, 0.5)]
        @test rate(e...) > order - 0.4
    end
    q = V3.run_mis(eu, q0, 5.0, 4; tab = V3.mis_tableau(:MIS4), dτ = dτ)
    @test abs(V3.total(eu, q, 1) - V3.total(eu, q0, 1)) < 1e-12 * V3.total(eu, q0, 1)
    @test abs(V3.total(eu, q, 3) - V3.total(eu, q0, 3)) < 1e-12 * V3.total(eu, q0, 3)

    s = V3.fast_operator(eu, q)                                              # re-frozen, non-smooth state
    λ = eigvals(Diagonal(1 ./ s.Mdiag) * Matrix(s.K + s.G * s.H))
    @test maximum(real.(λ)) < 1e-12
end

@testset "Euler2D: x–z slice with terrain" begin
    gas = V3.Gas()
    ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    for zs in (x -> 0.0, x -> 400.0 * exp(-((x - 10e3) / 2e3)^2))
        M = E2.Mesh2D(5, 3, 3, 20e3, 10e3; zs = zs)
        @test maximum(abs(sum(M.D[i, k] * M.Jξx[k, j, e] + M.D[j, k] * M.Jηx[i, k, e] for k in 1:M.np))
                      for i in 1:M.np, j in 1:M.np, e in 1:M.Ne) < 1e-9                    # metric identities
        m = E2.Euler2DModel(M, gas, ρr, Θr)
        qr = E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
        @test maximum(abs.(E2.rhs_nonlinear(m, qr))) < 1e-12                               # rest state
        q = E2.project2d(m, (x, z) -> (ρr(z) * (1 - 0.01 * exp(-((x - 8e3)^2 + (z - 4e3)^2) / 2e3^2)),
                                       10.0 * ρr(z), 0.0, Θr(z)))
        Nq = E2.rhs_nonlinear(m, q)
        @test abs(E2.total2d(m, Nq, 1)) < 1e-12 * E2.total2d(m, q, 1)                       # conservation
        @test abs(E2.total2d(m, Nq, 4)) < 1e-12 * E2.total2d(m, q, 4)
        s = E2.fast_operator(m, qr)                                                          # energy stable
        @test maximum(real.(eigvals(Matrix(Diagonal(1 ./ s.Mdiag) * (s.K + s.G * s.H))))) < 1e-12
        s = E2.fast_operator(m, q)                                                           # bounded growth
        @test maximum(real.(eigvals(Matrix(Diagonal(1 ./ s.Mdiag) * (s.K + s.G * s.H))))) < 1e-3
    end
    # MIS with explicit and HEVI fast part against RK4 on N (gravity-wave setup, 100 s)
    M = E2.Mesh2D(10, 2, 3, 100e3, 10e3)
    m = E2.Euler2DModel(M, gas, ρr, Θr)
    q0 = E2.project2d(m, (x, z) -> begin
        ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / 10e3) / (1 + ((x - 50e3) / 5e3)^2))
        (ρ, 20.0 * ρ, 0.0, Θr(z))
    end)
    c = E2.sound_speed_max(m, q0)
    dtr = 0.25 * 5e3 / ((c + 20) * 7)
    nref = ceil(Int, 100 / dtr)
    qref = V3.run_rk4(m, q0, 100 / nref, nref)
    dev = norm(qref .- q0)
    for (fast, dτ) in ((:imex, 0.1 * 5e3 / (c * 7)), (:hevi, 0.2 * 10e3 / (c * 7)))
        e = [norm(V3.run_mis(m, q0, Δt, round(Int, 100 / Δt); tab = V3.mis_tableau(:MIS4), dτ = dτ * Δt / 10,
                             fast = fast) .- qref) for Δt in (10.0, 5.0)]      # dτ refined with Δt
        @test e[2] < 0.05 * dev
        @test rate(e...) > 2.0
    end
    qb = E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
    q = V3.run_mis(m, qb, 20.0, 3; tab = V3.mis_tableau(:MIS4), dτ = 0.2 * 10e3 / (c * 7), fast = :hevi)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10                              # well balanced
end

end
