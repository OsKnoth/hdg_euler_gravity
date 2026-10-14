# Unit tests (fast). The detailed verification reports are in examples/v*_report.jl.
#
#     julia --project -e 'using Pkg; Pkg.test()'

using HDGEulerGravity
using LinearAlgebra, Random, Test

const V1 = HDGEulerGravity.HDGRelaxEuler
const V2 = HDGEulerGravity.HDGRelaxTheta
const V3 = HDGEulerGravity.SplitExplicitEuler

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

end
