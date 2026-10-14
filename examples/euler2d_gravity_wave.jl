# Inertia-gravity waves (Skamarock & Klemp 1994, non-rotating): nonlinear Euler in an x–z slice,
# split-explicit MIS with the 2D HDG fast operator re-frozen at every large step.
#
#     julia --project=examples examples/euler2d_gravity_wave.jl
#
# Channel 300 km × 10 km, periodic in x, walls at the bottom and top, N = 0.01 1/s, θ0 = 300 K,
# mean flow U = 20 m/s, θ' = Δθ sin(πz/H) / (1 + ((x - xc)/a)²) at constant pressure,
# Δθ = 0.01 K, a = 5 km, xc = 100 km, T = 3000 s. The wave packet is then centred at
# x = xc + U T = 160 km and symmetric about it.
# Writes gw_theta.csv (x, z, θ' at the nodes) and gw_section.csv (θ' at the node row nearest
# z = 5 km); with Plots.jl also gw_theta.png.

using HDGEulerGravity
using Printf
const E2 = HDGEulerGravity.Euler2D
const SE = HDGEulerGravity.SplitExplicitEuler

# ------------------------------------------------------------------ parameters
const P = 4                     # polynomial order
const NEX, NEZ = 30, 4          # elements (node spacing ~ 2.5 km × 0.6 km for P = 4)
const T_END = 3000.0
const DT = 10.0                 # large (slow) step
const TABLE = :MIS4             # :WSRK3, :MIS2, :MIS4, :MIS4a (see README for MIS4a)
const FAST = :hevi              # :hevi (vertical implicit) or :imex (fully explicit)
const CFL_FAST = 0.2            # dτ = CFL_FAST h / ((2p+1) c); h = hx for :hevi, hz for :imex
const U0, DTH, A, XC = 20.0, 0.01, 5e3, 100e3
const LX, H = 300e3, 10e3

# ------------------------------------------------------------------ setup
gas = SE.Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)
mesh = E2.Mesh2D(NEX, NEZ, P, LX, H)
model = E2.Euler2DModel(mesh, gas, ρr, Θr)
q0 = E2.project2d(model, (x, z) -> begin
    θp = DTH * sin(π * z / H) / (1 + ((x - XC) / A)^2)
    ρ = Θr(z) / (θr(z) + θp)                    # constant pressure: Θ unchanged
    (ρ, ρ * U0, 0.0, Θr(z))
end)
c = E2.sound_speed_max(model, q0)
dτ = CFL_FAST * (FAST == :hevi ? LX / NEX : H / NEZ) / ((2P + 1) * c)
nsteps = round(Int, T_END / DT)
@printf("p = %d, %d × %d elements, unknowns = %d, c_max = %.1f m/s\n", P, NEX, NEZ, model.nq, c)
@printf("%s, Δt = %.1f s (%d steps), fast = %s, dτ = %.3f s (%d substeps per Δt)\n", TABLE, DT, nsteps, FAST, dτ,
        ceil(Int, DT / dτ))

θprime(q) = E2.field(model, q, 4) ./ E2.field(model, q, 1) .- θr.(mesh.Z)
mass0 = E2.total2d(model, q0, 1)
t = @elapsed q = SE.run_mis(model, q0, DT, nsteps; tab = SE.mis_tableau(TABLE), dτ = dτ, fast = FAST,
                            monitor = (n, q) -> n % max(1, nsteps ÷ 10) == 0 &&
                                @printf("  t = %6.0f s  max θ' = %.3e  min θ' = %.3e\n", n * DT, maximum(θprime(q)), minimum(θprime(q))))
θp = θprime(q)
@printf("wall time %.1f s;  relative mass drift = %.1e\n", t, (E2.total2d(model, q, 1) - mass0) / mass0)

# ------------------------------------------------------------------ diagnostics and output
# section: node row nearest z = 5 km
jz = argmin(abs.(mesh.Z[1, :, 1:NEX:end][:] .- 5e3))
ez, j = divrem(jz - 1, P + 1) .+ (1, 1)
xs = Float64[]; ys = Float64[]
for ex in 1:NEX, i in 1:P+1
    e = (ez - 1) * NEX + ex
    (i == 1 && ex > 1) && continue
    push!(xs, mesh.X[i, j, e]); push!(ys, θp[i, j, e])
end
zsec = mesh.Z[1, j, (ez - 1) * NEX + 1]
# the packet is symmetric about x = XC + U T, with one maximum on each side
xc_t = XC + U0 * T_END
kl = argmax([x < xc_t ? y : -Inf for (x, y) in zip(xs, ys)])
kr = argmax([x > xc_t ? y : -Inf for (x, y) in zip(xs, ys)])
sym = maximum(abs(ys[k] - ys[argmin(abs.(xs .- (2xc_t - xs[k])))]) for k in eachindex(xs) if abs(xs[k] - xc_t) < 100e3)
@printf("section z = %.0f m: maxima θ' = %.3e K at x = %.1f km and %.3e K at x = %.1f km (centre %.1f km);\n",
        zsec, ys[kl], xs[kl] / 1e3, ys[kr], xs[kr] / 1e3, xc_t / 1e3)
@printf("  max |θ'(xc + d) - θ'(xc - d)| = %.1e K\n", sym)

open("gw_theta.csv", "w") do f
    println(f, "x,z,theta_prime")
    for k in eachindex(θp)
        println(f, mesh.X[k], ",", mesh.Z[k], ",", θp[k])
    end
end
open("gw_section.csv", "w") do f
    println(f, "x,theta_prime")
    foreach(k -> println(f, xs[k], ",", ys[k]), eachindex(xs))
end
println("wrote gw_theta.csv, gw_section.csv")

if Base.find_package("Plots") !== nothing
    @eval using Plots
    p1 = scatter(mesh.X[:] ./ 1e3, mesh.Z[:] ./ 1e3; zcolor = θp[:], markersize = 2.5, markerstrokewidth = 0,
                 color = :balance, clims = (-maximum(abs, θp), maximum(abs, θp)), xlabel = "x [km]", ylabel = "z [km]",
                 title = "θ' [K] at t = $(T_END) s", label = "")
    p2 = plot(xs ./ 1e3, ys; xlabel = "x [km]", ylabel = "θ' [K]", label = "z = $(round(Int, zsec)) m",
              title = "$(TABLE), Δt = $(DT) s, fast = $(FAST)")
    savefig(plot(p1, p2; layout = (2, 1), size = (1100, 800)), "gw_theta.png")
    println("wrote gw_theta.png")
end
