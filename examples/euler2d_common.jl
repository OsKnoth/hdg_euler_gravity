# Shared helpers for the 2D examples: MIS run with VTKHDF time series (ParaView) and Makie PNG,
# and the Skamarock–Klemp gravity-wave setup used by the short and the long channel.
#
# Loaded by euler2d_gravity_wave.jl, euler2d_gravity_wave_long.jl and euler2d_hills.jl.

using HDGEulerGravity
using HDF5, CairoMakie            # activate the VTKHDF and Makie package extensions
using Printf
const E2 = HDGEulerGravity.Euler2D
const SE = HDGEulerGravity.SplitExplicitEuler

"""
    run_with_output(model, q0; name, Δt, T, nsub=12, table=:MIS4, fast=:hevi, nout=10,
                    refreeze_every=1, plot_names=["theta_prime", "w"], zoom=nothing)

MIS run over [0, T] with `nsub` fast substeps per large step; writes `name.vtkhdf` (initial
state and `nout` further outputs) and `name.png` (final state). Returns (q, wall time).
"""
function run_with_output(model, q0; name, Δt, T, nsub = 12, table = :MIS4, fast = :hevi, nout = 10,
                         refreeze_every = 1, plot_names = ["theta_prime", "w"], zoom = nothing)
    nsteps = round(Int, T / Δt)
    every = max(1, nsteps ÷ nout)
    @printf("%s: %s, Δt = %.3f s (%d steps), fast = %s, nsub = %d (dτ ≈ %.3f s), re-freeze every %d\n",
            name, table, Δt, nsteps, fast, nsub, Δt / nsub, refreeze_every)
    vtk = E2.open_vtkhdf("$name.vtkhdf", model)
    E2.write_vtkhdf!(vtk, model, q0, 0.0)
    mass0 = E2.total2d(model, q0, 1)
    monitor = (n, q) -> begin
        F = Dict(E2.output_fields(model, q))
        @printf("  t = %9.0f s  θ' ∈ [%.3e, %.3e]  max|w| = %.3e  Δmass/mass = %.1e\n", n * Δt,
                extrema(F["theta_prime"])..., maximum(abs, F["w"]), (E2.total2d(model, q, 1) - mass0) / mass0)
        E2.write_vtkhdf!(vtk, model, q, n * Δt)
    end
    t = @elapsed q = SE.run_mis(model, q0, Δt, nsteps; tab = SE.mis_tableau(table), nsub = nsub, fast = fast,
                                refreeze_every = refreeze_every, monitor = monitor, every = every)
    close(vtk)
    fig = E2.plot_fields(model, q; names = plot_names, zoom = zoom,
                         title = @sprintf("%s, t = %.0f s", name, nsteps * Δt))
    save("$name.png", fig)
    @printf("wall time %.1f s; wrote %s.vtkhdf (%d outputs), %s.png\n", t, name, nsteps ÷ every + 1, name)
    return q, t
end

"""
Stable CFL limits of one fast step (spectral radius ≤ 1 at the reference state, CFL defined by
dτ = CFL h / ((2p+1) c), h = element size; HEVI: horizontal element size, element aspect 3),
measured for p = 2..5:
    p      imex    verlet  fb      hevi
    2      1.07    1.05    0.77    0.98
    3      0.24    0.79    0.56    0.28
    4      0.088   0.63    0.44    0.075
    5      0.050   0.52    0.36    0.060
Verlet ≈ 3.15/(p+1), fb ≈ 0.7 verlet. The explicit part of the IMEX/HEVI tableau is a 2-stage
2nd-order Runge–Kutta method, which is unstable on the imaginary axis (|R(iy)|² = 1 + y⁴/4); it is
stabilized only by the LMARS dissipation, hence the small limits for p ≥ 3. Beyond them the growth
is weak (1 + 3e-7 per step at CFL 0.2 for p = 4) but not zero.
"""
const CFL_LIMITS = Dict(2 => (1.07, 1.05, 0.77, 0.98), 3 => (0.24, 0.79, 0.56, 0.28),
                        4 => (0.088, 0.63, 0.44, 0.075), 5 => (0.050, 0.52, 0.36, 0.060))

"""Acoustic substep of the fast integrator: dτ = cfl h / ((2p+1) c), h = horizontal element size for
:hevi, smallest element size otherwise; default cfl = safety × limit (table above)."""
function fast_step(model, q, fast; cfl = nothing, safety = 0.85)
    M = model.mesh
    if cfl === nothing
        lim = get(CFL_LIMITS, M.p, (0.05, 3.15 / (M.p + 1), 2.2 / (M.p + 1), 0.05))
        cfl = safety * (fast == :verlet ? lim[2] : fast == :fb ? lim[3] : fast == :hevi ? lim[4] : lim[1])
    end
    hz = minimum(M.Z[:, end, :] .- M.Z[:, 1, :])
    h = fast == :hevi ? M.Lx / M.Nex : min(M.Lx / M.Nex, hz)
    return cfl * h / ((2M.p + 1) * E2.sound_speed_max(model, q))
end

"""
    gravity_wave(; name, L, H=10e3, a, xc=L/3, U=20, Δθ=0.01, N=0.01, θ0=300, T, P=4, NEX, NEZ,
                 nsub=12, table=:MIS4, fast=:hevi, cfl=nothing, nout=10)

Skamarock & Klemp (1994) gravity waves in a periodic channel [0, L] × [0, H] with walls:
θ' = Δθ sin(πz/H)/(1 + ((x - xc)/a)²) at constant pressure on a constant-N state with mean flow U.
The large step is Δt = nsub dτ (acoustic substep dτ, rounded so that T/Δt is an integer).
Prints the two maxima of θ' at the node row nearest z = H/2 and the asymmetry about xc + U T.
"""
function gravity_wave(; name, L, H = 10e3, a, xc = L / 3, U = 20.0, Δθ = 0.01, N = 0.01, θ0 = 300.0, T,
                      P = 4, NEX, NEZ, nsub = 12, table = :MIS4, fast = :hevi, cfl = nothing, nout = 10)
    gas = SE.Gas()
    ρr, Θr, θr = E2.stratified_reference(gas; θ0 = θ0, N = N)
    mesh = E2.Mesh2D(NEX, NEZ, P, L, H)
    model = E2.Euler2DModel(mesh, gas, ρr, Θr)
    q0 = E2.project2d(model, (x, z) -> begin
        ρ = Θr(z) / (θr(z) + Δθ * sin(π * z / H) / (1 + ((x - xc) / a)^2))
        (ρ, ρ * U, 0.0, Θr(z))
    end)
    Δt = T / ceil(T / (nsub * fast_step(model, q0, fast; cfl = cfl)))
    @printf("channel %.0f km × %.0f km, a = %.0f km, p = %d, %d × %d elements (node spacing ≈ %.1f km × %.0f m), unknowns = %d\n",
            L / 1e3, H / 1e3, a / 1e3, P, NEX, NEZ, L / NEX / P / 1e3, H / NEZ / P, model.nq)
    q, t = run_with_output(model, q0; name = name, Δt = Δt, T = T, nsub = nsub, table = table, fast = fast, nout = nout)

    # section at the node row nearest z = H/2
    θp = Dict(E2.output_fields(model, q))["theta_prime"]
    Zc = mesh.Z[1, :, 1:NEX:end][:]
    jz = argmin(abs.(Zc .- H / 2))
    ez, j = divrem(jz - 1, P + 1) .+ (1, 1)
    xs = Float64[]; ys = Float64[]
    for ex in 1:NEX, i in 1:P+1
        (i == 1 && ex > 1) && continue
        e = (ez - 1) * NEX + ex
        push!(xs, mesh.X[i, j, e]); push!(ys, θp[i, j, e])
    end
    xct = mod(xc + U * T, L)
    kl = argmax([x < xct ? y : -Inf for (x, y) in zip(xs, ys)])
    kr = argmax([x > xct ? y : -Inf for (x, y) in zip(xs, ys)])
    sym = maximum(abs(ys[k] - ys[argmin(abs.(xs .- (2xct - xs[k])))]) for k in eachindex(xs)
                  if abs(xs[k] - xct) < min(xct, L - xct))
    @printf("section z = %.0f m: maxima θ' = %.3e K at x = %.1f km and %.3e K at x = %.1f km (centre %.1f km)\n",
            Zc[jz], ys[kl], xs[kl] / 1e3, ys[kr], xs[kr] / 1e3, xct / 1e3)
    @printf("  max |θ'(xc + d) - θ'(xc - d)| = %.1e K\n", sym)
    open("$(name)_section.csv", "w") do f
        println(f, "x,theta_prime")
        foreach(k -> println(f, xs[k], ",", ys[k]), eachindex(xs))
    end
    return model, q, t
end
