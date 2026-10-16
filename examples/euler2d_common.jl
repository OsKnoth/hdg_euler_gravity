# Shared helpers for the 2D examples, for both discretizations (disc = :dg or :fv):
# model construction, split-explicit run with VTKHDF time series (ParaView) and Makie PNG, and
# the Skamarock–Klemp gravity-wave setup used by the short and the long channel.
#
# Loaded by euler2d_gravity_wave.jl, euler2d_gravity_wave_long.jl and euler2d_hills.jl.

using HDGEulerGravity
using HDGEulerGravity.Common
using HDF5, CairoMakie            # activate the VTKHDF and Makie package extensions
using Printf
const TS = HDGEulerGravity.TimeStepping
const E2 = HDGEulerGravity.Euler2D
const FV = HDGEulerGravity.FV2D

"""Default split-explicit settings per discretization: (table, nsub, fast)."""
default_scheme(disc) = disc == :fv ? (:WSRK3, 6, :wrf) : (:MIS4, 12, :hevi)

"""
    model = build_model(disc; L, H, P=4, NEX, NEZ, ρr, Θr, gas=Gas(), zs=x->0)

DG: NEX × NEZ elements of degree P (terrain zs). FV: (P NEX) × (P NEZ) cells, i.e. the same
number of unknowns per direction as the DG nodes (flat only).
"""
function build_model(disc; L, H, P = 4, NEX, NEZ, ρr, Θr, gas = Gas(), zs = x -> 0.0)
    if disc == :dg
        mesh = E2.Mesh2D(NEX, NEZ, P, L, H; zs = zs)
        return E2.Euler2DModel(mesh, gas, ρr, Θr)
    elseif disc == :fv
        zs(L / 2) == 0 || error("FV2D is flat (no terrain yet)")
        return FV.FVModel(FV.FVGrid(P * NEX, P * NEZ, L, H), gas, ρr, Θr)
    end
    error("unknown discretization $disc (:dg or :fv)")
end

grid_info(m::E2.Euler2DModel) = @sprintf("DG p = %d, %d × %d elements (node spacing ≈ %.0f m × %.0f m), unknowns = %d",
                                         m.mesh.p, m.mesh.Nex, m.mesh.Nez, m.mesh.Lx / (m.mesh.Nex * m.mesh.p),
                                         m.mesh.H / (m.mesh.Nez * m.mesh.p), m.nq)
grid_info(m::FV.FVModel) = @sprintf("FV C-grid %d × %d cells (Δx = %.0f m, Δz = %.0f m), unknowns = %d",
                                    m.grid.nx, m.grid.nz, m.grid.dx, m.grid.dz, m.nq)

"""
    run_with_output(model, q0; name, Δt, T, nsub, table, fast, nout=10, refreeze_every=1,
                    plot_names=["theta_prime", "w"], zoom=nothing)

Split-explicit run over [0, T] with `nsub` fast substeps per large step; writes `name.vtkhdf`
(initial state and `nout` further outputs) and `name.png` (final state). Returns (q, wall time).
"""
function run_with_output(model, q0; name, Δt, T, nsub, table, fast, nout = 10,
                         refreeze_every = 1, plot_names = ["theta_prime", "w"], zoom = nothing)
    nsteps = round(Int, T / Δt)
    every = max(1, nsteps ÷ nout)
    @printf("%s: %s, Δt = %.3f s (%d steps), fast = %s, nsub = %d (dτ ≈ %.3f s), re-freeze every %d\n",
            name, table, Δt, nsteps, fast, nsub, Δt / nsub, refreeze_every)
    vtk = open_vtkhdf("$name.vtkhdf", model)
    write_vtkhdf!(vtk, model, q0, 0.0)
    mass0 = total2d(model, q0, 1)
    monitor = (n, q) -> begin
        F = Dict(output_fields(model, q))
        @printf("  t = %9.0f s  θ' ∈ [%.3e, %.3e]  max|w| = %.3e  Δmass/mass = %.1e\n", n * Δt,
                extrema(F["theta_prime"])..., maximum(abs, F["w"]), (total2d(model, q, 1) - mass0) / mass0)
        write_vtkhdf!(vtk, model, q, n * Δt)
    end
    t = @elapsed q = TS.run_mis(model, q0, Δt, nsteps; tab = TS.mis_tableau(table), nsub = nsub, fast = fast,
                                refreeze_every = refreeze_every, monitor = monitor, every = every)
    close(vtk)
    fig = plot_fields(model, q; names = plot_names, zoom = zoom,
                      title = @sprintf("%s, t = %.0f s", name, nsteps * Δt))
    save("$name.png", fig)
    @printf("wall time %.1f s; wrote %s.vtkhdf (%d outputs), %s.png\n", t, name, nsteps ÷ every + 1, name)
    return q, t
end

"""
DG: stable CFL limits of one fast step (spectral radius ≤ 1 at the reference state, CFL defined
by dτ = CFL h / ((2p+1) c), h = element size; HEVI: horizontal element size, aspect 3):
    p      imex    verlet  fb      hevi
    2      1.07    1.05    0.77    0.98
    3      0.24    0.79    0.56    0.28
    4      0.088   0.63    0.44    0.075
    5      0.050   0.52    0.36    0.060
The explicit part of the IMEX/HEVI tableau is RK2, unstable on the imaginary axis; only the LMARS
dissipation stabilizes it.
FV (C-grid, no dissipation in the fast operator): verlet, fb: c dτ/min(Δx, Δz) ≤ 0.92;
wrf: c dτ/Δx ≤ 1.0 (vertically implicit); imex and hevi are unstable for every dτ.
"""
const CFL_LIMITS = Dict(2 => (1.07, 1.05, 0.77, 0.98), 3 => (0.24, 0.79, 0.56, 0.28),
                        4 => (0.088, 0.63, 0.44, 0.075), 5 => (0.050, 0.52, 0.36, 0.060))

"""Acoustic substep of the fast integrator (safety × stability limit, table above)."""
function fast_step(model::E2.Euler2DModel, q, fast; cfl = nothing, safety = 0.85)
    M = model.mesh
    if cfl === nothing
        lim = get(CFL_LIMITS, M.p, (0.05, 3.15 / (M.p + 1), 2.2 / (M.p + 1), 0.05))
        cfl = safety * (fast == :verlet ? lim[2] : fast == :fb ? lim[3] : fast == :hevi ? lim[4] : lim[1])
    end
    hz = minimum(M.Z[:, end, :] .- M.Z[:, 1, :])
    h = fast == :hevi ? M.Lx / M.Nex : min(M.Lx / M.Nex, hz)
    return cfl * h / ((2M.p + 1) * sound_speed_max(model, q))
end

function fast_step(model::FV.FVModel, q, fast; cfl = nothing, safety = 0.85)
    g = model.grid
    fast in (:wrf, :verlet, :fb) || error("FV: use :wrf, :verlet or :fb (:imex and :hevi are unstable on the C-grid)")
    lim = fast == :wrf ? 1.0 : 0.92
    cfl === nothing && (cfl = safety * lim)
    h = fast == :wrf ? g.dx : min(g.dx, g.dz)
    return cfl * h / sound_speed_max(model, q)
end

"""
    gravity_wave(; disc=:dg, name, L, H=10e3, a, xc=L/3, U=20, Δθ=0.01, N=0.01, θ0=300, T, P=4,
                 NEX, NEZ, nsub, table, fast, cfl=nothing, nout=10)

Skamarock & Klemp (1994) gravity waves in a periodic channel [0, L] × [0, H] with walls:
θ' = Δθ sin(πz/H)/(1 + ((x - xc)/a)²) at constant pressure on a constant-N state with mean flow U.
The large step is Δt = nsub dτ (acoustic substep dτ, rounded so that T/Δt is an integer).
Prints the two maxima of θ' at the row nearest z = H/2 and the asymmetry about xc + U T.
"""
function gravity_wave(; disc = :dg, name, L, H = 10e3, a, xc = L / 3, U = 20.0, Δθ = 0.01, N = 0.01, θ0 = 300.0, T,
                      P = 4, NEX, NEZ, nsub = nothing, table = nothing, fast = nothing, cfl = nothing, nout = 10)
    dt_, dn_, df_ = default_scheme(disc)
    table = something(table, dt_)
    nsub = something(nsub, dn_)
    fast = something(fast, df_)
    gas = Gas()
    ρr, Θr, θr = stratified_reference(gas; θ0 = θ0, N = N)
    model = build_model(disc; L = L, H = H, P = P, NEX = NEX, NEZ = NEZ, ρr = ρr, Θr = Θr, gas = gas)
    q0 = project2d(model, (x, z) -> begin
        ρ = Θr(z) / (θr(z) + Δθ * sin(π * z / H) / (1 + ((x - xc) / a)^2))
        (ρ, ρ * U, 0.0, Θr(z))
    end)
    Δt = T / ceil(T / (nsub * fast_step(model, q0, fast; cfl = cfl)))
    @printf("channel %.0f km × %.0f km, a = %.0f km, %s\n", L / 1e3, H / 1e3, a / 1e3, grid_info(model))
    q, t = run_with_output(model, q0; name = name, Δt = Δt, T = T, nsub = nsub, table = table, fast = fast, nout = nout)

    xs, ys, zl = field_section(model, q, "theta_prime", H / 2)
    xct = mod(xc + U * T, L)
    kl = argmax([x < xct ? y : -Inf for (x, y) in zip(xs, ys)])
    kr = argmax([x > xct ? y : -Inf for (x, y) in zip(xs, ys)])
    sym = maximum(abs(ys[k] - ys[argmin(abs.(xs .- (2xct - xs[k])))]) for k in eachindex(xs)
                  if abs(xs[k] - xct) < min(xct, L - xct))
    @printf("section z = %.0f m: maxima θ' = %.3e K at x = %.1f km and %.3e K at x = %.1f km (centre %.1f km)\n",
            zl, ys[kl], xs[kl] / 1e3, ys[kr], xs[kr] / 1e3, xct / 1e3)
    @printf("  max |θ'(xc + d) - θ'(xc - d)| = %.1e K\n", sym)
    open("$(name)_section.csv", "w") do f
        println(f, "x,theta_prime")
        foreach(k -> println(f, xs[k], ",", ys[k]), eachindex(xs))
    end
    return model, q, t
end
