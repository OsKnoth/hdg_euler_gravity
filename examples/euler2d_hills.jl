# Flow over a bell-shaped hill after Gallus & Klemp (2000, Mon. Wea. Rev. 128, 1153–1164),
# section 3: h(x) = H/(1 + x²/a²) with H = 400 m, U = 10 m/s, N = 0.01 1/s, half-widths
# a = 1, 10, 100 km (paper: Δx = 0.2 a; here 10 elements of p = 4 over 40 a), model top ≈ 15.6 km
# with a Rayleigh damping layer above
# 9 km, domain 40 a, integration time 21.6 a/U. Terrain-following mesh (Gal-Chen), node spacing
# ≈ 250 m in the vertical (as Dudhia 1993), split-explicit MIS with HEVI fast part.
#
#     julia --project=examples examples/euler2d_hills.jl a1  [T] [fast]   # Na/U = 1   (nonhydrostatic)
#     julia --project=examples examples/euler2d_hills.jl a10 [T] [fast]   # Na/U = 10  (hydrostatic)
#     julia --project=examples examples/euler2d_hills.jl a100 [T] [fast]  # Na/U = 100 (hydrostatic)
# fast = hevi (default), verlet, fb or imex.
#
# Differences to the paper: the open lateral boundaries are replaced by a periodic channel of
# width 40 a (inflow/outflow boundaries to come), and there is no rotation (the paper's linear
# theory uses f = 1e-4, fa/U = 1 for a = 100 km). The strength of the upper Rayleigh layer is not
# given in the paper; it is chosen as 5 c_gz / (layer depth) with the vertical group velocity
# c_gz = U min(1, U/(N a)). NH/U = 0.4, so the comparison with linear theory (Fourier integral
# with radiation condition and amplitude factor sqrt(ρ(0)/ρ(z)), as in eq. (3) of the paper,
# without rotation) is approximate.
# Output: hills_<case>_<fast>.vtkhdf, .png, _w.csv, _w.png

include(joinpath(@__DIR__, "euler2d_common.jl"))

case = isempty(ARGS) ? "a10" : ARGS[1]
a = Dict("a1" => 1e3, "a10" => 10e3, "a100" => 100e3)[case]

# ------------------------------------------------------------------ parameters
const U, NBV, θ0 = 10.0, 0.01, 300.0
const h0 = parse(Float64, get(ENV, "H0", "400.0"))   # hill height (paper: 400 m); H0=1 for the linear regime
const P = 4
const L = 40a                          # periodic, no lateral sponge
# horizontal elements: (p+1) NEX = 50 points over 40 a suffice for p = 4 (element 4 a); the
# paper's Δx = 0.2 a corresponds to NEX = 50. Override with the environment variable NEX.
const NEX = parse(Int, get(ENV, "NEX", "10"))
const H, NEZ = 15.6e3, 16              # model top 100 hPa, node spacing ≈ 245 m
const ZD = 9e3                         # Rayleigh damping above 9 km
const NSUB = 12                        # fast substeps per large step
const T = length(ARGS) > 1 && ARGS[2] != "-" ? parse(Float64, ARGS[2]) : 21.6a / U
const FAST = length(ARGS) > 2 ? Symbol(ARGS[3]) : :hevi
const ZCMP = 3e3                       # comparison level for w
# crest at the element centre nearest L/2 (CREST=face: at L/2, an element face for even NEX)
xm = get(ENV, "CREST", "center") == "center" ? (floor(NEX / 2) + 0.5) * L / NEX : L / 2
cgz = U * min(1.0, U / (NBV * a))
αsponge = parse(Float64, get(ENV, "SPONGE", string(5cgz / (H - ZD))))   # env SPONGE: rate [1/s]

gas = Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = θ0, N = NBV)
mesh = E2.Mesh2D(NEX, NEZ, P, L, H; zs = x -> h0 / (1 + ((x - xm) / a)^2))
# nonlinear fluxes: FLUX=tec|kg, GRAV=waruszewski|souza, CLMARS=c (constant LMARS sound speed, 0: local)
# (FLUX=kg GRAV=souza CLMARS=340 is the CGDycore setting)
const FLUX = Symbol(get(ENV, "FLUX", "tec"))
const GRAV = Symbol(get(ENV, "GRAV", "waruszewski"))
const CLMARS = parse(Float64, get(ENV, "CLMARS", "0"))
model0 = E2.Euler2DModel(mesh, gas, ρr, Θr; volume_flux = FLUX, gravity = GRAV, c_lmars = CLMARS)
q0 = E2.project2d(model0, (x, z) -> (ρr(z), ρr(z) * U, 0.0, Θr(z)))
model = E2.with_sponge(model0, q0; zD = ZD, αtop = αsponge)
Δt = T / ceil(T / (NSUB * fast_step(model, q0, FAST)))
@printf("hill %s: a = %.0f km, H = %.0f m, U = %.0f m/s, N = %.3f 1/s, Na/U = %.0f, NH/U = %.1f\n",
        case, a / 1e3, h0, U, NBV, NBV * a / U, NBV * h0 / U)
@printf("volume flux %s, gravity %s, LMARS sound speed %s\n", FLUX, GRAV, CLMARS > 0 ? string(CLMARS) : "local")
@printf("domain %.0f km × %.1f km, p = %d, %d × %d elements, Δx(node) = %.0f m, unknowns = %d, T = %.0f s, sponge α = %.1e 1/s\n",
        L / 1e3, H / 1e3, P, NEX, NEZ, L / (NEX * P), model.nq, T, αsponge)

q, _ = run_with_output(model, q0; name = "hills_$(case)_$(FAST)_nex$NEX", Δt = Δt, T = T, nsub = NSUB, table = :MIS4, fast = FAST,
                       nout = 10, refreeze_every = 10, plot_names = ["w", "theta_prime"],
                       zoom = ((xm - 10a, xm + 10a), (0.0, 10e3)))

# ------------------------------------------------------------------ linear theory for w
"""w(x, z) of linear steady flow over the Agnesi hill: w = Re ∫₀^∞ i k U h0 a e^{-ka} e^{ik(x-xm)} E(k, z) dk,
E = exp(i m z), m = sqrt(l² - k²) for k < l, exp(-sqrt(k² - l²) z) otherwise, l = N/U."""
function w_linear(x, z; nk = 40000)
    l = NBV / U
    kmax = 40 / a + 2l
    k = range(0, kmax; length = nk); dk = step(k)
    s = 0.0im
    for kk in k
        E = kk < l ? exp(im * sqrt(l^2 - kk^2) * z) : exp(-sqrt(kk^2 - l^2) * z)
        s += im * kk * U * h0 * a * exp(-kk * a) * exp(im * kk * (x - xm)) * E
    end
    return real(s) * dk * sqrt(ρr(0.0) / ρr(z))
end
# node row nearest z = ZCMP above the hill (terrain-following: take the nodes with ζ nearest)
W = Dict(E2.output_fields(model, q))["w"]
ζ = (mesh.Z .- (h0 ./ (1 .+ ((mesh.X .- xm) ./ a) .^ 2))) .* H ./ (H .- h0 ./ (1 .+ ((mesh.X .- xm) ./ a) .^ 2))
ζt = ζ[argmin(abs.(ζ .- ZCMP))]                 # node level (in ζ) nearest ZCMP
xs = Float64[]; zs_ = Float64[]; wm = Float64[]
for k in CartesianIndices(W)
    if abs(ζ[k] - ζt) < 1e-6 * H && abs(mesh.X[k] - xm) < 15a
        push!(xs, mesh.X[k]); push!(zs_, mesh.Z[k]); push!(wm, W[k])
    end
end
o = sortperm(xs); xs, zs_, wm = xs[o], zs_[o], wm[o]
wl = [w_linear(x, z) for (x, z) in zip(xs, zs_)]
jump = maximum((abs(wm[k+1] - wm[k]) for k in 1:length(xs)-1 if abs(xs[k+1] - xs[k]) < 1e-6 * L); init = 0.0)
@printf("w at z ≈ %.0f m, |x - xm| < 15 a: max|w_model| = %.3e, max|w_linear| = %.3e, rel. L2 difference = %.2f, max jump at element faces = %.2e\n",
        ZCMP, maximum(abs, wm), maximum(abs, wl), sqrt(sum(abs2, wm .- wl) / sum(abs2, wl)), jump)
open("hills_$(case)_$(FAST)_nex$(NEX)_w.csv", "w") do f
    println(f, "x,z,w_model,w_linear")
    foreach(k -> println(f, xs[k], ",", zs_[k], ",", wm[k], ",", wl[k]), eachindex(xs))
end
fig = Figure(size = (900, 350))
ax = Axis(fig[1, 1]; xlabel = "x - xm [km]", ylabel = "w [m/s]", title = "hill $case, z ≈ $(round(Int, ZCMP)) m")
lines!(ax, (xs .- xm) ./ 1e3, wl; label = "linear theory", color = :black)
scatter!(ax, (xs .- xm) ./ 1e3, wm; label = "model", markersize = 5, color = :tomato)
axislegend(ax)
save("hills_$(case)_$(FAST)_nex$(NEX)_w.png", fig)
println("wrote hills_$(case)_$(FAST)_nex$(NEX)_w.csv, hills_$(case)_$(FAST)_nex$(NEX)_w.png")
