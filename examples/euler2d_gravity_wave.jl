# Inertia-gravity waves (Skamarock & Klemp 1994, non-rotating, nonhydrostatic regime):
# channel 300 km × 10 km, a = 5 km, xc = 100 km, U = 20 m/s, Δθ = 0.01 K, N = 0.01 1/s, T = 3000 s.
# The wave packet is centred at x = xc + U T = 160 km and symmetric about it.
#
#     julia --project=examples examples/euler2d_gravity_wave.jl [dg|fv] [fast]
#
# Output: gravity_wave_<disc>.vtkhdf (ParaView ≥ 5.13, 11 time steps), gravity_wave_<disc>.png,
#         gravity_wave_<disc>_section.csv (θ' at z ≈ 5 km).

include(joinpath(@__DIR__, "euler2d_common.jl"))

disc = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :dg                # dg or fv
fast = length(ARGS) >= 2 ? Symbol(ARGS[2]) : nothing            # default: hevi (dg), wrf (fv)
gravity_wave(; disc = disc, name = "gravity_wave_$(disc)", L = 300e3, a = 5e3, T = 3000.0,
             P = 4, NEX = 30, NEZ = 4,          # DG nodes / FV cells: 120 × 16 (2.5 km × 625 m)
             fast = fast)
