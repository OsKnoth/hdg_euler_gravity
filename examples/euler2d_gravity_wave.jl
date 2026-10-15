# Inertia-gravity waves (Skamarock & Klemp 1994, non-rotating, nonhydrostatic regime):
# channel 300 km × 10 km, a = 5 km, xc = 100 km, U = 20 m/s, Δθ = 0.01 K, N = 0.01 1/s, T = 3000 s.
# The wave packet is centred at x = xc + U T = 160 km and symmetric about it.
#
#     julia --project=examples examples/euler2d_gravity_wave.jl [hevi|verlet|fb|imex]
#
# Output: gravity_wave_<fast>.vtkhdf (ParaView ≥ 5.13, 11 time steps), gravity_wave_<fast>.png,
#         gravity_wave_<fast>_section.csv (θ' at z ≈ 5 km).

include(joinpath(@__DIR__, "euler2d_common.jl"))

fast = isempty(ARGS) ? :hevi : Symbol(ARGS[1])          # hevi, verlet, fb, imex
gravity_wave(; name = "gravity_wave_$fast", L = 300e3, a = 5e3, T = 3000.0,
             P = 4, NEX = 30, NEZ = 4,          # node spacing ~ 2.5 km × 625 m
             nsub = 12, table = :MIS4, fast = fast)
