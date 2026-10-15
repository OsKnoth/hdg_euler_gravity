# Long-channel (hydrostatic-regime) gravity waves: the Skamarock & Klemp (1994) setup scaled to
# a 6000 km channel, a = 100 km, xc = L/3, U = 20 m/s, T = 60000 s (here without rotation).
# The grid aspect ratio (20 km × 625 m node spacing) makes explicit acoustics in the vertical
# very expensive; this is the regime in which HEVI pays off.
#
#     julia --project=examples examples/euler2d_gravity_wave_long.jl            # HEVI
#     julia --project=examples examples/euler2d_gravity_wave_long.jl verlet     # explicit Störmer–Verlet
#     julia --project=examples examples/euler2d_gravity_wave_long.jl imex       # explicit IMEX (CFL 0.2)
#
# Output: gravity_wave_long_<fast>.vtkhdf, .png, _section.csv

include(joinpath(@__DIR__, "euler2d_common.jl"))

fast = isempty(ARGS) ? :hevi : Symbol(ARGS[1])
gravity_wave(; name = "gravity_wave_long_$fast", L = 6000e3, a = 100e3, T = 60000.0,
             P = 4, NEX = 75, NEZ = 4,          # node spacing ~ 20 km × 625 m
             nsub = 12, table = :MIS4, fast = fast)
