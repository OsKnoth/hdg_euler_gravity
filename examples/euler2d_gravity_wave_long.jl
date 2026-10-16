# Long-channel (hydrostatic-regime) gravity waves: the Skamarock & Klemp (1994) setup scaled to
# a 6000 km channel, a = 100 km, xc = L/3, U = 20 m/s, T = 60000 s (here without rotation).
# The grid aspect ratio (20 km × 625 m node spacing) makes explicit acoustics in the vertical
# very expensive; this is the regime in which HEVI pays off.
#
#     julia --project=examples examples/euler2d_gravity_wave_long.jl [dg|fv] [fast]
# fast: dg: hevi (default), verlet, fb, imex;  fv: wrf (default), verlet, fb
#
# Output: gravity_wave_long_<disc>.vtkhdf, .png, _section.csv

include(joinpath(@__DIR__, "euler2d_common.jl"))

disc = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :dg                # dg or fv
fast = length(ARGS) >= 2 ? Symbol(ARGS[2]) : nothing            # default: hevi (dg), wrf (fv)
gravity_wave(; disc = disc, name = "gravity_wave_long_$(disc)", L = 6000e3, a = 100e3, T = 60000.0,
             P = 4, NEX = 75, NEZ = 4,          # DG nodes / FV cells: 300 × 16 (20 km × 625 m)
             fast = fast)
