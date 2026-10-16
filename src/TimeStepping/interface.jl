# Interface between the time integrators and a spatial discretization
#
# Model (slow + fast system):
#   rhs_nonlinear!(out, model, q)           full right-hand side N(q), in place
#   fast_operator(model, q; fast, kwargs)   fast operator L_n frozen at q (subtype of AbstractFastOperator)
#
# Fast operator s, all tendencies as dq/dt (already divided by the mass matrix):
#   fast_apply!(out, s, q, f)               out = L q + f/M         (:imex, MIS slow part)
#   fast_forcing(s), fast_mass(s)           constant forcing f_b of L and mass M (F_slow -> f = f_b + M F_slow)
#   fast_workspace(s)                       7 work vectors (cached for allocation-free loops)
# thermodynamic / momentum split (:verlet, :fb):
#   fast_rhs_thermo!(out, s, q, f)          (ρ, Θ) tendency, momentum rows 0
#   fast_rhs_momentum!(out, s, q, f)        momentum tendency, (ρ, Θ) rows 0
#   interface_update!(s, q; relax, dt)      interface variables from q (default: nothing to do)
# horizontal / vertical split (:hevi, :wrf):
#   fast_rhs_horizontal!(out, s, q, f)      explicit horizontal part + f/M
#   vertical_apply!(out, s, q)              vertical part A_v q (no forcing)
#   vertical_factor(s, c)                   factorization of I - c A_v (cached by the operator)
#   vertical_solve!(Y, s, F, Ys)            Y = (I - c A_v)⁻¹ Ys, Y may alias Ys
# WRF forward-backward (:wrf), additionally:
#   fast_rhs_hmomentum!(out, s, q, f)       horizontal-momentum tendency (explicit, forward), other rows 0
#   fast_rhs_hthermo!(out, s, q, f)         all other rows: explicit horizontal terms + f/M, horizontal momentum rows 0

"""Supertype of the fast (acoustic-gravity) operators of all discretizations."""
abstract type AbstractFastOperator end

function rhs_nonlinear! end
"""dq/dt of the full nonlinear operator N(q) (allocating)."""
rhs_nonlinear(model, q::AbstractVector) = rhs_nonlinear!(zeros(length(q)), model, q)
function fast_operator end
function fast_apply! end
function fast_rhs_thermo! end
function fast_rhs_momentum! end
function fast_rhs_horizontal! end
function vertical_apply! end
function vertical_factor end
function vertical_solve! end
function fast_rhs_hmomentum! end
function fast_rhs_hthermo! end

"""Interface variables of the fast operator (e.g. the DG face momentum); default: none."""
interface_update!(s::AbstractFastOperator, q; relax::Bool = false, dt = 0.0) = s

fast_forcing(s::AbstractFastOperator) = s.fb
fast_mass(s::AbstractFastOperator) = s.Mdiag
fast_workspace(s::AbstractFastOperator) = s.work
