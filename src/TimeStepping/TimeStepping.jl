"""
    TimeStepping

Time integrators shared by all spatial discretizations (DGSEM, FV):

  * split-explicit MIS / Wicker–Skamarock RK3 (`run_mis`, `mis_step!`) for q' = N(q) with
    the fast part L_n (frozen at qⁿ) and the slow part S = N - L_n,
  * fast integrators for q' = L_n q + f (`fast_solve!`): :imex, :hevi, :verlet, :fb, :wrf,
  * classical RK4 on N (`run_rk4`) as reference.

A discretization provides a model and a fast operator (subtype of `AbstractFastOperator`) and
implements the interface below (methods of the functions declared in `interface.jl`).
"""
module TimeStepping

export AbstractFastOperator,
       rhs_nonlinear, rhs_nonlinear!, fast_operator, fast_apply!, fast_workspace, fast_forcing, fast_mass,
       fast_rhs_thermo!, fast_rhs_momentum!, interface_update!,
       fast_rhs_horizontal!, vertical_apply!, vertical_factor, vertical_solve!,
       fast_rhs_hmomentum!, fast_rhs_hthermo!,
       fast_solve!, IMEX_GM, IMEX_DL, IMEX_AE, IMEX_AI, WRF_BETA,
       MISTableau, mis_tableau, nstages, MISWork, stage_substeps, mis_step!, mis_step, run_mis, run_rk4

include("interface.jl")
include("fast.jl")
include("tableaus.jl")
include("mis.jl")

end # module
