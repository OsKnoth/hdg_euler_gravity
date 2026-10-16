"""
    HDGRelaxEuler

1D hybridized DG (HDG) for the Euler equations with gravity, linearized about a
hydrostatic background. The HDG trace ("additional") conditions are treated by
RELAXATION, so the coupled (element + trace) system is integrated in time WITHOUT
eliminating the element unknowns (no static condensation / Schur complement).

Model, q = (ρ', m = ρ₀u', p'):

    q_t + (A(z) q)_z = S q
    A(z) q = (m, p', c₀²(z) m),   S q = (0, -g ρ', -(γ-1) g m)

HDG flux on element K:  F̂·n = A(z_f) q̂ n + T (q - q̂)

Interface flux (keyword `flux`):
  :lmars (default)  T = |A(z_f)| = A²/c_f = [0 0 1/c; 0 c 0; 0 0 c].  With the trace
                    condition q̂ = mean(q) and the face flux is exactly LMARS
                    (Chen et al. 2013) linearized about u₀ = 0:
                      m* = {m} - [p']/(2c),  p* = {p'} - c[m]/2,  F* = (m*, p*, c² m*)
  :llf              T = c_K diag(τ_ρ, 1, 1)  (element Rusanov-type penalty)

Trace conditions:
  interior face: τ_L (q_L - q̂) + τ_R (q_R - q̂) = 0   (conservativity)
  wall face    : q̂ = P q_int,  P = diag(1, 0, 1)      (m̂ = 0)
Relaxed:  ε dq̂/dt = H q - q̂.

Semi-discrete system, U = (q, q̂):
    [M 0; 0 εI] U_t = [K G; H -I] U

Time integrators (both act on the coupled system):
  :dirk  stiffly accurate L-stable SDIRK4 (Hairer–Wanner), one sparse LU of the full
         block matrix, valid for any ε ≥ 0 (ε = 0 → index-1 DAE = pure HDG)
  :imex  IMEX ARS(2,2,2): DG part explicit, relaxation implicit and local per face

Only the standard libraries LinearAlgebra, SparseArrays and Printf are used.
1-based indexing: variable r = 1 (ρ'), 2 (m), 3 (p').
"""
module HDGRelaxEuler

using LinearAlgebra, SparseArrays, Printf

export Atmosphere, c2, temperature, p0, rho0,
       HDGRelax, gauss_legendre, legendre_basis, iq, it,
       project, evaluate, l2_error, total_mass, consistent_trace, trace_residual,
       initial_state, run!

const NV = 3

# ---------------------------------------------------------------------------
# Background atmosphere
# ---------------------------------------------------------------------------
"""Hydrostatic background with constant lapse rate Γ (Γ = 0 → isothermal)."""
Base.@kwdef struct Atmosphere
    T0::Float64  = 288.0
    Γ::Float64   = 0.0
    R::Float64   = 287.0
    γ::Float64   = 1.4
    g::Float64   = 9.81
    p00::Float64 = 1.0e5
end

temperature(a::Atmosphere, z) = a.T0 - a.Γ * z
c2(a::Atmosphere, z) = a.γ * a.R * temperature(a, z)
p0(a::Atmosphere, z) = a.Γ == 0.0 ? a.p00 * exp(-a.g * z / (a.R * a.T0)) :
                                    a.p00 * (temperature(a, z) / a.T0)^(a.g / (a.R * a.Γ))
rho0(a::Atmosphere, z) = p0(a, z) / (a.R * temperature(a, z))

# ---------------------------------------------------------------------------
# Quadrature and Legendre basis
# ---------------------------------------------------------------------------
"""Gauss–Legendre nodes and weights on [-1,1] (Golub–Welsch)."""
function gauss_legendre(n::Int)
    n == 1 && return [0.0], [2.0]
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

"""Legendre P_{i-1} and its derivative at points ξ → (length(ξ) × np) matrices."""
function legendre_basis(ξ::AbstractVector, np::Int)
    nx = length(ξ)
    V = zeros(nx, np); D = zeros(nx, np)
    for (k, x) in enumerate(ξ)
        P = zeros(np); dP = zeros(np)
        P[1] = 1.0
        if np > 1
            P[2] = x; dP[2] = 1.0
        end
        for n in 2:np-1                       # degree n stored at index n+1
            P[n+1]  = ((2n - 1) * x * P[n] - (n - 1) * P[n-1]) / n
            dP[n+1] = dP[n-1] + (2n - 1) * P[n]
        end
        V[k, :] = P; D[k, :] = dP
    end
    return V, D
end

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
struct HDGRelax
    Ne::Int; p::Int; np::Int
    zmin::Float64; zmax::Float64; h::Float64; xe::Vector{Float64}
    atm::Atmosphere; bc::Symbol; flux::Symbol; eps::Float64
    Nf::Int; nq::Int; nt::Int
    tw::Vector{Float64}             # per-variable penalty weights (τ_ρ, 1, 1), :llf only
    tau::Vector{Float64}            # element penalty τ_K = c_K
    Mdiag::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    G::SparseMatrixCSC{Float64,Int}
    H::SparseMatrixCSC{Float64,Int}
    L::SparseMatrixCSC{Float64,Int}
    Mbig::SparseMatrixCSC{Float64,Int}
end

# element dof (e, r, i) and trace dof (f, r)
iq(np::Int, e, r, i) = ((e - 1) * NV + (r - 1)) * np + i
iq(s::HDGRelax, e, r, i) = iq(s.np, e, r, i)
it(f, r) = (f - 1) * NV + r
coeffs(s::HDGRelax, q, e, r) = view(q, iq(s, e, r, 1):iq(s, e, r, s.np))

"""
    HDGRelax(Ne, p, zmin, zmax, atm; bc=:wall, flux=:lmars, eps=0.0,
             tau_scale=1.0, tau_rho=0.0)

bc = :wall (reflecting at both ends) or :periodic; flux = :lmars or :llf.
"""
function HDGRelax(Ne::Int, p::Int, zmin::Real, zmax::Real, atm::Atmosphere;
                  bc::Symbol = :wall, flux::Symbol = :lmars, eps::Real = 0.0,
                  tau_scale::Real = 1.0, tau_rho::Real = 0.0)
    flux in (:lmars, :llf) || error("flux must be :lmars or :llf")
    np = p + 1
    h  = (zmax - zmin) / Ne
    xe = collect(zmin .+ h .* (0:Ne))
    Nf = bc == :wall ? Ne + 1 : Ne
    nq = Ne * NV * np
    nt = Nf * NV
    tw = [Float64(tau_rho), 1.0, 1.0]
    g, γ = atm.g, atm.γ

    xq, wq = gauss_legendre(p + 3)
    V, D = legendre_basis(xq, np)
    phiR = ones(np)
    phiL = [(-1.0)^(i - 1) for i in 1:np]
    Mref = [2.0 / (2(i - 1) + 1) for i in 1:np]       # ∫ P_i P_i  (diagonal)
    Sder = D' * (wq .* V)                               # ∫ P_i' P_j (Jacobians cancel)
    Mdiag = repeat(Mref .* (h / 2), Ne * NV)
    tau = zeros(Ne)

    KI, KJ, KV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    HI, HJ, HV = Int[], Int[], Float64[]
    addK!(i, j, v) = v == 0 || (push!(KI, i); push!(KJ, j); push!(KV, v))
    addG!(i, j, v) = v == 0 || (push!(GI, i); push!(GJ, j); push!(GV, v))
    addH!(i, j, v) = v == 0 || (push!(HI, i); push!(HJ, j); push!(HV, v))

    Aface(z) = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 c2(atm, z) 0.0]
    face_R(e) = bc == :wall ? e + 1 : mod1(e + 1, Ne)
    # 3×3 penalty at a face of an element with element penalty τ
    function Tpen(z, τ)
        if flux == :lmars
            cf = sqrt(c2(atm, z))
            return tau_scale .* [0.0 0.0 1/cf; 0.0 cf 0.0; 0.0 0.0 cf]
        end
        return τ .* Matrix(Diagonal(tw))
    end

    for e in 1:Ne
        zl, zr = xe[e], xe[e+1]
        zq = 0.5 * (zl + zr) .+ 0.5 * h .* xq
        c2q = [c2(atm, z) for z in zq]
        Sc2 = D' * ((wq .* c2q) .* V)                   # ∫ P_i' c₀² P_j
        τ = tau_scale * sqrt(max(c2(atm, zl), c2(atm, zr)))
        tau[e] = τ
        TL, TR = Tpen(zl, τ), Tpen(zr, τ)
        for i in 1:np, j in 1:np
            # volume flux terms (v_z, A q)
            addK!(iq(np, e, 1, i), iq(np, e, 2, j), Sder[i, j])
            addK!(iq(np, e, 2, i), iq(np, e, 3, j), Sder[i, j])
            addK!(iq(np, e, 3, i), iq(np, e, 2, j), Sc2[i, j])
            # gravity sources (v, S q)
            if i == j
                addK!(iq(np, e, 2, i), iq(np, e, 1, j), -g * h / 2 * Mref[i])
                addK!(iq(np, e, 3, i), iq(np, e, 2, j), -(γ - 1) * g * h / 2 * Mref[i])
            end
            # T (q - q̂), element part
            for r in 1:NV, s in 1:NV
                addK!(iq(np, e, r, i), iq(np, e, s, j),
                      -(TR[r, s] * phiR[i] * phiR[j] + TL[r, s] * phiL[i] * phiL[j]))
            end
        end
        fl, fr = e, face_R(e)
        AL, AR = Aface(zl), Aface(zr)
        for i in 1:np, r in 1:NV, s in 1:NV
            # right face, n = +1:  -φ_i(1) [A q̂ - T q̂]
            addG!(iq(np, e, r, i), it(fr, s), -phiR[i] * (AR[r, s] - TR[r, s]))
            # left face, n = -1:   -φ_i(-1) [-A q̂ - T q̂]
            addG!(iq(np, e, r, i), it(fl, s), -phiL[i] * (-AL[r, s] - TL[r, s]))
        end
    end

    # trace (additional) conditions: q̂_target = H q
    for f in 1:Nf
        if bc == :wall && (f == 1 || f == Ne + 1)
            e, phi = f == 1 ? (1, phiL) : (Ne, phiR)
            for s in (1, 3), j in 1:np                  # P = diag(1,0,1) → m̂ = 0
                addH!(it(f, s), iq(np, e, s, j), phi[j])
            end
            continue
        end
        eL, eR = mod1(f - 1, Ne), f
        # :lmars → same T on both sides → plain mean; :llf → τ-weighted mean
        tL, tR = flux == :lmars ? (1.0, 1.0) : (tau[eL], tau[eR])
        for s in 1:NV, j in 1:np
            addH!(it(f, s), iq(np, eL, s, j), tL / (tL + tR) * phiR[j])
            addH!(it(f, s), iq(np, eR, s, j), tR / (tL + tR) * phiL[j])
        end
    end

    K = sparse(KI, KJ, KV, nq, nq)
    G = sparse(GI, GJ, GV, nq, nt)
    H = sparse(HI, HJ, HV, nt, nq)
    L = [K G; H -sparse(1.0I, nt, nt)]
    Mbig = spdiagm(0 => vcat(Mdiag, fill(Float64(eps), nt)))
    return HDGRelax(Ne, p, np, zmin, zmax, h, xe, atm, bc, flux, Float64(eps), Nf, nq, nt,
                    tw, tau, Mdiag, K, G, H, L, Mbig)
end

# ---------------------------------------------------------------------------
# Projection, evaluation, diagnostics
# ---------------------------------------------------------------------------
elem_points(s::HDGRelax, e, ξ) = 0.5 * (s.xe[e] + s.xe[e+1]) .+ 0.5 * s.h .* ξ

"""L2 projection; `fun(z::Vector)` must return a 3 × length(z) matrix."""
function project(s::HDGRelax, fun; nquad::Int = s.p + 5)
    xq, wq = gauss_legendre(nquad)
    V, _ = legendre_basis(xq, s.np)
    Minv = [(2(i - 1) + 1) / 2 for i in 1:s.np]
    q = zeros(s.nq)
    for e in 1:s.Ne
        F = fun(elem_points(s, e, xq))
        for r in 1:NV
            coeffs(s, q, e, r) .= Minv .* (V' * (wq .* F[r, :]))
        end
    end
    return q
end

"""Point values: returns (z, Q) with Q a 3 × N matrix."""
function evaluate(s::HDGRelax, q; npts::Int = 8)
    ξ = collect(range(-1, 1; length = npts))
    V, _ = legendre_basis(ξ, s.np)
    Z = Float64[]; Q = zeros(NV, 0)
    for e in 1:s.Ne
        append!(Z, elem_points(s, e, ξ))
        Q = hcat(Q, reduce(vcat, [(V * coeffs(s, q, e, r))' for r in 1:NV]))
    end
    return Z, Q
end

"""Component-wise L2 errors against `fun(z) -> 3 × length(z)`."""
function l2_error(s::HDGRelax, q, fun)
    xq, wq = gauss_legendre(s.p + 6)
    V, _ = legendre_basis(xq, s.np)
    err = zeros(NV)
    for e in 1:s.Ne
        F = fun(elem_points(s, e, xq))
        for r in 1:NV
            d = V * coeffs(s, q, e, r) .- F[r, :]
            err[r] += s.h / 2 * sum(wq .* d .^ 2)
        end
    end
    return sqrt.(err)
end

total_mass(s::HDGRelax, q) = s.h * sum(q[iq(s, e, 1, 1)] for e in 1:s.Ne)
consistent_trace(s::HDGRelax, q) = s.H * q
initial_state(s::HDGRelax, q) = vcat(q, consistent_trace(s, q))
trace_residual(s::HDGRelax, U) = maximum(abs.(s.H * U[1:s.nq] .- U[s.nq+1:end]))

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
"""
    run!(s, U0, dt, nsteps; method=:dirk, monitor=nothing, every=1)

Returns the state after `nsteps` steps; `monitor(n, U)` is called every `every` steps.
"""
function run!(s::HDGRelax, U0, dt, nsteps; method::Symbol = :dirk,
              monitor = nothing, every::Int = 1)
    method == :dirk && return run_dirk(s, U0, dt, nsteps, monitor, every)
    method == :imex && return run_imex(s, U0, dt, nsteps, monitor, every)
    error("unknown method $method")
end

function run_dirk(s::HDGRelax, U0, dt, nsteps, monitor, every)
    # Hairer–Wanner SDIRK4, 5 stages, L-stable, stiffly accurate (U^{n+1} = Y_5)
    a = [1/4       0          0       0       0;
         1/2       1/4        0       0       0;
         17/50     (-1/25)    1/4     0       0;
         371/1360  (-137/2720) 15/544 1/4     0;
         25/24     (-49/48)   125/16  (-85/12) 1/4]
    F = lu(s.Mbig - (dt * a[1, 1]) * s.L)              # full coupled matrix, factored once
    U = copy(U0)
    LY = [zeros(length(U0)) for _ in 1:5]
    for n in 1:nsteps
        MU = s.Mbig * U
        Y = U
        for i in 1:5
            rhs = copy(MU)
            for j in 1:i-1
                rhs .+= (dt * a[i, j]) .* LY[j]
            end
            Y = F \ rhs
            LY[i] = s.L * Y
        end
        U = Y
        if monitor !== nothing && n % every == 0
            monitor(n, U)
        end
    end
    return U
end

function run_imex(s::HDGRelax, U0, dt, nsteps, monitor, every)
    # ARS(2,2,2). Explicit: element DG operator. Implicit: trace relaxation (local).
    gm = 1 - 1 / sqrt(2); dl = 1 - 1 / (2gm)
    AE = [0.0 0.0 0.0; gm 0.0 0.0; dl (1 - dl) 0.0]
    AI = [0.0 0.0 0.0; 0.0 gm 0.0; 0.0 (1 - gm) gm]
    nq, nt, ε = s.nq, s.nt, s.eps
    N = nq + nt
    fE(U) = vcat((s.K * U[1:nq] .+ s.G * U[nq+1:end]) ./ s.Mdiag, zeros(nt))
    fI(U) = ε > 0 ? vcat(zeros(nq), (s.H * U[1:nq] .- U[nq+1:end]) ./ ε) : zeros(N)
    U = copy(U0)
    for n in 1:nsteps
        FE = Vector{Vector{Float64}}()
        FI = Vector{Vector{Float64}}()
        Y = U
        for i in 1:3
            Ystar = copy(U)
            for j in 1:i-1
                Ystar .+= dt .* (AE[i, j] .* FE[j] .+ AI[i, j] .* FI[j])
            end
            # implicit solve Y = Ystar + dt*AI[i,i]*fI(Y): q-part explicit, trace part local
            Y = copy(Ystar)
            Hq = s.H * Y[1:nq]
            if ε == 0.0
                Y[nq+1:end] .= Hq                       # exact HDG constraint
            else
                k = dt * AI[i, i] / ε
                Y[nq+1:end] .= (Ystar[nq+1:end] .+ k .* Hq) ./ (1 + k)
            end
            push!(FE, fE(Y)); push!(FI, fI(Y))
        end
        U = Y                                           # stiffly accurate
        if monitor !== nothing && n % every == 0
            monitor(n, U)
        end
    end
    return U
end

end # module
