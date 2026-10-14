"""
    HDGRelaxTheta

1D hybridized DGSEM (nodal, Gauss–Lobatto collocation) with LMARS interface flux for the
LINEARIZED (fast) part of the compressible Euler equations in potential-temperature form,
with relaxed trace conditions integrated in time WITHOUT elimination.

Linearized system (frozen coefficients; momentum advection removed, θ frozen in the Θ
flux, p = (p/Θ) Θ with p/Θ frozen), q = (ρ, m = ρw, Θ = ρθ):

    ρ_t + m_z = 0,   m_t + (a Θ + b)_z = -g ρ,   Θ_t + (θ̄ m)_z = 0
    a = pfac p̄/Θ̄,  b = (1 - pfac) p̄        (pfac = 1: c² = R T̄;  pfac = γ: c² = γ R T̄)

Energy (p = aΘ, χ = Θ - θ̄ρ, c² = aθ̄, φ = θ̄ exp(∫ g/c² dz)):
    E = ½ ∫ φ [ m² + p²/c² + g χ²/(θ̄ θ̄_z) ] dz

DGSEM, form = :split (default, energy stable), per element with SBP matrix D, J = h/2:
    ρ_i' = -(D m)_i / J
    Θ_i' = -(θ̄_i (D m)_i + m_i (D θ̄)_i) / J                       (product form)
    m_i' = -[(D(φ a Θ))_i / φ_i - a_i Θ_i (D θ̄)_i / θ̄_i + (D b)_i] / J + g Θ_i/θ̄_i - g ρ_i
plus SAT / HDG coupling at the end nodes:  J w_b q_b' += (A_f n - T_f)(q_b - q̂_f),
    A_f = [0 1 0; 0 0 a; 0 θ̄ 0],  T_f = |A_f| = [0 0 a/c; 0 c 0; 0 0 c]   (LMARS).
form = :standard: plain collocation weak form (only O(h)-stable for variable θ̄).

Trace conditions: interior q̂ = mean(q), wall q̂ = diag(1,0,1) q_int; relaxed as
ε dq̂/dt = H q - q̂.  Semi-discrete system, U = (q, q̂):
    [M 0; 0 εI] U_t = [K G; H -I] U + [f_b + M F_slow; 0],   M = diag(J w_i)

Standard library only (LinearAlgebra, SparseArrays). Variables r = 1 (ρ), 2 (m), 3 (Θ);
node i = 1 is the bottom (ξ = -1), node p+1 the top (ξ = +1) of each element.
"""
module HDGRelaxTheta

using LinearAlgebra, SparseArrays

export Atmosphere, FrozenState, frozen_from_atmosphere, HDGRelaxTh,
       gauss_legendre, gll, lagrange_basis, iq, it, blk, nodes,
       hydrostatic_rho, balanced_state, forcing_vector, tendency, energy,
       project, evaluate, l2_error, total, initial_state, trace_residual, run!,
       atm_T, atm_p, atm_rho, atm_theta, atm_Theta, atm_pi, coef_a, coef_b

const NV = 3

# ---------------------------------------------------------------------------
# Background atmosphere and frozen coefficients
# ---------------------------------------------------------------------------
Base.@kwdef struct Atmosphere
    T0::Float64  = 288.0
    Γ::Float64   = 0.0
    R::Float64   = 287.0
    γ::Float64   = 1.4
    g::Float64   = 9.81
    p00::Float64 = 1.0e5
end
atm_T(a::Atmosphere, z) = a.T0 - a.Γ * z
atm_p(a::Atmosphere, z) = a.Γ == 0.0 ? a.p00 * exp(-a.g * z / (a.R * a.T0)) :
                                       a.p00 * (atm_T(a, z) / a.T0)^(a.g / (a.R * a.Γ))
atm_rho(a::Atmosphere, z) = atm_p(a, z) / (a.R * atm_T(a, z))
atm_theta(a::Atmosphere, z) = atm_T(a, z) * (a.p00 / atm_p(a, z))^((a.γ - 1) / a.γ)
atm_Theta(a::Atmosphere, z) = atm_rho(a, z) * atm_theta(a, z)
atm_pi(a::Atmosphere, z) = atm_p(a, z) / atm_Theta(a, z)        # p/(ρθ) = R × Exner

"""Frozen coefficients as scalar functions of z: pi = p̄/Θ̄, theta = θ̄, pbar = p̄."""
struct FrozenState
    pi::Function
    theta::Function
    pbar::Function
    g::Float64
    pfac::Float64
end
FrozenState(pi, theta; pbar = z -> 0.0, g = 9.81, pfac = 1.0) =
    FrozenState(pi, theta, pbar, Float64(g), Float64(pfac))
frozen_from_atmosphere(atm::Atmosphere; pfac = 1.0) =
    FrozenState(z -> atm_pi(atm, z), z -> atm_theta(atm, z), z -> atm_p(atm, z), atm.g, Float64(pfac))
coef_a(fr::FrozenState, z) = fr.pfac * fr.pi(z)
coef_b(fr::FrozenState, z) = (1.0 - fr.pfac) * fr.pbar(z)

# ---------------------------------------------------------------------------
# Quadrature, GLL nodes, Lagrange basis
# ---------------------------------------------------------------------------
function gauss_legendre(n::Int)
    n == 1 && return [0.0], [2.0]
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

legendre_P(n, x) = n == 0 ? 1.0 : n == 1 ? x :
    begin
        P0, P1 = 1.0, x
        for k in 2:n
            P0, P1 = P1, ((2k - 1) * x * P1 - (k - 1) * P0) / k
        end
        P1
    end

"""GLL nodes (±1 and zeros of P_p') and weights w = 2/(p(p+1) P_p(x)²)."""
function gll(p::Int)
    if p == 1
        x = [-1.0, 1.0]
    else
        n = p - 1
        inner = n == 1 ? [0.0] :
            eigvals(SymTridiagonal(zeros(n), [sqrt(k * (k + 2) / ((2k + 1) * (2k + 3))) for k in 1:n-1]))
        x = vcat(-1.0, sort(inner), 1.0)
    end
    w = [2.0 / (p * (p + 1) * legendre_P(p, xi)^2) for xi in x]
    return x, w
end

"""Lagrange basis on `nodes` and its derivative at points x → (length(x) × n) matrices."""
function lagrange_basis(nodes::AbstractVector, x::AbstractVector)
    n = length(nodes)
    lam = [1.0 / prod(nodes[j] - nodes[m] for m in 1:n if m != j) for j in 1:n]
    V = zeros(length(x), n); D = zeros(length(x), n)
    for (k, xk) in enumerate(x)
        d = xk .- nodes
        i = findfirst(abs.(d) .< 1e-14)
        if i !== nothing
            V[k, i] = 1.0
            for j in 1:n
                j != i && (D[k, j] = lam[j] / lam[i] / (nodes[i] - nodes[j]))
            end
            D[k, i] = -sum(D[k, :])
        else
            V[k, :] = prod(d) .* lam ./ d
            D[k, :] = V[k, :] .* (sum(1.0 ./ d) .- 1.0 ./ d)
        end
    end
    return V, D
end

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
struct HDGRelaxTh
    Ne::Int; p::Int; np::Int
    zmin::Float64; zmax::Float64; h::Float64; J::Float64; xe::Vector{Float64}
    fr::FrozenState; bc::Symbol; form::Symbol; eps::Float64; g::Float64
    Nf::Int; nq::Int; nt::Int
    xg::Vector{Float64}; wg::Vector{Float64}; Dg::Matrix{Float64}
    Za::Matrix{Float64}; Zth::Matrix{Float64}; Zb::Matrix{Float64}; Zphi::Matrix{Float64}  # (np × Ne)
    cface::Vector{Float64}; cmax::Float64
    Mdiag::Vector{Float64}; fb::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    G::SparseMatrixCSC{Float64,Int}
    H::SparseMatrixCSC{Float64,Int}
    L::SparseMatrixCSC{Float64,Int}
    Mbig::SparseMatrixCSC{Float64,Int}
end

iq(np::Int, e, r, i) = ((e - 1) * NV + (r - 1)) * np + i
iq(s::HDGRelaxTh, e, r, i) = iq(s.np, e, r, i)
it(f, r) = (f - 1) * NV + r
blk(s::HDGRelaxTh, q, e, r) = view(q, iq(s, e, r, 1):iq(s, e, r, s.np))
zpts(xe, h, e, ξ) = 0.5 * (xe[e] + xe[e+1]) .+ 0.5 * h .* ξ
zpts(s::HDGRelaxTh, e, ξ) = zpts(s.xe, s.h, e, ξ)
"""Node coordinates, (np × Ne)."""
nodes(s::HDGRelaxTh) = reduce(hcat, [zpts(s, e, s.xg) for e in 1:s.Ne])

"""Energy weight φ = θ̄ exp(∫_{zmin}^z g/c² dz'), c² = a θ̄, at the nodes (np × Ne)."""
function energy_weight(fr::FrozenState, xe, h, xg)
    Ne = length(xe) - 1
    xq, wq = gauss_legendre(12)
    f(z) = fr.g / (coef_a(fr, z) * fr.theta(z))
    Φ = zeros(length(xg), Ne)
    base = 0.0
    for e in 1:Ne
        z0 = xe[e]
        for (i, z) in enumerate(zpts(xe, h, e, xg))
            I = base
            if z > z0
                zz = 0.5 * (z0 + z) .+ 0.5 * (z - z0) .* xq
                I += 0.5 * (z - z0) * sum(wq .* f.(zz))
            end
            Φ[i, e] = fr.theta(z) * exp(I)
        end
        base = log(Φ[end, e] / fr.theta(xe[e+1]))
    end
    return Φ
end

"""
    HDGRelaxTh(Ne, p, zmin, zmax, frozen; bc=:wall, eps=0.0, tau_scale=1.0, form=:split)

bc = :wall or :periodic; form = :split (energy stable) or :standard.
"""
function HDGRelaxTh(Ne::Int, p::Int, zmin::Real, zmax::Real, fr::FrozenState;
                    bc::Symbol = :wall, eps::Real = 0.0, tau_scale::Real = 1.0,
                    form::Symbol = :split)
    p >= 1 || error("p >= 1 required")
    form in (:split, :standard) || error("form must be :split or :standard")
    np = p + 1
    h = (zmax - zmin) / Ne
    J = h / 2
    xe = collect(zmin .+ h .* (0:Ne))
    Nf = bc == :wall ? Ne + 1 : Ne
    nq = Ne * NV * np
    nt = Nf * NV
    g = fr.g
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)

    Z = reduce(hcat, [zpts(xe, h, e, xg) for e in 1:Ne])
    Za = coef_a.(Ref(fr), Z); Zth = fr.theta.(Z); Zb = coef_b.(Ref(fr), Z)
    Zphi = energy_weight(fr, xe, h, xg)
    Mdiag = repeat(w .* J, Ne * NV)

    zf = bc == :wall ? xe : xe[1:end-1]
    af = [coef_a(fr, z) for z in zf]
    thf = [fr.theta(z) for z in zf]
    cf = sqrt.(af .* thf)
    Af(f, n) = n .* [0.0 1.0 0.0; 0.0 0.0 af[f]; 0.0 thf[f] 0.0]
    Tf(f) = tau_scale .* [0.0 0.0 af[f]/cf[f]; 0.0 cf[f] 0.0; 0.0 0.0 cf[f]]
    face_R(e) = bc == :wall ? e + 1 : mod1(e + 1, Ne)

    KI, KJ, KV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    addK!(i, j, v) = v == 0 || (push!(KI, i); push!(KJ, j); push!(KV, v))
    addG!(i, j, v) = v == 0 || (push!(GI, i); push!(GJ, j); push!(GV, v))
    fb = zeros(nq)

    for e in 1:Ne
        a, th, b, ph = Za[:, e], Zth[:, e], Zb[:, e], Zphi[:, e]
        Dth = D * th
        for i in 1:np
            if form == :split
                for j in 1:np
                    addK!(iq(np, e, 1, i), iq(np, e, 2, j), -w[i] * D[i, j])                       # -(Dm)
                    addK!(iq(np, e, 3, i), iq(np, e, 2, j), -w[i] * th[i] * D[i, j])               # -θ̄ Dm
                    addK!(iq(np, e, 2, i), iq(np, e, 3, j), -w[i] * D[i, j] * ph[j] * a[j] / ph[i]) # -D(φaΘ)/φ
                end
                addK!(iq(np, e, 3, i), iq(np, e, 2, i), -w[i] * Dth[i])                            # -m Dθ̄
                addK!(iq(np, e, 2, i), iq(np, e, 3, i), w[i] * a[i] * Dth[i] / th[i] + J * w[i] * g / th[i])
            else
                for j in 1:np
                    addK!(iq(np, e, 1, i), iq(np, e, 2, j), w[j] * D[j, i])
                    addK!(iq(np, e, 2, i), iq(np, e, 3, j), w[j] * D[j, i] * a[j])
                    addK!(iq(np, e, 3, i), iq(np, e, 2, j), w[j] * D[j, i] * th[j])
                end
            end
            addK!(iq(np, e, 2, i), iq(np, e, 1, i), -J * w[i] * g)                                 # -g ρ
        end
        r2 = iq(np, e, 2, 1):iq(np, e, 2, np)
        if form == :split
            fb[r2] .+= -w .* (D * b)
        else
            fb[r2] .+= D' * (w .* b)
        end
        for (node, f, n) in ((np, face_R(e), 1.0), (1, e, -1.0))
            A, T = Af(f, n), Tf(f)
            if form == :split
                Mq, Mh = A .- T, -(A .- T)               # (A n - T)(q_b - q̂)
            else
                Mq, Mh = -T, -(A .- T)                   # -(A n q̂ + T(q_b - q̂))
                fb[iq(np, e, 2, node)] += n > 0 ? -coef_b(fr, xe[e+1]) : coef_b(fr, xe[e])
            end
            for r in 1:NV, s in 1:NV
                addK!(iq(np, e, r, node), iq(np, e, s, node), Mq[r, s])
                addG!(iq(np, e, r, node), it(f, s), Mh[r, s])
            end
        end
    end

    HI, HJ, HV = Int[], Int[], Float64[]
    for f in 1:Nf
        if bc == :wall && (f == 1 || f == Ne + 1)
            e, node = f == 1 ? (1, 1) : (Ne, np)
            for s in (1, 3)
                push!(HI, it(f, s)); push!(HJ, iq(np, e, s, node)); push!(HV, 1.0)
            end
            continue
        end
        eL, eR = mod1(f - 1, Ne), f
        for s in 1:NV
            append!(HI, [it(f, s), it(f, s)])
            append!(HJ, [iq(np, eL, s, np), iq(np, eR, s, 1)])
            append!(HV, [0.5, 0.5])
        end
    end

    K = sparse(KI, KJ, KV, nq, nq)
    G = sparse(GI, GJ, GV, nq, nt)
    H = sparse(HI, HJ, HV, nt, nq)
    L = [K G; H -sparse(1.0I, nt, nt)]
    Mbig = spdiagm(0 => vcat(Mdiag, fill(Float64(eps), nt)))
    return HDGRelaxTh(Ne, p, np, zmin, zmax, h, J, xe, fr, bc, form, Float64(eps), g, Nf, nq, nt,
                      xg, w, D, Za, Zth, Zb, Zphi, cf, maximum(cf), Mdiag, fb, K, G, H, L, Mbig)
end

# ---------------------------------------------------------------------------
# Operators and helpers
# ---------------------------------------------------------------------------
function forcing_vector(s::HDGRelaxTh, F_slow = nothing)
    f = copy(s.fb)
    F_slow === nothing || (f .+= s.Mdiag .* F_slow)
    return vcat(f, zeros(s.nt))
end

"""dq/dt of the fast operator (HDG limit q̂ = H q unless qh is given)."""
function tendency(s::HDGRelaxTh, q; qh = nothing, F_slow = nothing)
    qh === nothing && (qh = s.H * q)
    out = (s.K * q .+ s.G * qh .+ s.fb) ./ s.Mdiag
    F_slow === nothing || (out .+= F_slow)
    return out
end

"""Nodal interpolation (DGSEM initialization); `fun(z::Vector)` returns 3 × length(z)."""
function project(s::HDGRelaxTh, fun)
    q = zeros(s.nq)
    for e in 1:s.Ne
        F = fun(zpts(s, e, s.xg))
        for r in 1:NV
            blk(s, q, e, r) .= F[r, :]
        end
    end
    return q
end

"""Discrete hydrostatic ρ (np × Ne) for m = 0 and continuous nodal Θ (np × Ne):
chosen such that the discrete momentum residual vanishes exactly."""
function hydrostatic_rho(s::HDGRelaxTh, Th::AbstractMatrix)
    q = zeros(s.nq)
    for e in 1:s.Ne
        blk(s, q, e, 3) .= Th[:, e]
    end
    r = s.K * q .+ s.G * (s.H * q) .+ s.fb
    return reduce(hcat, [blk(s, r, e, 2) ./ (s.g * s.J .* s.wg) for e in 1:s.Ne])
end

"""Discrete hydrostatic state for Θ(z) (scalar function): m = 0, nodal Θ, balanced ρ."""
function balanced_state(s::HDGRelaxTh, Θfun)
    Th = Θfun.(nodes(s))
    rho = hydrostatic_rho(s, Th)
    q = zeros(s.nq)
    for e in 1:s.Ne
        blk(s, q, e, 1) .= rho[:, e]
        blk(s, q, e, 3) .= Th[:, e]
    end
    return q
end

"""Discrete weighted energy of a deviation dq (non-increasing for form=:split, ε = 0)."""
function energy(s::HDGRelaxTh, dq)
    E = 0.0
    for e in 1:s.Ne
        rho, m, Th = blk(s, dq, e, 1), blk(s, dq, e, 2), blk(s, dq, e, 3)
        a, th, ph = s.Za[:, e], s.Zth[:, e], s.Zphi[:, e]
        thz = (s.Dg * th) ./ s.J
        chi = Th .- th .* rho
        E += 0.5 * s.J * sum(s.wg .* ph .* (m .^ 2 .+ a .* Th .^ 2 ./ th .+ s.g .* chi .^ 2 ./ (th .* thz)))
    end
    return E
end

function evaluate(s::HDGRelaxTh, q; npts::Int = 8)
    ξ = collect(range(-1, 1; length = npts))
    V, _ = lagrange_basis(s.xg, ξ)
    Z = Float64[]; Q = zeros(NV, 0)
    for e in 1:s.Ne
        append!(Z, zpts(s, e, ξ))
        Q = hcat(Q, reduce(vcat, [(V * blk(s, q, e, r))' for r in 1:NV]))
    end
    return Z, Q
end

function l2_error(s::HDGRelaxTh, q, fun)
    xq, wq = gauss_legendre(s.p + 6)
    V, _ = lagrange_basis(s.xg, xq)
    err = zeros(NV)
    for e in 1:s.Ne
        F = fun(zpts(s, e, xq))
        for r in 1:NV
            d = V * blk(s, q, e, r) .- F[r, :]
            err[r] += s.J * sum(wq .* d .^ 2)
        end
    end
    return sqrt.(err)
end

"""Discrete integral Σ J w_i q_i of variable r (1 mass, 2 momentum, 3 Θ)."""
total(s::HDGRelaxTh, q, r) = s.J * sum(dot(s.wg, blk(s, q, e, r)) for e in 1:s.Ne)
initial_state(s::HDGRelaxTh, q) = vcat(q, s.H * q)
trace_residual(s::HDGRelaxTh, U) = maximum(abs.(s.H * U[1:s.nq] .- U[s.nq+1:end]))

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
"""
    run!(s, U0, dt, nsteps; method=:dirk, F_slow=nothing, monitor=nothing, every=1)

`F_slow`: constant tendency of the element unknowns (slow forcing, split-explicit).
"""
function run!(s::HDGRelaxTh, U0, dt, nsteps; method::Symbol = :dirk, F_slow = nothing,
              monitor = nothing, every::Int = 1)
    f = forcing_vector(s, F_slow)
    method == :dirk && return run_dirk(s, U0, dt, nsteps, f, monitor, every)
    method == :imex && return run_imex(s, U0, dt, nsteps, f, monitor, every)
    error("unknown method $method")
end

function run_dirk(s::HDGRelaxTh, U0, dt, nsteps, f, monitor, every)
    a = [1/4       0           0       0        0;
         1/2       1/4         0       0        0;
         17/50     (-1/25)     1/4     0        0;
         371/1360  (-137/2720) 15/544  1/4      0;
         25/24     (-49/48)    125/16  (-85/12) 1/4]
    F = lu(s.Mbig - (dt * a[1, 1]) * s.L)
    U = copy(U0)
    R = [zeros(length(U0)) for _ in 1:5]
    for n in 1:nsteps
        MU = s.Mbig * U
        Y = U
        for i in 1:5
            rhs = MU .+ (dt * a[i, i]) .* f
            for j in 1:i-1
                rhs .+= (dt * a[i, j]) .* R[j]
            end
            Y = F \ rhs
            R[i] = s.L * Y .+ f
        end
        U = Y
        if monitor !== nothing && n % every == 0
            monitor(n, U)
        end
    end
    return U
end

function run_imex(s::HDGRelaxTh, U0, dt, nsteps, f, monitor, every)
    gm = 1 - 1 / sqrt(2); dl = 1 - 1 / (2gm)
    AE = [0.0 0.0 0.0; gm 0.0 0.0; dl (1 - dl) 0.0]
    AI = [0.0 0.0 0.0; 0.0 gm 0.0; 0.0 (1 - gm) gm]
    nq, nt, ε = s.nq, s.nt, s.eps
    N = nq + nt
    fq = f[1:nq]
    fE(U) = vcat((s.K * U[1:nq] .+ s.G * U[nq+1:end] .+ fq) ./ s.Mdiag, zeros(nt))
    fI(U) = ε > 0 ? vcat(zeros(nq), (s.H * U[1:nq] .- U[nq+1:end]) ./ ε) : zeros(N)
    U = copy(U0)
    for n in 1:nsteps
        FE = Vector{Vector{Float64}}()
        FI = Vector{Vector{Float64}}()
        Y = U
        for i in 1:3
            Ys = copy(U)
            for j in 1:i-1
                Ys .+= dt .* (AE[i, j] .* FE[j] .+ AI[i, j] .* FI[j])
            end
            Y = copy(Ys)
            Hq = s.H * Y[1:nq]
            if ε == 0.0
                Y[nq+1:end] .= Hq
            else
                k = dt * AI[i, i] / ε
                Y[nq+1:end] .= (Ys[nq+1:end] .+ k .* Hq) ./ (1 + k)
            end
            push!(FE, fE(Y)); push!(FI, fI(Y))
        end
        U = Y
        if monitor !== nothing && n % every == 0
            monitor(n, U)
        end
    end
    return U
end

end # module
