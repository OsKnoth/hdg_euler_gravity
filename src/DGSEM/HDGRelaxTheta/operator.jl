# Fast HDG operator (1D DGSEM, split form, LMARS SAT, relaxed traces)

"""Fast HDG operators (alias of TimeStepping.AbstractFastOperator). Sparse operators have the
fields K, G, H and A = K + G H (ε = 0), matrix-free ones implement `fast_apply!` directly."""
const AbstractHDGOperator = AbstractFastOperator

struct HDGRelaxTh <: AbstractFastOperator
    Ne::Int
    p::Int
    np::Int
    zmin::Float64
    zmax::Float64
    h::Float64
    J::Float64
    xe::Vector{Float64}
    fr::Union{FrozenState,Nothing}
    bc::Symbol
    form::Symbol
    eps::Float64
    g::Float64
    Nf::Int
    nq::Int
    nt::Int
    xg::Vector{Float64}
    wg::Vector{Float64}
    Dg::Matrix{Float64}
    Za::Matrix{Float64}
    Zth::Matrix{Float64}
    Zb::Matrix{Float64}
    Zphi::Matrix{Float64}  # (np × Ne)
    cface::Vector{Float64}
    cmax::Float64
    Mdiag::Vector{Float64}
    fb::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    G::SparseMatrixCSC{Float64,Int}
    H::SparseMatrixCSC{Float64,Int}
    L::SparseMatrixCSC{Float64,Int}
    Mbig::SparseMatrixCSC{Float64,Int}
    A::SparseMatrixCSC{Float64,Int}       # K + G H (traces eliminated, ε = 0)
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

"""Energy weight from nodal coefficients: φ = θ̄ exp(∫ g/(aθ̄)), with g/(aθ̄) replaced by its
GLL interpolant and integrated exactly; the running integral is continuous across faces."""
function energy_weight_nodal(Za, Zth, g, J, xg)
    np, Ne = size(Za)
    xq, wq = gauss_legendre(np)
    Iint = zeros(np, np)                               # Iint[i, j] = ∫_{-1}^{ξ_i} ℓ_j
    for i in 1:np
        V, _ = lagrange_basis(xg, -1.0 .+ 0.5 * (xg[i] + 1) .* (xq .+ 1))
        Iint[i, :] = 0.5 * (xg[i] + 1) .* (V' * wq)
    end
    Φ = zeros(np, Ne)
    base = 0.0
    for e in 1:Ne
        I = base .+ J .* (Iint * (g ./ (Za[:, e] .* Zth[:, e])))
        Φ[:, e] = Zth[:, e] .* exp.(I)
        base = I[end]
    end
    return Φ
end

"""Average the end-node values of a nodal field (np × Ne) at interior faces, making it
continuous (direct stiffness summation); with bc = :periodic also across the outer face."""
function dss!(F::AbstractMatrix, bc::Symbol)
    Ne = size(F, 2)
    for e in 1:Ne-1
        F[end, e] = F[1, e+1] = 0.5 * (F[end, e] + F[1, e+1])
    end
    bc == :periodic && (F[end, Ne] = F[1, 1] = 0.5 * (F[end, Ne] + F[1, 1]))
    return F
end

"""
    HDGRelaxTh(Ne, p, zmin, zmax, frozen; bc=:wall, eps=0.0, tau_scale=1.0, form=:split)

bc = :wall or :periodic; form = :split (energy stable) or :standard.
"""
function HDGRelaxTh(Ne::Int, p::Int, zmin::Real, zmax::Real, fr::FrozenState;
                    bc::Symbol = :wall, eps::Real = 0.0, tau_scale::Real = 1.0,
                    form::Symbol = :split)
    p >= 1 || error("p >= 1 required")
    h = (zmax - zmin) / Ne
    xe = collect(zmin .+ h .* (0:Ne))
    xg, _ = gll(p)
    Z = reduce(hcat, [zpts(xe, h, e, xg) for e in 1:Ne])
    Za = coef_a.(Ref(fr), Z)
    Zth = fr.theta.(Z)
    Zb = coef_b.(Ref(fr), Z)
    Zphi = energy_weight(fr, xe, h, xg)
    zf = bc == :wall ? xe : xe[1:end-1]
    af = [coef_a(fr, z) for z in zf]
    thf = [fr.theta(z) for z in zf]
    bf = [coef_b(fr, z) for z in xe]
    return assemble(Ne, p, zmin, zmax, fr, fr.g, Za, Zth, Zb, Zphi, af, thf, bf;
                    bc = bc, eps = eps, tau_scale = tau_scale, form = form)
end

"""
    HDGRelaxTh(Ne, p, zmin, zmax, a, theta, b; g=9.81, bc=:wall, eps=0.0, tau_scale=1.0, form=:split)

Fast operator with frozen coefficients given as nodal arrays (np × Ne) on the GLL nodes, e.g.
taken from the current state of a split-explicit method. The fields are first made continuous
(`dss!`, on copies); face values are the common end-node values, and φ is built from the nodal
interpolant of g/(a θ̄). With continuous coefficients the discrete energy proof applies unchanged.
"""
function HDGRelaxTh(Ne::Int, p::Int, zmin::Real, zmax::Real, Za::AbstractMatrix,
                    Zth::AbstractMatrix, Zb::AbstractMatrix; g::Real = 9.81, bc::Symbol = :wall,
                    eps::Real = 0.0, tau_scale::Real = 1.0, form::Symbol = :split)
    p >= 1 || error("p >= 1 required")
    size(Za) == size(Zth) == size(Zb) == (p + 1, Ne) || error("coefficient arrays must be (p+1) × Ne")
    Za, Zth, Zb = dss!(Matrix{Float64}(Za), bc), dss!(Matrix{Float64}(Zth), bc), dss!(Matrix{Float64}(Zb), bc)
    h = (zmax - zmin) / Ne
    xg, _ = gll(p)
    Zphi = energy_weight_nodal(Za, Zth, Float64(g), h / 2, xg)
    Nf = bc == :wall ? Ne + 1 : Ne
    face(Zc) = [f <= Ne ? Zc[1, f] : Zc[end, Ne] for f in 1:Nf]
    bf = vcat(Zb[1, :], Zb[end, Ne])
    return assemble(Ne, p, zmin, zmax, nothing, Float64(g), Za, Zth, Zb, Zphi, face(Za), face(Zth), bf;
                    bc = bc, eps = eps, tau_scale = tau_scale, form = form)
end

function assemble(Ne, p, zmin, zmax, fr, g, Za, Zth, Zb, Zphi, af, thf, bf;
                  bc, eps, tau_scale, form)
    form in (:split, :standard) || error("form must be :split or :standard")
    np = p + 1
    h = (zmax - zmin) / Ne
    J = h / 2
    xe = collect(zmin .+ h .* (0:Ne))
    Nf = bc == :wall ? Ne + 1 : Ne
    nq = Ne * NV * np
    nt = Nf * NV
    xg, w = gll(p)
    _, D = lagrange_basis(xg, xg)
    Mdiag = repeat(w .* J, Ne * NV)
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
                fb[iq(np, e, 2, node)] += n > 0 ? -bf[e+1] : bf[e]
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
                push!(HI, it(f, s))
                push!(HJ, iq(np, e, s, node))
                push!(HV, 1.0)
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
                      xg, w, D, Za, Zth, Zb, Zphi, cf, maximum(cf), Mdiag, fb, K, G, H, L, Mbig, K + G * H)
end
