# Time integration of the fast system (run! wrapper, relaxed-trace IMEX); fast_solve! is in TimeStepping

"""
    run!(s, U0, dt, nsteps; method=:imex, F_slow=nothing, monitor=nothing, every=1)

Fast system M U_t = L U + f. method = :imex (explicit for the element unknowns; with ε > 0 the
relaxed traces are integrated implicitly, with ε = 0 they are the averages H q) or :hevi
(operators that split horizontal/vertical). `F_slow`: constant tendency of the element unknowns
(slow forcing, split-explicit).
"""
function run!(s::AbstractHDGOperator, U0, dt, nsteps; method::Symbol = :imex, F_slow = nothing,
              monitor = nothing, every::Int = 1)
    method in (:imex, :hevi) || error("unknown method $method (available: :imex, :hevi)")
    if s.eps > 0
        method == :imex || error("relaxed traces (eps > 0) only with :imex")
        return run_imex_relaxed(s, U0, dt, nsteps, forcing_vector(s, F_slow), monitor, every)
    end
    q = Vector{Float64}(U0[1:s.nq])
    if monitor === nothing
        fast_solve!(q, s, q, F_slow, dt, nsteps; method = method)
    else
        for n in 1:nsteps
            fast_solve!(q, s, q, F_slow, dt, 1; method = method)
            n % every == 0 && monitor(n, initial_state(s, q))
        end
    end
    return initial_state(s, q)
end

"""out = (K q + G H q + f) ./ M (ε = 0 fast tendency; f has length nq). Sparse version;
matrix-free operators provide their own method."""
function fast_apply!(out, s::AbstractHDGOperator, q, f)
    mul!(out, s.A, q)
    @inbounds for k in eachindex(out)
        out[k] = (out[k] + f[k]) / s.Mdiag[k]
    end
    return out
end

"""Work vectors (length nq) for the fast integrators (allocated per call for the 1D operator)."""
fast_workspace(s::HDGRelaxTh) = [zeros(s.nq) for _ in 1:7]

"""IMEX with relaxed traces (ε > 0): ε dq̂/dt = H q - q̂ implicit (diagonal), the rest explicit."""
function run_imex_relaxed(s::AbstractHDGOperator, U0, dt, nsteps, f, monitor, every)
    AE, AI = IMEX_AE, IMEX_AI
    nq, nt, ε = s.nq, s.nt, s.eps
    fq = view(f, 1:nq)
    U = copy(U0)
    FE = [zeros(nq + nt) for _ in 1:3]
    FI = [zeros(nq + nt) for _ in 1:3]
    Ys = similar(U)
    Hq = zeros(nt)
    for n in 1:nsteps
        for i in 1:3
            Ys .= U
            for j in 1:i-1
                @. Ys += dt * (AE[i, j] * FE[j] + AI[i, j] * FI[j])
            end
            q = view(Ys, 1:nq)
            qh = view(Ys, nq+1:nq+nt)
            mul!(Hq, s.H, q)
            k = dt * AI[i, i] / ε
            @. qh = (qh + k * Hq) / (1 + k)
            fe = view(FE[i], 1:nq)
            mul!(fe, s.K, q)
            mul!(fe, s.G, qh, 1.0, 1.0)
            @. fe = (fe + fq) / s.Mdiag
            mul!(Hq, s.H, q)
            @. FI[i][nq+1:end] = (Hq - qh) / ε
            i == 3 && (U .= Ys)
        end
        if monitor !== nothing && n % every == 0
            monitor(n, U)
        end
    end
    return U
end
