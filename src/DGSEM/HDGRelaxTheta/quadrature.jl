# Quadrature, Gauss–Lobatto nodes, Lagrange basis

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
    V = zeros(length(x), n)
    D = zeros(length(x), n)
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
