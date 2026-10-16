# Metric terms from the LGL nodes of the elements

"""
    Jac, Jξx, Jξz, Jηx, Jηz = metric_terms(X, Z, D)

Collocation metric terms of the (curvilinear) elements from the physical node coordinates
X, Z (np × np × Ne) and the LGL differentiation matrix D:
    x_ξ = D X,  x_η = X Dᵀ,  z_ξ = D Z,  z_η = Z Dᵀ   (per element)
    Ja^ξ = (z_η, -x_η),  Ja^η = (-z_ξ, x_ξ),  J = x_ξ z_η - x_η z_ξ.
In 2D these satisfy the discrete metric identities D_ξ(Ja^ξ) + D_η(Ja^η) = 0 exactly.
"""
function metric_terms(X, Z, D)
    Jξx = similar(X)
    Jξz = similar(X)
    Jηx = similar(X)
    Jηz = similar(X)
    Jac = similar(X)
    for e in axes(X, 3)
        Xξ = D * X[:, :, e]
        Xη = X[:, :, e] * D'
        Zξ = D * Z[:, :, e]
        Zη = Z[:, :, e] * D'
        Jξx[:, :, e] = Zη
        Jξz[:, :, e] = -Xη
        Jηx[:, :, e] = -Zξ
        Jηz[:, :, e] = Xξ
        Jac[:, :, e] = Xξ .* Zη .- Xη .* Zξ
    end
    return Jac, Jξx, Jξz, Jηx, Jηz
end
