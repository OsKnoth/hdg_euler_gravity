# Upwind-biased interpolation of advected quantities to the flux points (WRF flux3)

"""Third-order upwind-biased value at the point between qm1 and q0 (stencil qm2, qm1, q0, qp1)
for the transport velocity sign of `v` (WRF flux3 / v)."""
@inline function interp3(qm2, qm1, q0, qp1, v)
    centred = (7 * (q0 + qm1) - (qp1 + qm2)) / 12
    upwind = ((qp1 - qm2) - 3 * (q0 - qm1)) / 12
    return centred + copysign(1.0, v) * upwind
end

"""Second-order (centred) value between qm1 and q0."""
@inline interp2(qm1, q0) = 0.5 * (qm1 + q0)
