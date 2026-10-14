"""Verification and demo for hdg_relax_euler_gravity.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdg_relax_euler_gravity import Atmosphere, HDGRelax

np.set_printoptions(precision=3)


# ---------------------------------------------------------------- 1) convergence
def normal_mode(atm, Lz):
    """Exact normal mode of q_t + A q_z = S q for constant c0^2 (isothermal), periodic."""
    c2, g, gam = atm.c2(0.0), atm.g, atm.gamma
    k = 2 * np.pi / Lz
    A = np.array([[0, 1, 0], [0, 0, 1], [0, c2, 0]], float)
    S = np.array([[0, 0, 0], [-g, 0, 0], [0, -(gam - 1) * g, 0]], float)
    lam, vec = np.linalg.eig(-1j * k * A + S)
    i = np.argmax(lam.imag)                        # upward acoustic branch
    v = vec[:, i] / vec[2, i] * 100.0              # p' amplitude 100 Pa
    return lambda t: (lambda z: np.real(v[:, None] * np.exp(1j * k * z[None, :] + lam[i] * t)))


def convergence(method, eps, ps=(1, 2, 3), Nes=(8, 16, 32, 64), flux="lmars"):
    atm = Atmosphere(T0=250.0, Gamma=0.0)
    Lz = 2000.0
    ex = normal_mode(atm, Lz)
    c = np.sqrt(atm.c2(0.0))
    Tend = Lz / c
    out = {}
    for p in ps:
        errs, erho = [], []
        for Ne in Nes:
            s = HDGRelax(Ne, p, 0.0, Lz, atm, bc="periodic", eps=eps * (Lz / Ne) / c, flux=flux)
            q0 = s.project(ex(0.0))
            U0 = np.concatenate([q0, s.consistent_trace(q0)])
            if method == "dirk":
                cfl = 1.0 / (2 * p + 1)
            else:
                cfl = 0.2 / (2 * p + 1)
            dt0 = cfl * s.h / c
            n = int(np.ceil(Tend / dt0)); dt = Tend / n
            U = s.run(U0, dt, n, method=method)
            e3 = s.l2_error(U[: s.nq], ex(Tend))
            errs.append(e3[2]); erho.append(e3[0] * c**2)
        errs = np.array(errs); erho = np.array(erho)
        rates = np.log2(errs[:-1] / errs[1:])
        out[p] = (errs, rates)
        rr = np.log2(erho[:-1] / erho[1:])
        print(f"  {flux:5s} {method:4s} eps={eps:g}*h/c  p={p}: err(p')={errs}  rates={rates}  rates(rho')={rr}")
    return Nes, out


# ---------------------------------------------------------------- 1b) eps sweep
def eps_sweep(Ne=16, p=3):
    atm = Atmosphere(T0=250.0, Gamma=0.0)
    Lz = 2000.0; ex = normal_mode(atm, Lz); c = np.sqrt(atm.c2(0.0)); Tend = Lz / c
    facs = np.logspace(-8, 0, 9)
    out = {}
    for method in ("dirk", "imex"):
        errs = []
        for fac in facs:
            s = HDGRelax(Ne, p, 0.0, Lz, atm, bc="periodic", eps=fac * (Lz / Ne) / c)
            q0 = s.project(ex(0.0)); U0 = np.concatenate([q0, s.consistent_trace(q0)])
            cfl = (1.0 if method == "dirk" else 0.2) / (2 * p + 1)
            n = int(np.ceil(Tend / (cfl * s.h / c))); dt = Tend / n
            U = s.run(U0, dt, n, method=method)
            errs.append(s.l2_error(U[: s.nq], ex(Tend))[2])
        out[method] = np.array(errs)
        print(f"  {method}: eps*c/h = {facs}\n        err(p') = {out[method]}")
    s0 = HDGRelax(Ne, p, 0.0, Lz, atm, bc="periodic", eps=0.0)
    q0 = s0.project(ex(0.0)); U0 = np.concatenate([q0, s0.consistent_trace(q0)])
    n = int(np.ceil(Tend / (s0.h / c / (2 * p + 1)))); U = s0.run(U0, Tend / n, n, "dirk")
    e0 = s0.l2_error(U[: s0.nq], ex(Tend))[2]
    print(f"  eps=0 reference err(p') = {e0:.3e}")
    return facs, out, e0


# ---------------------------------------------------------------- 2) well-balancing
def well_balanced(method, eps, flux="lmars"):
    """Discrete hydrostatic perturbation in a non-isothermal atmosphere (walls)."""
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    Ne, p = 40, 3
    s = HDGRelax(Ne, p, 0.0, 10000.0, atm, bc="wall", eps=eps, flux=flux)
    # continuous piecewise-P_p pressure perturbation, rho' = -p'_z/g element-wise (exact)
    pfun = lambda z: 50.0 * np.cos(2 * np.pi * z / 7000.0) + 30.0 * (z / 1e4) ** 2
    q = np.zeros(s.nq)
    xi = np.cos(np.pi * np.arange(p + 1) / p)[::-1]          # interpolation nodes incl. endpoints
    V, D = s.basis(xi)
    for e in range(Ne):
        z = 0.5 * (s.xe[e] + s.xe[e + 1]) + 0.5 * s.h * xi
        cp = np.linalg.solve(V, pfun(z))                    # interpolant coefficients
        dcp = np.linalg.solve(V, (D @ cp) * 2 / s.h)        # derivative (degree p-1, exact)
        q[s.iq(e, 2, 0): s.iq(e, 2, 0) + p + 1] = cp
        q[s.iq(e, 0, 0): s.iq(e, 0, 0) + p + 1] = -dcp / atm.g
    U0 = np.concatenate([q, s.consistent_trace(q)])
    c = np.sqrt(atm.c2(0.0))
    dt = (1.0 if method == "dirk" else 0.2 / (2 * p + 1)) * s.h / c
    U = s.run(U0, dt, 200, method=method)
    dev = np.max(np.abs(U - U0)) / np.max(np.abs(U0))
    print(f"  {flux:5s} {method:4s} eps={eps:g}: max rel. deviation from hydrostatic state after 200 steps = {dev:.2e}")
    return dev


# ---------------------------------------------------------------- 3) wall demo
def wall_demo():
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    Lz, Ne, p = 10000.0, 80, 3
    c = np.sqrt(atm.c2(0.0))
    pulse = lambda z: np.array([0 * z, 0 * z, 100.0 * np.exp(-((z - 3000.0) / 300.0) ** 2)])
    res = {}
    for label, eps_fac, method in [("HDG eps=0 (DIRK)", 0.0, "dirk"),
                                   ("relaxed eps=1e-4 h/c (DIRK)", 1e-4, "dirk"),
                                   ("relaxed eps=1e-4 h/c (IMEX)", 1e-4, "imex"),
                                   ("relaxed eps=1e-2 h/c (DIRK)", 1e-2, "dirk")]:
        s = HDGRelax(Ne, p, 0.0, Lz, atm, bc="wall", eps=eps_fac * (Lz / Ne) / c)
        q0 = s.project(pulse)
        U0 = np.concatenate([q0, s.consistent_trace(q0)])
        m0 = s.total_mass(q0)
        Tend = 40.0
        cfl = 0.5 / (2 * p + 1) if method == "dirk" else 0.2 / (2 * p + 1)
        n = int(np.ceil(Tend / (cfl * s.h / c))); dt = Tend / n
        t_hist, mass, resid = [], [], []

        def mon(k, U):
            t_hist.append(k * dt)
            mass.append(s.total_mass(U[: s.nq]) - m0)
            resid.append(s.trace_residual(U))

        U = s.run(U0, dt, n, method=method, monitor=mon, every=max(1, n // 200))
        res[label] = (s, U, np.array(t_hist), np.array(mass), np.array(resid))
        print(f"  {label:28s}: |mass drift|max = {np.max(np.abs(mass)):.2e},  "
              f"trace residual max = {np.max(resid):.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, (s, U, t, m, r) in res.items():
        z, Q = s.evaluate(U[: s.nq])
        ax[0].plot(Q[2], z / 1000, label=label)
        ax[1].semilogy(t, np.abs(m) + 1e-30, label=label)
        ax[2].semilogy(t, r + 1e-30, label=label)
    ax[0].set_xlabel("p' [Pa] at t = 40 s"); ax[0].set_ylabel("z [km]")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("|total mass drift|")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("max |H q - q^| (trace residual)")
    for a in ax: a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Acoustic pulse, lapse-rate atmosphere, reflecting walls (Ne=80, p=3, LMARS)")
    fig.tight_layout(); fig.savefig("wall_demo.png", dpi=130)


if __name__ == "__main__":
    print("1) Convergence, periodic isothermal normal mode, one acoustic transit")
    conv = {}
    for method, eps in [("dirk", 0.0), ("dirk", 1e-9), ("imex", 1e-9)]:
        conv[(method, eps)] = convergence(method, eps, ps=(1, 2, 3) if method == "dirk" else (1, 2))
    print("  -- comparison: LLF penalty c*diag(0,1,1) --")
    convergence("dirk", 0.0, flux="llf")
    print("\n1b) Relaxation error vs eps (Ne=16, p=3)")
    facs, sweep, e0 = eps_sweep()
    print("\n2) Well-balancing (discrete hydrostatic perturbation, lapse-rate atmosphere)")
    for method, eps in [("dirk", 0.0), ("dirk", 1e-3), ("imex", 1e-3), ("imex", 0.0)]:
        well_balanced(method, eps)
    well_balanced("dirk", 0.0, flux="llf")
    print("\n3) Wall demo: pulse, conservation and constraint residual")
    wall_demo()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    mk = {"dirk": "o-", "imex": "s--"}
    for (method, eps), (Nes, out) in conv.items():
        for p, (errs, _) in out.items():
            ax[0].loglog(2000.0 / np.array(Nes), errs, mk[method], label=f"{method}, eps={eps:g} h/c, p={p}")
    ax[0].set_xlabel("h [m]"); ax[0].set_ylabel("L2 error in p' [Pa m^1/2]")
    ax[0].set_title("Convergence, periodic normal mode"); ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    for method, errs in sweep.items():
        ax[1].loglog(facs, errs, mk[method], label=method)
    ax[1].axhline(e0, color="k", lw=0.8, ls=":", label="eps = 0 (HDG)")
    ax[1].set_xlabel("eps c / h"); ax[1].set_ylabel("L2 error in p'")
    ax[1].set_title("Relaxation error, Ne=16, p=3"); ax[1].grid(alpha=0.3, which="both"); ax[1].legend()
    fig.tight_layout(); fig.savefig("convergence.png", dpi=130)
