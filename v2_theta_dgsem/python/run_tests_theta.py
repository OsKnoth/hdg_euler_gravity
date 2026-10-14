"""Verification and demo for hdg_relax_theta.py (DGSEM, energy-stable split form)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdg_relax_theta import Atmosphere, FrozenState, HDGRelaxTheta

np.set_printoptions(precision=3)
R_, TH0, G_ = 287.0, 300.0, 9.81
f3 = lambda v: " ".join(f"{x:.3e}" for x in v)
f2 = lambda v: " ".join(f"{x:.2f}" for x in v)


# ---------------------------------------------------------------- 1) convergence
def const_frozen():
    """Constant frozen coefficients: π̄ = R (Exner = 1), θ̄ = 300 K -> c² = R θ̄."""
    return FrozenState(pi=lambda z: R_ + 0 * z, theta=lambda z: TH0 + 0 * z, g=G_)


def normal_mode(fr, Lz):
    a, th, g = fr.a(0.0), fr.theta(0.0), fr.g
    k = 2 * np.pi / Lz
    A = np.array([[0, 1, 0], [0, 0, a], [0, th, 0]], float)
    S = np.array([[0, 0, 0], [-g, 0, 0], [0, 0, 0]], float)
    lam, vec = np.linalg.eig(-1j * k * A + S)
    i = np.argmax(lam.imag)
    v = vec[:, i] / vec[2, i] * (100.0 / a)          # pressure amplitude a Θ' = 100 Pa
    return lambda t: (lambda z: np.real(v[:, None] * np.exp(1j * k * np.asarray(z)[None, :] + lam[i] * t)))


def convergence(method, epsfac, ps=(1, 2, 3), Nes=(8, 16, 32, 64), form="split"):
    fr = const_frozen(); Lz = 2000.0; ex = normal_mode(fr, Lz)
    c = np.sqrt(fr.a(0.0) * fr.theta(0.0)); Tend = Lz / c
    out = {}
    for p in ps:
        E = []
        for Ne in Nes:
            s = HDGRelaxTheta(Ne, p, 0.0, Lz, fr, bc="periodic", eps=epsfac * (Lz / Ne) / c, form=form)
            U0 = s.initial_state(s.project(ex(0.0)))
            cfl = (1.0 if method == "dirk" else 0.2) / (2 * p + 1)
            n = int(np.ceil(Tend / (cfl * s.h / c))); dt = Tend / n
            U = s.run(U0, dt, n, method=method)
            E.append(s.l2_error(U[: s.nq], ex(Tend)))
        E = np.array(E)
        out[p] = E[:, 2]
        print(f"  {form:8s} {method:4s} eps={epsfac:g}*h/c p={p}: err(Θ)= {f3(E[:, 2])} | "
              f"rates ρ,m,Θ (last) = {f2(np.log2(E[-2] / E[-1]))}")
    return Nes, out


def eps_sweep(Ne=16, p=3):
    fr = const_frozen(); Lz = 2000.0; ex = normal_mode(fr, Lz)
    c = np.sqrt(fr.a(0.0) * fr.theta(0.0)); Tend = Lz / c
    facs = np.logspace(-8, 0, 9); res = {}
    for method in ("dirk", "imex"):
        errs = []
        for fac in list(facs) + [0.0]:
            s = HDGRelaxTheta(Ne, p, 0.0, Lz, fr, bc="periodic", eps=fac * (Lz / Ne) / c)
            U0 = s.initial_state(s.project(ex(0.0)))
            cfl = (1.0 if method == "dirk" else 0.2) / (2 * p + 1)
            n = int(np.ceil(Tend / (cfl * s.h / c)))
            U = s.run(U0, Tend / n, n, method=method)
            errs.append(s.l2_error(U[: s.nq], ex(Tend))[2])
        res[method] = np.array(errs)
        print(f"  {method}: err(Θ) = {f3(res[method][:-1])}   (eps=0: {res[method][-1]:.3e})")
    print(f"  eps*c/h  = {f3(facs)}")
    return facs, res


# ---------------------------------------------------------------- 2) stability
def semidiscrete_matrix(s):
    L = s.L.toarray(); nq = s.nq
    return (L[:nq, :nq] + L[:nq, nq:] @ L[nq:, :nq]) / s.Mdiag[:, None]


def stability_table():
    rows = []
    for name, atm in [("isothermal", Atmosphere(T0=288.0)), ("lapse", Atmosphere(T0=288.0, Gamma=0.0065))]:
        for pfac in (1.0, 1.4):
            for form in ("split", "standard"):
                mr = []
                for Ne in (8, 16, 32):
                    s = HDGRelaxTheta(Ne, 3, 0.0, 1e4, FrozenState.from_atmosphere(atm, pfac=pfac), bc="wall", form=form)
                    mr.append(np.linalg.eigvals(semidiscrete_matrix(s)).real.max())
                rows.append((name, pfac, form, mr))
                print(f"  {name:10s} pfac={pfac:.1f} {form:8s}: max Re λ (Ne=8,16,32) = {f3(mr)}")
    # relaxed system, split form
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    for ef in (1e-3, 0.1, 1.0):
        s = HDGRelaxTheta(16, 3, 0.0, 1e4, FrozenState.from_atmosphere(atm), bc="wall", eps=ef * 625 / 340)
        Mi = np.concatenate([1 / s.Mdiag, np.ones(s.nt) / s.eps])
        print(f"  relaxed eps={ef:g} h/c (split, lapse): max Re λ = {np.linalg.eigvals(Mi[:, None] * s.L.toarray()).real.max():.2e}")
    return rows


# ---------------------------------------------------------------- 3) well-balancing
def well_balanced(method, eps, pfac):
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    s = HDGRelaxTheta(40, 3, 0.0, 10000.0, FrozenState.from_atmosphere(atm, pfac=pfac), bc="wall", eps=eps)
    q = s.balanced_state(atm.Theta)
    U0 = s.initial_state(q)
    dt = (1.0 if method == "dirk" else 0.2 / 7) * s.h / s.cmax
    U = s.run(U0, dt, 200, method=method)
    dev = np.max(np.abs(U - U0)) / np.max(np.abs(U0))
    z, Q = s.evaluate(q)
    rho_err = np.max(np.abs(Q[0] - atm.rho(z))) / atm.rho(0.0)
    print(f"  pfac={pfac:.1f} {method:4s} eps={eps:g}: max rel. deviation after 200 steps = {dev:.2e}"
          f"   (discrete vs exact ρ: {rho_err:.1e})")


def projected_background_residual():
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    out = []
    for Ne in (10, 20, 40, 80):
        s = HDGRelaxTheta(Ne, 3, 0.0, 10000.0, FrozenState.from_atmosphere(atm), bc="wall")
        q = s.project(lambda z: np.array([atm.rho(z), 0 * z, atm.Theta(z)]))
        out.append(np.max(np.abs(s.tendency(q))))
    out = np.array(out)
    print(f"  interpolated analytic background (pfac=1), p=3, Ne=10..80: max|dq/dt| = {f3(out)}  (exact balance)")
    out = []
    for Ne in (10, 20, 40, 80):
        s = HDGRelaxTheta(Ne, 3, 0.0, 10000.0, FrozenState.from_atmosphere(atm, pfac=atm.gamma), bc="wall")
        q = s.project(lambda z: np.array([atm.rho(z), 0 * z, atm.Theta(z)]))
        out.append(np.max(np.abs(s.tendency(q))))
    out = np.array(out)
    print(f"  interpolated analytic background (pfac=γ), p=3, Ne=10..80: max|dq/dt| = {f3(out)} | rates = {f2(np.log2(out[:-1] / out[1:]))}")


# ---------------------------------------------------------------- 4) split-explicit hook
def slow_forcing_check():
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    s = HDGRelaxTheta(20, 2, 0.0, 10000.0, FrozenState.from_atmosphere(atm), bc="wall", eps=0.0)
    rng = np.random.default_rng(0)
    qs = s.project(lambda z: np.array([atm.rho(z), 0 * z, atm.Theta(z)])) + 1e-3 * rng.normal(size=s.nq)
    F = -s.tendency(qs)
    for method in ("dirk", "imex"):
        U = s.run(s.initial_state(qs), 0.05 * s.h / s.cmax, 50, method=method, F_slow=F)
        print(f"  {method}: q* with F_slow = -L(q*) stays steady, max|q - q*| = {np.max(np.abs(U[:s.nq] - qs)):.2e}")


# ---------------------------------------------------------------- 5) wall demo
def wall_demo():
    atm = Atmosphere(T0=288.0, Gamma=0.0065)
    Lz, Ne, p = 10000.0, 80, 3
    res = {}
    cases = [("pfac=1, eps=0, DIRK, split", 1.0, 0.0, "dirk", "split"),
             ("pfac=1, eps=1e-4 h/c, IMEX, split", 1.0, 1e-4, "imex", "split"),
             ("pfac=γ, eps=0, DIRK, split", atm.gamma, 0.0, "dirk", "split"),
             ("pfac=1, eps=0, DIRK, standard", 1.0, 0.0, "dirk", "standard")]
    for label, pfac, efac, method, form in cases:
        fr = FrozenState.from_atmosphere(atm, pfac=pfac)
        s0 = HDGRelaxTheta(Ne, p, 0.0, Lz, fr, bc="wall", form=form)
        s = HDGRelaxTheta(Ne, p, 0.0, Lz, fr, bc="wall", eps=efac * (Lz / Ne) / s0.cmax, form=form)
        qb = s.balanced_state(atm.Theta)
        dq = s.project(lambda z: np.array([0 * z, 0 * z, 100.0 * np.exp(-((z - 3000.0) / 300.0) ** 2) / fr.a(z)]))
        U0 = s.initial_state(qb + dq)
        tot0 = [s.total(qb + dq, r) for r in (0, 2)]
        E0 = s.energy(dq)
        Tend = 25.0
        cfl = (0.5 if method == "dirk" else 0.2) / (2 * p + 1)
        n = int(np.ceil(Tend / (cfl * s.h / s.cmax))); dt = Tend / n
        hist = []

        def mon(k, U):
            q = U[: s.nq]
            hist.append([k * dt, s.total(q, 0) - tot0[0], s.total(q, 2) - tot0[1],
                         s.trace_residual(U), s.energy(q - qb) / E0])

        U = s.run(U0, dt, n, method=method, monitor=mon, every=max(1, n // 200))
        z, Q = s.evaluate(U[: s.nq] - qb)
        hist = np.array(hist)
        res[label] = (z, fr.a(z) * Q[2], hist)
        dE = np.diff(np.concatenate([[1.0], hist[:, 4]]))
        print(f"  {label:34s}: c(0)={np.sqrt(fr.a(0.0) * fr.theta(0.0)):.1f} m/s, |Δmass|={np.max(np.abs(hist[:, 1])):.1e}, "
              f"|ΔΘ|={np.max(np.abs(hist[:, 2])):.1e}, E(T)/E0={hist[-1, 4]:.4f}, max ΔE>0: {max(dE.max(), 0):.1e}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for label, (z, pp, hist) in res.items():
        ax[0].plot(pp, z / 1000, label=label)
        ax[1].semilogy(hist[:, 0], np.maximum(np.abs(hist[:, 1]), 1e-16), label=label)
        ax[2].semilogy(hist[:, 0], np.maximum(1.0 - hist[:, 4], 1e-16), label=label)
    ax[0].set_xlabel("p' = a (Θ - Θ_bg) [Pa] at t = 25 s"); ax[0].set_ylabel("z [km]")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("|total mass drift|")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("1 - E(t)/E(0)  (weighted energy loss)")
    for a_ in ax: a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Fast operator, DGSEM + LMARS, lapse-rate atmosphere, walls (Ne=80, p=3)")
    fig.tight_layout(); fig.savefig("theta_wall_demo.png", dpi=130)


def long_run_growth():
    """Long integration of a grid-scale perturbation: split vs standard form."""
    atm = Atmosphere(T0=288.0)
    for form in ("split", "standard"):
        s = HDGRelaxTheta(16, 3, 0.0, 1e4, FrozenState.from_atmosphere(atm), bc="wall", form=form)
        qb = s.balanced_state(atm.Theta)
        rng = np.random.default_rng(1)
        dq = 1e-3 * rng.normal(size=s.nq)
        E0 = s.energy(dq)
        U = s.run(s.initial_state(qb + dq), 20.0, 1080, method="dirk")   # 6 h, dt = 20 s
        print(f"  {form:8s}: random perturbation, 6 h with dt=20 s: E(6h)/E(0) = {s.energy(U[:s.nq] - qb) / E0:.3e}")


if __name__ == "__main__":
    print("1) Convergence, constant frozen coefficients, periodic normal mode")
    conv = {}
    for method, ef in [("dirk", 0.0), ("dirk", 1e-9), ("imex", 1e-9)]:
        conv[(method, ef)] = convergence(method, ef, ps=(1, 2, 3) if method == "dirk" else (1, 2))
    convergence("dirk", 0.0, ps=(3,), form="standard")
    print("\n1b) Relaxation error vs eps (Ne=16, p=3)")
    facs, sweep = eps_sweep()
    print("\n2) Stability: largest real part of the spectrum (walls, p=3, eps=0)")
    stability_table()
    long_run_growth()
    print("\n3) Well-balancing (discrete hydrostatic state, lapse-rate atmosphere, walls)")
    for pfac in (1.0, 1.4):
        for method, eps in [("dirk", 0.0), ("dirk", 1e-3), ("imex", 1e-3)]:
            well_balanced(method, eps, pfac)
    projected_background_residual()
    print("\n4) Split-explicit hook: constant slow forcing")
    slow_forcing_check()
    print("\n5) Wall demo")
    wall_demo()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    mk = {"dirk": "o-", "imex": "s--"}
    for (method, ef), (Nes, out) in conv.items():
        for p, errs in out.items():
            ax[0].loglog(2000.0 / np.array(Nes), errs, mk[method], label=f"{method}, eps={ef:g} h/c, p={p}")
    ax[0].set_xlabel("h [m]"); ax[0].set_ylabel("L2 error in Θ"); ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7); ax[0].set_title("Convergence, normal mode (DGSEM split form)")
    for method, errs in sweep.items():
        ax[1].loglog(facs, errs[:-1], mk[method], label=method)
    ax[1].axhline(sweep["dirk"][-1], color="k", ls=":", lw=0.8, label="eps = 0")
    ax[1].set_xlabel("eps c / h"); ax[1].set_ylabel("L2 error in Θ"); ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(); ax[1].set_title("Relaxation error, Ne=16, p=3")
    fig.tight_layout(); fig.savefig("theta_convergence.png", dpi=130)
