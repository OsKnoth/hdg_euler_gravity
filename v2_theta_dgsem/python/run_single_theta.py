"""
Single run of the fast operator (DGSEM split form, LMARS, relaxed traces) with IMEX.

    python3 run_single_theta.py

Setup: lapse-rate atmosphere, reflecting walls, one resolution with polynomial order 4
(5 Gauss-Lobatto nodes per element).  Initial state = discrete hydrostatic background
plus a pressure pulse (in ρθ) and optionally a warm bubble (θ' at constant pressure).
Plots initial and final profiles of ρ, ρw and ρθ (deviations from the background and
full fields) and writes them to CSV.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdg_relax_theta import Atmosphere, FrozenState, HDGRelaxTheta

# ------------------------------------------------------------------ parameters
P = 4                    # polynomial order (5 GLL nodes)
NE = 40                  # elements  -> h = 250 m, nodal spacing ~ 60 m
ZTOP = 10000.0           # domain [0, ZTOP], walls at both ends
T_END = 30.0             # final time [s]
CFL = 0.1                # dt = CFL * h / (c_max (2p+1)).  IMEX ARS(2,2,2) is stable up to ~0.2;
                         # <= 0.1 keeps its slight RK2 anti-dissipation below the LMARS dissipation
EPS_FAC = 0.0            # relaxation time eps = EPS_FAC * h / c_max  (0: exact HDG traces)
PFAC = 1.0               # 1: frozen p/(ρθ) (c² = R T);  1.4: tangent linear (c² = γ R T)
PULSE_PA = 100.0         # pressure pulse amplitude [Pa], centre / width [m]
PULSE_Z, PULSE_W = 3000.0, 300.0
BUBBLE_K = 0.5           # warm bubble θ' amplitude [K] (0 to switch off)
BUBBLE_Z, BUBBLE_W = 6000.0, 500.0
NPLOT = 10               # plot points per element

# ------------------------------------------------------------------ setup
atm = Atmosphere(T0=288.0, Gamma=0.0065)
fr = FrozenState.from_atmosphere(atm, pfac=PFAC)
s0 = HDGRelaxTheta(NE, P, 0.0, ZTOP, fr, bc="wall")
eps = EPS_FAC * s0.h / s0.cmax
s = HDGRelaxTheta(NE, P, 0.0, ZTOP, fr, bc="wall", eps=eps) if eps > 0 else s0

qb = s.balanced_state(atm.Theta)                 # discrete hydrostatic background


def perturbation(z):
    """Pressure pulse: Θ' = p'/a (ρ' = 0).  Warm bubble: θ' at constant pressure,
    i.e. Θ' = 0, ρ' = -ρ̄ θ'/θ̄ (to first order)."""
    pulse = PULSE_PA * np.exp(-((z - PULSE_Z) / PULSE_W) ** 2) / fr.a(z)
    thp = BUBBLE_K * np.exp(-((z - BUBBLE_Z) / BUBBLE_W) ** 2)
    rho_p = -atm.rho(z) * thp / atm.theta(z)
    return np.array([rho_p, 0 * z, pulse])


dq = s.project(perturbation)
q0 = qb + dq
U0 = s.initial_state(q0)

dt0 = CFL * s.h / (s.cmax * (2 * P + 1))
nsteps = int(np.ceil(T_END / dt0)); dt = T_END / nsteps

print(f"p = {P} ({P + 1} nodes), Ne = {NE}, h = {s.h:.1f} m, unknowns = {s.nq} + {s.nt} traces")
print(f"c_max = {s.cmax:.1f} m/s, dt = {dt:.4f} s, steps = {nsteps}, eps = {eps:.2e} s, pfac = {PFAC}")

# ------------------------------------------------------------------ run
hist = []
E0 = s.energy(dq)
mass0, Th0 = s.total(q0, 0), s.total(q0, 2)


def monitor(n, U):
    q = U[: s.nq]
    hist.append([n * dt, s.total(q, 0) - mass0, s.total(q, 2) - Th0, s.energy(q - qb) / E0,
                 s.trace_residual(U), np.max(np.abs(q[[s.iq(e, 1, i) for e in range(s.Ne) for i in range(s.np)]]))])


U = s.run(U0, dt, nsteps, method="imex", monitor=monitor, every=max(1, nsteps // 200))
q1 = U[: s.nq]
hist = np.array(hist)
print(f"mass drift = {hist[-1, 1]:.2e} (total {mass0:.4e}),  ρθ drift = {hist[-1, 2]:.2e} (total {Th0:.4e})")
print(f"E(T)/E(0) - 1 = {hist[-1, 3] - 1:.2e}   (weighted energy of the deviation; <= 0 expected)")

# ------------------------------------------------------------------ output
z, Qb = s.evaluate(qb, NPLOT)
_, Q0 = s.evaluate(q0, NPLOT)
_, Q1 = s.evaluate(q1, NPLOT)
np.savetxt("single_run_profiles.csv",
           np.column_stack([z, Qb.T, Q0.T, Q1.T]), delimiter=",",
           header="z,rho_b,m_b,Theta_b,rho_0,m_0,Theta_0,rho_T,m_T,Theta_T", comments="")

names = [r"$\rho'$ [kg m$^{-3}$]", r"$\rho w$ [kg m$^{-2}$ s$^{-1}$]", r"$(\rho\theta)'$ [K kg m$^{-3}$]"]
zk = z / 1000
fig, ax = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
for r in range(3):
    ax[0, r].plot(Q0[r] - Qb[r], zk, color="0.55", lw=1.6, label="initial")
    ax[0, r].plot(Q1[r] - Qb[r], zk, color="C0", lw=1.4, label=f"t = {T_END:g} s")
    ax[0, r].set_xlabel(names[r] + (" (deviation)" if r != 1 else ""))
    ax[0, r].axvline(0, color="k", lw=0.5)
xn = s.nodes().ravel() / 1000
for r, lab in enumerate([r"$\rho$ [kg m$^{-3}$]", r"$\rho w$ [kg m$^{-2}$ s$^{-1}$]", r"$\rho\theta$ [K kg m$^{-3}$]"]):
    ax[1, r].plot(Q0[r], zk, color="0.55", lw=1.6, label="initial")
    ax[1, r].plot(Q1[r], zk, color="C0", lw=1.4, ls="--", label=f"t = {T_END:g} s")
    ax[1, r].set_xlabel(lab + " (full field)")
for a in ax.ravel():
    a.grid(alpha=0.3)
for a in ax[:, 0]:
    a.set_ylabel("z [km]")
ax[0, 0].legend(loc="lower left")
from matplotlib.ticker import MaxNLocator
for a in ax.ravel():
    a.xaxis.set_major_locator(MaxNLocator(5))
    a.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
fig.suptitle(f"Fast operator, DGSEM (p={P}, {P + 1} GLL nodes, Ne={NE}, h={s.h:.0f} m), LMARS, "
             f"IMEX ARS(2,2,2), eps={EPS_FAC:g} h/c, pfac={PFAC:g}")
fig.tight_layout()
fig.savefig("single_run_profiles.png", dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(hist[:, 0], hist[:, 1] / mass0, label="mass")
ax[0].plot(hist[:, 0], hist[:, 2] / Th0, label=r"$\rho\theta$")
ax[0].set_ylabel("relative drift of totals"); ax[0].legend()
ax[1].plot(hist[:, 0], hist[:, 3] - 1)
ax[1].set_ylabel("E(t)/E(0) - 1   (weighted energy)")
if eps > 0:
    ax[2].semilogy(hist[:, 0], np.maximum(hist[:, 4], 1e-18))
    ax[2].set_ylabel(r"max $|Hq - \hat q|$  (trace residual)")
else:
    ax[2].plot(hist[:, 0], hist[:, 5])
    ax[2].set_ylabel(r"max $|\rho w|$ [kg m$^{-2}$ s$^{-1}$]")
for a in ax:
    a.set_xlabel("t [s]"); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("single_run_diagnostics.png", dpi=130)
print("wrote single_run_profiles.png, single_run_diagnostics.png, single_run_profiles.csv")
