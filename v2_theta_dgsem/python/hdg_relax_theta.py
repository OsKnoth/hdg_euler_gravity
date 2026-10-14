"""
1D hybridized DGSEM (nodal, Gauss-Lobatto collocation) with LMARS interface flux for the
LINEARIZED (fast) part of the compressible Euler equations in potential-temperature form,
with relaxed trace conditions integrated in time WITHOUT elimination.

----------------------------------------------------------------------------
Model
----------------------------------------------------------------------------
Full equations (Θ = ρθ, m = ρw):
    ρ_t + m_z = 0,   m_t + (m w + p)_z = -ρ g,   Θ_t + (θ m)_z = 0,   p = p00 (RΘ/p00)^γ
Linearization (fast operator for a split-explicit method), frozen coefficients:
  * momentum advection removed
  * θ frozen in the Θ equation:   (θ m)_z -> (θ̄ m)_z
  * p = (p/Θ) Θ with p/Θ frozen:  p -> a Θ + b,  a = pfac p̄/Θ̄,  b = (1-pfac) p̄
    (pfac = 1: the linearization above, c² = a θ̄ = R T̄;  pfac = γ: tangent linear)

    ρ_t + m_z = 0,   m_t + (a Θ + b)_z = -g ρ,   Θ_t + (θ̄ m)_z = 0         (*)

Linear (affine) in the FULL state q = (ρ, m, Θ):  q_t = L(q) + F_slow.

----------------------------------------------------------------------------
Energy of (*)
----------------------------------------------------------------------------
With p = aΘ, χ = Θ - θ̄ρ (θ-perturbation), c² = aθ̄, φ = θ̄ exp(∫ g/c² dz):
    E = 1/2 ∫ φ [ m² + p²/c² + g χ² / (θ̄ θ̄_z) ] dz,    dE/dt = -[φ p m]_boundary
(positive for stable stratification θ̄_z > 0).  Note χ_t = -θ̄_z m.

----------------------------------------------------------------------------
DGSEM (form="split", default): energy-stable split form
----------------------------------------------------------------------------
Per element: GLL nodes ξ_i, weights w_i, SBP derivative D (W D + D^T W = B), J = h/2,
all products evaluated at nodes, ∂z = D/J:
    ρ_i' = -(D m)_i / J                                              + SAT
    Θ_i' = -(θ̄_i (D m)_i + m_i (D θ̄)_i) / J        (product form)    + SAT
    m_i' = -[ (D(φ a Θ))_i / φ_i - a_i Θ_i (D θ̄)_i / θ̄_i + (D b)_i ] / J
           + g Θ_i / θ̄_i - g ρ_i                                     + SAT
The product form of the Θ flux is conservative by SBP; with it χ_i' = -(Dθ̄)_i m_i / J
holds exactly, and the pressure-gradient form makes the discrete energy identity exact.
form="standard": plain collocation weak form (conservative a Θ, θ̄ m fluxes); only
O(h)-stable for variable θ̄ with gravity (kept for comparison).

SAT / HDG coupling at the end node b of an element (outward normal n):
    J w_b q_b' += (A_f n - T_f)(q_b - q̂_f)
    A_f = [0 1 0; 0 0 a; 0 θ̄ 0],   T_f = |A_f| = [0 0 a/c; 0 c 0; 0 0 c]
Trace conditions: interior q̂ = mean(q), wall q̂ = diag(1,0,1) q_int (m̂ = 0); relaxed as
ε dq̂/dt = H q - q̂.  For ε = 0 the face flux is LMARS:
    m* = {m} - [p]/(2c),   p* = {p} + b - c[m]/2,   Θ-flux = θ̄ m*,
and the face energy production is φ(-c[m]²/2 - [p]²/(2c)) <= 0 (walls: -φ c m²).

Discrete hydrostatic states (m = 0, Θ continuous) are obtained from the operator itself
(hydrostatic_rho / balanced_state) and are steady to round-off.

Semi-discrete system, U = (q, q̂):
    [M 0; 0 εI] U_t = [K G; H -I] U + [f_b + M F_slow; 0],   M = diag(J w_i)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as Lg


# ---------------------------------------------------------------------------
class Atmosphere:
    """Hydrostatic background with constant lapse rate (Gamma = 0 -> isothermal)."""

    def __init__(self, T0=288.0, Gamma=0.0, R=287.0, gamma=1.4, g=9.81, p00=1e5):
        self.T0, self.Gamma, self.R, self.gamma, self.g, self.p00 = T0, Gamma, R, gamma, g, p00
        self.kappa = (gamma - 1) / gamma

    def T(self, z):
        return self.T0 - self.Gamma * z

    def p(self, z):
        if self.Gamma == 0.0:
            return self.p00 * np.exp(-self.g * z / (self.R * self.T0))
        return self.p00 * (self.T(z) / self.T0) ** (self.g / (self.R * self.Gamma))

    def rho(self, z):
        return self.p(z) / (self.R * self.T(z))

    def theta(self, z):
        return self.T(z) * (self.p00 / self.p(z)) ** self.kappa

    def Theta(self, z):
        return self.rho(z) * self.theta(z)

    def pi(self, z):
        """p / (rho theta) = R * Exner."""
        return self.p(z) / self.Theta(z)


class FrozenState:
    """Frozen coefficients: pi(z) = p̄/Θ̄, theta(z) = θ̄, pbar(z) = p̄ (only for pfac != 1)."""

    def __init__(self, pi, theta, pbar=None, g=9.81, pfac=1.0):
        self.pi, self.theta, self.g, self.pfac = pi, theta, g, float(pfac)
        self.pbar = pbar if pbar is not None else (lambda z: 0.0 * np.asarray(z, float))

    @classmethod
    def from_atmosphere(cls, atm, pfac=1.0):
        return cls(atm.pi, atm.theta, atm.p, g=atm.g, pfac=pfac)

    def a(self, z):
        return self.pfac * self.pi(z)

    def b(self, z):
        return (1.0 - self.pfac) * self.pbar(z)


# ---------------------------------------------------------------------------
def gll(p):
    """Gauss-Lobatto-Legendre nodes and weights on [-1,1] (p+1 points)."""
    c = np.zeros(p + 1); c[-1] = 1.0
    inner = np.sort(np.real(Lg.legroots(Lg.legder(c)))) if p > 1 else np.array([])
    x = np.concatenate([[-1.0], inner, [1.0]])
    w = 2.0 / (p * (p + 1) * Lg.legval(x, c) ** 2)
    return x, w


def lagrange_basis(nodes, x):
    """Lagrange basis ℓ_j on `nodes` and its derivative at points x -> (len(x), n)."""
    n = len(nodes)
    lam = np.array([1.0 / np.prod(nodes[j] - np.delete(nodes, j)) for j in range(n)])
    V = np.zeros((len(x), n)); D = np.zeros((len(x), n))
    for k, xk in enumerate(np.atleast_1d(x)):
        d = xk - nodes
        hit = np.where(np.abs(d) < 1e-14)[0]
        if len(hit):
            i = hit[0]
            V[k, i] = 1.0
            for j in range(n):
                if j != i:
                    D[k, j] = lam[j] / lam[i] / (nodes[i] - nodes[j])
            D[k, i] = -np.sum(D[k, :])
        else:
            V[k] = np.prod(d) * lam / d
            D[k] = V[k] * (np.sum(1.0 / d) - 1.0 / d)
    return V, D


# ---------------------------------------------------------------------------
class HDGRelaxTheta:
    NV = 3  # rho, m = rho w, Theta = rho theta

    def __init__(self, Ne, p, zmin, zmax, frozen, bc="wall", eps=0.0, tau_scale=1.0,
                 form="split"):
        assert p >= 1 and form in ("split", "standard")
        self.Ne, self.p, self.np = Ne, p, p + 1
        self.zmin, self.zmax, self.fr, self.bc, self.form = zmin, zmax, frozen, bc, form
        self.g = frozen.g
        self.eps = float(eps)
        self.h = (zmax - zmin) / Ne
        self.J = self.h / 2
        self.xe = zmin + self.h * np.arange(Ne + 1)
        self.Nf = Ne + 1 if bc == "wall" else Ne
        self.nq = Ne * self.NV * self.np
        self.nt = self.Nf * self.NV
        self.tau_scale = tau_scale
        self.xg, self.wg = gll(p)
        self.Dg = lagrange_basis(self.xg, self.xg)[1]       # SBP differentiation matrix
        self._build()

    # indices / geometry ---------------------------------------------------
    def iq(self, e, r, i):
        return (e * self.NV + r) * self.np + i

    def it(self, f, s):
        return f * self.NV + s

    def face_R(self, e):
        return e + 1 if self.bc == "wall" else (e + 1) % self.Ne

    def blk(self, q, e, r):
        i0 = self.iq(e, r, 0)
        return q[i0:i0 + self.np]

    def zpts(self, e, xi):
        return 0.5 * (self.xe[e] + self.xe[e + 1]) + 0.5 * self.h * np.asarray(xi)

    def nodes(self):
        """(Ne, np) array of node coordinates."""
        return np.array([self.zpts(e, self.xg) for e in range(self.Ne)])

    def basis(self, xi):
        return lagrange_basis(self.xg, np.asarray(xi, float))

    def _phi(self, Z):
        """Energy weight φ = θ̄ exp(∫ g/c² dz), c² = a θ̄ (continuous, nodal)."""
        fr = self.fr
        xq, wq = Lg.leggauss(12)
        f = lambda z: fr.g / (fr.a(z) * fr.theta(z))
        I = np.zeros_like(Z)
        base = 0.0
        for e in range(self.Ne):
            z0 = self.xe[e]
            for i, z in enumerate(Z[e]):
                if z > z0:
                    zz = 0.5 * (z0 + z) + 0.5 * (z - z0) * xq
                    I[e, i] = base + 0.5 * (z - z0) * np.sum(wq * f(zz))
                else:
                    I[e, i] = base
            base = I[e, -1]
        return fr.theta(Z) * np.exp(I)

    # assembly -------------------------------------------------------------
    def _build(self):
        NV, npp, J, g = self.NV, self.np, self.J, self.g
        w, D = self.wg, self.Dg
        fr = self.fr
        Z = self.nodes()
        self.Za, self.Zth, self.Zb = fr.a(Z), fr.theta(Z), fr.b(Z)
        self.Zphi = self._phi(Z)
        self.Mdiag = np.tile(w * J, self.Ne * NV)

        zf = self.xe if self.bc == "wall" else self.xe[:-1]
        af, thf = fr.a(zf), fr.theta(zf)
        cf = np.sqrt(af * thf)
        self.cface = cf
        Af = lambda f, n: n * np.array([[0, 1, 0], [0, 0, af[f]], [0, thf[f], 0]], float)
        Tf = lambda f: self.tau_scale * np.array([[0, 0, af[f] / cf[f]], [0, cf[f], 0], [0, 0, cf[f]]])

        rows, cols, vals = [], [], []
        grow, gcol, gval = [], [], []

        def addK(i, j, v):
            if v != 0.0:
                rows.append(i); cols.append(j); vals.append(v)

        def addG(i, j, v):
            if v != 0.0:
                grow.append(i); gcol.append(j); gval.append(v)

        self.fb = np.zeros(self.nq)
        for e in range(self.Ne):
            a, th, b, ph = self.Za[e], self.Zth[e], self.Zb[e], self.Zphi[e]
            Dth = D @ th
            iR, iM, iT = (lambda i: self.iq(e, 0, i)), (lambda i: self.iq(e, 1, i)), (lambda i: self.iq(e, 2, i))
            for i in range(npp):
                if self.form == "split":
                    for j in range(npp):
                        addK(iR(i), iM(j), -w[i] * D[i, j])                         # -(Dm)
                        addK(iT(i), iM(j), -w[i] * th[i] * D[i, j])                 # -θ̄ Dm
                        addK(iM(i), iT(j), -w[i] * D[i, j] * ph[j] * a[j] / ph[i])  # -D(φaΘ)/φ
                    addK(iT(i), iM(i), -w[i] * Dth[i])                              # -m Dθ̄
                    addK(iM(i), iT(i), w[i] * a[i] * Dth[i] / th[i] + J * w[i] * g / th[i])
                else:  # standard collocation weak form: (v_z, f) with f = m, aΘ, θ̄m
                    for j in range(npp):
                        addK(iR(i), iM(j), w[j] * D[j, i])
                        addK(iM(i), iT(j), w[j] * D[j, i] * a[j])
                        addK(iT(i), iM(j), w[j] * D[j, i] * th[j])
                addK(iM(i), iR(i), -J * w[i] * g)                                  # -g ρ
            # affine part b
            if self.form == "split":
                self.fb[iM(0):iM(0) + npp] += -w * (D @ b)
            else:
                self.fb[iM(0):iM(0) + npp] += D.T @ (w * b)
            # faces: strong-form SAT (split) or weak-form flux (standard)
            for (node, f, n) in ((npp - 1, self.face_R(e), 1.0), (0, e, -1.0)):
                A, T = Af(f, n), Tf(f)
                if self.form == "split":
                    Mq, Mh = A - T, -(A - T)            # (A n - T)(q_b - q̂)
                else:
                    Mq, Mh = -T, -(A - T)               # -(A n q̂ + T(q_b - q̂))
                    if n > 0:
                        self.fb[self.iq(e, 1, node)] += -fr.b(self.xe[e + 1])
                    else:
                        self.fb[self.iq(e, 1, node)] += fr.b(self.xe[e])
                for r in range(NV):
                    for s in range(NV):
                        addK(self.iq(e, r, node), self.iq(e, s, node), Mq[r, s])
                        addG(self.iq(e, r, node), self.it(f, s), Mh[r, s])

        hrow, hcol, hval = [], [], []
        for f in range(self.Nf):
            if self.bc == "wall" and f in (0, self.Ne):
                e, node = (0, 0) if f == 0 else (self.Ne - 1, npp - 1)
                for s in (0, 2):
                    hrow.append(self.it(f, s)); hcol.append(self.iq(e, s, node)); hval.append(1.0)
                continue
            eL, eR = (f - 1) % self.Ne, f % self.Ne
            for s in range(NV):
                hrow += [self.it(f, s)] * 2
                hcol += [self.iq(eL, s, npp - 1), self.iq(eR, s, 0)]
                hval += [0.5, 0.5]

        self.K = sp.csr_matrix((vals, (rows, cols)), shape=(self.nq, self.nq))
        self.G = sp.csr_matrix((gval, (grow, gcol)), shape=(self.nq, self.nt))
        self.H = sp.csr_matrix((hval, (hrow, hcol)), shape=(self.nt, self.nq))
        self.L = sp.bmat([[self.K, self.G], [self.H, -sp.identity(self.nt)]], format="csr")
        self.Mbig = sp.diags(np.concatenate([self.Mdiag, self.eps * np.ones(self.nt)]), format="csr")
        self.cmax = float(np.max(cf))

    # operators / helpers --------------------------------------------------
    def forcing_vector(self, F_slow=None):
        f = self.fb.copy()
        if F_slow is not None:
            f += self.Mdiag * F_slow
        return np.concatenate([f, np.zeros(self.nt)])

    def tendency(self, q, qh=None, F_slow=None):
        """dq/dt of the fast operator (HDG limit if qh is None)."""
        qh = self.H @ q if qh is None else qh
        out = (self.K @ q + self.G @ qh + self.fb) / self.Mdiag
        return out + F_slow if F_slow is not None else out

    def project(self, fun):
        """Nodal interpolation at the GLL nodes (DGSEM initialization)."""
        q = np.zeros(self.nq)
        for e in range(self.Ne):
            F = fun(self.zpts(e, self.xg))
            for r in range(self.NV):
                self.blk(q, e, r)[:] = F[r]
        return q

    def hydrostatic_rho(self, Theta_nodal):
        """Discrete hydrostatic density for m = 0 and continuous nodal Θ (Ne, np):
        ρ is chosen such that the discrete momentum residual vanishes exactly."""
        q = np.zeros(self.nq)
        for e in range(self.Ne):
            self.blk(q, e, 2)[:] = Theta_nodal[e]
        r = self.K @ q + self.G @ (self.H @ q) + self.fb
        return np.array([self.blk(r, e, 1) / (self.g * self.J * self.wg) for e in range(self.Ne)])

    def balanced_state(self, Theta_fun):
        Th = Theta_fun(self.nodes())
        rho = self.hydrostatic_rho(Th)
        q = np.zeros(self.nq)
        for e in range(self.Ne):
            self.blk(q, e, 0)[:] = rho[e]
            self.blk(q, e, 2)[:] = Th[e]
        return q

    def energy(self, dq):
        """Discrete energy 1/2 Σ J w φ [m² + p²/c² + g χ²/(θ̄ Dθ̄/J)] of a deviation dq
        (meaningful for stable stratification; exactly non-increasing for form='split', eps=0)."""
        E = 0.0
        for e in range(self.Ne):
            rho, m, Th = (self.blk(dq, e, r) for r in range(3))
            a, th, ph = self.Za[e], self.Zth[e], self.Zphi[e]
            thz = (self.Dg @ th) / self.J
            chi = Th - th * rho
            E += 0.5 * self.J * np.sum(self.wg * ph * (m ** 2 + a * Th ** 2 / th + self.g * chi ** 2 / (th * thz)))
        return E

    def evaluate(self, q, npts=8):
        xi = np.linspace(-1, 1, npts)
        V, _ = self.basis(xi)
        Z, Q = [], []
        for e in range(self.Ne):
            Z.append(self.zpts(e, xi))
            Q.append(np.array([V @ self.blk(q, e, r) for r in range(self.NV)]))
        return np.concatenate(Z), np.concatenate(Q, axis=1)

    def l2_error(self, q, fun):
        xq, wq = Lg.leggauss(self.p + 6)
        V, _ = self.basis(xq)
        err = np.zeros(self.NV)
        for e in range(self.Ne):
            F = fun(self.zpts(e, xq))
            for r in range(self.NV):
                d = V @ self.blk(q, e, r) - F[r]
                err[r] += self.J * np.sum(wq * d * d)
        return np.sqrt(err)

    def total(self, q, r):
        """Discrete integral Σ J w_i q_i of variable r (0 mass, 1 momentum, 2 Θ)."""
        return self.J * sum(self.wg @ self.blk(q, e, r) for e in range(self.Ne))

    def initial_state(self, q):
        return np.concatenate([q, self.H @ q])

    def trace_residual(self, U):
        return np.max(np.abs(self.H @ U[: self.nq] - U[self.nq:]))

    # time integration -----------------------------------------------------
    def run(self, U0, dt, nsteps, method="dirk", F_slow=None, monitor=None, every=1):
        f = self.forcing_vector(F_slow)
        if method == "dirk":
            return self._run_dirk(U0, dt, nsteps, f, monitor, every)
        if method == "imex":
            return self._run_imex(U0, dt, nsteps, f, monitor, every)
        raise ValueError(method)

    def _run_dirk(self, U0, dt, nsteps, f, monitor, every):
        # Hairer-Wanner SDIRK4, 5 stages, L-stable, stiffly accurate
        a = np.array([[1/4, 0, 0, 0, 0],
                      [1/2, 1/4, 0, 0, 0],
                      [17/50, -1/25, 1/4, 0, 0],
                      [371/1360, -137/2720, 15/544, 1/4, 0],
                      [25/24, -49/48, 125/16, -85/12, 1/4]])
        lu = spla.splu((self.Mbig - dt * a[0, 0] * self.L).tocsc())
        U = U0.copy()
        for n in range(nsteps):
            MU = self.Mbig @ U
            R = []
            for i in range(5):
                rhs = MU + dt * a[i, i] * f
                for j in range(i):
                    rhs += dt * a[i, j] * R[j]
                Y = lu.solve(rhs)
                R.append(self.L @ Y + f)
            U = Y
            if monitor and (n + 1) % every == 0:
                monitor(n + 1, U)
        return U

    def _run_imex(self, U0, dt, nsteps, f, monitor, every):
        # ARS(2,2,2): element part explicit, trace relaxation implicit (local per face)
        gm = 1 - 1 / np.sqrt(2); dl = 1 - 1 / (2 * gm)
        AE = np.array([[0, 0, 0], [gm, 0, 0], [dl, 1 - dl, 0]])
        AI = np.array([[0, 0, 0], [0, gm, 0], [0, 1 - gm, gm]])
        nq, eps = self.nq, self.eps
        fq = f[:nq]

        def fE(U):
            out = np.zeros_like(U)
            out[:nq] = (self.K @ U[:nq] + self.G @ U[nq:] + fq) / self.Mdiag
            return out

        def fI(U):
            out = np.zeros_like(U)
            if eps > 0:
                out[nq:] = (self.H @ U[:nq] - U[nq:]) / eps
            return out

        U = U0.copy()
        for n in range(nsteps):
            FE, FI = [], []
            for i in range(3):
                Ys = U.copy()
                for j in range(i):
                    Ys += dt * (AE[i, j] * FE[j] + AI[i, j] * FI[j])
                Y = Ys.copy()
                Hq = self.H @ Y[:nq]
                if eps == 0.0:
                    Y[nq:] = Hq
                else:
                    k = dt * AI[i, i] / eps
                    Y[nq:] = (Ys[nq:] + k * Hq) / (1 + k)
                FE.append(fE(Y)); FI.append(fI(Y))
            U = Y
            if monitor and (n + 1) % every == 0:
                monitor(n + 1, U)
        return U
