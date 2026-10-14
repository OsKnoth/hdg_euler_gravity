"""
1D hybridized DG (HDG) for the Euler equations with gravity, linearized about a
hydrostatic background, with the HDG trace ("additional") conditions treated by
RELAXATION, so the coupled (element + trace) system is integrated in time
WITHOUT eliminating the element unknowns (no static condensation / Schur complement).

----------------------------------------------------------------------------
Model
----------------------------------------------------------------------------
Background (vertical coordinate z): u0 = 0,  dp0/dz = -rho0 g,  c0^2(z) = gamma p0/rho0.
Perturbations rho', u', p'.  With m = rho0 u' the linear system becomes

    q_t + (A(z) q)_z = S q ,      q = (rho', m, p')

    A(z) q = ( m , p' , c0^2(z) m )
    S q    = ( 0 , -g rho' , -(gamma-1) g m )

(derivation: p'_t + u' p0_z + gamma p0 u'_z = 0 and (c0^2 m)_z = gamma(p0 u')_z,
 together with p0_z = -rho0 g).  rho0 itself never appears, only c0^2(z).

----------------------------------------------------------------------------
HDG discretization (element K, test v in P_p(K)^3)
----------------------------------------------------------------------------
    (v, q_t)_K = (v_z, A q)_K + (v, S q)_K - < v, F^ n >_dK
    F^ n = A(z_f) q^ n + tau_K (q - q^)            (one-sided HDG flux)
    Interface flux (flux=...):
    'lmars' (default): tau_f = |A(z_f)| = A^2/c_f = [[0,0,1/c],[0,c,0],[0,0,c]].
        With the conservativity condition this gives q^ = mean(q), and the face flux
        is exactly LMARS (Chen et al. 2013), linearized about u0 = 0:
            m*  = {m} - [p']/(2c),   p* = {p'} - c [m]/2,   F* = (m*, p*, c^2 m*)
        ([.] = right - left).  This is the exact Godunov flux of the linear system.
    'llf': tau_K = c_K * diag(tau_rho, 1, 1) (element-based Rusanov-type penalty).
    Both keep discrete hydrostatic states exactly steady (no penalty on the
    rho'-jump; LMARS only penalizes pressure jumps).

Additional conditions on the single-valued traces q^:
  interior face :  R_f = tau_L (q_L - q^) + tau_R (q_R - q^)  = 0   (conservativity)
  wall face     :  R_f = P q_int - q^                         = 0,   P = diag(1,0,1)
                   (q^ = mirror average  ->  m^ = 0, no mass/energy flux)

Relaxation:  instead of solving R_f = 0 (and eliminating q), the trace evolves as
    eps dq^/dt = W_f q - q^         (W_f q = tau-weighted average, i.e. R_f/(tau_L+tau_R))
eps -> 0 recovers the HDG scheme; eps = 0 is allowed (index-1 DAE).

Full semi-discrete system, U = (q, q^):
    [ M  0     ] dU/dt = [ K  G ] U
    [ 0  eps I ]         [ H -I ]
Time integration (both act on the coupled system, no elimination):
  * 'dirk' : stiffly accurate L-stable SDIRK4 (Hairer-Wanner), monolithic sparse LU
             of the full block matrix; valid for any eps >= 0.
  * 'imex' : IMEX ARS(2,2,2): DG part explicit, relaxation implicit.  The implicit
             stage is LOCAL per face (no global solve at all); asymptotic-preserving
             as eps -> 0.
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

    def T(self, z):
        return self.T0 - self.Gamma * z

    def c2(self, z):
        return self.gamma * self.R * self.T(z)

    def p0(self, z):
        if self.Gamma == 0.0:
            return self.p00 * np.exp(-self.g * z / (self.R * self.T0))
        return self.p00 * (self.T(z) / self.T0) ** (self.g / (self.R * self.Gamma))

    def rho0(self, z):
        return self.p0(z) / (self.R * self.T(z))


# ---------------------------------------------------------------------------
class HDGRelax:
    NV = 3  # rho', m, p'

    def __init__(self, Ne, p, zmin, zmax, atm, bc="wall", eps=0.0, tau_scale=1.0, tau_rho=0.0, flux="lmars"):
        self.Ne, self.p, self.np = Ne, p, p + 1
        self.zmin, self.zmax, self.atm, self.bc = zmin, zmax, atm, bc
        self.eps = float(eps)
        self.h = (zmax - zmin) / Ne
        self.xe = zmin + self.h * np.arange(Ne + 1)          # element vertices
        self.Nf = Ne + 1 if bc == "wall" else Ne
        self.nq = Ne * self.NV * self.np
        self.nt = self.Nf * self.NV
        self.tau_scale = tau_scale
        self.tw = np.array([tau_rho, 1.0, 1.0])   # per-variable penalty weights (llf)
        assert flux in ("lmars", "llf")
        self.flux = flux
        self._build()

    # dof indices -----------------------------------------------------------
    def iq(self, e, r, i):
        return (e * self.NV + r) * self.np + i

    def it(self, f, s):
        return f * self.NV + s

    def face_R(self, e):
        return e + 1 if self.bc == "wall" else (e + 1) % self.Ne

    # basis helpers ---------------------------------------------------------
    def basis(self, xi):
        """Legendre P_i and dP_i/dxi at points xi -> arrays (len(xi), np)."""
        V = np.zeros((len(xi), self.np)); D = np.zeros_like(V)
        for i in range(self.np):
            c = np.zeros(i + 1); c[i] = 1.0
            V[:, i] = Lg.legval(xi, c)
            D[:, i] = Lg.legval(xi, Lg.legder(c))
        return V, D

    # assembly --------------------------------------------------------------
    def _build(self):
        NV, npp, h, atm = self.NV, self.np, self.h, self.atm
        g, gam = atm.g, atm.gamma
        xq, wq = Lg.leggauss(self.p + 3)
        V, D = self.basis(xq)
        phiR = np.ones(npp)
        phiL = (-1.0) ** np.arange(npp)
        Mref = np.diag(2.0 / (2 * np.arange(npp) + 1))     # int_{-1}^{1} P_i P_j
        Sder = D.T @ (wq[:, None] * V)                      # int P_i' P_j   (jacobians cancel)

        self.Mdiag = np.tile(np.diag(Mref) * h / 2, self.Ne * NV)
        self.tau = np.zeros(self.Ne)

        K = sp.lil_matrix((self.nq, self.nq))
        G = sp.lil_matrix((self.nq, self.nt))
        H = sp.lil_matrix((self.nt, self.nq))

        def Aface(z):
            return np.array([[0, 1, 0], [0, 0, 1], [0, atm.c2(z), 0]], float)

        def Tpen(z, tau):
            """3x3 penalty matrix at a face of an element with element penalty tau."""
            if self.flux == "lmars":
                cf = np.sqrt(atm.c2(z))
                return self.tau_scale * np.array([[0, 0, 1 / cf], [0, cf, 0], [0, 0, cf]])
            return tau * np.diag(self.tw)

        for e in range(self.Ne):
            zl, zr = self.xe[e], self.xe[e + 1]
            zq = 0.5 * (zl + zr) + 0.5 * h * xq
            Sc2 = D.T @ ((wq * atm.c2(zq))[:, None] * V)   # int P_i' c0^2 P_j
            tau = self.tau_scale * np.sqrt(max(atm.c2(zl), atm.c2(zr)))
            self.tau[e] = tau
            TL, TR = Tpen(zl, tau), Tpen(zr, tau)
            for i in range(npp):
                for j in range(npp):
                    # volume flux terms (v_z, A q)
                    K[self.iq(e, 0, i), self.iq(e, 1, j)] += Sder[i, j]
                    K[self.iq(e, 1, i), self.iq(e, 2, j)] += Sder[i, j]
                    K[self.iq(e, 2, i), self.iq(e, 1, j)] += Sc2[i, j]
                    # gravity sources (v, S q)
                    K[self.iq(e, 1, i), self.iq(e, 0, j)] += -g * h / 2 * Mref[i, j]
                    K[self.iq(e, 2, i), self.iq(e, 1, j)] += -(gam - 1) * g * h / 2 * Mref[i, j]
                    # T (q - q^) part of the face flux, element side
                    for r in range(NV):
                        for s_ in range(NV):
                            v = TR[r, s_] * phiR[i] * phiR[j] + TL[r, s_] * phiL[i] * phiL[j]
                            if v != 0.0:
                                K[self.iq(e, r, i), self.iq(e, s_, j)] += -v
            fl, fr = e, self.face_R(e)
            AL, AR = Aface(zl), Aface(zr)
            for i in range(npp):
                for r in range(NV):
                    for s in range(NV):
                        # right face, n=+1 :  -phi_i(1) [A q^ - T q^]
                        G[self.iq(e, r, i), self.it(fr, s)] += -phiR[i] * (AR[r, s] - TR[r, s])
                        # left face,  n=-1 :  -phi_i(-1) [-A q^ - T q^]
                        G[self.iq(e, r, i), self.it(fl, s)] += -phiL[i] * (-AL[r, s] - TL[r, s])

        # trace (additional) conditions: q^_target = H q
        for f in range(self.Nf):
            if self.bc == "wall" and f in (0, self.Ne):
                e, phi = (0, phiL) if f == 0 else (self.Ne - 1, phiR)
                for s in (0, 2):                            # P = diag(1,0,1): m^ = 0
                    for j in range(npp):
                        H[self.it(f, s), self.iq(e, s, j)] += phi[j]
                continue
            eL = (f - 1) % self.Ne
            eR = f % self.Ne
            # lmars: same T on both sides -> plain mean; llf: tau-weighted mean
            tL, tR = (1.0, 1.0) if self.flux == "lmars" else (self.tau[eL], self.tau[eR])
            for s in range(NV):
                for j in range(npp):
                    H[self.it(f, s), self.iq(eL, s, j)] += tL / (tL + tR) * phiR[j]
                    H[self.it(f, s), self.iq(eR, s, j)] += tR / (tL + tR) * phiL[j]

        self.K, self.G, self.H = K.tocsr(), G.tocsr(), H.tocsr()
        self.L = sp.bmat([[self.K, self.G], [self.H, -sp.identity(self.nt)]], format="csr")
        self.Mbig = sp.diags(np.concatenate([self.Mdiag, self.eps * np.ones(self.nt)]), format="csr")

    # projections / evaluation ---------------------------------------------
    def project(self, fun, nquad=None):
        """L2 projection of fun(z) -> (rho', m, p') onto the DG space."""
        xq, wq = Lg.leggauss(nquad or self.p + 5)
        V, _ = self.basis(xq)
        Minv = (2 * np.arange(self.np) + 1) / 2.0
        q = np.zeros(self.nq)
        for e in range(self.Ne):
            zq = 0.5 * (self.xe[e] + self.xe[e + 1]) + 0.5 * self.h * xq
            F = fun(zq)
            for r in range(self.NV):
                q[self.iq(e, r, 0): self.iq(e, r, 0) + self.np] = Minv * (V.T @ (wq * F[r]))
        return q

    def evaluate(self, q, npts=8):
        xi = np.linspace(-1, 1, npts)
        V, _ = self.basis(xi)
        Z, Q = [], []
        for e in range(self.Ne):
            Z.append(0.5 * (self.xe[e] + self.xe[e + 1]) + 0.5 * self.h * xi)
            Q.append(np.array([V @ q[self.iq(e, r, 0): self.iq(e, r, 0) + self.np] for r in range(self.NV)]))
        return np.concatenate(Z), np.concatenate(Q, axis=1)

    def l2_error(self, q, fun):
        xq, wq = Lg.leggauss(self.p + 6)
        V, _ = self.basis(xq)
        err = np.zeros(self.NV)
        for e in range(self.Ne):
            zq = 0.5 * (self.xe[e] + self.xe[e + 1]) + 0.5 * self.h * xq
            F = fun(zq)
            for r in range(self.NV):
                d = V @ q[self.iq(e, r, 0): self.iq(e, r, 0) + self.np] - F[r]
                err[r] += self.h / 2 * np.sum(wq * d * d)
        return np.sqrt(err)

    def total_mass(self, q):
        return self.h * sum(q[self.iq(e, 0, 0)] for e in range(self.Ne))

    def consistent_trace(self, q):
        return self.H @ q

    def trace_residual(self, U):
        q, qh = U[: self.nq], U[self.nq:]
        return np.max(np.abs(self.H @ q - qh))

    # time integration ------------------------------------------------------
    def run(self, U0, dt, nsteps, method="dirk", monitor=None, every=1):
        if method == "dirk":
            return self._run_dirk(U0, dt, nsteps, monitor, every)
        if method == "imex":
            return self._run_imex(U0, dt, nsteps, monitor, every)
        raise ValueError(method)

    def _run_dirk(self, U0, dt, nsteps, monitor, every):
        # Hairer-Wanner SDIRK4, 5 stages, L-stable, stiffly accurate (U^{n+1} = Y_5)
        a = np.array([[1/4, 0, 0, 0, 0],
                      [1/2, 1/4, 0, 0, 0],
                      [17/50, -1/25, 1/4, 0, 0],
                      [371/1360, -137/2720, 15/544, 1/4, 0],
                      [25/24, -49/48, 125/16, -85/12, 1/4]])
        lu = spla.splu((self.Mbig - dt * a[0, 0] * self.L).tocsc())   # full coupled matrix
        U = U0.copy()
        for n in range(nsteps):
            MU = self.Mbig @ U
            LY = []
            for i in range(5):
                rhs = MU + dt * sum(a[i, j] * LY[j] for j in range(i))
                Y = lu.solve(rhs)
                LY.append(self.L @ Y)
            U = Y
            if monitor and (n + 1) % every == 0:
                monitor(n + 1, U)
        return U

    def _run_imex(self, U0, dt, nsteps, monitor, every):
        # ARS(2,2,2). Explicit: element DG operator. Implicit: trace relaxation (local).
        gm = 1 - 1 / np.sqrt(2); dl = 1 - 1 / (2 * gm)
        AE = np.array([[0, 0, 0], [gm, 0, 0], [dl, 1 - dl, 0]])
        AI = np.array([[0, 0, 0], [0, gm, 0], [0, 1 - gm, gm]])
        nq, eps = self.nq, self.eps

        def fE(U):  # explicit part, nonzero only in q-rows
            out = np.zeros_like(U)
            out[:nq] = (self.K @ U[:nq] + self.G @ U[nq:]) / self.Mdiag
            return out

        def fI(U):  # relaxation, nonzero only in trace rows
            out = np.zeros_like(U)
            out[nq:] = (self.H @ U[:nq] - U[nq:]) / eps if eps > 0 else 0.0
            return out

        U = U0.copy()
        for n in range(nsteps):
            FE, FI, Y = [], [], None
            for i in range(3):
                Ystar = U + dt * sum(AE[i, j] * FE[j] + AI[i, j] * FI[j] for j in range(i))
                # implicit solve  Y = Ystar + dt*AI[i,i]*fI(Y): q-part explicit, trace part local
                Y = Ystar.copy()
                Hq = self.H @ Y[:nq]
                if eps == 0.0:
                    Y[nq:] = Hq                       # exact HDG constraint
                else:
                    k = dt * AI[i, i] / eps
                    Y[nq:] = (Ystar[nq:] + k * Hq) / (1 + k)
                FE.append(fE(Y))
                FI.append(fI(Y) if eps > 0 else np.zeros_like(Y))
            U = Y                                     # stiffly accurate
            if monitor and (n + 1) % every == 0:
                monitor(n + 1, U)
        return U
