"""Cloud EHR environment and PSO controller (Chapter 3, Sections 3.4 and 3.7).

Every function cites the Chapter 3 equation it implements. Calibrated
parameters (Tables 3.6, 3.7 and phi) are passed in through a `Params`
object so they can be re-fitted from the workload data and swapped in
without touching the model code.
"""
from dataclasses import dataclass, field
import numpy as np

DELTA_S = 15 * 60  # interval length in seconds (Delta = 15 min)


@dataclass
class Params:
    # Demand mappings d = alpha + beta*y (Table 3.6)
    alpha: dict = field(default_factory=lambda: dict(cpu=22.738, mem=21.639, bw=2.590, users=10.149, enc=-0.252))
    beta: dict = field(default_factory=lambda: dict(cpu=0.1328, mem=0.1223, bw=0.0768, users=0.1703, enc=0.0743))
    # Response-time surrogate (Table 3.7), ms
    theta0: float = 60.15
    theta: tuple = (1.28, 0.97, 23.74)  # cpu, mem, bw
    rho_clip: float = 0.95
    # Erlang-B offered-load scaling (Eq. 3.10)
    phi: float = 0.454
    # Reference instance (Section 3.4.1)
    x0: tuple = (8, 32.0, 100.0, 50)
    c0: float = 8.0
    m0: float = 32.0
    # Prices (Eq. 3.21)
    prices: tuple = (1.0, 0.25, 0.05, 0.02)
    # Optimisation constants (Table 3.11)
    w: tuple = (0.40, 0.30, 0.15, 0.15)  # R, C, B, U
    R_sla: float = 100.0
    eps_B: float = 0.01
    rho_max: float = 0.85
    rho_star: float = 0.65
    C_ref: float = 22.0
    C_max: float = 60.0
    x_min: tuple = (2, 4.0, 10.0, 10)
    x_max: tuple = (64, 256.0, 300.0, 200)
    penalty: float = 1e3
    z: float = 1.645  # Eq. 3.14


# ---------------------------------------------------------------- queueing
def erlang_b_table(A, kmax):
    """B(n, A) for n = 0..kmax by the stable recursion of Eq. 3.9."""
    B = np.empty(kmax + 1)
    B[0] = 1.0
    for n in range(1, kmax + 1):
        B[n] = A * B[n - 1] / (n + A * B[n - 1])
    return B


def erlang_b(k, A):
    return erlang_b_table(A, int(k))[int(k)]


def erlang_b_vec(k, A):
    """Vectorised over arrays of A for a single integer k (used in calibration)."""
    A = np.asarray(A, dtype=float)
    B = np.ones_like(A)
    for n in range(1, int(k) + 1):
        B = A * B / (n + A * B)
    return B


def erlang_c(c, a):
    """Probability of waiting in M/M/c (Eq. 3.6); requires a < c."""
    rho = a / c
    if rho >= 1:
        return 1.0
    terms = np.cumprod(np.r_[1.0, a / np.arange(1, c)])  # a^n/n!, n = 0..c-1
    last = terms[-1] * a / c / (1 - rho)  # a^c/c! * 1/(1-rho)
    return last / (terms.sum() + last)


def availability(mtbf, mttr):
    """Two-state Markov availability (Eq. 3.12)."""
    return mtbf / (mtbf + mttr)


# ---------------------------------------------------------- demand mapping
def demand(y_hat, sigma_e, tau_svc, P: Params, inflate=True):
    """Map a request forecast to resource demand (Eqs. 3.14-3.15).

    Returns D_c (vCPU), D_m (GB), D_b (Mbps), users u, encounters n.
    """
    y = y_hat + P.z * sigma_e if inflate else y_hat
    a, b = P.alpha, P.beta
    n = a["enc"] + b["enc"] * y
    Dc = P.c0 / 100 * (a["cpu"] + b["cpu"] * y) + max(n, 0.0) * tau_svc / DELTA_S
    Dm = P.m0 / 100 * (a["mem"] + b["mem"] * y)
    Db = a["bw"] + b["bw"] * y
    u = a["users"] + b["users"] * y
    return np.array([Dc, Dm, Db]), u, n


def g(rho, clip):
    r = np.minimum(rho, clip)
    return r / (1 - r)  # Eq. 3.17


def response_time(rho, P: Params):
    """Calibrated surrogate R_hat (Eq. 3.18); rho has shape (..., 3)."""
    return P.theta0 + (np.asarray(P.theta) * g(rho, P.rho_clip)).sum(-1)


def cost(X, P: Params):
    return X @ np.asarray(P.prices)  # Eq. 3.21


# ------------------------------------------------------------- objective
def evaluate(X, D, Btab, P: Params):
    """Objective, penalised objective and diagnostics for allocations X (S x 4).

    D: demand vector [Dc, Dm, Db]; Btab: Erlang-B table for this interval's
    offered load, indexed by pool size k.
    """
    X = np.atleast_2d(X)
    rho = D[None, :] / X[:, :3]  # Eq. 3.16
    R = response_time(rho, P)
    C = cost(X, P)
    PB = Btab[X[:, 3].astype(int)]
    wR, wC, wB, wU = P.w
    U = ((rho - P.rho_star) ** 2).sum(1) / (3 * P.rho_star ** 2)
    J = wR * R / P.R_sla + wC * C / P.C_ref + wB * PB / P.eps_B + wU * U  # Eq. 3.42
    gc = np.column_stack([
        rho[:, 0] - P.rho_max, rho[:, 1] - P.rho_max, rho[:, 2] - P.rho_max,
        R / P.R_sla - 1, PB / P.eps_B - 1, C / P.C_max - 1,
    ])  # Eq. 3.43
    Jp = J + P.penalty * (np.maximum(0, gc) ** 2).sum(1)  # Eq. 3.45
    return dict(J=J, Jp=Jp, R=R, C=C, PB=PB, rho=rho, g=gc, feasible=(gc <= 0).all(1))


# -------------------------------------------------------------------- PSO
@dataclass
class PSOConfig:
    S: int = 30
    iters: int = 60
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    vclamp: float = 0.20
    stall_tol: float = 1e-3
    stall_iters: int = 10


def _round(X):
    X[:, 0] = np.round(X[:, 0])
    X[:, 3] = np.round(X[:, 3])
    return X


def pso(D, A, P: Params, cfg: PSOConfig = PSOConfig(), rng=None, last_feasible=None, trace=False):
    """Algorithm 3.1: one PSO allocation for one facility interval."""
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = np.asarray(P.x_min, float), np.asarray(P.x_max, float)
    vmax = cfg.vclamp * (hi - lo)
    Btab = erlang_b_table(A, int(hi[3]))
    X = _round(rng.uniform(lo, hi, (cfg.S, 4)))
    V = rng.uniform(-vmax, vmax, (cfg.S, 4))
    f = evaluate(X, D, Btab, P)["Jp"]
    Pb, Pf = X.copy(), f.copy()
    gi = np.argmin(Pf)
    G, Gf = Pb[gi].copy(), Pf[gi]
    hist, stall, it = [Gf], 0, 0
    for it in range(1, cfg.iters + 1):
        r1, r2 = rng.random((cfg.S, 4)), rng.random((cfg.S, 4))
        V = np.clip(cfg.w * V + cfg.c1 * r1 * (Pb - X) + cfg.c2 * r2 * (G - X), -vmax, vmax)  # Eq. 3.46
        X = _round(np.clip(X + V, lo, hi))  # Eq. 3.47
        f = evaluate(X, D, Btab, P)["Jp"]
        better = f < Pf
        Pb[better], Pf[better] = X[better], f[better]  # Eq. 3.48
        gi = np.argmin(Pf)
        prev = Gf
        G, Gf = Pb[gi].copy(), Pf[gi]
        hist.append(Gf)
        stall = stall + 1 if abs(prev - Gf) < cfg.stall_tol else 0
        if stall >= cfg.stall_iters:
            break
    ev = evaluate(G, D, Btab, P)
    out = dict(x=G, iters=it, feasible=bool(ev["feasible"][0]), Jp=float(Gf))
    if not out["feasible"] and last_feasible is not None:
        out["x"] = np.asarray(last_feasible, float)
        out["fallback"] = True
    if trace:
        out["history"] = np.array(hist)
    return out


# --------------------------------------------------------- PSO stability
def pso_stability(w, c1, c2):
    """First-order (Eq. 3.51) and second-order (Eq. 3.52/3.53) conditions."""
    phibar = (c1 + c2) / 2
    first = abs(w) < 1 and 0 < phibar < 2 * (1 + w)
    poli = 24 * (1 - w ** 2) / (7 - 5 * w)
    Ea = 1 + w - phibar
    Ea2 = Ea ** 2 + (c1 ** 2 + c2 ** 2) / 12
    M = np.array([[Ea2, -2 * w * Ea, w ** 2], [Ea, -w, 0], [1, 0, 0]])
    sr = max(abs(np.linalg.eigvals(M)))
    return dict(first_order_ok=first, first_order_limit_sum=4 * (1 + w),
                poli_limit_sum=poli, spectral_radius=sr, second_order_ok=sr < 1)


def numeric_boundary(w, tol=1e-6):
    """c1 + c2 (with c1 = c2) at which the second-moment spectral radius reaches 1."""
    lo, hi = 0.0, 4 * (1 + w)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if pso_stability(w, mid / 2, mid / 2)["spectral_radius"] < 1:
            lo = mid
        else:
            hi = mid
    return lo
