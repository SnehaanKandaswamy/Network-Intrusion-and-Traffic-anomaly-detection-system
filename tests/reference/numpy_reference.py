"""Independent NumPy implementation of the subspace method (reference only).

Deliberately uses a different numerical route from the C++ code: an SVD of the
centred matrix instead of covariance + symmetric eigensolver.
  Y_c = U S V^T   ->  principal axes V, covariance eigenvalues S^2 / (t-1)
"""
import numpy as np
from statistics import NormalDist


def fit(Y, r_method="sigma", sigma=3.0, fixed_r=4, alpha=0.001):
    Y = np.asarray(Y, dtype=float)
    t, m = Y.shape
    mean = Y.mean(axis=0)
    Yc = Y - mean
    _, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    lam = np.zeros(m)
    lam[: len(s)] = s**2 / (t - 1)
    V = Vt.T
    if r_method == "fixed":
        r = fixed_r
    else:
        rank = int(np.sum(lam > 1e-12 * lam[0]))
        r = rank
        for i in range(rank):
            u = Yc @ V[:, i]
            dev = np.abs(u - u.mean()).max() / u.std(ddof=1)
            if dev > sigma:
                r = i
                break
    r = min(max(r, 1), m - 1)
    return dict(mean=mean, eigenvalues=lam, P=V[:, :r], r=r, delta2=q_statistic(lam, r, alpha))


def q_statistic(lam, r, alpha):
    res = np.clip(lam[r:], 0, None)
    phi1, phi2, phi3 = (np.sum(res**i) for i in (1, 2, 3))
    total = np.clip(lam, 0, None).sum()
    if phi1 <= 1e-12 * total:
        return 1e-12 * total
    c = NormalDist().inv_cdf(1 - alpha)
    h0 = 1 - 2 * phi1 * phi3 / (3 * phi2**2)
    if h0 > 1e-6:
        term = c * np.sqrt(2 * phi2 * h0**2) / phi1 + 1 + phi2 * h0 * (h0 - 1) / phi1**2
        if term > 0:
            return phi1 * term ** (1 / h0)
    g, h = phi2 / phi1, phi1**2 / phi2
    return g * h * (1 - 2 / (9 * h) + c * np.sqrt(2 / (9 * h))) ** 3


def spe(Y, model):
    Yc = np.asarray(Y, dtype=float) - model["mean"]
    resid = Yc - (Yc @ model["P"]) @ model["P"].T
    return np.sum(resid**2, axis=1)
