"""
Low-rank matrix completion via Alternating Least Squares (ALS).

Used to fill unobserved entries in the K-sparse cross-evaluation score
matrix.  The model:

    S_ij ≈ Σ_k u_ik v_jk  +  b_i  +  c_j  +  μ

where rank r is a small constant (default 3).  ALS alternates between
fixing U and solving for V (plus biases) and vice versa.

Reference: Keshavan, Montanari & Oh (2010) — noisy matrix completion
with O(rN) observations suffices for approximate recovery.
"""

import numpy as np
from typing import Optional, Tuple


def als_matrix_completion(
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int = 3,
    max_iter: int = 30,
    tol: float = 1e-4,
    reg: float = 0.1,
) -> np.ndarray:
    """Complete a partially-observed matrix using ALS with biases.

    Parameters
    ----------
    observed : (N, N) array
        Score matrix.  Entries where mask==0 are unobserved (value ignored).
    mask : (N, N) binary array
        1 = observed, 0 = unobserved.
    rank : int
        Latent rank for the UV decomposition (excluding biases).
    max_iter : int
        Maximum ALS iterations.
    tol : float
        Early stop when RMSE improvement < tol.
    reg : float
        L2 regularisation coefficient λ for U, V rows.

    Returns
    -------
    completed : (N, N) array
        Fully-filled matrix with observed entries unchanged and unobserved
        entries imputed.
    """
    N = observed.shape[0]
    assert observed.shape == (N, N) and mask.shape == (N, N)

    # If fully observed, nothing to do.
    if mask.all():
        return observed.copy()

    # ── Initialise ────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    # Global mean over observed entries
    obs_vals = observed[mask.astype(bool)]
    mu = obs_vals.mean() if len(obs_vals) > 0 else 0.0

    # Row and column biases initialised to marginal deviations
    b = np.zeros(N)  # row bias (answer quality)
    c = np.zeros(N)  # col bias (rubric strictness)
    for i in range(N):
        row_mask = mask[i].astype(bool)
        if row_mask.any():
            b[i] = observed[i, row_mask].mean() - mu
    for j in range(N):
        col_mask = mask[:, j].astype(bool)
        if col_mask.any():
            c[j] = observed[col_mask, j].mean() - mu

    U = rng.normal(0, 0.1, (N, rank))
    V = rng.normal(0, 0.1, (N, rank))

    prev_rmse = float("inf")

    for iteration in range(max_iter):
        # Residual after biases
        R = observed - mu - b[:, None] - c[None, :]

        # ── Fix V, solve for each row of U ────────────────────────
        for i in range(N):
            idx = np.where(mask[i].astype(bool))[0]
            if len(idx) == 0:
                continue
            V_sub = V[idx]  # (|idx|, rank)
            r_sub = R[i, idx]  # (|idx|,)
            # Solve (V_sub^T V_sub + λI) u_i = V_sub^T r_sub
            A = V_sub.T @ V_sub + reg * np.eye(rank)
            U[i] = np.linalg.solve(A, V_sub.T @ r_sub)

        # ── Fix U, solve for each column of V ─────────────────────
        for j in range(N):
            idx = np.where(mask[:, j].astype(bool))[0]
            if len(idx) == 0:
                continue
            U_sub = U[idx]  # (|idx|, rank)
            r_sub = R[idx, j]  # (|idx|,)
            A = U_sub.T @ U_sub + reg * np.eye(rank)
            V[j] = np.linalg.solve(A, U_sub.T @ r_sub)

        # ── Update biases ─────────────────────────────────────────
        pred_uv = U @ V.T
        for i in range(N):
            idx = np.where(mask[i].astype(bool))[0]
            if len(idx) > 0:
                b[i] = (observed[i, idx] - mu - c[idx] - pred_uv[i, idx]).mean()
        for j in range(N):
            idx = np.where(mask[:, j].astype(bool))[0]
            if len(idx) > 0:
                c[j] = (observed[idx, j] - mu - b[idx] - pred_uv[idx, j]).mean()

        # ── Convergence check ─────────────────────────────────────
        full_pred = mu + b[:, None] + c[None, :] + pred_uv
        residuals = (observed - full_pred) * mask
        rmse = np.sqrt((residuals ** 2).sum() / max(mask.sum(), 1))

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    # ── Build completed matrix ────────────────────────────────────────
    completed = mu + b[:, None] + c[None, :] + U @ V.T
    # Clamp to valid score range [0, 10]
    completed = np.clip(completed, 0.0, 10.0)
    # Keep observed entries exact
    completed[mask.astype(bool)] = observed[mask.astype(bool)]

    return completed
