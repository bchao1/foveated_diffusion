"""K-stage generalization of RALU's NT-DM (noise-timestep rescheduling).

RALU_NTDM.py hardcodes three stages: P_target/P index E[0], E[1], S[1], S[2] explicitly
and the optimizer solves for exactly 3 shifts. The 2-stage RALU-foveated variant needs an
arbitrary stage count, so eq. (8) and eq. (9) are rewritten here as sums over stages.

This is a direct transcription, not a redesign: at K=3 it must reproduce the shift/Z
constants published in pipeline_flux_RALU.py. See test_ntdm_regression.py.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.optimize import minimize


# Flow-matching PDF
def pdf_shift(t, h):
    return h / ((1 + (h - 1) * t) ** 2)


# Flow-matching CDF
def cdf_shift(t, h):
    return (h * t) / (1 + (h - 1) * t)


# Truncated PDF (eq. (7) in the paper)
def pdf_trunc_shift(t, h, s, e):
    z = cdf_shift(e, h) - cdf_shift(s, h)
    return np.where((t >= s) & (t <= e), pdf_shift(t, h) / z, 0)


def stage_starts(E, Z, K):
    """S[0] = 0; S[i+1] = E[i] / (Z(1-E[i]) + E[i]) for i < K-1."""
    S = [0.0]
    for i in range(K - 1):
        S.append(E[i] / (Z * (1 - E[i]) + E[i]))
    return S


# P_target(t) (eq. (8)), generalized to K stages
def P_target(t, h_ori, E, Z):
    K = len(E)
    S = stage_starts(E, Z, K)
    num = pdf_shift(t, h_ori)
    denom = 1.0
    for i in range(K - 1):
        w = E[i] - S[i + 1]
        num = num + w * pdf_trunc_shift(t, h_ori, S[i + 1], E[i])
        denom += w
    return num / denom


# P(t) (eq. (9)), generalized to K stages
def P(t, N, H, E, Z):
    K = len(N)
    S = stage_starts(E, Z, K)
    num = 0.0
    for i in range(K):
        num = num + N[i] * pdf_trunc_shift(t, H[i], S[i], E[i])
    return num / sum(N)


# Jensen-Shannon divergence
def js_objective(params, N, E, h_ori, grid):
    K = len(N)
    H = params[:K]
    Z = params[K]

    p_vals = P_target(grid, h_ori, E, Z)
    q_vals = P(grid, N, H, E, Z)

    epsilon = 1e-10
    p_vals = np.clip(p_vals, epsilon, None)
    q_vals = np.clip(q_vals, epsilon, None)
    p_vals /= p_vals.sum()
    q_vals /= q_vals.sum()

    return jensenshannon(p_vals, q_vals)


# NT-DM: minimize JSD(P_target(t), P(t)) over K stage shifts plus Z
def NT_DM(N, E, h_ori=3, shift_max=None, z_max=100.0):
    """Solve for per-stage shifts and Z.

    shift_max: upper bound on the per-stage shift. The original code hardcodes 10.0, which
    is calibrated to FLUX.1-dev (h_ori=3). FLUX.2 has h_ori=exp(mu)~7.56 at 1024^2/50 steps
    and pins stage-1 against that bound, so it scales with h_ori by default. Pass 10.0 to
    reproduce the original 3-stage behaviour exactly.
    """
    assert len(N) == len(E), "N and E must have the same length"
    assert abs(E[-1] - 1.0) < 1e-12, "last entry of E must be 1.0"
    K = len(N)
    if shift_max is None:
        shift_max = max(10.0, 5.0 * h_ori)
    grid = np.linspace(0, 1, 1000)
    initial_guess = [min(3.0, shift_max)] * K + [2.0]
    bounds = [(0.5, shift_max)] * K + [(2.0, z_max)]
    result = minimize(js_objective, initial_guess, args=(N, E, h_ori, grid), bounds=bounds)

    return result.x[:K], result.x[K]
