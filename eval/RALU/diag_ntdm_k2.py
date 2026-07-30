"""Is the K=2 NT-DM Z-blowup a real optimum or a flat/degenerate objective?

Scans JSD vs Z (with the stage shifts re-optimized at each Z) and reports the implied
re-noising coefficients alpha, beta and stage start s1.
"""
import numpy as np
from scipy.optimize import minimize

from RALU_NTDM_general import P_target, P
from scipy.spatial.distance import jensenshannon

GRID = np.linspace(0, 1, 1000)


def jsd(N, H, E, Z, h_ori):
    p = np.clip(P_target(GRID, h_ori, E, Z), 1e-10, None)
    q = np.clip(P(GRID, N, H, E, Z), 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    return jensenshannon(p, q)


def best_H_at_Z(N, E, Z, h_ori):
    K = len(N)
    f = lambda H: jsd(N, H, E, Z, h_ori)
    r = minimize(f, [3.0] * K, bounds=[(0.5, 10.0)] * K)
    return r.x, r.fun


def coeffs(e0, Z):
    """RALU transition coefficients, pipeline_flux_RALU.py:79-82."""
    s1 = e0 / (Z * (1 - e0) + e0)
    alpha = 1 - s1
    beta = 1 / (Z * (1 - e0) + e0)
    return s1, alpha, beta


def main():
    h_ori = 3.0
    for e0 in (0.2, 0.3, 0.4):
        N, E = [15, 35], [e0, 1.0]
        print(f"\n=== K=2, N={N}, e=[{e0}, 1.0], h_ori={h_ori} ===")
        print(f"{'Z':>8} {'JSD':>12} {'shift':>22} {'s1':>8} {'alpha':>8} {'beta':>8}")
        for Z in (2.0, 3.0, 4.0, 6.0, 6.317, 8.0, 12.0, 20.0, 50.0, 100.0):
            H, v = best_H_at_Z(N, E, Z, h_ori)
            s1, a, b = coeffs(e0, Z)
            print(f"{Z:8.3f} {v:12.6f} {str(np.round(H,3).tolist()):>22} {s1:8.4f} {a:8.4f} {b:8.4f}")

    # Same scan for the published 3-stage level-4 config, as a reference for what a
    # well-conditioned objective looks like.
    print("\n=== K=3 reference, level 4 (published Z=6.317) ===")
    N, E = [5, 6, 7], [0.3, 0.45, 1.0]
    print(f"{'Z':>8} {'JSD':>12} {'shift':>28}")
    for Z in (2.0, 4.0, 6.317, 10.0, 20.0, 50.0, 100.0):
        H, v = best_H_at_Z(N, E, Z, h_ori)
        print(f"{Z:8.3f} {v:12.6f} {str(np.round(H,3).tolist()):>28}")


if __name__ == "__main__":
    main()
