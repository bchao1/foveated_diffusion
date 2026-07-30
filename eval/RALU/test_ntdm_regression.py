"""Gate for the K-stage NT-DM generalization.

At K=3 the generalized solver must reproduce (a) the original 3-stage RALU_NTDM.NT_DM
output exactly, and (b) the shift/Z constants published in pipeline_flux_RALU.set_params.
Only if both hold is the K=2 solve trustworthy.
"""
import numpy as np

from RALU_NTDM import NT_DM as NT_DM_orig
from RALU_NTDM_general import NT_DM as NT_DM_gen

# Published constants, pipeline_flux_RALU.py:46-65 (h_ori=3 for FLUX.1-dev)
PUBLISHED = {
    4: dict(N=[5, 6, 7], e=[0.3, 0.45, 1.0], shift=[5.019, 2.589, 2.226], Z=6.317),
    7: dict(N=[2, 3, 5], e=[0.2, 0.3, 1.0], shift=[8.141, 2.856, 2.193], Z=6.261),
}


def main():
    ok = True

    for level, cfg in PUBLISHED.items():
        N, e = cfg["N"], cfg["e"]
        h_gen, z_gen = NT_DM_gen(N, e, h_ori=3)
        h_ori_impl, z_ori_impl = NT_DM_orig(N, e, h_ori=3)

        # (a) generalized == original implementation
        d_impl = max(np.max(np.abs(h_gen - h_ori_impl)), abs(z_gen - z_ori_impl))
        # (b) generalized == published constants
        d_pub = max(
            np.max(np.abs(h_gen - np.array(cfg["shift"]))),
            abs(z_gen - cfg["Z"]),
        )

        print(f"level {level}: N={N} e={e}")
        print(f"  generalized shift={np.round(h_gen, 3).tolist()}  Z={z_gen:.3f}")
        print(f"  original    shift={np.round(h_ori_impl, 3).tolist()}  Z={z_ori_impl:.3f}")
        print(f"  published   shift={cfg['shift']}  Z={cfg['Z']}")
        print(f"  max|gen-orig|={d_impl:.2e}   max|gen-published|={d_pub:.2e}")

        if d_impl > 1e-6:
            print("  FAIL: generalized solver disagrees with the original 3-stage code")
            ok = False
        if d_pub > 5e-3:
            print("  FAIL: does not reproduce published constants")
            ok = False
        print()

    # K=2 solve for the RALU-foveated variant (h_ori placeholder; real runs use exp(mu))
    for e1 in (0.2, 0.3, 0.4):
        N2 = [15, 35]
        e2 = [e1, 1.0]
        h2, z2 = NT_DM_gen(N2, e2, h_ori=3)
        print(f"K=2 N={N2} e={e2}: shift={np.round(h2, 3).tolist()} Z={z2:.3f}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
