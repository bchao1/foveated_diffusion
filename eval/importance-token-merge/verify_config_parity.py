"""
Runtime parity check: does the ToMe arm receive the same conditioning as our foveated arm?

Static reading of the two pipelines is not enough — they are separate `__call__`
implementations with separate defaults, and the RALU port showed how easily those drift.
This intercepts the real DiT calls and records what actually arrives:

  * cfg_scale, and the resulting number of DiT calls per step (CFG structure)
  * embedded_guidance as a scalar, per call
  * timestep as a scalar, per call, plus the full sigma schedule
  * text conditioning: prompt_embeds / txt_ids shapes and checksums for the
    positive and negative branches

Run one arm per process, then compare:

    python verify_config_parity.py --arm tome --out parity/tome.json
    python verify_config_parity.py --arm ours --out parity/ours.json
    python verify_config_parity.py --compare parity/

Nothing here imports or mutates DiffSynth.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROMPT = ("A lion lounges calmly, displaying its powerful physique, with a soft focus "
          "highlighting its majestic mane and intense gaze.")
NOFOV_LORA = "/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors"
FOV_LORA = "/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors"


def _sum(t):
    return None if t is None else round(float(t.double().sum().item()), 4)


def _shape(t):
    return None if t is None else list(t.shape)


def record(dit, log):
    """Wrap the bound forward to capture what each DiT call actually receives."""
    original = dit.forward

    def traced(*a, **kw):
        log.append({
            "guidance": None if kw.get("guidance") is None else round(float(kw["guidance"].flatten()[0]), 6),
            "timestep": None if kw.get("timestep") is None else round(float(torch.as_tensor(kw["timestep"]).flatten()[0]), 6),
            "hidden_states_shape": _shape(kw.get("hidden_states")),
            "encoder_hidden_states_shape": _shape(kw.get("encoder_hidden_states")),
            "encoder_hidden_states_sum": _sum(kw.get("encoder_hidden_states")),
            "txt_ids_shape": _shape(kw.get("txt_ids")),
            "img_ids_shape": _shape(kw.get("img_ids")),
        })
        return original(*a, **kw)

    dit.forward = traced
    return original


def run(a):
    from inference import create_foveation_mask, create_foveation_mask_full_res

    log = []
    if a.arm == "tome":
        from tome_fov_inference import load_pipeline
        pipe = load_pipeline(argparse.Namespace(
            model_id=a.model_id, model_path=a.model_path, lora_checkpoint=NOFOV_LORA))
        mask = create_foveation_mask(a.height, a.width, center=(0, 0), r=a.radius,
                                     shape=a.mask, device=pipe.device, lr_factor=a.lr_factor)
        from flux2_tome import apply_patch
        apply_patch(pipe.dit, mask, a.height // 16, a.width // 16,
                    lr_factor=a.lr_factor, merge_mlp=True)
        record(pipe.dit, log)
        # Exactly the call tome_fov_inference.py makes.
        pipe(prompt=PROMPT, height=a.height, width=a.width, seed=a.seed,
             rand_device="cuda", num_inference_steps=a.steps, cfg_scale=a.cfg)
    else:
        from inference import load_pipeline as load_fov
        pipe = load_fov(argparse.Namespace(model_id=a.model_id), use_foveated_pipeline=True)
        pipe.load_lora(pipe.dit, FOV_LORA)
        mask = create_foveation_mask(a.height, a.width, center=(0, 0), r=a.radius,
                                     shape=a.mask, device=pipe.device, lr_factor=a.lr_factor)
        full_mask = create_foveation_mask_full_res(a.height, a.width, center=(0, 0),
                                                   r=a.radius, shape=a.mask, device=pipe.device)
        record(pipe.dit, log)
        # Exactly the call eval/inference.py run_ours makes.
        pipe(prompt=PROMPT, height=a.height, width=a.width, seed=a.seed,
             rand_device="cuda", num_inference_steps=a.steps, cfg_scale=a.cfg,
             foveation_mask=mask, full_res_foveation_mask=full_mask,
             token_upsampler=None, token_downsampler=None,
             decode_mode="merge", prediction_type="clean", tokenAE_training=False,
             soft_foveation_blend=True, lr_downsample_factor=a.lr_factor)

    sched = pipe.scheduler
    result = {
        "arm": a.arm,
        "cfg_scale": a.cfg,
        "steps": a.steps,
        "dit_calls": len(log),
        "calls_per_step": len(log) / a.steps,
        "timesteps": [round(float(t), 4) for t in sched.timesteps.flatten().tolist()],
        "sigmas": [round(float(s), 6) for s in torch.as_tensor(sched.sigmas).flatten().tolist()],
        "guidance_values": sorted({c["guidance"] for c in log}),
        "calls": log[:4],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=2))


FIELDS = [
    ("cfg_scale", "cfg_scale"),
    ("dit_calls", "DiT calls"),
    ("calls_per_step", "DiT calls / step (CFG structure)"),
    ("guidance_values", "embedded_guidance value(s)"),
    ("timesteps", "timestep schedule"),
    ("sigmas", "sigma schedule"),
]


def compare(folder):
    with open(os.path.join(folder, "tome.json")) as f:
        t = json.load(f)
    with open(os.path.join(folder, "ours.json")) as f:
        o = json.load(f)

    print(f"\n{'field':<38} {'ToMe':<22} {'ours':<22} match")
    print("-" * 92)
    ok = True
    for key, label in FIELDS:
        a, b = t[key], o[key]
        same = a == b
        ok &= same
        fmt = (lambda v: f"[{len(v)} vals]" if isinstance(v, list) and len(v) > 3 else str(v))
        print(f"{label:<38} {fmt(a):<22} {fmt(b):<22} {'OK' if same else 'MISMATCH'}")

    # Text conditioning: the first two calls are the positive and negative CFG branches.
    print()
    for i in range(min(2, len(t["calls"]), len(o["calls"]))):
        ta, oa = t["calls"][i], o["calls"][i]
        for k in ("encoder_hidden_states_shape", "encoder_hidden_states_sum", "txt_ids_shape"):
            same = ta[k] == oa[k]
            ok &= same
            print(f"call{i} {k:<32} {str(ta[k]):<22} {str(oa[k]):<22} {'OK' if same else 'MISMATCH'}")

    print()
    for i in range(min(2, len(t["calls"]))):
        print(f"call{i} img tokens: ToMe {t['calls'][i]['hidden_states_shape']}  "
              f"ours {o['calls'][i]['hidden_states_shape']}   "
              "(expected to differ: ToMe merges inside blocks, ours enters short)")

    print("\n" + ("ALL CONDITIONING PARAMETERS MATCH" if ok else "*** MISMATCH FOUND ***"))
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("tome", "ours"))
    p.add_argument("--compare", type=str, default=None)
    p.add_argument("--out", type=str, default="parity/result.json")
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--mask", type=str, default="circular")
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--lr_factor", type=int, default=2)
    args = p.parse_args()

    if args.compare:
        sys.exit(0 if compare(args.compare) else 1)
    elif args.arm:
        run(args)
    else:
        p.error("pass --arm or --compare")
