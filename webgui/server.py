"""Minimal Flask GUI for the foveated FLUX2 pipeline.

Run from the repo root:

    python webgui/server.py [--model_id ...] [--lora_checkpoint ...] [--dit_checkpoint ...]

then open http://localhost:5000 in a browser.

The page lets the user paint a foveation mask (or drag a circular preset) on a
1024x1024 canvas, previews the resulting tokenization grid, and runs a single
inference pass via the same pipeline used by `inference.py`. Soft foveation
blend is always on. Generation runs in a background thread so the page can
poll `/progress` for a denoising progress bar.
"""

import argparse
import base64
import io
import os
import sys
import threading
import traceback
from types import SimpleNamespace

# Match inference.py: disable sageattention before importing torch-deps.
sys.modules["sageattention"] = None

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# Make the repo importable when running this file from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from diffsynth.core import load_state_dict  # noqa: E402
from src.inference import load_pipeline  # noqa: E402
from src.inference.visualize import create_tokenization_mask_vis  # noqa: E402
from src.masks import create_foveation_mask_full_res  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fixed generation settings (per user spec): 1024x1024, 50 steps, soft blend on.
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
DECODE_MODE = "merge"  # merge needed for soft blend to take effect
PREDICTION_TYPE = "clean"
LR_DOWNSAMPLE_FACTOR = 2
SOFT_FOVEATION_BLEND = True

_pipe = None
_pipe_device = None

# LoRA registry. Populated from --lora args at startup. State dicts are kept on
# CPU; we fuse/unfuse in place against pipe.dit's base weights so switching is
# fast (no pipeline reload) and bounded in VRAM.
_lora_registry = {}     # name -> {"path": str, "state_dict": dict or None (lazy)}
_lora_order = []        # preserves the dropdown order
_current_lora = None    # name of the LoRA currently fused into pipe.dit
_lora_lock = threading.Lock()

# Generation state (single-GPU server: one job at a time).
_job_lock = threading.Lock()
_job_state = {
    "running": False,
    "step": 0,
    "total": NUM_INFERENCE_STEPS,
    "image": None,        # base64 PNG once done
    "mask": None,         # base64 PNG (full-res mask) once done
    "error": None,
    "job_id": 0,
}


def _parse_lora_specs(specs):
    """Parse repeated --lora NAME=PATH args into an ordered dict."""
    parsed = []
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--lora must be NAME=PATH, got: {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f"--lora must be NAME=PATH, got: {spec!r}")
        parsed.append((name, path))
    return parsed


def _convert_lora_sd(state_dict):
    """Run state dict through the pipeline's lora_loader.convert_state_dict so
    keys match base-model module names. Returned tensors stay on whatever device
    the input lived on."""
    loader = _pipe.lora_loader(torch_dtype=_pipe.torch_dtype, device="cpu")
    return loader.convert_state_dict(state_dict)


def _apply_lora_delta(state_dict, sign):
    """Fuse (sign=+1) or unfuse (sign=-1) a converted LoRA state dict into pipe.dit
    in-place. Mirrors GeneralLoRALoader.fuse_lora_to_base_model but supports a
    sign so the same code path handles both directions."""
    lora_layer_names = {
        k[: -len(".lora_B.weight")]
        for k in state_dict if k.endswith(".lora_B.weight")
    }
    updated = 0
    for name, module in _pipe.dit.named_modules():
        if name not in lora_layer_names:
            continue
        wb = state_dict[name + ".lora_B.weight"].to(device=_pipe.device, dtype=_pipe.torch_dtype)
        wa = state_dict[name + ".lora_A.weight"].to(device=_pipe.device, dtype=_pipe.torch_dtype)
        if wb.dim() == 4:
            wb = wb.squeeze(3).squeeze(2)
            wa = wa.squeeze(3).squeeze(2)
            delta = torch.mm(wb, wa).unsqueeze(2).unsqueeze(3)
        else:
            delta = torch.mm(wb, wa)
        base = module.weight.data
        base.add_(delta.to(device=base.device, dtype=base.dtype), alpha=float(sign))
        updated += 1
    return updated


def _switch_lora_unlocked(target_name):
    """Unfuse the currently-fused LoRA (if any) and fuse the target one.
    Caller must hold _lora_lock and ensure no generation is running."""
    global _current_lora
    if target_name == _current_lora:
        return
    if target_name not in _lora_registry:
        raise ValueError(f"unknown lora: {target_name}")

    if _current_lora is not None:
        cur = _lora_registry[_current_lora]
        n = _apply_lora_delta(cur["state_dict"], sign=-1)
        print(f"[lora] unfused {_current_lora} ({n} tensors)")

    nxt = _lora_registry[target_name]
    if nxt["state_dict"] is None:
        raw = load_state_dict(nxt["path"], torch_dtype=_pipe.torch_dtype)
        nxt["state_dict"] = _convert_lora_sd(raw)
    n = _apply_lora_delta(nxt["state_dict"], sign=+1)
    print(f"[lora] fused {target_name} ({n} tensors)")
    _current_lora = target_name


def _load_pipe(args):
    global _pipe, _pipe_device, _current_lora
    loader_args = SimpleNamespace(
        model_id=args.model_id,
        lora_checkpoint=None,  # registry handles LoRA, not the loader
        dit_checkpoint=args.dit_checkpoint,
        experiment="ours",
    )
    _pipe = load_pipeline(loader_args)
    _pipe_device = _pipe.device
    print(f"Pipeline loaded on {_pipe_device}.")

    for name, path in _parse_lora_specs(args.lora):
        if name in _lora_registry:
            raise SystemExit(f"duplicate --lora name: {name}")
        _lora_registry[name] = {"path": path, "state_dict": None}
        _lora_order.append(name)

    if not _lora_order:
        raise SystemExit("at least one --lora NAME=PATH is required")

    default = args.default_lora or _lora_order[0]
    if default not in _lora_registry:
        raise SystemExit(f"--default_lora {default!r} not in registered loras")
    with _lora_lock:
        _switch_lora_unlocked(default)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def _quantize_to_lr_blocks(raw_mask: torch.Tensor):
    """Snap any binary [H, W] mask to LR-block resolution and expand back.

    Rule: an LR block is HR iff *any* pixel inside it is set (max-pool, not
    average). This ensures a single HR-token "touch" upgrades the full 2x2
    HR-token block, never leaving 1/2/3 HR tokens inside an LR cell.
    """
    if raw_mask.shape != (HEIGHT, WIDTH):
        raw_mask = F.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0), size=(HEIGHT, WIDTH), mode="nearest",
        ).squeeze(0).squeeze(0)

    lr_h = HEIGHT // 16 // LR_DOWNSAMPLE_FACTOR
    lr_w = WIDTH // 16 // LR_DOWNSAMPLE_FACTOR
    lr_mask = F.adaptive_max_pool2d(
        (raw_mask > 0).float().unsqueeze(0).unsqueeze(0), (lr_h, lr_w),
    )

    token_grid = F.interpolate(
        lr_mask, size=(HEIGHT // 16, WIDTH // 16), mode="nearest",
    ).squeeze(0).squeeze(0)
    full_res = F.interpolate(
        lr_mask, size=(HEIGHT, WIDTH), mode="nearest",
    ).squeeze(0).squeeze(0)
    return token_grid, full_res


def _drawn_mask_to_tensors(mask_b64: str):
    """User-painted mask -> (token_grid_mask [H/16, W/16], full_res_mask [H, W])."""
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",", 1)[1]
    raw = base64.b64decode(mask_b64)
    img = Image.open(io.BytesIO(raw))
    if img.mode == "RGBA":
        arr = np.array(img.split()[-1])  # alpha channel
    else:
        arr = np.array(img.convert("L"))
    raw_mask = torch.from_numpy((arr > 0).astype(np.float32)).to(_pipe_device)
    return _quantize_to_lr_blocks(raw_mask)


def _preset_mask_to_tensors(shape: str, cx: float, cy: float, r: float):
    """Preset (circular) mask: rasterize the circle at full resolution, then
    quantize to LR blocks under the same "any-touch -> full block" rule used
    by the drawn path. We bypass `create_foveation_mask` here because that
    helper point-samples at the LR-grid center, which can drop LR cells that
    the circle's edge clips through.
    """
    if shape != "circular":
        raise ValueError(f"unsupported preset shape: {shape}")
    full_res = create_foveation_mask_full_res(
        HEIGHT, WIDTH, (cx, cy), r, shape, _pipe_device,
    )
    return _quantize_to_lr_blocks(full_res)


def _empty_mask_tensors():
    """All-LR mask (no foveal region) — used for the initial tokenization preview."""
    token_grid = torch.zeros(
        HEIGHT // 16, WIDTH // 16, device=_pipe_device, dtype=torch.float32,
    )
    full_res = torch.zeros(HEIGHT, WIDTH, device=_pipe_device, dtype=torch.float32)
    return token_grid, full_res


def _build_mask_from_payload(mask_spec):
    """Resolve a mask payload to (token_grid_mask, full_res_mask). Raises ValueError."""
    kind = mask_spec.get("kind", "drawn")
    if kind == "empty":
        return _empty_mask_tensors()
    if kind == "drawn":
        data = mask_spec.get("data") or ""
        if not data:
            return _empty_mask_tensors()
        return _drawn_mask_to_tensors(data)
    if kind == "preset":
        shape = mask_spec.get("shape", "circular")
        if shape != "circular":
            # Per UI spec: presets are circular-only.
            raise ValueError(f"unsupported preset shape: {shape}")
        cx = float(mask_spec.get("cx", 0.0))
        cy = float(mask_spec.get("cy", 0.0))
        r = float(mask_spec.get("r", 0.3))
        return _preset_mask_to_tensors(shape, cx, cy, r)
    raise ValueError(f"unknown mask kind: {kind}")


def _tokenization_vis_b64(token_mask) -> str:
    vis = create_tokenization_mask_vis(token_mask, HEIGHT, WIDTH, lr_factor=LR_DOWNSAMPLE_FACTOR)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.route("/")
def index():
    resp = send_from_directory(_STATIC_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/static/<path:fname>")
def static_files(fname):
    resp = send_from_directory(_STATIC_DIR, fname)
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/config")
def config():
    return jsonify({
        "height": HEIGHT, "width": WIDTH,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    })


@app.route("/loras")
def loras():
    with _lora_lock:
        return jsonify({"loras": list(_lora_order), "current": _current_lora})


@app.route("/switch_lora", methods=["POST"])
def switch_lora():
    if _pipe is None:
        return jsonify({"error": "pipeline not loaded"}), 503
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    # Block switching while a generation is running — the DiT weights are being
    # read by the background thread, mutating them mid-pass would corrupt output.
    with _job_lock:
        if _job_state["running"]:
            return jsonify({"error": "cannot switch LoRA while a job is running"}), 409
        with _lora_lock:
            try:
                _switch_lora_unlocked(name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"switch failed: {e}"}), 500
            return jsonify({"current": _current_lora})


@app.route("/tokenization", methods=["POST"])
def tokenization():
    """Build a token-grid mask from the payload and return its visualization PNG."""
    if _pipe is None:
        return jsonify({"error": "pipeline not loaded"}), 503
    payload = request.get_json(silent=True) or {}
    mask_spec = payload.get("mask") or {}
    try:
        token_mask, _ = _build_mask_from_payload(mask_spec)
    except Exception as e:
        return jsonify({"error": f"failed to build mask: {e}"}), 400
    return jsonify({"tokenization": _tokenization_vis_b64(token_mask)})


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------

def _make_progress_cb(job_id: int):
    """Return an iterator wrapper that updates `_job_state['step']` per step."""
    def wrap(iterable):
        for i, item in enumerate(iterable):
            yield item
            with _job_lock:
                if _job_state["job_id"] == job_id:
                    _job_state["step"] = i + 1
    return wrap


def _run_generation(job_id, prompt, seed, token_mask, full_res_mask):
    try:
        torch.cuda.empty_cache()
        image = _pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            seed=seed,
            rand_device="cuda" if torch.cuda.is_available() else "cpu",
            num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_scale=GUIDANCE_SCALE,
            foveation_mask=token_mask,
            full_res_foveation_mask=full_res_mask,
            decode_mode=DECODE_MODE,
            prediction_type=PREDICTION_TYPE,
            soft_foveation_blend=SOFT_FOVEATION_BLEND,
            lr_downsample_factor=LR_DOWNSAMPLE_FACTOR,
            progress_bar_cmd=_make_progress_cb(job_id),
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mask_arr = (full_res_mask.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        mask_buf = io.BytesIO()
        Image.fromarray(mask_arr).save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("ascii")

        with _job_lock:
            if _job_state["job_id"] == job_id:
                _job_state.update(image=img_b64, mask=mask_b64,
                                  step=NUM_INFERENCE_STEPS, running=False)
    except Exception as e:
        traceback.print_exc()
        with _job_lock:
            if _job_state["job_id"] == job_id:
                _job_state.update(error=str(e), running=False)


@app.route("/generate", methods=["POST"])
def generate():
    if _pipe is None:
        return jsonify({"error": "pipeline not loaded"}), 503

    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    seed = int(payload.get("seed", 0))

    try:
        token_mask, full_res_mask = _build_mask_from_payload(payload.get("mask") or {})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if float(token_mask.sum().item()) == 0:
        return jsonify({"error": "mask is empty — paint a region or pick a preset"}), 400

    with _job_lock:
        if _job_state["running"]:
            return jsonify({"error": "another job is running"}), 409
        _job_state["job_id"] += 1
        job_id = _job_state["job_id"]
        _job_state.update(
            running=True, step=0, total=NUM_INFERENCE_STEPS,
            image=None, mask=None, error=None,
        )

    threading.Thread(
        target=_run_generation,
        args=(job_id, prompt, seed, token_mask, full_res_mask),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "total": NUM_INFERENCE_STEPS})


@app.route("/progress")
def progress():
    with _job_lock:
        return jsonify({
            "running": _job_state["running"],
            "step": _job_state["step"],
            "total": _job_state["total"],
            "image": _job_state["image"],
            "mask": _job_state["mask"],
            "error": _job_state["error"],
            "job_id": _job_state["job_id"],
        })


def _build_argparser():
    p = argparse.ArgumentParser(description="Foveated diffusion web GUI")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument(
        "--lora", action="append", default=[],
        help="NAME=PATH for a LoRA available in the dropdown. Repeatable.",
    )
    p.add_argument(
        "--default_lora", default=None,
        help="Which --lora name to fuse at startup (default: first --lora).",
    )
    p.add_argument("--dit_checkpoint", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    _load_pipe(args)
    # threaded=True so /progress requests don't block while /generate's background
    # thread is running (the thread itself holds the GPU; the lock keeps it serial).
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
