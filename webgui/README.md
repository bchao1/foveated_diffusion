# Foveated Diffusion — Web GUI

A minimal browser interface for the foveated FLUX2 pipeline. Paint a foveation
mask (or drag a circle) on a 1024×1024 canvas, watch the resulting tokenization
grid update live, then run a single inference pass via the same
`Flux2FoveatedImagePipeline` used by `inference.py`.

![layout: left = prompt + mask controls; right = tokenization mask | generated image]

## Install

The server only adds Flask to the repo's existing inference deps:

```bash
pip install flask
```

Everything else (`torch`, `diffsynth`, the foveated pipeline) is shared with
the inference entry point.

## Run

Use the launcher script, which bakes in the same env vars, conda env, GPU
pinning, and checkpoint paths as `tests/_common.sh`:

```bash
bash webgui/run.sh                      # default ckpt (random-mask LoRA), GPU 1, port 5000
GPU_ID=2 bash webgui/run.sh             # pick GPU
PORT=8080 bash webgui/run.sh            # pick port
CKPT=saliency bash webgui/run.sh        # one of: no_fov | random | saliency | bbox | none
bash webgui/run.sh --port 8080          # extra flags forwarded to server.py
```

**The model checkpoint paths inside `run.sh` (`CKPT_NO_FOV`, `CKPT_RANDOM`,
`CKPT_SALIENCY`, `CKPT_BBOX`) must be set to point at valid `.safetensors`
files on your machine before launching.** Edit the script to match your setup.

The pipeline loads once on startup and stays resident in GPU memory. Open
[http://localhost:5000](http://localhost:5000) in a browser.

## Using the UI

**Left panel — controls**

- **Prompt + seed.**
- **Mask source** (two modes):
  - *Draw* — drag on the canvas to paint the foveal (HR) region. Shift-drag to
    erase. Brush size, Clear, and Fill-all are available.
  - *Preset (circle)* — drag on the canvas: **press** sets the center,
    **release** sets the radius. The circle previews live as you drag.
- **Generate** kicks off a background denoising job. A progress bar polls
  `/progress` and shows step `k / 50`.

**Right panel — two stages, side by side**

- **Tokenization mask** (left stage) — the actual LR-block grid the pipeline
  will use, rendered by `create_tokenization_mask_vis`. White cells = HR
  tokens, gray cells = LR tokens. Refreshes (debounced) whenever the mask
  changes, so you can see exactly which blocks "snap" to HR before
  committing.
- **Generated image** (right stage) — populated after generation finishes.

### Mask quantization

Any HR pixel inside an LR block upgrades the whole block to HR (max-pool, not
average). A single brush touch therefore promotes the full 2×2 HR-token block
— you never end up with 1/2/3 stranded HR tokens inside an LR cell. The same
rule applies to the preset circle, so circle edges that clip an LR cell
upgrade that cell rather than dropping it.

## Fixed pipeline settings

Baked into the server to keep the GUI minimal:

| Setting | Value |
|---|---|
| Resolution | 1024 × 1024 |
| Steps | 50 |
| CFG | 4.0 |
| Decode mode | `merge` |
| Prediction type | `clean` |
| LR downsample factor | 2 |
| Soft foveation blend | **on** |

To vary these, edit the constants at the top of `webgui/server.py`, or pass
`--lora_checkpoint` / `--dit_checkpoint` through `run.sh` (extra args are
forwarded to `server.py`).

## HTTP endpoints

- `GET /` — page.
- `GET /config` — `{height, width, num_inference_steps}`.
- `POST /tokenization` — body `{mask: {kind, ...}}`; returns the tokenization
  visualization PNG (base64). Used by the live preview.
- `POST /generate` — body `{prompt, seed, mask: {kind, ...}}`. Returns
  `{job_id, total}` and starts a background thread.
- `GET /progress` — `{running, step, total, image, mask, error, job_id}`.
  Poll until `running` is false.

`mask.kind` is one of `drawn` (base64 PNG of the painted alpha channel),
`preset` (`{shape: "circular", cx, cy, r}` with `cx, cy ∈ [-0.5, 0.5]`,
`r` relative to half the image diagonal), or `empty`.

## Notes

- The Flask app runs `threaded=True` so `/progress` doesn't block while
  `/generate`'s worker thread holds the GPU. A `_job_lock` keeps generation
  itself serial — submitting a second job while one is in flight returns 409.
- This is a demo UI: no auth, no rate-limiting, no upload size cap beyond
  Flask defaults.
