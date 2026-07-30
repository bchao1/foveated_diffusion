"""Summarize the cfg-4.0 RALU-vs-ours rerun.

Per config: mean/std pixel brightness, paired brightness ratio to the full-res reference,
CLIP score (ViT-L-14-336), schedule/timing from stats json. Also writes a contact sheet.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

ROOT = "/home/brianchc/foveated_diffusion-private/results/ralu_compare/cfg4"
CONFIGS = ["plain", "ours", "ralu3_free", "ralu3_z6.5", "ralu2_free", "ralu2_z6.5", "ralu3_pub"]
REF = "plain"


def load_dir(d):
    """Return {global_idx: (path, prompt)} from all metadata shards."""
    out = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("metadata_") and f.endswith(".csv"):
            for _, row in pd.read_csv(os.path.join(d, f)).iterrows():
                idx = int(os.path.splitext(row["image"])[0].split("_")[-1])
                out[idx] = (os.path.join(d, row["image"]), row["prompt"])
    return out


def brightness(path):
    return float(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).mean())


def clip_scores(pairs, device="cuda:0"):
    """pairs: list[(path, prompt)] -> list[float] cosine sims."""
    import torch
    from metric_computation.CLIP import _load_openclip, _clip_sim_batch_loop
    model, tokenizer = _load_openclip(device=device)
    image_size = model.visual.image_size
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    image_size = int(image_size)
    image_mean = torch.tensor(model.visual.image_mean, device=device).view(3, 1, 1)
    image_std = torch.tensor(model.visual.image_std, device=device).view(3, 1, 1)
    dtype = torch.float16
    sims = []
    for p, t in pairs:
        with torch.no_grad():
            sims.append(float(_clip_sim_batch_loop(
                [p], [t], model, tokenizer, image_size, image_mean, image_std,
                dtype, 1, device)))
    return sims


def main():
    do_clip = "--clip" in sys.argv
    data = {}
    for c in CONFIGS:
        d = os.path.join(ROOT, c)
        if not os.path.isdir(d):
            print(f"[skip] {c}: missing")
            continue
        data[c] = load_dir(d)

    ref = data.get(REF, {})
    rows = []
    for c, imgs in data.items():
        idxs = sorted(set(imgs) & set(ref)) if ref else sorted(imgs)
        b = np.array([brightness(imgs[i][0]) for i in idxs])
        rb = np.array([brightness(ref[i][0]) for i in idxs]) if ref else None
        row = {"config": c, "n": len(idxs), "bright": b.mean(), "bright_sd": b.std()}
        if rb is not None:
            row["ratio_mean"] = b.mean() / rb.mean()
            row["paired_ratio"] = float(np.mean(b / rb))
        st = os.path.join(ROOT, c, "stats_00000.json")
        if os.path.exists(st):
            s = json.load(open(st))
            row["N"] = str(s["params"]["N"])
            row["steps"] = sum(s["params"]["N"])
            row["Z"] = round(s["params"]["Z"], 2)
            row["beta"] = [round(x, 3) for x in s["params"]["betas"]]
            row["sec"] = round(s["mean_denoise_time_s"], 2)
            si = s.get("schedule_info", {})
            row["cost_ratio"] = si.get("cost_ratio")
        if do_clip:
            row["clip"] = float(np.mean(clip_scores([imgs[i] for i in idxs])))
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(ROOT, "summary.csv"), index=False)

    # contact sheet: rows = config, cols = first 6 shared prompts
    idxs = sorted(set.intersection(*[set(v) for v in data.values()]))[:6]
    thumb = 256
    sheet = Image.new("RGB", (thumb * len(idxs), thumb * len(data)), "black")
    for r, c in enumerate(data):
        for k, i in enumerate(idxs):
            im = Image.open(data[c][i][0]).convert("RGB").resize((thumb, thumb))
            sheet.paste(im, (k * thumb, r * thumb))
    sheet.save(os.path.join(ROOT, "contact_sheet.png"))
    print("rows (top->bottom):", list(data))
    print("wrote", os.path.join(ROOT, "contact_sheet.png"))


if __name__ == "__main__":
    main()
