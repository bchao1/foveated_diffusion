"""Aesthetic-Train-V2 loader for foveated DMD.

Yields raw images in [-1, 1] plus the caption string; VAE and text encoding happen on GPU
inside the training step (measured at ~53 ms/prompt for the Qwen3-4B encoder, versus
~7.9 MB/prompt = 749 GB to cache the full 95k-prompt split at the 512-token padding the
pipeline uses -- so caching is not worth it here).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset

PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AestheticFoveatedDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        height: int = 1024,
        width: int = 1024,
        repeat: int = 1,
        max_samples: Optional[int] = None,
        seed: int = 0,
    ):
        df = pd.read_csv(metadata_path)
        assert {"image", "prompt"} <= set(df.columns), df.columns
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        self.rows = df.to_dict("records")
        self.base_path = base_path
        self.height, self.width = height, width
        self.repeat = repeat

    def __len__(self):
        return len(self.rows) * self.repeat

    def _load(self, rel_path: str) -> torch.Tensor:
        img = Image.open(os.path.join(self.base_path, rel_path)).convert("RGB")
        # resize preserving aspect, then centre crop -- same convention as the SFT operator
        scale = max(self.width / img.width, self.height / img.height)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.BICUBIC,
        )
        left = (img.width - self.width) // 2
        top = (img.height - self.height) // 2
        img = img.crop((left, top, left + self.width, top + self.height))
        x = torch.from_numpy(np.array(img, dtype=np.float32))
        return (x / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        row = self.rows[idx % len(self.rows)]
        try:
            image = self._load(row["image"])
        except Exception as exc:  # skip unreadable files rather than crash a long run
            print(f"[data] failed to load {row['image']}: {exc}; using next sample")
            return self.__getitem__((idx + 1) % len(self))
        return {"image": image, "prompt": str(row["prompt"])}


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }
