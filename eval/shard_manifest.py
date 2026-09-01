"""Resumable per-shard manifest, shared by every eval driver.

Needed for preemptible Slurm arrays (see marlowe_scripts/marlowe_launch_rules.md): a shard
gets a one-hour allocation, is requeued when preempted, and must pick up where it left off.
The contract from those rules:

  - one manifest per shard, so concurrent shards never write the same file
  - a row is flushed only after its image is fully written
  - a sample is skipped only when BOTH its manifest row and its output file exist
  - a `.generation_shard<id>_complete` marker is written only when the shard has no
    pending samples

Repeating the single interrupted sample is harmless, since every sample's filename, seed
and configuration are a deterministic function of its global prompt index.

Rows are appended as they are produced, so the file is identical in name and schema to the
metadata_%05d.csv the drivers used to write in one shot at the end.
"""

import csv
import os

# Timing columns every driver appends, so a runtime comparison is one pandas read over the
# eval tree instead of a per-arm special case.
TIMING_COLUMNS = ["dit_time_s", "vae_time_s"]


def pipe_timings(pipe):
    """Read the last call's DiT-loop and VAE timings off any of the three pipelines.

    All of them clock the same boundaries (CUDA-synchronized, text encoding already done,
    stopping before VAE decode), they just spell the attribute differently:
    `vit_timing` in flux2_image.py / flux2_image_foveated.py, `denoise_time` in RALU's.
    Missing attributes come back blank rather than raising, so an un-instrumented pipeline
    still produces a well-formed row.
    """
    dit = getattr(pipe, "vit_timing", None)
    if dit is None:
        dit = getattr(pipe, "denoise_time", None)
    vae = getattr(pipe, "vae_timing", None)
    return {
        "dit_time_s": "" if dit is None else round(float(dit), 4),
        "vae_time_s": "" if vae is None else round(float(vae), 4),
    }


class ShardManifest:
    """Append-only metadata_<subset_idx>.csv with resume support."""

    def __init__(self, output_dir, subset_idx, columns=("image", "prompt")):
        self.output_dir = output_dir
        self.subset_idx = subset_idx
        self.columns = list(columns)
        self.path = os.path.join(output_dir, f"metadata_{subset_idx:05d}.csv")
        self.marker = os.path.join(output_dir, f".generation_shard{subset_idx}_complete")
        os.makedirs(output_dir, exist_ok=True)
        self._done = self._load_and_repair()
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns)
        if self._fh.tell() == 0:
            self._writer.writeheader()
            self._flush()

    def _load_and_repair(self):
        """Keep only rows whose image actually exists, and rewrite the file if any dropped.

        A row without its image means the shard died between the flush and the save (or the
        image was deleted); regenerating is correct, and dropping the row here is what keeps
        the rerun from appending a duplicate.
        """
        if not os.path.exists(self.path):
            return set()
        with open(self.path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("image")]
        keep = [r for r in rows if os.path.exists(os.path.join(self.output_dir, r["image"]))]
        if len(keep) != len(rows):
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.columns)
                w.writeheader()
                for r in keep:
                    w.writerow({c: r.get(c, "") for c in self.columns})
            os.replace(tmp, self.path)
            print(f"[manifest] dropped {len(rows) - len(keep)} row(s) with no image; "
                  f"{len(keep)} complete sample(s) retained")
        return {r["image"] for r in keep}

    def _flush(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def is_done(self, image_name):
        """True only when the row and the image are both present — the rules' condition."""
        return (image_name in self._done
                and os.path.exists(os.path.join(self.output_dir, image_name)))

    def append(self, row):
        """Record one finished sample. Call only after its image is on disk."""
        missing = set(self.columns) - set(row)
        if missing:
            raise ValueError(f"manifest row missing columns {sorted(missing)}")
        self._writer.writerow({c: row[c] for c in self.columns})
        self._flush()
        self._done.add(row["image"])

    def mark_complete(self):
        """Write the shard's completion marker. Only when nothing is pending."""
        with open(self.marker, "w") as f:
            f.write(f"{len(self._done)}\n")
        self._flush_dir()

    def _flush_dir(self):
        fd = os.open(self.output_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self):
        if not self._fh.closed:
            self._flush()
            self._fh.close()

    def __len__(self):
        return len(self._done)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
