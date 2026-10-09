#!/usr/bin/env python
"""Web picker for 4-video qualitative sets.

Video sibling of LoT's ``scripts/qual_picker.py``. Serves an eval tree laid out
as ``<set>/<method>/<run>/<step>/videos/<id>.mp4`` (the layout under
``/local/brian/foveated_diffusion_drive/{eval200,eval200_720p,spline}``) as a
local web page, so that picking "the best 10 sets of 200" is a sequence of
small decisions instead of one impossible one.

An *item* here is a prompt id, shown as the whole 4-video set (fullres / naive /
tome / ours) playing in sync. You are picking the id, not a single video.

Three stages, switched with ``t`` / ``r`` / ``c``:

1. **Triage** -- one set at a time, as large as the window allows. Keep / maybe
   / reject with one key, auto-advancing. 200 sets is a few minutes, and you
   never compare more than one thing against your own taste.
2. **Shortlist** -- a grid of only the keeps (and maybes), hover to play, click
   to put a set in the final pick list.
3. **Compare** -- pairwise duels over the shortlist, Elo-ranked. You never rank
   200 sets, you just answer "left or right?" a few dozen times and commit the
   top of the leaderboard.

State (verdicts, pick order, Elo ratings) is persisted to
``results/video_picks/<set>/video_picks.json`` plus a flat ``.csv`` and a
``picks.txt`` of bare ids for downstream figure/video scripts. It is autosaved
and reloaded on restart, so a picking session is resumable.

Usage
-----
    conda activate imp_tome    # the env that has flask
    python scripts/video_picker.py /local/brian/foveated_diffusion_drive/eval200

Nothing is ever written into the video tree; the state dir and the poster cache
live under ``results/video_picks/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

# Methods are discovered from disk; this only fixes the display order/labels of
# the ones we know about. Unknown methods are appended alphabetically.
METHOD_ORDER = ["fullres", "naive", "tome", "ours"]
METHOD_LABELS = {
    "fullres": "Full-res (reference)",
    "naive": "Naive foveation",
    "tome": "Token merging",
    "ours": "Ours",
}
# Video variants, in toggle order; the first is what a session opens on. The
# fovea-circle overlay comes first because that is what makes a foveation
# artifact legible. fullres has no overlay render, so a set falls back to the
# plain video for any method that lacks the current variant.
VARIANTS = ["videos_with_circle", "videos"]
VARIANT_LABELS = {"videos": "plain", "videos_with_circle": "with fovea circle"}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


@dataclass
class Item:
    sid: str                                   # prompt id, e.g. "137"
    videos: dict[str, dict[str, str]] = field(default_factory=dict)
    prompt: str = ""
    meta: dict = field(default_factory=dict)   # from runtimes.jsonl, per method

    @property
    def n_methods(self) -> int:
        return len(self.videos)


def _sortkey(sid: str) -> tuple[int, str]:
    return (int(sid), "") if sid.isdigit() else (1 << 30, sid)


def find_video_dirs(method_dir: Path) -> dict[str, Path]:
    """{variant: dir} for the (single) run/step under a method dir.

    The tree is ``<method>/<run_name>/<step>/videos[_with_circle]``; the run and
    step names differ per method (ours is a different lora at a different step),
    so they are globbed rather than assumed.
    """
    out: dict[str, Path] = {}
    for variant in VARIANTS:
        hits = sorted(p for p in method_dir.glob(f"*/*/{variant}") if p.is_dir())
        if hits:
            out[variant] = hits[-1]
    return out


def read_runtimes(method_dir: Path) -> dict[str, dict]:
    """``runtimes.jsonl`` next to the videos dir -> {stem: record}."""
    out: dict[str, dict] = {}
    for p in sorted(method_dir.glob("*/*/runtimes.jsonl")):
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stem = str(rec.get("stem", rec.get("idx", "")))
            if stem:
                out[stem] = rec
    return out


def read_prompts(path: Path) -> dict[str, str]:
    """``id,prompt`` (or a bare prompt column) -> {id: prompt}.

    A csv without an ``id`` column is taken as 1-indexed by row, which is how
    the generation scripts name their outputs.
    """
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            keys = {k.lower(): v for k, v in row.items() if k}
            prompt = keys.get("prompt") or keys.get("text") or ""
            sid = keys.get("id") or keys.get("idx") or keys.get("index") or str(i)
            if prompt:
                out[str(sid).strip()] = prompt
    return out


def scan(root: Path, prompts: dict[str, str]) -> tuple[list[Item], list[str]]:
    methods_seen: dict[str, dict[str, Path]] = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        dirs = find_video_dirs(mdir)
        if dirs:
            methods_seen[mdir.name] = dirs

    methods = ([m for m in METHOD_ORDER if m in methods_seen]
               + sorted(set(methods_seen) - set(METHOD_ORDER)))

    items: dict[str, Item] = {}
    for method in methods:
        runtimes = read_runtimes(root / method)
        for variant, vdir in methods_seen[method].items():
            for mp4 in vdir.glob("*.mp4"):
                it = items.setdefault(mp4.stem, Item(sid=mp4.stem))
                it.videos.setdefault(method, {})[variant] = str(mp4.relative_to(root))
        for sid, rec in runtimes.items():
            if sid in items:
                items[sid].meta[method] = {
                    "gen_time_sec": rec.get("gen_time_sec"),
                    "token_ratio": rec.get("token_ratio"),
                }

    ordered = [items[k] for k in sorted(items, key=_sortkey)]
    for it in ordered:
        it.prompt = prompts.get(it.sid, "")
    return ordered, methods


# ---------------------------------------------------------------------------
# state persistence
# ---------------------------------------------------------------------------

TAB = "set"          # one tab: we only ever pick whole 4-video sets
EMPTY_TAB = {"picks": [], "verdicts": {}, "elo": {}, "duels": 0, "n": {}}


class Conflict(Exception):
    """The state file changed on disk since we last read or wrote it."""


class State:
    """{picks, verdicts, elo, duels} for the one tab, mirrored to json/csv/txt.

    Loaded from disk at startup and re-loadable at any time, so a saved
    selection comes back populated instead of having to be re-picked. `mtime`
    tracks the file we last read or wrote, so an edit made behind our back (a
    second browser tab, an external script) is refused rather than silently
    overwritten.
    """

    def __init__(self, path: Path, target: int):
        self.path = path
        self.csv_path = path.with_suffix(".csv")
        self.txt_path = path.with_name("picks.txt")
        self.target = target
        self.lock = threading.Lock()
        self.tabs: dict[str, dict] = {}
        self.mtime: float | None = None
        self.load()

    def _disk_mtime(self) -> int | None:
        return self.path.stat().st_mtime_ns if self.path.is_file() else None

    def load(self) -> dict:
        """(Re-)read the state file. Returns a summary of what came back."""
        tabs: dict[str, dict] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
            except Exception:
                loaded = {}
            for tab, blob in (loaded.get("tabs") or {}).items():
                tabs[tab] = {**EMPTY_TAB, **blob}
        self.tabs = tabs
        self.mtime = self._disk_mtime()
        return self.summary()

    def summary(self) -> dict:
        return {tab: {"picks": len(blob.get("picks", [])),
                      "verdicts": len(blob.get("verdicts", {})),
                      "duels": blob.get("duels", 0)}
                for tab, blob in sorted(self.tabs.items())}

    def tab(self) -> dict:
        return self.tabs.setdefault(TAB, {k: (v.copy() if hasattr(v, "copy") else v)
                                          for k, v in EMPTY_TAB.items()})

    def as_json(self, root: Path) -> dict:
        return {
            "root": str(root),
            "target": self.target,
            "tabs": {
                tab: {
                    "target": self.target,
                    "picks": blob.get("picks", []),
                    "verdicts": blob.get("verdicts", {}),
                    "elo": {k: round(v, 1) for k, v in (blob.get("elo") or {}).items()},
                    "duels": blob.get("duels", 0),
                    "n": blob.get("n", {}),
                }
                for tab, blob in sorted(self.tabs.items())
            },
        }

    def save(self, root: Path, *, force: bool = False) -> None:
        with self.lock:
            if not force and self._disk_mtime() != self.mtime:
                raise Conflict(
                    f"{self.path} changed on disk since it was loaded; "
                    f"reload before saving (or save with force)")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.as_json(root), indent=2) + "\n")
            with self.csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["tab", "sid", "rank", "verdict", "elo"])
                for tab, blob in sorted(self.tabs.items()):
                    picks = blob.get("picks", [])
                    verdicts = blob.get("verdicts", {})
                    elo = blob.get("elo", {})
                    for i, sid in enumerate(picks):
                        w.writerow([tab, sid, i + 1, verdicts.get(sid, "keep"),
                                    round(elo.get(sid, 0.0), 1) or ""])
                    # Non-picked verdicts are kept so a session stays resumable
                    # from the csv alone; rank is blank for them.
                    for sid, v in sorted(verdicts.items(), key=lambda kv: _sortkey(kv[0])):
                        if sid not in picks:
                            w.writerow([tab, sid, "", v, round(elo.get(sid, 0.0), 1) or ""])
            picks = self.tabs.get(TAB, {}).get("picks", [])
            self.txt_path.write_text("".join(f"{sid}\n" for sid in picks))
            self.mtime = self._disk_mtime()


# ---------------------------------------------------------------------------
# posters
# ---------------------------------------------------------------------------


def poster_jpeg(src: Path, dst: Path, width: int, time_s: float) -> None:
    """One frame of `src` at `time_s`, scaled to `width`, as a jpeg at `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-ss", f"{time_s}", "-i", str(src), "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not dst.is_file():
        # A seek past the end of a short clip returns nothing; retry from 0.
        subprocess.run([c if c != f"{time_s}" else "0" for c in cmd], capture_output=True)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def build_app(root: Path, state: State, items: list[Item], methods: list[str],
              poster_at: float) -> Flask:
    app = Flask(__name__)
    poster_dir = state.path.parent / ".posters"
    poster_dir.mkdir(parents=True, exist_ok=True)

    def resolve(rel: str) -> Path:
        p = (root / rel).resolve()
        if root not in p.parents or not p.is_file():
            abort(404)
        return p

    @app.get("/")
    def index() -> Response:
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/data")
    def api_data():
        return jsonify({
            "root": str(root),
            "set_name": root.name,
            "methods": [{"key": m, "label": METHOD_LABELS.get(m, m)} for m in methods],
            "variants": [{"key": v, "label": VARIANT_LABELS.get(v, v)} for v in VARIANTS],
            "target": state.target,
            "state_path": str(state.path),
            "items": [{"sid": it.sid, "prompt": it.prompt, "videos": it.videos,
                       "meta": it.meta} for it in items],
            "state": state.as_json(root),
        })

    @app.get("/video")
    def video():
        # conditional=True gives Range support, which is what lets the browser
        # seek and loop a dozen clips without refetching each one.
        return send_file(resolve(request.args.get("p", "")),
                         mimetype="video/mp4", conditional=True)

    @app.get("/poster")
    def poster():
        rel = request.args.get("p", "")
        width = request.args.get("w", default=480, type=int)
        path = resolve(rel)
        st = path.stat()
        key = hashlib.sha1(
            f"{path}|{st.st_mtime_ns}|{st.st_size}|{width}|{poster_at}".encode()).hexdigest()
        cached = poster_dir / f"{key}.jpg"
        if not cached.is_file():
            poster_jpeg(path, cached, width, poster_at)
        if not cached.is_file():
            abort(404)
        return send_file(cached, mimetype="image/jpeg")

    @app.get("/api/reload")
    def api_reload():
        """Re-read the state file, so an externally edited selection shows up."""
        summary = state.load()
        print(f"reloaded {state.path}: "
              + ", ".join(f"{tab} {info['picks']} picks" for tab, info in summary.items()),
              flush=True)
        return jsonify({"ok": True, "state": state.as_json(root), "summary": summary})

    @app.post("/api/state")
    def api_state():
        body = request.get_json(force=True) or {}
        previous = state.tabs
        with state.lock:
            state.tabs = {tab: {**EMPTY_TAB, **blob}
                          for tab, blob in (body.get("tabs") or {}).items()}
        try:
            state.save(root, force=bool(request.args.get("force")))
        except Conflict as exc:
            state.tabs = previous          # keep serving what we loaded
            return jsonify({"ok": False, "conflict": str(exc)}), 409
        if request.args.get("explicit"):
            picks = state.tabs.get(TAB, {}).get("picks", [])
            print(f"saved {state.path}", flush=True)
            print(f"  picks ({len(picks)}/{state.target}): {' '.join(picks) or '-'}",
                  flush=True)
            print(f"  ids   : {state.txt_path}", flush=True)
        return jsonify({"ok": True, "json": str(state.path), "csv": str(state.csv_path),
                        "txt": str(state.txt_path)})

    return app


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Video picker</title>
<style>
:root { --bg:#0f1116; --panel:#171a21; --fg:#e6e8ee; --dim:#98a0b0; --line:#272b35;
        --accent:#4ea1ff; --keep:#28c07a; --maybe:#e0a33a; --reject:#d0505a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-sans-serif,system-ui,sans-serif; }
header { position:sticky; top:0; z-index:5; background:#0b0d11f2; backdrop-filter:blur(6px);
         border-bottom:1px solid var(--line); padding:7px 12px; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.spacer { flex:1; }
button, select, input { background:#1a1d24; color:var(--fg); border:1px solid var(--line);
         border-radius:6px; padding:5px 9px; cursor:pointer; font:inherit; }
button:hover { border-color:#3a4150; }
button.on { background:var(--accent); border-color:var(--accent); color:#08101c; font-weight:600; }
button.primary { background:var(--keep); border-color:var(--keep); color:#04150d; font-weight:600; }
.hint { color:var(--dim); font-size:11.5px; }
kbd { background:#1e222b; border:1px solid var(--line); border-radius:4px; padding:0 4px; }
.bar { height:3px; background:#1e222b; border-radius:2px; overflow:hidden; margin-top:6px; }
.bar > div { height:100%; background:var(--accent); width:0; transition:width .15s; }
main { padding:10px 12px 40px; }

/* a 4-video set, used by triage / duels / lightbox */
.set { display:grid; gap:6px; width:100%; }
.tile { position:relative; background:#0b0d11; border-radius:6px; overflow:hidden;
        border:1px solid var(--line); }
.tile video, .tile img { display:block; width:100%; background:#0b0d11; }
.tile .tag { position:absolute; top:4px; left:4px; background:#000000b0; border-radius:4px;
             padding:1px 6px; font-size:11px; color:#dfe4ee; }
.tile .sub { position:absolute; bottom:4px; right:5px; background:#000000b0; border-radius:4px;
             padding:1px 5px; font-size:10.5px; color:var(--dim); }
.tile.missing { display:flex; align-items:center; justify-content:center; min-height:120px;
                color:var(--reject); font-size:11.5px; }

/* triage */
#triage { display:none; } #triage.on { display:block; }
.stage { display:flex; flex-direction:column; align-items:center; gap:8px; }
#tSet { max-width:1500px; }
.verdict { display:flex; gap:8px; align-items:center; }
.vchip { border-radius:6px; padding:5px 11px; border:1px solid var(--line); }
.vchip.keep.on { background:var(--keep); border-color:var(--keep); color:#04150d; font-weight:700; }
.vchip.maybe.on { background:var(--maybe); border-color:var(--maybe); color:#1c1204; font-weight:700; }
.vchip.reject.on { background:var(--reject); border-color:var(--reject); color:#180406; font-weight:700; }
.meta { max-width:1200px; text-align:center; color:var(--dim); }
.meta b { color:var(--fg); }

/* shortlist */
#shortlist { display:none; } #shortlist.on { display:block; }
#grid { display:grid; gap:9px; }
.card { border:2px solid transparent; border-radius:8px; background:var(--panel); overflow:hidden;
        cursor:pointer; position:relative; }
.card.cur { border-color:var(--accent); }
.card.picked { border-color:var(--keep); }
.card video, .card img { display:block; width:100%; background:#0b0d11; }
.card .cap { padding:4px 7px; font-size:11px; color:var(--dim); display:flex; gap:6px; }
.card .cap b { color:var(--fg); font-weight:600; }
.card .cap .p { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
.badge { position:absolute; top:6px; left:6px; background:var(--keep); color:#04150d; font-weight:700;
         border-radius:999px; min-width:20px; height:20px; padding:0 6px; display:flex;
         align-items:center; justify-content:center; font-size:11px; z-index:2; }
.vdot { position:absolute; top:6px; right:6px; width:10px; height:10px; border-radius:50%; z-index:2; }
.vdot.keep { background:var(--keep); } .vdot.maybe { background:var(--maybe); }
.vdot.reject { background:var(--reject); } .vdot.none { background:#3a4150; }

/* compare */
#compare { display:none; } #compare.on { display:flex; gap:12px; }
#duel { flex:1; display:flex; gap:10px; align-items:flex-start; }
.duelpane { flex:1; border:2px solid var(--line); border-radius:8px; overflow:hidden;
            background:var(--panel); cursor:pointer; padding:6px; }
.duelpane:hover { border-color:var(--accent); }
.duelpane .cap { padding:5px 4px 2px; font-size:11.5px; color:var(--dim); }
#board { width:320px; flex:0 0 320px; border:1px solid var(--line); border-radius:8px;
         background:var(--panel); padding:8px; max-height:82vh; overflow:auto; }
.brow { display:flex; gap:7px; align-items:center; padding:3px 2px; font-size:11.5px;
        border-bottom:1px solid #1e222b; cursor:pointer; border-radius:4px; }
.brow:hover { background:#1e222b; }
.brow img { width:56px; height:34px; object-fit:cover; border-radius:3px; background:#0b0d11;
            flex:0 0 56px; }
.brow .rk { color:var(--dim); width:22px; text-align:right; flex:0 0 22px; }
.brow .nm { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.brow .el { color:var(--dim); font-variant-numeric:tabular-nums; }
.brow .pk { flex:0 0 24px; width:24px; height:22px; border-radius:5px; font-size:10.5px;
            border:1px solid var(--line); color:var(--dim); background:#1a1d24;
            display:flex; align-items:center; justify-content:center; }
.brow .pk:hover { border-color:var(--keep); color:var(--keep); }
.brow .pk.on { background:var(--keep); border-color:var(--keep); color:#04150d; font-weight:700; }
.brow.top .nm { color:var(--keep); font-weight:700; }
.brow.picked .nm::before { content:"\2605 "; }
.brow.duelling { background:#1b2430; }

#lb { position:fixed; inset:0; background:#000000ee; display:none; z-index:20; }
#lb.on { display:flex; flex-direction:column; }
#lb .top { padding:8px 12px; color:var(--dim); display:flex; gap:12px; }
#lb .wrap { flex:1; overflow:auto; padding:0 12px 12px; }
#toast { position:fixed; bottom:14px; right:14px; background:#1a1d24; border:1px solid var(--line);
         padding:8px 12px; border-radius:8px; opacity:0; transition:opacity .2s; z-index:30; }
#toast.on { opacity:1; }
</style></head>
<body>
<header>
  <div class="row">
    <b id="title">video picker</b>
    <div class="spacer"></div>
    <div id="modes">
      <button data-mode="triage">1. Triage</button>
      <button data-mode="shortlist">2. Shortlist</button>
      <button data-mode="compare">3. Compare</button>
    </div>
    <button id="variant">variant</button>
    <button id="reload" title="re-read the state file from disk">Reload</button>
    <button class="primary" id="save">Save</button>
  </div>
  <div class="row hint" id="status"></div>
  <div class="row hint" id="keys"></div>
  <div class="bar"><div id="prog"></div></div>
</header>
<main>
  <section id="triage"><div class="stage">
    <div class="set" id="tSet"></div>
    <div class="verdict">
      <button class="vchip reject" data-v="reject">Reject <kbd>x</kbd></button>
      <button class="vchip maybe" data-v="maybe">Maybe <kbd>m</kbd></button>
      <button class="vchip keep" data-v="keep">Keep <kbd>f</kbd></button>
      <button id="tPrev">&larr;</button><button id="tNext">&rarr;</button>
      <button id="tUndecided">next undecided</button>
      <button id="tPick">pick <kbd>space</kbd></button>
    </div>
    <div class="meta" id="tMeta"></div>
  </div></section>

  <section id="shortlist">
    <div class="row" style="margin-bottom:8px">
      <span class="hint">show</span>
      <select id="sFilter"><option value="keep">keeps</option>
        <option value="keepmaybe" selected>keeps + maybes</option>
        <option value="picked">picked only</option>
        <option value="all">everything</option></select>
      <span class="hint">sort</span>
      <select id="sSort"><option value="elo">elo / pick order</option>
        <option value="order" selected>id order</option></select>
      <span class="hint">method</span><select id="sMethod"></select>
      <span class="hint">cols</span><input id="cols" type="number" min="1" max="8" value="4" style="width:52px">
      <label class="hint"><input type="checkbox" id="sPlay" style="width:auto"> play all</label>
      <button id="sCommit">fill picks from top of list</button>
      <button id="sClear">clear picks</button>
    </div>
    <div id="grid"></div>
  </section>

  <section id="compare">
    <div id="duel"></div>
    <div id="board"></div>
  </section>
</main>
<div id="lb"><div class="top"><b id="lbTitle"></b><span id="lbSub"></span></div>
  <div class="wrap"><div class="set" id="lbSet"></div></div></div>
<div id="toast"></div>
<script>
const $ = s => document.querySelector(s);
let D = null, mode = 'triage', variant = 0, cur = 0, dirty = false, method = 'ours';
let S = {};                       // tab -> {picks, verdicts, elo, duels, n}
let hist = [];                    // undo stack of {sid, prev}
let duelPair = null, lastPair = '';
const TAB = 'set', K_ELO = 24, ELO0 = 1200;

function blank(){ return {picks:[], verdicts:{}, elo:{}, duels:0, n:{}}; }
function T(){ if (!S[TAB]) S[TAB] = blank(); return S[TAB]; }
function eloOf(sid){ const e = T().elo[sid]; return e === undefined ? ELO0 : e; }
function nOf(sid){ return (T().n || {})[sid] || 0; }
function target(){ return D.target; }
function toast(m){ const t=$('#toast'); t.textContent=m; t.classList.add('on');
                   setTimeout(()=>t.classList.remove('on'), 1600); }

/* ---------- item helpers ---------- */
function pool(){ return D.items; }
function byId(sid){ return D.items.find(i=>i.sid===sid); }
function variantKey(){ return D.variants[variant].key; }
// fullres has no overlay render, so any method missing the current variant
// falls back to its plain video rather than dropping out of the set.
function src(it, m){
  const v = it.videos[m]; if (!v) return null;
  return v[variantKey()] || v['videos'] || Object.values(v)[0];
}
function label(it){ return '#' + it.sid; }
function nMissing(it){ return D.methods.filter(m=>!it.videos[m.key]).length; }

function setHTML(it, opts){
  const o = opts || {}, cols = o.cols || 2, poster = o.poster;
  const tiles = D.methods.map(m => {
    const p = src(it, m.key);
    if (!p) return `<div class="tile missing">${m.label}: missing</div>`;
    const meta = (it.meta && it.meta[m.key]) || {};
    const sub = meta.gen_time_sec ? `${meta.gen_time_sec.toFixed(1)}s`
              + (meta.token_ratio ? ` · ${(100*meta.token_ratio).toFixed(0)}% tok` : '') : '';
    const media = poster
      ? `<img loading="lazy" src="/poster?w=${o.w||480}&p=${encodeURIComponent(p)}">`
      : `<video src="/video?p=${encodeURIComponent(p)}" muted loop playsinline autoplay
                preload="auto" data-sync="1"></video>`;
    return `<div class="tile">${media}<span class="tag">${m.label}</span>`
         + (sub ? `<span class="sub">${sub}</span>` : '') + `</div>`;
  }).join('');
  return {html: tiles, cols};
}
function fillSet(el, it, opts){
  const r = setHTML(it, opts);
  el.style.gridTemplateColumns = `repeat(${r.cols}, minmax(0,1fr))`;
  el.innerHTML = r.html;
  syncPlay(el);
}
// Restart every clip in a container together: a foveation artifact is only
// legible when the four videos are on the same frame.
function syncPlay(el){
  const vs = [...el.querySelectorAll('video[data-sync]')];
  vs.forEach(v => {
    const go = () => { v.currentTime = 0; v.play().catch(()=>{}); };
    if (v.readyState >= 2) go(); else v.addEventListener('loadeddata', go, {once:true});
    v.addEventListener('ended', ()=>{ v.currentTime = 0; v.play().catch(()=>{}); });
  });
}
function restartAll(){ document.querySelectorAll('video[data-sync]').forEach(v=>{
  v.currentTime = 0; v.play().catch(()=>{}); }); }
function togglePlayAll(){
  const vs = [...document.querySelectorAll('video[data-sync]')];
  const anyPlaying = vs.some(v=>!v.paused);
  vs.forEach(v => anyPlaying ? v.pause() : v.play().catch(()=>{}));
}

/* ---------- state ---------- */
function setVerdict(sid, v){
  const t = T();
  hist.push({sid, prev: t.verdicts[sid]});
  if (t.verdicts[sid] === v) delete t.verdicts[sid]; else t.verdicts[sid] = v;
  if (t.verdicts[sid] !== 'keep' && t.verdicts[sid] !== 'maybe'){
    const i = t.picks.indexOf(sid); if (i>=0) t.picks.splice(i,1);
  }
  markDirty();
}
function undo(){
  const h = hist.pop(); if (!h) return;
  const t = T();
  if (h.prev === undefined) delete t.verdicts[h.sid]; else t.verdicts[h.sid] = h.prev;
  const p = pool(), i = p.findIndex(x=>x.sid===h.sid); if (i>=0) cur = i;
  markDirty(); render();
}
function togglePick(sid){
  const t = T(), i = t.picks.indexOf(sid);
  if (i>=0) t.picks.splice(i,1);
  else { t.picks.push(sid); if (!t.verdicts[sid]) t.verdicts[sid] = 'keep'; }
  markDirty();
}
let saveTimer = null;
function markDirty(){ dirty = true; clearTimeout(saveTimer);
  saveTimer = setTimeout(()=>save(false, false), 900); }
async function save(explicit, force){
  const q = [explicit ? 'explicit=1' : '', force ? 'force=1' : ''].filter(Boolean).join('&');
  const r = await fetch('/api/state' + (q ? '?' + q : ''),
    {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tabs:S})});
  const j = await r.json();
  if (r.status === 409){
    // Someone else wrote the file. Never clobber it silently: let the user
    // choose between their in-browser picks and what is on disk.
    if (confirm('The state file changed on disk since it was loaded.\n\n'
                + 'OK = overwrite it with the picks in this window\n'
                + 'Cancel = keep editing (use Reload to take the file instead)'))
      return save(explicit, true);
    toast('not saved — file changed on disk');
    return;
  }
  dirty = false; renderStatus();
  if (explicit) toast('saved ' + j.txt);
}
async function reloadFromDisk(){
  if (dirty && !confirm('Discard unsaved changes and reload from disk?')) return;
  const j = await (await fetch('/api/reload')).json();
  S = {};
  for (const [k,v] of Object.entries(j.state.tabs || {}))
    S[k] = {picks:v.picks||[], verdicts:v.verdicts||{}, elo:v.elo||{},
            duels:v.duels||0, n:v.n||{}};
  hist = []; duelPair = null; dirty = false; cur = 0; render();
  const loaded = Object.entries(j.summary).map(([t,i])=>`${t} ${i.picks}`).join(', ');
  toast('loaded from disk — picks: ' + (loaded || 'none'));
}

/* ---------- triage ---------- */
function renderTriage(){
  const its = pool();
  if (!its.length){ $('#tSet').innerHTML = '<p class="hint">nothing to triage</p>'; return; }
  cur = Math.max(0, Math.min(cur, its.length-1));
  const it = its[cur], t = T(), v = t.verdicts[it.sid];
  fillSet($('#tSet'), it, {cols: D.methods.length <= 2 ? D.methods.length : 2});
  document.querySelectorAll('.vchip').forEach(b=> b.classList.toggle('on', b.dataset.v===v));
  const rank = t.picks.indexOf(it.sid);
  $('#tMeta').innerHTML = `<b>${label(it)}</b>`
    + (rank>=0 ? ` <span style="color:var(--keep)">★ pick #${rank+1}</span>` : '')
    + ` <span class="hint">${cur+1}/${its.length} · ${D.variants[variant].label}`
    + (nMissing(it) ? ` · <span style="color:var(--reject)">${nMissing(it)} missing</span>` : '')
    + `</span><div class="hint" style="margin-top:3px">${it.prompt||''}</div>`;
}
function step(d, skipDecided){
  const its = pool();
  let i = cur + d;
  if (skipDecided){ const t = T(); while (i>=0 && i<its.length && t.verdicts[its[i].sid]) i += d; }
  cur = Math.max(0, Math.min(i, its.length-1));
  render();
}
function judge(v){
  const its = pool(); if (!its[cur]) return;
  setVerdict(its[cur].sid, v);
  if (cur < its.length-1) cur++;
  render();
}

/* ---------- shortlist ---------- */
function shortlistItems(){
  const t = T(), f = $('#sFilter').value;
  let its = pool().filter(it => {
    const v = t.verdicts[it.sid];
    if (f==='all') return true;
    if (f==='picked') return t.picks.includes(it.sid);
    if (f==='keep') return v==='keep';
    return v==='keep' || v==='maybe';
  });
  if ($('#sSort').value==='elo') its = its.slice().sort((a,b)=>{
      const pa=t.picks.indexOf(a.sid), pb=t.picks.indexOf(b.sid);
      if (pa>=0 || pb>=0) return (pa<0?1e6:pa) - (pb<0?1e6:pb);
      return eloOf(b.sid) - eloOf(a.sid); });
  return its;
}
function renderShortlist(){
  const its = shortlistItems(), t = T(), g = $('#grid'), playAll = $('#sPlay').checked;
  g.style.gridTemplateColumns = `repeat(${$('#cols').value}, minmax(0,1fr))`;
  cur = Math.max(0, Math.min(cur, its.length-1));
  g.innerHTML = its.map((it,idx)=>{
    const r = t.picks.indexOf(it.sid), v = t.verdicts[it.sid] || 'none';
    const p = src(it, method) || src(it, D.methods[0].key);
    const media = !p ? '<div class="tile missing">missing</div>'
      : playAll ? `<video src="/video?p=${encodeURIComponent(p)}" muted loop playsinline autoplay
                          preload="metadata" data-sync="1"></video>`
                : `<img loading="lazy" src="/poster?w=420&p=${encodeURIComponent(p)}"
                        data-src="${p}">`;
    return `<div class="card ${idx===cur?'cur':''} ${r>=0?'picked':''}" data-i="${idx}">
      ${r>=0?`<div class="badge">${r+1}</div>`:''}<div class="vdot ${v}"></div>
      ${media}
      <div class="cap"><b>${label(it)}</b><span class="p">${it.prompt||''}</span></div></div>`;
  }).join('');
  [...g.children].forEach(c => {
    c.onclick = e => {
      cur = +c.dataset.i;
      if (e.shiftKey) zoom(); else { togglePick(its[cur].sid); render(); }
    };
    // Hover swaps the poster for the real clip, so scanning 100 cards does not
    // mean decoding 100 videos at once.
    const im = c.querySelector('img[data-src]');
    if (im) c.onmouseenter = () => {
      const v = document.createElement('video');
      v.src = '/video?p=' + encodeURIComponent(im.dataset.src);
      v.muted = v.loop = v.autoplay = v.playsInline = true;
      v.dataset.sync = '1';
      im.replaceWith(v);
    };
  });
}

/* ---------- compare (elo duels) ---------- */
function duelPool(){
  const t = T();
  const its = pool().filter(it => ['keep','maybe'].includes(t.verdicts[it.sid])
                                  || t.picks.includes(it.sid));
  return its.length >= 2 ? its : pool();
}
function nextPair(){
  const p = duelPool();
  if (p.length < 2) return null;
  // least-compared first, opponent = closest rating (informative duels)
  const sorted = p.slice().sort((a,b)=>nOf(a.sid)-nOf(b.sid));
  for (let tries=0; tries<40; tries++){
    const a = sorted[Math.floor(Math.random()*Math.min(6, sorted.length))];
    const others = p.filter(x=>x.sid!==a.sid)
      .sort((x,y)=>Math.abs(eloOf(x.sid)-eloOf(a.sid)) - Math.abs(eloOf(y.sid)-eloOf(a.sid)));
    const b = others[Math.floor(Math.random()*Math.min(4, others.length))];
    const key = [a.sid,b.sid].sort().join('|');
    if (key !== lastPair) { lastPair = key; return [a,b]; }
  }
  return [sorted[0], p.find(x=>x.sid!==sorted[0].sid)];
}
function renderCompare(){
  const t = T();
  if (!duelPair) duelPair = nextPair();
  const d = $('#duel');
  if (!duelPair){ d.innerHTML = '<p class="hint">keep at least two sets in triage first</p>'; }
  else {
    d.innerHTML = duelPair.map((it,i)=>
      `<div class="duelpane" data-i="${i}"><div class="cap"><b>${label(it)}</b>
        — elo ${Math.round(eloOf(it.sid))}
        <span class="hint">${(it.prompt||'').slice(0,90)}</span></div>
       <div class="set" data-set="${i}"></div></div>`).join('');
    d.querySelectorAll('.set[data-set]').forEach(el =>
      fillSet(el, duelPair[+el.dataset.set], {cols:2}));
    d.querySelectorAll('.duelpane').forEach(p => p.onclick = e => {
      if (e.shiftKey) showLightbox(duelPair, +p.dataset.i); else win(+p.dataset.i);
    });
  }
  const board = duelPool().map(it=>({sid:it.sid, it, r:eloOf(it.sid)}))
                          .sort((a,b)=>b.r-a.r);
  // Rows carry a thumbnail and open the set on click: an id and a rating alone
  // are not enough to tell what you are ranking.
  const inDuel = duelPair ? duelPair.map(it=>it.sid) : [];
  $('#board').innerHTML = `<div class="row" style="margin-bottom:6px">
      <b>leaderboard</b><span class="hint">${t.duels||0} duels · click a row to see it</span></div>
    <button id="bCommit" style="width:100%;margin-bottom:6px">commit top ${target()} to picks</button>
    <div id="boardRows">${board.map((b,i)=>{
      const p = src(b.it, method) || src(b.it, D.methods[0].key);
      return `<div class="brow ${i<target()?'top':''} ${t.picks.includes(b.sid)?'picked':''} `
      + `${inDuel.includes(b.sid)?'duelling':''}" data-i="${i}">`
      + (p?`<img loading="lazy" src="/poster?w=160&p=${encodeURIComponent(p)}">`:'')
      + `<span class="rk">${i+1}</span><span class="nm" title="${(b.it.prompt||'').replace(/"/g,'&quot;')}">`
      + `${label(b.it)}</span><span class="el">${Math.round(b.r)}</span>`
      + `<span class="pk ${t.picks.includes(b.sid)?'on':''}" data-pick="${i}" `
      + `title="pick / unpick this one">${t.picks.indexOf(b.sid) >= 0
            ? t.picks.indexOf(b.sid) + 1 : '+'}</span></div>`; }).join('')}</div>`;
  $('#boardRows').querySelectorAll('.brow').forEach(row => {
    row.onclick = () => showLightbox(board.map(b=>b.it), +row.dataset.i);
  });
  // The pick toggle sits inside the row, so it must not also open the set.
  $('#boardRows').querySelectorAll('.pk').forEach(button => {
    button.onclick = event => {
      if (event && event.stopPropagation) event.stopPropagation();
      togglePick(board[+button.dataset.pick].sid);
      render();
    };
  });
  $('#bCommit').onclick = ()=>{ T().picks = board.slice(0, target()).map(b=>b.sid);
    board.slice(0, target()).forEach(b=>{ if(!T().verdicts[b.sid]) T().verdicts[b.sid]='keep'; });
    markDirty(); toast('picks = top ' + target()); render(); };
}
function win(i){
  if (!duelPair) return;
  const t = T(), a = duelPair[i].sid, b = duelPair[1-i].sid;
  const ra = eloOf(a), rb = eloOf(b);
  const ea = 1/(1+Math.pow(10,(rb-ra)/400));
  t.elo[a] = ra + K_ELO*(1-ea); t.elo[b] = rb - K_ELO*(1-ea);
  t.n = t.n || {}; t.n[a] = (t.n[a]||0)+1; t.n[b] = (t.n[b]||0)+1;
  t.duels = (t.duels||0)+1;
  duelPair = nextPair(); markDirty(); render();
}

/* ---------- chrome ---------- */
const KEYS = {
  triage: '<kbd>f</kbd> keep · <kbd>m</kbd> maybe · <kbd>x</kbd> reject · <kbd>&larr;</kbd><kbd>&rarr;</kbd> move · <kbd>n</kbd> next undecided · <kbd>space</kbd> pick · <kbd>u</kbd> undo · <kbd>g</kbd> variant · <kbd>0</kbd> restart clips · <kbd>p</kbd> play/pause · <kbd>z</kbd> zoom',
  shortlist: 'click = pick/unpick · shift-click = zoom · hover = play · arrows move · <kbd>g</kbd> variant · <kbd>z</kbd> zoom',
  compare: '<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or click = winner · <kbd>=</kbd> skip pair · <kbd>0</kbd> restart clips · <kbd>z</kbd> zoom',
};
function renderStatus(){
  const t = T(), its = pool();
  const v = Object.entries(t.verdicts).filter(([sid])=>its.some(i=>i.sid===sid)).map(([,x])=>x);
  const keep = v.filter(x=>x==='keep').length, maybe = v.filter(x=>x==='maybe').length,
        rej = v.filter(x=>x==='reject').length, done = keep+maybe+rej;
  $('#status').innerHTML =
      `<span>${its.length} sets</span>`
    + `<span style="color:var(--keep)">keep ${keep}</span>`
    + `<span style="color:var(--maybe)">maybe ${maybe}</span>`
    + `<span style="color:var(--reject)">reject ${rej}</span>`
    + `<span>triaged ${done}/${its.length}</span>`
    + `<span style="color:${t.picks.length===target()?'var(--keep)':'var(--dim)'}">`
    + `<b>picked ${t.picks.length}/${target()}</b></span>`
    + `<span class="hint">${t.picks.join(' ')}</span>`
    + `<span class="spacer"></span><span class="hint">${D.state_path}</span>`
    + (dirty ? '<span style="color:#ffb454">saving…</span>' : '<span class="hint">saved</span>');
  $('#prog').style.width = (100*done/Math.max(1,its.length)) + '%';
  $('#keys').innerHTML = KEYS[mode];
  $('#variant').textContent = 'variant: ' + D.variants[variant].label;
  document.querySelectorAll('#modes button').forEach(b=>b.classList.toggle('on', b.dataset.mode===mode));
}
function render(){
  ['triage','shortlist','compare'].forEach(m => $('#'+m).classList.toggle('on', m===mode));
  if (mode==='triage') renderTriage();
  else if (mode==='shortlist') renderShortlist();
  else renderCompare();
  renderStatus();
  if (mode==='shortlist'){ const c = $('#grid').children[cur]; if (c) c.scrollIntoView({block:'nearest'}); }
}
function setMode(m){ mode = m; render(); }
function cycleVariant(){ variant = (variant+1) % D.variants.length; render(); }

let lb = {list: [], i: 0};
function showLightbox(list, index){
  if (!list.length) return;
  lb = {list, i: Math.max(0, Math.min(index, list.length - 1))};
  const it = lb.list[lb.i], rank = T().picks.indexOf(it.sid);
  $('#lbTitle').textContent = (rank >= 0 ? `★ pick #${rank + 1}  ` : '') + label(it);
  $('#lbSub').textContent = `${lb.i + 1}/${lb.list.length}   ${it.prompt || ''}`;
  fillSet($('#lbSet'), it, {cols: 2});
  $('#lbSet').style.maxWidth = '100%';
  $('#lb').classList.add('on');
}
function closeLightbox(){ $('#lb').classList.remove('on'); $('#lbSet').innerHTML = ''; }
function stepLightbox(d){ showLightbox(lb.list, lb.i + d); }
function zoom(){
  if (mode === 'compare') return showLightbox(duelPair || pool(), 0);
  const its = mode === 'shortlist' ? shortlistItems() : pool();
  showLightbox(its, cur);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if ($('#lb').classList.contains('on')){
    if (e.key==='Escape' || e.key==='z') closeLightbox();
    else if (e.key==='ArrowRight' || e.key==='l') stepLightbox(+1);
    else if (e.key==='ArrowLeft' || e.key==='h') stepLightbox(-1);
    else if (e.key==='0') restartAll();
    else if (e.key===' '){ e.preventDefault();
      const it = lb.list[lb.i]; if (it){ togglePick(it.sid); render(); showLightbox(lb.list, lb.i); } }
    return;
  }
  const k = e.key;
  if (k==='s'){ e.preventDefault(); save(true, false); return; }
  if (k==='t'){ setMode('triage'); return; }
  if (k==='r'){ setMode('shortlist'); return; }
  if (k==='c'){ setMode('compare'); return; }
  if (k==='g'){ cycleVariant(); return; }
  if (k==='z'){ zoom(); return; }
  if (k==='u'){ undo(); return; }
  if (k==='0'){ restartAll(); return; }
  if (k==='p'){ togglePlayAll(); return; }
  if (mode==='triage'){
    if (k==='f') judge('keep');
    else if (k==='m') judge('maybe');
    else if (k==='x') judge('reject');
    else if (k===' '){ e.preventDefault(); const it = pool()[cur];
      if (it){ togglePick(it.sid); render(); } }
    else if (k==='ArrowRight'||k==='l') step(+1,false);
    else if (k==='ArrowLeft'||k==='h') step(-1,false);
    else if (k==='n') step(+1,true);
  } else if (mode==='shortlist'){
    const cols = +$('#cols').value, n = shortlistItems().length;
    if (k===' '){ e.preventDefault(); const it = shortlistItems()[cur];
      if (it){ togglePick(it.sid); render(); } }
    else if (k==='ArrowRight'||k==='l'){ cur=Math.min(cur+1,n-1); render(); }
    else if (k==='ArrowLeft'||k==='h'){ cur=Math.max(cur-1,0); render(); }
    else if (k==='ArrowDown'||k==='j'){ cur=Math.min(cur+cols,n-1); render(); }
    else if (k==='ArrowUp'||k==='k'){ cur=Math.max(cur-cols,0); render(); }
  } else {
    if (k==='ArrowLeft') win(0);
    else if (k==='ArrowRight') win(1);
    else if (k==='='){ duelPair = nextPair(); render(); }
  }
});
window.addEventListener('beforeunload', e => { if (dirty){ e.preventDefault(); e.returnValue=''; } });

(async () => {
  D = await (await fetch('/api/data')).json();
  for (const [k,v] of Object.entries(D.state.tabs || {}))
    S[k] = {picks:v.picks||[], verdicts:v.verdicts||{}, elo:v.elo||{}, duels:v.duels||0, n:v.n||{}};
  method = (D.methods.find(m=>m.key==='ours') || D.methods[0] || {key:''}).key;
  $('#title').textContent = `video picker — ${D.set_name} (${D.items.length} sets × ${D.methods.length} methods)`;
  document.title = `picker · ${D.set_name}`;
  $('#sMethod').innerHTML = D.methods.map(m=>
    `<option value="${m.key}" ${m.key===method?'selected':''}>${m.label}</option>`).join('');
  $('#sMethod').onchange = e => { method = e.target.value; render(); };
  document.querySelectorAll('#modes button').forEach(b=> b.onclick = ()=>setMode(b.dataset.mode));
  document.querySelectorAll('.vchip').forEach(b=> b.onclick = ()=>judge(b.dataset.v));
  $('#tPrev').onclick = ()=>step(-1,false); $('#tNext').onclick = ()=>step(+1,false);
  $('#tUndecided').onclick = ()=>step(+1,true);
  $('#tPick').onclick = ()=>{ const it = pool()[cur]; if (it){ togglePick(it.sid); render(); } };
  $('#sFilter').onchange = $('#sSort').onchange = $('#sPlay').onchange
    = $('#cols').oninput = ()=>{ cur=0; render(); };
  $('#sCommit').onclick = ()=>{ const its = shortlistItems().slice(0, target());
    T().picks = its.map(i=>i.sid); markDirty();
    toast('picks = top '+its.length+' of current list'); render(); };
  $('#sClear').onclick = ()=>{ T().picks = []; markDirty(); render(); };
  $('#variant').onclick = cycleVariant;
  $('#save').onclick = ()=>save(true);
  $('#reload').onclick = reloadFromDisk;
  $('#lb').onclick = e => { if (!e.target.closest('#lbSet')) closeLightbox(); };
  render();
})();
</script></body></html>
"""


def warm_posters(root: Path, items: list[Item], methods: list[str],
                 state: State, widths: tuple[int, ...], poster_at: float) -> None:
    """Pre-render the shortlist/leaderboard thumbnails, so the first scroll
    through 200 sets is not 200 ffmpeg calls in the request path."""
    poster_dir = state.path.parent / ".posters"
    poster_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in items:
        for m in methods:
            # Same variant fallback the UI uses, so the warmed files are the
            # ones the grid actually asks for.
            v = it.videos.get(m) or {}
            rel = v.get(VARIANTS[0]) or v.get("videos")
            if not rel:
                continue
            src = root / rel
            st = src.stat()
            for w in widths:
                key = hashlib.sha1(
                    f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{w}|{poster_at}".encode()
                ).hexdigest()
                dst = poster_dir / f"{key}.jpg"
                if not dst.is_file():
                    todo.append((src, dst, w))
    print(f"warming {len(todo)} posters into {poster_dir}", flush=True)
    for i, (src, dst, w) in enumerate(todo, 1):
        poster_jpeg(src, dst, w, poster_at)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("posters warm", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="eval set dir, e.g. /local/brian/foveated_diffusion_drive/eval200")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--target", type=int, default=10,
                    help="how many 4-video sets to end up picking (default 10)")
    ap.add_argument("--prompts", type=Path, default=None,
                    help="csv with columns [id,]prompt, to caption each set")
    ap.add_argument("--picks", type=Path, default=None,
                    help="state json (default results/video_picks/<set>/video_picks.json)")
    ap.add_argument("--poster-at", type=float, default=0.5,
                    help="timestamp (s) of the frame used for grid thumbnails")
    ap.add_argument("--warm-posters", action="store_true",
                    help="pre-render every thumbnail with ffmpeg before serving")
    ap.add_argument("--warm-only", action="store_true",
                    help="warm the poster cache and exit without serving")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    prompts = read_prompts(args.prompts) if args.prompts else {}
    items, methods = scan(root, prompts)
    if not items:
        raise SystemExit(f"no <method>/<run>/<step>/videos/*.mp4 found under {root}")

    repo = Path(__file__).resolve().parent.parent
    state_path = (args.picks or (repo / "results" / "video_picks" / root.name
                                 / "video_picks.json")).expanduser().resolve()
    state = State(state_path, args.target)

    complete = sum(1 for it in items if it.n_methods == len(methods))
    print(f"root      : {root}")
    print(f"sets      : {len(items)} ({complete} complete)   methods: {', '.join(methods)}")
    print(f"target    : {args.target} sets")
    print(f"state file: {state_path}")
    if state.tabs:
        print("loaded    : " + ", ".join(
            f"{info['picks']} picks / {info['verdicts']} triaged"
            for info in state.summary().values()))
    else:
        print("loaded    : nothing yet (fresh selection)")
    if prompts:
        print(f"prompts   : {args.prompts} ({len(prompts)} rows)")

    if args.warm_posters or args.warm_only:
        warm_posters(root, items, methods, state, (160, 420, 480), args.poster_at)
    if args.warm_only:
        return

    app = build_app(root, state, items, methods, args.poster_at)
    print(f"serving   : http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
