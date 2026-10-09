# Picking and joining the qualitative 4-video sets

`video_picker.py` picks the ids; `join_baseline_videos.py` turns the picks
into labelled side-by-side videos.

## Video picker

`scripts/video_picker.py` is the video sibling of LoT's `scripts/qual_picker.py`.
It serves an eval tree of generated videos as a local web page and turns *"the
best 10 of 200"* into a sequence of small decisions.

An **item** is a prompt id, shown as the whole **4-video set** — `fullres`,
`naive`, `tome`, `ours` — playing in sync in a 2×2 grid. You are picking the id,
not a single video.

## Run it

```bash
scripts/serve_video_picker.sh                       # eval200 on :8090, 10 picks
scripts/serve_video_picker.sh spline 8091 --target 3
scripts/serve_video_picker.sh eval200_720p 8092
```

Then open `http://localhost:<port>` (over ssh: `ssh -L 8090:localhost:8090 …`).

The wrapper activates the `imp_tome` conda env (the one that has flask) and
resolves a bare set name under `$DRIVE`
(`/local/brian/foveated_diffusion_drive` by default). To call the script
directly:

```bash
conda activate imp_tome
python scripts/video_picker.py /local/brian/foveated_diffusion_drive/eval200 \
    --port 8090 --target 10 [--prompts eval/prompts/<file>.csv]
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--target N` | how many sets you intend to end up with (default 10); drives the progress readout and the "commit top N" button |
| `--prompts <csv>` | captions each set; columns `[id,]prompt`, ids default to 1-indexed row order |
| `--picks <path>` | state file location (default below) |
| `--warm-posters` | pre-render every grid thumbnail with ffmpeg before serving, instead of on first scroll (`--warm-only` warms the cache and exits) |
| `--poster-at 0.5` | timestamp (s) of the frame used as the thumbnail |

## Input layout

Discovered from disk, nothing hardcoded except the display order of the four
known method names:

```
<set>/<method>/<run_name>/<step>/videos/<id>.mp4
                                 /videos_with_circle/<id>.mp4   # not for fullres
                                 /runtimes.jsonl                # gen time, token ratio
```

The picker opens on the **fovea-circle overlays**; `g` toggles to the plain
videos and back. A method with no overlay render (i.e. `fullres`) falls back to
its plain video rather than dropping out of the set. Ids missing for a method
render as a red "missing" tile — `eval200_720p/tome` is only partly downloaded, for instance —
so an incomplete set is visible rather than silently different.

## The three stages

Switch with `t` / `r` / `c`.

1. **Triage** — one set at a time, as large as the window allows. Keep (`f`) /
   maybe (`m`) / reject (`x`) auto-advances. 200 sets is a few minutes, and you
   never compare more than one thing against your own taste.
2. **Shortlist** — a grid of only the keeps (and maybes), one method per cell
   (pick which in the toolbar). Cells are ffmpeg posters that swap to the real
   clip on hover, so scanning 100 cards does not decode 100 videos. Click to
   pick/unpick, shift-click to open the full 2×2 set.
3. **Compare** — pairwise duels between two full sets, Elo-ranked, with a
   leaderboard beside them. You never rank 200 sets; you answer "left or right?"
   a few dozen times and hit **commit top N to picks**.

Keys: `f`/`m`/`x` verdict · `←`/`→` move (or pick a duel winner) · `n` next
undecided · `space` pick/unpick · `u` undo · `g` variant · `p` play/pause all ·
`0` restart every clip in sync · `z` zoom · `=` skip duel · `s` save now.

## Output

State is written to (`<set>` = the input dir's name):

```
results/video_picks/<set>/video_picks.json    # verdicts, pick order, elo, duel counts
results/video_picks/<set>/video_picks.csv     # flat: tab,sid,rank,verdict,elo
results/video_picks/<set>/picks.txt           # the picked ids, one per line, in pick order
results/video_picks/<set>/.posters/           # ffmpeg thumbnail cache (disposable)
```

It is autosaved ~1 s after every change and reloaded at startup, so a session is
resumable and can be handed to a figure script directly:

```bash
readarray -t IDS < results/video_picks/eval200/picks.txt
python eval/make_video.py --ids "${IDS[@]}"      # or whatever consumes them
```

Nothing is ever written into the video tree itself.

Two behaviours worth knowing:

- **Conflict guard.** If the state file changed on disk since the server read it
  (a second browser tab, an external edit), the save is refused with a 409 and
  the browser asks whether to overwrite or reload — picks are never silently
  clobbered. **Reload** re-reads the file, which is also how you pick up an
  externally edited selection.
- **Elo is only as good as the duels.** Ratings start at 1200 and move by
  K = 24, so a set with two duels is not meaningfully ranked against one with
  twenty. Read the leaderboard as a shortlist ordering, not a score.

## Joining the picks — `join_baseline_videos.py`

Turns a `picks.txt` into one labelled side-by-side video per id.

The horizontal join is **not** reimplemented: the script imports and calls
`_stack_single_set` from
`/home/brianchc/fov_video/examples/wanvideo/model_training/validate_foveation/postprocess_videos.py`
(the `stack` subcommand behind `eval_scripts/stack_videos.sh`), so the
resize-to-common-height and the white `--margin` gap are exactly the logic
already used for the other stacked videos. On top of that it adds the fixed
column order, a white label band, and the prompt list.

```bash
conda activate imp_tome
python scripts/join_baseline_videos.py \
    --set /local/brian/foveated_diffusion_drive/eval200 \
    --picks results/video_picks/eval200/picks.txt \
    --out results/joined_videos/eval200_circle \
    --prompts eval/prompts/eval200_prompts_en.csv
```

Columns, left to right, fixed: **Full-Res | Token Merging | Naive
Mixed-Resolution | Ours**. `--variant` chooses the fovea-circle renders
(default) or the plain ones; `fullres` has no overlay render, so that column
falls back to its plain video.

| Flag | Meaning |
|---|---|
| `--variant` | `videos_with_circle` (default) or `videos` |
| `--margin 10` | white gap in px between columns, passed straight to the stacker |
| `--quality 8` | imageio/ffmpeg quality, 1-10 |
| `--label-height` / `--font-size` | override the band (defaults: 11% of frame height, 55% of that) |
| `--prompts <csv>` | `[id,]prompt`; without it the prompt column is left blank |
| `--prompts-only` | rewrite the prompt list in `--out` without re-encoding |
| `--stacker <path>` | where `postprocess_videos.py` lives |

Each output folder holds `<id>.mp4` plus `ids.txt`, `prompts.csv`
(`rank,id,video,prompt`) and a human-readable `prompts.txt`.

One gotcha baked in: the h264 writer silently **resizes** frames that are not a
multiple of 16, which would stretch the labels off their columns — so the final
frame is white-padded up to a macro block instead.

### Prompts

`eval/prompts/eval200_prompts.csv` is the source list for both `eval200` and
`eval200_720p` (ids 1-200) and is in Chinese;
`eval/prompts/eval200_prompts_en.csv` is the English translation, same ids, and
is what the joined folders quote.
