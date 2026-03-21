# make_video

Use this command whenever the user invokes `/make_video` or asks to:
- Stack / tile / arrange videos in a grid
- Concatenate / join / chain videos together (with an optional pause/sleep between them)
- Add text / label / caption onto a video

All video operations must be run using the `imp_tome` conda environment:
```
conda run -n imp_tome python /home/brianchc/foveated_diffusion/eval/make_video.py <subcommand> ...
```

---

## Functions

### `stack_videos` — grid layout

Call when the user wants to arrange multiple videos side by side or in a grid.

**CLI subcommand:** `stack`

```bash
conda run -n imp_tome python /home/brianchc/foveated_diffusion/eval/make_video.py stack \
  video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
  --output stacked.mp4 \
  --grid-x <columns> \
  --grid-y <rows> \
  --margin <pixels>         # optional, default 10
  --bg-color <R,G,B>        # optional, default 0,0,0 (black)
  --target-size <WxH>       # optional, default 1920x1080; pass 'none' to disable
```

- `--grid-x` × `--grid-y` must be >= number of input videos.
- Videos are filled left-to-right, top-to-bottom.
- All clips are resized to fit within `--target-size` (≈1080p) maintaining aspect ratio.
- Shorter clips hold their last frame until the longest clip ends.

### `concat_videos` — sequential with pause

Call when the user wants to join videos one after another, with a gap between them.

**CLI subcommand:** `concat`

```bash
conda run -n imp_tome python /home/brianchc/foveated_diffusion/eval/make_video.py concat \
  video1.mp4 video2.mp4 video3.mp4 \
  --output concatenated.mp4 \
  --sleep <seconds>   # optional, default 1.0
  --fps <fps>         # optional, default from first video
  --freeze            # optional: hold last frame during gap instead of black
```

- Clips play in the order given.
- Gap between clips is black by default; `--freeze` holds the last frame instead.
- All clips are resized to the first video's dimensions.

### `add_text_overlay` — burn text onto a video

Call when the user wants to add a label, caption, or title to a video.

**CLI subcommand:** `text`

```bash
conda run -n imp_tome python /home/brianchc/foveated_diffusion/eval/make_video.py text \
  input.mp4 \
  --output output.mp4 \
  --text "My Label" \
  --font /home/brianchc/fonts/cmu-serif/cmunrm.ttf   # optional, this is the default
  --font-size 48          # optional, default 48
  --color 255,255,255     # optional, R,G,B default white
  --position bottom-center  # optional, default bottom-center
  --padding 20            # optional, pixel margin from edge
```

Available fonts in `/home/brianchc/fonts/cmu-serif/`:
- `cmunrm.ttf` — CMU Serif Regular (default)
- `cmunbx.ttf` — CMU Serif Bold
- `cmunti.ttf` — CMU Serif Italic
- `cmunbi.ttf` — CMU Serif Bold Italic

Positions: `top-left`, `top-center`, `top-right`, `center`, `bottom-left`, `bottom-center`, `bottom-right`

---

### `images_to_video` — compile images into a video

Call when the user wants to turn a folder of images into a video.

**CLI subcommand:** `images`

```bash
conda run -n imp_tome python /home/brianchc/foveated_diffusion/eval/make_video.py images \
  /path/to/folder \
  --output video.mp4 \
  --fps 24              # optional, default 24
  --pattern "*.png"     # optional, default *.png
  --order 003 001 002   # optional: explicit filenames or bare IDs in desired order
```

- Default: all images matching `--pattern` sorted by filename.
- `--order` accepts full filenames (`img_003.png`) or bare IDs (`003`) — zero-padding is handled automatically.

---

## Decision guide

| User intent | Subcommand |
|---|---|
| "stack", "tile", "grid", "side by side", "arrange", "X×Y" | `stack` |
| "concatenate", "join", "chain", "one after another", "append", "pause between" | `concat` |
| "add text", "label", "caption", "overlay text", "title" | `text` |
| "images to video", "compile images", "slideshow", "frames to video" | `images` |

When the user's request is ambiguous, ask whether they want a grid layout or sequential concatenation before running.

---

## Information to gather before running

**For all subcommands:**
- Input video paths (ask if not provided)
- Output path: always save to `/home/brianchc/foveated_diffusion/twitter_videos/<descriptive_name>.mp4` unless the user specifies otherwise. Create the directory if it doesn't exist.

**For `stack` only:**
- Grid dimensions (`--grid-x`, `--grid-y`) — infer from number of videos if obvious (e.g. 4 videos → 2×2), otherwise ask
- Margin (optional, default 10px)

**For `concat` only:**
- Sleep/pause duration between clips (optional, default 1 second)
- Whether gap should be black or frozen last frame (default: black)

**For `text` only:**
- The text string to render
- Font variant, size, color, position (all optional — defaults are CMU Serif Regular, 48pt, white, bottom-center)
