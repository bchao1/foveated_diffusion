#!/bin/bash
# Serve the 4-video-set picker for one eval set.
#
#   scripts/serve_video_picker.sh [set] [port] [extra args...]
#
#   set   name under $DRIVE (eval200 | eval200_720p | spline) or an absolute
#         path to a set dir. default: eval200
#   port  default 8090
#
# Examples
#   scripts/serve_video_picker.sh                       # eval200 on :8090
#   scripts/serve_video_picker.sh spline 8091 --target 3
#   scripts/serve_video_picker.sh eval200 8090 --warm-posters
set -euo pipefail

DRIVE=${DRIVE:-/local/brian/foveated_diffusion_drive}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SET=${1:-eval200}; shift || true
PORT=${1:-8090}; shift || true
[[ "$SET" = /* ]] || SET="$DRIVE/$SET"

# flask lives in the imp_tome env (the one the drive downloader uses)
if [[ -z "${CONDA_PREFIX:-}" || ! -x "$CONDA_PREFIX/bin/python" ]] \
   || ! "$CONDA_PREFIX/bin/python" -c "import flask" 2>/dev/null; then
  source /home/brianchc/miniconda3/etc/profile.d/conda.sh
  conda activate "${PICKER_ENV:-imp_tome}"
fi

exec python -u "$REPO/scripts/video_picker.py" "$SET" --port "$PORT" "$@"
