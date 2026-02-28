#!/usr/bin/env python3
"""
User study GUI for comparing foveated diffusion image quality.

Usage:
    python user_study.py <user_study_dir> [--num_pairs 60] [--output_dir .]

Each subdirectory of user_study_dir must contain:
    fixation_point.png, img_high_res.png, img_naive.png, img_ours.png, prompt.txt

num_pairs real comparisons are shown (must be divisible by 3) so that each of
the three method pairs (naive/high_res, high_res/ours, naive/ours) appears
exactly num_pairs//3 times. For each comparison a subfolder is sampled
independently and uniformly at random.
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

from psychopy import core, event, visual
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXATION_DURATION      = 5.0           # seconds (≥5 as specified)
FLASH_DURATION_S       = 1.000         # 1000 ms per image
NUM_PRACTICE_TRIALS    = 2             # practice comparisons shown before real ones
INTER_IMAGE_FIXATION_S = 1.0           # fixation shown between img1 and img2
BG_COLOR               = (-1, -1, -1)  # PsychoPy [-1,1] range: black

IMAGE_KEYS = ['high_res', 'naive', 'ours']
IMAGE_FILES = {
    'high_res': 'img_high_res.png',
    'naive':    'img_naive.png',
    'ours':     'img_ours.png',
}
ALL_PAIRS = list(itertools.combinations(IMAGE_KEYS, 2))  # 3 distinct pairs

QUESTION = ('quality', 'Which image has better overall visual quality?')
CSV_FIELDNAMES = ['trial_dir', 'pair_index', 'img1', 'img2', 'prompt', 'question', 'response']


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Foveated diffusion user study')
    parser.add_argument('user_study_dir', type=str,
                        help='Directory containing per-trial subdirectories')
    parser.add_argument('--num_pairs', type=int, default=60,
                        help='Total real comparisons shown; must be divisible by 3 (default: 60)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save result CSVs (default: current directory)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode: do not save a participant CSV')
    args = parser.parse_args()
    if args.num_pairs % 3 != 0:
        parser.error(f'--num_pairs must be divisible by 3, got {args.num_pairs}')
    return args


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def next_participant_path(output_dir):
    """Return the path for the next participant CSV (participant_0.csv, participant_1.csv, ...)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_id = 0
    while (output_dir / f'participant_{participant_id}.csv').exists():
        participant_id += 1
    return output_dir / f'participant_{participant_id}.csv'


# ---------------------------------------------------------------------------
# Trial directory loading & schedule building
# ---------------------------------------------------------------------------

def load_all_dirs(user_study_dir):
    """Return all subdirectories inside user_study_dir."""
    base = Path(user_study_dir)
    dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not dirs:
        raise ValueError(f'No subdirectories found in {user_study_dir}')
    return dirs


def build_schedule(all_dirs, num_pairs):
    """Return a shuffled list of (trial_dir, (key_a, key_b)) for real comparisons.

    Each of the 3 canonical pairs appears exactly num_pairs // 3 times.
    Subfolders are sampled independently with replacement.
    """
    per_pair = num_pairs // 3
    pair_types = ALL_PAIRS * per_pair
    random.shuffle(pair_types)
    return [(random.choice(all_dirs), pair) for pair in pair_types]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def compute_display_size(win, img_path):
    """Return (w, h) in PsychoPy 'height' units that fills the shorter screen dimension."""
    screen_w, screen_h = win.size
    screen_aspect = screen_w / screen_h
    with Image.open(img_path) as img:
        img_aspect = img.size[0] / img.size[1]
    if img_aspect <= screen_aspect:
        return (img_aspect, 1.0)
    else:
        return (screen_aspect, screen_aspect / img_aspect)


def load_image_stims(win, trial_dir, display_size):
    """Pre-load all comparison images and the fixation image as ImageStim objects."""
    stims = {}
    for key, filename in IMAGE_FILES.items():
        stims[key] = visual.ImageStim(
            win,
            image=str(trial_dir / filename),
            size=display_size,
            units='height',
        )
    stims['fixation'] = visual.ImageStim(
        win,
        image=str(trial_dir / 'fixation_point.png'),
        size=display_size,
        units='height',
    )
    return stims


def read_prompt(trial_dir):
    return (trial_dir / 'prompt.txt').read_text().strip()


# ---------------------------------------------------------------------------
# Display routines
# ---------------------------------------------------------------------------

def show_instructions(win):
    """Opening screen before the experiment starts."""
    msg = visual.TextStim(
        win,
        text=(
            'Foveated Image Quality Study\n\n'
            'You will see pairs of images flashed briefly.\n'
            'After each pair, answer 1 question about image quality.\n\n'
            'Use keys  1  and  2  to respond.\n'
            'Press  Escape  at any time to quit.\n\n'
            'Press any key to begin.'
        ),
        color='white',
        height=28,
        wrapWidth=900,
        alignText='center',
    )
    msg.draw()
    win.flip()
    keys = event.waitKeys(keyList=None)
    if 'escape' in keys:
        core.quit()


def show_fixation(win, stims, pair_idx, total_pairs, is_practice):
    """Display fixation_point.png with a 5-second countdown overlay."""
    label = visual.TextStim(
        win,
        text='Focus your gaze on the red dot.',
        pos=(0, -(win.size[1] // 2 - 50)),
        color='white',
        height=26,
        anchorVert='bottom',
    )
    countdown_stim = visual.TextStim(
        win,
        pos=(0, win.size[1] // 2 - 60),
        color='white',
        height=40,
        bold=True,
        anchorVert='top',
    )
    if is_practice:
        progress_text = 'PRACTICE'
    else:
        progress_text = f'Comparison {pair_idx} of {total_pairs}'
    progress = visual.TextStim(
        win,
        text=progress_text,
        pos=(-(win.size[0] // 2 - 20), win.size[1] // 2 - 20),
        color='gray',
        height=22,
        anchorHoriz='left',
        anchorVert='top',
    )

    total_seconds = int(FIXATION_DURATION)
    for remaining in range(total_seconds, 0, -1):
        countdown_stim.text = str(remaining)
        stims['fixation'].draw()
        label.draw()
        countdown_stim.draw()
        progress.draw()
        win.flip()
        for _ in range(10):
            if event.getKeys(keyList=['escape']):
                core.quit()
            core.wait(0.1)


def flash_pair(win, stim1, fixation_stim, stim2, flash_frames):
    """Flash stim1, show fixation for 1s, then flash stim2."""
    for _ in range(flash_frames):
        stim1.draw()
        win.flip()
    t_start = core.getTime()
    while core.getTime() - t_start < INTER_IMAGE_FIXATION_S:
        fixation_stim.draw()
        win.flip()
        if event.getKeys(keyList=['escape']):
            core.quit()
    for _ in range(flash_frames):
        stim2.draw()
        win.flip()
    win.flip()


def ask_question(win, question_text, pair_idx, total_pairs,
                 allow_repeat=False, is_practice=False):
    """Show a question and wait for a keypress.

    Returns '1', '2', or 'r' (only when allow_repeat=True and R is pressed).
    """
    if is_practice:
        progress_text = 'PRACTICE'
    else:
        progress_text = f'Comparison {pair_idx} of {total_pairs}'
    progress = visual.TextStim(
        win,
        text=progress_text,
        pos=(0, win.size[1] // 2 - 40),
        color='gray',
        height=22,
        anchorVert='top',
    )
    if is_practice:
        practice_banner = visual.TextStim(
            win,
            text='PRACTICE ROUND — responses will not be recorded',
            pos=(0, 180),
            color='yellow',
            height=26,
            bold=True,
            alignText='center',
        )
    question = visual.TextStim(
        win,
        text=question_text,
        pos=(0, 60),
        color='white',
        height=32,
        bold=True,
        wrapWidth=900,
        alignText='center',
    )
    options_text = 'Press  1  for  FIRST       Press  2  for  SECOND'
    if allow_repeat:
        options_text += '       Press  R  to  repeat'
    options = visual.TextStim(
        win,
        text=options_text,
        pos=(0, -60),
        color='lightgray',
        height=28,
        alignText='center',
    )

    progress.draw()
    if is_practice:
        practice_banner.draw()
    question.draw()
    options.draw()
    win.flip()

    key_list = ['1', '2', 'escape'] + (['r'] if allow_repeat else [])
    event.clearEvents()
    while True:
        keys = event.getKeys(keyList=key_list)
        if 'escape' in keys:
            core.quit()
        for k in keys:
            if k in ('1', '2'):
                return k
            if k == 'r':
                return 'r'
        core.wait(0.005)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(win, practice_schedule, real_schedule, csv_writer, csvfile):
    flash_frames = max(1, int(round(FLASH_DURATION_S * win.getActualFrameRate())))
    total_real = len(real_schedule)

    show_instructions(win)

    # --- Practice comparisons ---
    for trial_dir, (key_a, key_b) in practice_schedule:
        display_size = compute_display_size(win, trial_dir / 'img_high_res.png')
        stims = load_image_stims(win, trial_dir, display_size)

        ordered = [key_a, key_b]
        random.shuffle(ordered)
        img1_key, img2_key = ordered

        while True:
            show_fixation(win, stims, pair_idx=None, total_pairs=total_real, is_practice=True)
            flash_pair(win, stims[img1_key], stims['fixation'], stims[img2_key], flash_frames)
            response = ask_question(
                win, QUESTION[1],
                pair_idx=None, total_pairs=total_real,
                allow_repeat=True, is_practice=True,
            )
            if response != 'r':
                break

    # --- Real comparisons ---
    for pair_idx, (trial_dir, (key_a, key_b)) in enumerate(real_schedule, start=1):
        display_size = compute_display_size(win, trial_dir / 'img_high_res.png')
        stims = load_image_stims(win, trial_dir, display_size)
        prompt = read_prompt(trial_dir)

        ordered = [key_a, key_b]
        random.shuffle(ordered)
        img1_key, img2_key = ordered

        while True:
            show_fixation(win, stims, pair_idx, total_real, is_practice=False)
            flash_pair(win, stims[img1_key], stims['fixation'], stims[img2_key], flash_frames)
            response = ask_question(
                win, QUESTION[1],
                pair_idx=pair_idx, total_pairs=total_real,
                allow_repeat=True, is_practice=False,
            )
            if response != 'r':
                break

        csv_writer.writerow({
            'trial_dir':  trial_dir.name,
            'pair_index': pair_idx - 1,
            'img1':       img1_key,
            'img2':       img2_key,
            'prompt':     prompt,
            'question':   QUESTION[0],
            'response':   response,
        })
        csvfile.flush()

    # end screen
    end = visual.TextStim(
        win,
        text='Study complete.\n\nThank you for participating!\n\nPress any key to exit.',
        color='white',
        height=32,
        wrapWidth=900,
        alignText='center',
    )
    end.draw()
    win.flip()
    event.waitKeys()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    all_dirs = load_all_dirs(args.user_study_dir)
    real_schedule = build_schedule(all_dirs, args.num_pairs)
    practice_schedule = [
        (random.choice(all_dirs), random.choice(ALL_PAIRS))
        for _ in range(NUM_PRACTICE_TRIALS)
    ]

    if args.test:
        print('TEST MODE: results will not be saved')
    else:
        output_path = next_participant_path(args.output_dir)
        print(f'Saving results to: {output_path}')
    print(f'Schedule: {NUM_PRACTICE_TRIALS} practice + {args.num_pairs} real comparisons '
          f'({args.num_pairs // 3} per pair type)')

    win = visual.Window(
        fullscr=True,
        color=BG_COLOR,
        units='pix',
        allowGUI=False,
    )

    try:
        if args.test:
            import io
            dummy = io.StringIO()
            writer = csv.DictWriter(dummy, fieldnames=CSV_FIELDNAMES)
            run_experiment(win, practice_schedule, real_schedule, writer, dummy)
        else:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                csvfile.flush()
                run_experiment(win, practice_schedule, real_schedule, writer, csvfile)
    finally:
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
