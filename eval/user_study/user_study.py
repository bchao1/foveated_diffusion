#!/usr/bin/env python3
"""
User study GUI for comparing foveated diffusion image quality.

Usage:
    python user_study.py <user_study_dir> [--num_trials 5] [--imgs_per_trial 3] [--output results.csv]

Each subdirectory of user_study_dir must contain:
    fixation_point.png, img_high_res.png, img_naive.png, img_ours.png, prompt.txt
"""

import argparse
import csv
import itertools
import os
import random
import sys
from pathlib import Path

from psychopy import core, event, visual
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXATION_DURATION = 5.0          # seconds (≥5 as specified)
FLASH_DURATION_S  = 0.500        # 500 ms per image
INTER_IMAGE_FIXATION_S = 1.0    # fixation shown between img1 and img2
BG_COLOR          = (-1, -1, -1) # PsychoPy [-1,1] range: black

IMAGE_KEYS = ['high_res', 'naive', 'ours']  # short labels
IMAGE_FILES = {
    'high_res': 'img_high_res.png',
    'naive':    'img_naive.png',
    'ours':     'img_ours.png',
}
ALL_PAIRS = list(itertools.combinations(IMAGE_KEYS, 2))  # 3 distinct pairs

QUESTIONS = [
    ('quality', 'Which image has better overall visual quality?'),
    ('errors',  'Which image contains fewer visual distortions or errors?'),
    ('natural', 'Which image looks more realistic and natural?'),
]
CSV_FIELDNAMES = ['trial_dir', 'pair_index', 'img1', 'img2', 'prompt', 'question', 'response']

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Foveated diffusion user study')
    parser.add_argument('user_study_dir', type=str,
                        help='Directory containing per-trial subdirectories')
    parser.add_argument('--num_trials', type=int, default=5,
                        help='Number of randomly selected trials (default: 5)')
    parser.add_argument('--imgs_per_trial', type=int, default=3,
                        help='Image pairs per trial (default: 3)')
    parser.add_argument('--output', type=str, default='user_study_results.csv',
                        help='Output CSV file path (default: user_study_results.csv)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trial directory loading
# ---------------------------------------------------------------------------

def load_trial_dirs(user_study_dir, num_trials):
    base = Path(user_study_dir)
    dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if len(dirs) < num_trials:
        raise ValueError(
            f'Requested {num_trials} trials but only {len(dirs)} directories found in {user_study_dir}'
        )
    return random.sample(dirs, num_trials)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def compute_display_size(win, img_path):
    """Return (w, h) in PsychoPy 'height' units that fills the shorter screen dimension.

    Using 'height' units (where 1.0 = window height in both axes) avoids any
    pixel-scaling ambiguity on Retina / HiDPI displays. The image fills whichever
    screen dimension is proportionally smaller while preserving its aspect ratio;
    the remaining screen area stays black.
    """
    screen_w, screen_h = win.size
    screen_aspect = screen_w / screen_h
    with Image.open(img_path) as img:
        img_aspect = img.size[0] / img.size[1]
    # In 'height' units, 1.0 = full window height in both x and y.
    if img_aspect <= screen_aspect:
        # Screen is wider than the image: fill height, black bars on sides.
        return (img_aspect, 1.0)
    else:
        # Screen is taller than the image: fill width, black bars top/bottom.
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
            'After each pair, answer 3 questions about image quality.\n\n'
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


def show_fixation(win, stims, trial_idx, num_trials):
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
    progress = visual.TextStim(
        win,
        text=f'Trial {trial_idx} of {num_trials}',
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
        # poll escape every 0.1s so exit is responsive during the countdown
        for _ in range(10):
            if event.getKeys(keyList=['escape']):
                core.quit()
            core.wait(0.1)


def flash_pair(win, stim1, fixation_stim, stim2, flash_frames):
    """Flash stim1, show fixation for 1s, then flash stim2."""
    for _ in range(flash_frames):
        stim1.draw()
        win.flip()
    # 1-second fixation between the two images
    t_start = core.getTime()
    while core.getTime() - t_start < INTER_IMAGE_FIXATION_S:
        fixation_stim.draw()
        win.flip()
        if event.getKeys(keyList=['escape']):
            core.quit()
    for _ in range(flash_frames):
        stim2.draw()
        win.flip()
    # one blank frame to mark end of flash
    win.flip()


def ask_question(win, question_text, q_num, num_questions, pair_idx, imgs_per_trial, allow_repeat=False):
    """Show a question and wait for a keypress.

    Returns '1', '2', or 'r' (only when allow_repeat=True and R is pressed).
    """
    progress_text = f'Comparison {pair_idx + 1} of {imgs_per_trial}  |  Question {q_num} of {num_questions}'
    progress = visual.TextStim(
        win,
        text=progress_text,
        pos=(0, win.size[1] // 2 - 40),
        color='gray',
        height=22,
        anchorVert='top',
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

def run_experiment(win, trial_dirs, imgs_per_trial, csv_writer, csvfile):
    flash_frames = max(1, int(round(FLASH_DURATION_S * win.getActualFrameRate())))
    num_trials = len(trial_dirs)

    show_instructions(win)

    for trial_idx, trial_dir in enumerate(trial_dirs, start=1):
        prompt = read_prompt(trial_dir)

        # compute display size from the first image in this trial
        display_size = compute_display_size(win, trial_dir / 'img_high_res.png')

        # pre-load all images
        stims = load_image_stims(win, trial_dir, display_size)

        # select pairs with replacement
        pairs = [random.choice(ALL_PAIRS) for _ in range(imgs_per_trial)]

        for pair_idx, (key_a, key_b) in enumerate(pairs):
            show_fixation(win, stims, trial_idx, num_trials)
            # randomize order (fixed for this pair even on repeat)
            ordered = [key_a, key_b]
            random.shuffle(ordered)
            img1_key, img2_key = ordered

            # flash loop: R on first question repeats fixation + flash
            while True:
                flash_pair(win, stims[img1_key], stims['fixation'], stims[img2_key], flash_frames)
                first_response = ask_question(
                    win, QUESTIONS[0][1],
                    q_num=1,
                    num_questions=len(QUESTIONS),
                    pair_idx=pair_idx,
                    imgs_per_trial=imgs_per_trial,
                    allow_repeat=True,
                )
                if first_response != 'r':
                    break
                show_fixation(win, stims, trial_idx, num_trials)

            # record first question response
            csv_writer.writerow({
                'trial_dir':  trial_dir.name,
                'pair_index': pair_idx,
                'img1':       img1_key,
                'img2':       img2_key,
                'prompt':     prompt,
                'question':   QUESTIONS[0][0],
                'response':   first_response,
            })
            csvfile.flush()

            # remaining questions (no repeat option)
            for q_label, q_text in QUESTIONS[1:]:
                response = ask_question(
                    win, q_text,
                    q_num=QUESTIONS.index((q_label, q_text)) + 1,
                    num_questions=len(QUESTIONS),
                    pair_idx=pair_idx,
                    imgs_per_trial=imgs_per_trial,
                )
                csv_writer.writerow({
                    'trial_dir':  trial_dir.name,
                    'pair_index': pair_idx,
                    'img1':       img1_key,
                    'img2':       img2_key,
                    'prompt':     prompt,
                    'question':   q_label,
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

    trial_dirs = load_trial_dirs(args.user_study_dir, args.num_trials)

    win = visual.Window(
        fullscr=True,
        color=BG_COLOR,
        units='pix',
        allowGUI=False,
    )

    try:
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            csvfile.flush()
            run_experiment(win, trial_dirs, args.imgs_per_trial, writer, csvfile)
    finally:
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
