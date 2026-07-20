"""Demo for the Image paradigm display — Diffusion Roundtrip sequence.

Shows one diffusion folder through the full roundtrip:

    Clean-to-Noise direction:
        1s Black | Fixation (wait button) |
        Repeat 4 times (
            1s Black | 2s Clean |
            Repeat 8 times ( 1s Black | 1s +Noise ) )

    Noise-to-Clean direction:
        1s Black | Fixation (wait button) |
        Repeat 4 times (
            1s Black | 2s Clean |
            Repeat 8 times ( 1s Black | 1s -Noise ) )

Run from the repository root:
    python bcipy/display/demo/image/demo_image_display.py
"""

from pathlib import Path

from psychopy import core, event

from bcipy.display import InformationProperties, StimuliProperties, init_display_window
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.image.display import ImageDisplay
from bcipy.helpers.clock import Clock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Diffusion folder to use for the demo
DEMO_FOLDER = 'diffusion_output/cat_lynx'

# Number of outer repetitions (clean-image breaks) per direction
NOISE_GROUPS = 4

# Number of noise images per group (4 × 8 = all 32 noisy timesteps)
NOISE_PER_GROUP = 8

# Stimulus durations (seconds)
TIME_BLACK = 1.0   # blank inter-stimulus interval
TIME_CLEAN = 2.0   # clean reference / break image
TIME_NOISE = 2.0   # each noise-step image


# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------

folder = Path(DEMO_FOLDER)

# Sort PNGs by filename: t000 comes first, then t031 … t992 in ascending order
all_images = sorted(folder.glob('*.png'), key=lambda p: p.name)

# t000 — zero-noise reference used as the clean break image
clean_image = str(all_images[0])

# t031 through t992 — all 32 noise steps (exactly NOISE_GROUPS × NOISE_PER_GROUP)
noisy_images = [str(p) for p in all_images[1: NOISE_GROUPS * NOISE_PER_GROUP + 1]]

print(f'Clean image : {clean_image}')
print(f'Noise images: {len(noisy_images)} steps')


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

window_parameters = {
    'full_screen': False,
    'window_height': 600,
    'window_width': 600,
    'stim_screen': 0,
    'background_color': 'black',
}

# Open the PsychoPy display window via the BciPy helper
win = init_display_window(window_parameters)
win.recordFrameIntervals = True

# getActualFrameRate() returns None when vsync is unavailable; fall back to 60 Hz
frame_rate = win.getActualFrameRate() or 60.0
print(f'Frame rate: {frame_rate}')


# ---------------------------------------------------------------------------
# ImageDisplay construction
# ---------------------------------------------------------------------------

# StaticPeriod for precise frame-locked timing inside ImageDisplay
static_clock = core.StaticPeriod(screenHz=frame_rate)

# Experiment-wide clock; timestamps are recorded relative to this
experiment_clock = Clock()

# Persistent corner label visible throughout the demo
info = InformationProperties(
    info_color=['white'],
    info_pos=[(-.5, -.75)],
    info_height=[0.06],
    info_font=['Arial'],
    info_text=['Image Paradigm Demo'],
)

# StimuliProperties holds defaults; schedule_to() overwrites them before each inquiry
stimuli = StimuliProperties(
    stim_font='Arial',
    stim_pos=(0, 0),            # stimuli drawn at window centre
    stim_height=0.8,            # height fraction passed to resize_image for images
    stim_inquiry=[clean_image], # placeholder — overwritten by schedule_to()
    stim_colors=['white'],      # placeholder
    stim_timing=[TIME_NOISE],   # placeholder
    is_txt_stim=False,          # primary content is images
)

# Task bar shows current direction and group progress in the top strip
task_bar = CalibrationTaskBar(
    win,
    inquiry_count=NOISE_GROUPS * 2,  # 4 groups × 2 directions
    current_index=0,
    font='Arial',
)

# ImageDisplay orchestrates all stimulus presentation via schedule_to / do_inquiry
display = ImageDisplay(
    win,
    static_clock,
    experiment_clock,
    stimuli,
    task_bar,
    info,
)


# ---------------------------------------------------------------------------
# Helpers that delegate entirely to ImageDisplay
# ---------------------------------------------------------------------------

def show_black() -> None:
    """Present a 1 s blank frame using ImageDisplay.

    Scheduling an empty string as a text stimulus with color='black' renders
    invisible text on the black background — effectively a black screen.
    draw_static() inside do_inquiry() still draws the task bar and info text.
    """
    display.schedule_to(
        stimuli=[''],            # empty text → invisible on black background
        timing=[TIME_BLACK],
        colors=['black'],
    )
    display.do_inquiry()


def show_fixation_and_wait() -> None:
    """Show a fixation cross, then block until the subject presses a key.

    _create_stimulus() is used to stay within the ImageDisplay API.
    draw_static() ensures the task bar and info text are visible while waiting.
    """
    # Build the fixation cross via ImageDisplay's own stimulus factory
    fix = display._create_stimulus(
        mode='text',
        height=0.1,
        stimulus='+',
        color='white',
    )
    fix.draw()                  # draw the cross into the back buffer
    display.draw_static()       # draw task bar and info on top
    display.window.flip()       # present the frame to the screen
    event.waitKeys(keyList=['space', 'return'])   # hold until button press


def run_group(clean: str, noise_group: list, label: str) -> list:
    """Run one group: 1s Black → 2s Clean → [1s Black → 1s Noise] × 8.

    The entire group is packed into a single schedule_to / do_inquiry call.
    ImageDisplay._generate_inquiry() detects .png extensions to create
    ImageStims and falls back to TextStims for the blank '' entries.

    Args:
        clean      : path to the zero-noise reference image.
        noise_group: list of noise-image paths for this group (length 8).
        label      : task-bar text shown during this group.

    Returns:
        List of (stimulus_label, onset_time) tuples from do_inquiry().
    """
    # Build a flat stimulus list that mirrors the nested sequence:
    #   1s Black, 2s Clean, then repeat(1s Black, 1s Noise) × NOISE_PER_GROUP
    stims = ['', clean]                                         # black + clean break
    timing = [TIME_BLACK, TIME_CLEAN]
    colors = ['black', 'white']

    for noise_path in noise_group:
        stims += ['', noise_path]                               # black + noise image
        timing += [TIME_BLACK, TIME_NOISE]
        colors += ['black', 'white']

    # Update task bar with the current group label
    display.update_task_bar(text=label)

    # Load the full group into ImageDisplay's stimulus buffer
    display.schedule_to(stimuli=stims, timing=timing, colors=colors)

    # Present all stimuli in sequence; returns onset timing for analysis
    timings = display.do_inquiry()
    print(f'  {label}: {len(timings)} stimuli presented')
    return timings


# ---------------------------------------------------------------------------
# Roundtrip sequence
# ---------------------------------------------------------------------------

# --- Clean-to-Noise direction --------------------------------------------

print('\n=== Clean → Noise ===')

# 1s Black before fixation cross
show_black()

# Fixation cross: subject presses space/return to begin the direction
display.update_task_bar(text='Clean → Noise  (press SPACE)')
display.draw_static()
display.window.flip()
show_fixation_and_wait()

# 4 groups, each showing images in ascending noise order (low → high)
for group_idx in range(NOISE_GROUPS):
    start = group_idx * NOISE_PER_GROUP
    group_noisy = noisy_images[start: start + NOISE_PER_GROUP]   # e.g. t031–t248
    run_group(
        clean=clean_image,
        noise_group=group_noisy,           # ascending: low → high noise
        label=f'C→N  group {group_idx + 1}/{NOISE_GROUPS}',
    )

# --- Noise-to-Clean direction ---------------------------------------------

print('\n=== Noise → Clean ===')

# 1s Black before fixation cross
show_black()

# Fixation cross: subject presses space/return to begin the direction
display.update_task_bar(text='Noise → Clean  (press SPACE)')
display.draw_static()
display.window.flip()
show_fixation_and_wait()

# 4 groups in reverse order, each showing images in descending noise order (high → low)
for group_idx in range(NOISE_GROUPS - 1, -1, -1):
    start = group_idx * NOISE_PER_GROUP
    group_noisy = list(reversed(noisy_images[start: start + NOISE_PER_GROUP]))  # descending
    run_group(
        clean=clean_image,
        noise_group=group_noisy,           # descending: high → low noise
        label=f'N→C  group {group_idx + 1}/{NOISE_GROUPS}',
    )


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

# Close the PsychoPy window once both directions are complete
win.close()
