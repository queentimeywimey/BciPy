"""Diffusion Roundtrip task module.

Presents diffusion model images in a roundtrip sequence for EEG data collection.
For each folder in diffusion_output, all 32 noisy timesteps are shown across two
directions (clean-to-noise then noise-to-clean), divided into 4 groups of 8 with a
clean-image break before each group.

Roundtrip sequence per folder (~3 min, 156 s stim):

    Clean-to-Noise:
        1s Black | Fixation (wait button) |
        Repeat 4 times (
            1s Black | 2s Clean |
            Repeat 8 times ( 1s Black | 1s +Noise )
        )

    Noise-to-Clean:
        1s Black | Fixation (wait button) |
        Repeat 4 times (
            1s Black | 2s Clean |
            Repeat 8 times ( 1s Black | 1s -Noise )
        )
        1s Black | 2s Clean  ← final break
"""
import logging
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from psychopy import core, event, visual

from bcipy.config import SESSION_LOG_FILENAME, TRIGGER_FILENAME
from bcipy.core.stimuli import resize_image
from bcipy.core.triggers import (
    FlushFrequency,
    Trigger,
    TriggerCallback,
    TriggerHandler,
    TriggerType,
    _calibration_trigger,
    offset_label,
)
from bcipy.display import init_display_window
from bcipy.helpers.acquisition import init_acquisition
from bcipy.helpers.clock import Clock
from bcipy.task import Task, TaskData, TaskMode

logger = logging.getLogger(SESSION_LOG_FILENAME)


class DiffusionRoundtripTask(Task):
    """Diffusion Roundtrip Task.

    For each folder in diffusion_output, shows all 32 noisy timesteps in two
    directions separated by a fixation-and-button checkpoint:

    - Clean-to-Noise (t000 → t992): four groups of eight +noise images, each
      preceded by a two-second clean-image break.
    - Noise-to-Clean (t992 → t000): same structure in reverse, followed by a
      final two-second clean-image break.

    Pressing escape at any point ends the task early and closes the window,
    same as the RSVP/Matrix/VEP calibration tasks.

    Parameters
    ----------
    parameters : Parameters
        BciPy parameter object. Reads ``'diffusion_image_path'``,
        ``'diffusion_noise_groups'``, ``'diffusion_noise_per_group'``,
        ``'diffusion_image_scale'``, ``'diffusion_time_black'``,
        ``'diffusion_time_clean'``, and ``'diffusion_time_noise'``.
    file_save : str
        Directory path for saving trigger and session data.
    fake : bool
        When True, uses simulated acquisition data.
    exit_callback : Optional[Callable]
        Called once when the user presses escape, so a multi-task protocol
        (via the orchestrator) also stops after this task ends.
    """

    name = 'Diffusion Roundtrip'
    paradigm = 'Image'
    mode = TaskMode.CALIBRATION

    def __init__(
            self,
            parameters,
            file_save: str,
            fake: bool = False,
            exit_callback: Optional[Callable] = None,
            **kwargs) -> None:
        super().__init__()
        self.parameters = parameters
        self.file_save = file_save
        self.fake = fake
        self.exit_callback = exit_callback
        self.should_stop = False

        self.daq, self.servers, self.window = self._setup(
            parameters, file_save, fake)

        self.experiment_clock = Clock()
        self.trigger_callback = TriggerCallback()
        self.trigger_handler = TriggerHandler(
            file_save, TRIGGER_FILENAME, FlushFrequency.EVERY)
        self.initialized = True
        self._first_stim_time: Optional[float] = None

        self.diffusion_dir = Path(parameters['diffusion_image_path'])
        self.noise_groups = parameters['diffusion_noise_groups']
        self.noise_per_group = parameters['diffusion_noise_per_group']
        self.image_scale = parameters['diffusion_image_scale']
        self.time_black = parameters['diffusion_time_black']
        self.time_clean = parameters['diffusion_time_clean']
        self.time_noise = parameters['diffusion_time_noise']

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self, parameters, file_save: str, fake: bool):
        daq, servers = init_acquisition(parameters, file_save, server=fake)
        window = init_display_window(parameters)
        return daq, servers, window

    def cleanup(self) -> None:
        """Stop acquisition and close the display window."""
        self.trigger_handler.close()
        if self.initialized:
            try:
                self.daq.stop_acquisition()
                self.daq.cleanup()
                for server in self.servers:
                    server.stop()
                self.window.close()
                self.initialized = False
            except Exception as e:
                logger.exception(str(e))

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _get_folders(self) -> List[Path]:
        """Return image folders under diffusion_output in randomized order."""
        folders = [p for p in self.diffusion_dir.iterdir() if p.is_dir()]
        random.shuffle(folders)
        return folders

    def _get_images(self, folder: Path) -> Tuple[str, List[str]]:
        """Return ``(clean_path, sorted_noisy_paths)`` for a folder.

        The clean image is t000 (no noise). The noisy list contains the
        remaining 32 timesteps in ascending order (t031 … t992).
        """
        images = sorted(folder.glob('*.png'), key=lambda p: p.name)
        required = self.noise_groups * self.noise_per_group + 1
        if len(images) < required:
            raise ValueError(
                f'{folder}: expected at least '
                f'{required} PNG images, '
                f'found {len(images)}.')
        clean = str(images[0])
        noisy = [str(p) for p in images[1:required]]
        return clean, noisy

    # ------------------------------------------------------------------
    # Low-level display helpers
    # ------------------------------------------------------------------

    def _show_black(self) -> None:
        """Flip to a black frame and hold for self.time_black seconds."""
        self.window.flip()
        core.wait(self.time_black)

    def _show_fixation_wait(self) -> None:
        """Show a fixation cross and block until space, return, or escape is pressed."""
        fix = visual.TextStim(
            win=self.window,
            text='+',
            color='white',
            height=0.1,
        )
        fix.draw()
        self.window.flip()
        keys = event.waitKeys(keyList=['space', 'return', 'escape'])
        if keys and 'escape' in keys:
            logger.info('Escape pressed. Ending Diffusion Roundtrip early.')
            self.should_stop = True
            if self.exit_callback:
                self.exit_callback()

    def _ensure_calibration(self) -> None:
        """Send a one-shot calibration trigger for EEG clock synchronisation."""
        if self._first_stim_time is not None:
            return
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type='image',
            display=self.window)
        self._first_stim_time = calibration_time[-1]
        if hasattr(self.daq, 'clients_by_type'):
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = (client.offset(self._first_stim_time)
                        - self._first_stim_time)
                self.trigger_handler.add_triggers(
                    [Trigger(label, TriggerType.OFFSET, time)])

    def _record_system_trigger(self, label: str) -> None:
        """Record a non-visual system event at the current clock time."""
        time = self.experiment_clock.getTime()
        self.trigger_handler.add_triggers(
            [Trigger(label, TriggerType.SYSTEM, time)])

    def _show_stimulus(
            self,
            image_path: str,
            duration: float,
            label: str,
            trigger_type: TriggerType) -> None:
        """Display an image, record its onset trigger, then wait *duration* s.

        Parameters
        ----------
        image_path : str
            Path to the PNG image file.
        duration : float
            Time in seconds to hold the image on screen.
        label : str
            Trigger label written to the trigger file.
        trigger_type : TriggerType
            Semantic type attached to this trigger.
        """
        self._ensure_calibration()

        img = visual.ImageStim(win=self.window, image=image_path, pos=(0, 0))
        img.size = resize_image(img.image, self.window.size, self.image_scale)

        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            label)
        img.draw()
        self.window.flip()
        core.wait(duration)

        stim_label, stim_time = self.trigger_callback.timing
        self.trigger_handler.add_triggers(
            [Trigger(stim_label, trigger_type, stim_time)])
        self.trigger_callback.reset()

    # ------------------------------------------------------------------
    # Roundtrip directions
    # ------------------------------------------------------------------

    def _run_clean_to_noise(
            self, clean: str, noisy: List[str], folder_name: str) -> None:
        """Clean-to-noise direction.

        1s Black | Fixation (wait button) |
        Repeat 4 times (
            1s Black | 2s Clean |
            Repeat 8 times ( 1s Black | 1s +Noise )
        )
        """
        self._record_system_trigger(f'{folder_name}_c2n_start')
        self._show_black()
        self._show_fixation_wait()
        if self.should_stop:
            return

        for group_idx in range(self.noise_groups):
            self._show_black()
            self._show_stimulus(
                clean, self.time_clean, 'clean_break', TriggerType.NONTARGET)

            start = group_idx * self.noise_per_group
            for noise_path in noisy[start:start + self.noise_per_group]:
                self._show_black()
                label = os.path.splitext(os.path.basename(noise_path))[0]
                self._show_stimulus(
                    noise_path, self.time_noise, label, TriggerType.TARGET)

    def _run_noise_to_clean(
            self, clean: str, noisy: List[str], folder_name: str) -> None:
       
        self._record_system_trigger(f'{folder_name}_n2c_start')
        self._show_black()
        self._show_fixation_wait()
        if self.should_stop:
            return

        for group_idx in range(self.noise_groups - 1, -1, -1):
            self._show_black()
            self._show_stimulus(
                clean, self.time_clean, 'clean_break', TriggerType.NONTARGET)

            start = group_idx * self.noise_per_group
            for noise_path in reversed(noisy[start:start + self.noise_per_group]):
                self._show_black()
                label = os.path.splitext(os.path.basename(noise_path))[0]
                self._show_stimulus(
                    noise_path, self.time_noise, label, TriggerType.TARGET)

        # Final clean-image break at end of noise-to-clean direction
        self._show_black()
        self._show_stimulus(
            clean, self.time_clean, 'clean_end', TriggerType.NONTARGET)

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def execute(self) -> TaskData:
        """Run the full roundtrip task across all diffusion_output folders."""
        logger.info(f'Starting {self.name}!')

        for folder in self._get_folders():
            if self.should_stop:
                break
            logger.info(f'Roundtrip: {folder.name}')
            clean, noisy = self._get_images(folder)
            self._run_clean_to_noise(clean, noisy, folder.name)
            if self.should_stop:
                break
            self._run_noise_to_clean(clean, noisy, folder.name)

        self.cleanup()
        return TaskData(save_path=self.file_save, task_dict={})
