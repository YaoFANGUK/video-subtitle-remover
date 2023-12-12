# -*- coding: utf-8 -*-
#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2014-2023 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
""":class:`ThresholdDetector` uses a set intensity as a threshold to scene_detect cuts, which are
triggered when the average pixel intensity exceeds or falls below this threshold.

This detector is available from the command-line as the `scene_detect-threshold` command.
"""

from enum import Enum
from logging import getLogger
from typing import List, Optional

import numpy

from backend.scenedetect.scene_detector import SceneDetector

logger = getLogger('pyscenedetect')

##
## ThresholdDetector Helper Functions
##


def _compute_frame_average(frame: numpy.ndarray) -> float:
    """Computes the average pixel value/intensity for all pixels in a frame.

    The value is computed by adding up the 8-bit R, G, and B values for
    each pixel, and dividing by the number of pixels multiplied by 3.

    Arguments:
        frame: Frame representing the RGB pixels to average.

    Returns:
        Average pixel intensity across all 3 channels of `frame`
    """
    num_pixel_values = float(frame.shape[0] * frame.shape[1] * frame.shape[2])
    avg_pixel_value = numpy.sum(frame[:, :, :]) / num_pixel_values
    return avg_pixel_value


##
## ThresholdDetector Class Implementation
##


class ThresholdDetector(SceneDetector):
    """Detects fast cuts/slow fades in from and out to a given threshold level.

    Detects both fast cuts and slow fades so long as an appropriate threshold
    is chosen (especially taking into account the minimum grey/black level).
    """

    class Method(Enum):
        """Method for ThresholdDetector to use when comparing frame brightness to the threshold."""
        FLOOR = 0
        """Fade out happens when frame brightness falls below threshold."""
        CEILING = 1
        """Fade out happens when frame brightness rises above threshold."""

    THRESHOLD_VALUE_KEY = 'average_rgb'

    def __init__(
        self,
        threshold: float = 12,
        min_scene_len: int = 15,
        fade_bias: float = 0.0,
        add_final_scene: bool = False,
        method: Method = Method.FLOOR,
        block_size=None,
    ):
        """
        Arguments:
            threshold:  8-bit intensity value that each pixel value (R, G, and B)
                must be <= to in order to trigger a fade in/out.
            min_scene_len:  FrameTimecode object or integer greater than 0 of the
                minimum length, in frames, of a scene (or subsequent scene cut).
            fade_bias:  Float between -1.0 and +1.0 representing the percentage of
                timecode skew for the start of a scene (-1.0 causing a cut at the
                fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be
                right at the position where the threshold is passed).
            add_final_scene:  Boolean indicating if the video ends on a fade-out to
                generate an additional scene at this timecode.
            method: How to treat `threshold` when detecting fade events.
            block_size: [DEPRECATED] DO NOT USE. For backwards compatibility.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `block_size` will be removed in v0.8.
        if block_size is not None:
            logger.error('block_size is deprecated.')

        super().__init__()
        self.threshold = int(threshold)
        self.method = ThresholdDetector.Method(method)
        self.fade_bias = fade_bias
        self.min_scene_len = min_scene_len
        self.processed_frame = False
        self.last_scene_cut = None
        # Whether to add an additional scene or not when ending on a fade out
        # (as cuts are only added on fade ins; see post_process() for details).
        self.add_final_scene = add_final_scene
        # Where the last fade (threshold crossing) was detected.
        self.last_fade = {
            'frame': 0,  # frame number where the last detected fade is
            'type': None  # type of fade, can be either 'in' or 'out'
        }
        self._metric_keys = [ThresholdDetector.THRESHOLD_VALUE_KEY]

    def get_metrics(self) -> List[str]:
        return self._metric_keys

    def process_frame(self, frame_num: int, frame_img: Optional[numpy.ndarray]) -> List[int]:
        """
        Args:
            frame_num (int): Frame number of frame that is being passed.
            frame_img (numpy.ndarray or None): Decoded frame image (numpy.ndarray) to perform
                scene detection with. Can be None *only* if the self.is_processing_required()
                method (inhereted from the base SceneDetector class) returns True.
        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self.last_scene_cut is None:
            self.last_scene_cut = frame_num

        # Compare the # of pixels under threshold in current_frame & last_frame.
        # If absolute value of pixel intensity delta is above the threshold,
        # then we trigger a new scene cut/break.

        # List of cuts to return.
        cut_list = []

        # The metric used here to scene_detect scene breaks is the percent of pixels
        # less than or equal to the threshold; however, since this differs on
        # user-supplied values, we supply the average pixel intensity as this
        # frame metric instead (to assist with manually selecting a threshold)
        if (self.stats_manager is not None) and (self.stats_manager.metrics_exist(
                frame_num, self._metric_keys)):
            frame_avg = self.stats_manager.get_metrics(frame_num, self._metric_keys)[0]
        else:
            frame_avg = _compute_frame_average(frame_img)
            if self.stats_manager is not None:
                self.stats_manager.set_metrics(frame_num, {self._metric_keys[0]: frame_avg})

        if self.processed_frame:
            if self.last_fade['type'] == 'in' and ((
                (self.method == ThresholdDetector.Method.FLOOR and frame_avg < self.threshold) or
                (self.method == ThresholdDetector.Method.CEILING and frame_avg >= self.threshold))):
                # Just faded out of a scene, wait for next fade in.
                self.last_fade['type'] = 'out'
                self.last_fade['frame'] = frame_num

            elif self.last_fade['type'] == 'out' and (
                (self.method == ThresholdDetector.Method.FLOOR and frame_avg >= self.threshold) or
                (self.method == ThresholdDetector.Method.CEILING and frame_avg < self.threshold)):
                # Only add the scene if min_scene_len frames have passed.
                if (frame_num - self.last_scene_cut) >= self.min_scene_len:
                    # Just faded into a new scene, compute timecode for the scene
                    # split based on the fade bias.
                    f_out = self.last_fade['frame']
                    f_split = int(
                        (frame_num + f_out + int(self.fade_bias * (frame_num - f_out))) / 2)
                    cut_list.append(f_split)
                    self.last_scene_cut = frame_num
                self.last_fade['type'] = 'in'
                self.last_fade['frame'] = frame_num
        else:
            self.last_fade['frame'] = 0
            if frame_avg < self.threshold:
                self.last_fade['type'] = 'out'
            else:
                self.last_fade['type'] = 'in'
        self.processed_frame = True
        return cut_list

    def post_process(self, frame_num: int):
        """Writes a final scene cut if the last detected fade was a fade-out.

        Only writes the scene cut if add_final_scene is true, and the last fade
        that was detected was a fade-out.  There is no bias applied to this cut
        (since there is no corresponding fade-in) so it will be located at the
        exact frame where the fade-out crossed the detection threshold.
        """

        # If the last fade detected was a fade out, we add a corresponding new
        # scene break to indicate the end of the scene.  This is only done for
        # fade-outs, as a scene cut is already added when a fade-in is found.
        cut_times = []
        if self.last_fade['type'] == 'out' and self.add_final_scene and (
            (self.last_scene_cut is None and frame_num >= self.min_scene_len) or
            (frame_num - self.last_scene_cut) >= self.min_scene_len):
            cut_times.append(self.last_fade['frame'])
        return cut_times
