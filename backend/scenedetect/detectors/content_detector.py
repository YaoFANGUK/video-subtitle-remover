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
""":class:`ContentDetector` compares the difference in content between adjacent frames against a
set threshold/score, which if exceeded, triggers a scene cut.

This detector is available from the command-line as the `scene_detect-content` command.
"""
from dataclasses import dataclass
import math
from typing import List, NamedTuple, Optional

import numpy
import cv2

from backend.scenedetect.scene_detector import SceneDetector


def _mean_pixel_distance(left: numpy.ndarray, right: numpy.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (numpy.sum(numpy.abs(left.astype(numpy.int32) - right.astype(numpy.int32))) / num_pixels)


def _estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    # TODO: This equation is based on manual estimation from a few videos.
    # Create a more comprehensive test suite to optimize against.
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size


class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To scene_detect slow fades between
    content scenes still using HSV information, use the DissolveDetector.
    """

    # TODO: Come up with some good weights for a new default if there is one that can pass
    # a wider variety of test cases.
    class Components(NamedTuple):
        """Components that make up a frame's score, and their default values."""
        delta_hue: float = 1.0
        """Difference between pixel hue values of adjacent frames."""
        delta_sat: float = 1.0
        """Difference between pixel saturation values of adjacent frames."""
        delta_lum: float = 1.0
        """Difference between pixel luma (brightness) values of adjacent frames."""
        delta_edges: float = 0.0
        """Difference between calculated edges of adjacent frames.

        Edge differences are typically larger than the other components, so the detection
        threshold may need to be adjusted accordingly."""

    DEFAULT_COMPONENT_WEIGHTS = Components()
    """Default component weights. Actual default values are specified in :class:`Components`
    to allow adding new components without breaking existing usage."""

    LUMA_ONLY_WEIGHTS = Components(
        delta_hue=0.0,
        delta_sat=0.0,
        delta_lum=1.0,
        delta_edges=0.0,
    )
    """Component weights to use if `luma_only` is set."""

    FRAME_SCORE_KEY = 'content_val'
    """Key in statsfile representing the final frame score after weighed by specified components."""

    METRIC_KEYS = [FRAME_SCORE_KEY, *Components._fields]
    """All statsfile keys this detector produces."""

    @dataclass
    class _FrameData:
        """Data calculated for a given frame."""
        hue: numpy.ndarray
        """Frame hue map [2D 8-bit]."""
        sat: numpy.ndarray
        """Frame saturation map [2D 8-bit]."""
        lum: numpy.ndarray
        """Frame luma/brightness map [2D 8-bit]."""
        edges: Optional[numpy.ndarray]
        """Frame edge map [2D 8-bit, edges are 255, non edges 0]. Affected by `kernel_size`."""

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        weights: 'ContentDetector.Components' = DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: Optional[int] = None,
    ):
        """
        Arguments:
            threshold: Threshold the average change in pixel intensity must exceed to trigger a cut.
            min_scene_len: Once a cut is detected, this many frames must pass before a new one can
                be added to the scene list.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel for expanding detected edges. Must be odd integer
                greater than or equal to 3. If None, automatically set using video resolution.
        """
        super().__init__()
        self._threshold: float = threshold
        self._min_scene_len: int = min_scene_len
        self._last_scene_cut: Optional[int] = None
        self._last_frame: Optional[ContentDetector._FrameData] = None
        self._weights: ContentDetector.Components = weights
        if luma_only:
            self._weights = ContentDetector.LUMA_ONLY_WEIGHTS
        self._kernel: Optional[numpy.ndarray] = None
        if kernel_size is not None:
            print(kernel_size)
            if kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError('kernel_size must be odd integer >= 3')
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        self._frame_score: Optional[float] = None

    def get_metrics(self):
        return ContentDetector.METRIC_KEYS

    def is_processing_required(self, frame_num):
        return True

    def _calculate_frame_score(self, frame_num: int, frame_img: numpy.ndarray) -> float:
        """Calculate score representing relative amount of motion in `frame_img` compared to
        the last time the function was called (returns 0.0 on the first call)."""
        # TODO: Add option to enable motion estimation before calculating score components.
        # TODO: Investigate methods of performing cheaper alternatives, e.g. shifting or resizing
        # the frame to simulate camera movement, using optical flow, etc...

        # Convert image into HSV colorspace.
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))

        # Performance: Only calculate edges if we have to.
        calculate_edges: bool = ((self._weights.delta_edges > 0.0)
                                 or self.stats_manager is not None)
        edges = self._detect_edges(lum) if calculate_edges else None

        if self._last_frame is None:
            # Need another frame to compare with for score calculation.
            self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
            return 0.0

        score_components = ContentDetector.Components(
            delta_hue=_mean_pixel_distance(hue, self._last_frame.hue),
            delta_sat=_mean_pixel_distance(sat, self._last_frame.sat),
            delta_lum=_mean_pixel_distance(lum, self._last_frame.lum),
            delta_edges=(0.0 if edges is None else _mean_pixel_distance(
                edges, self._last_frame.edges)),
        )

        frame_score: float = (
            sum(component * weight for (component, weight) in zip(score_components, self._weights))
            / sum(abs(weight) for weight in self._weights))

        # Record components and frame score if needed for analysis.
        if self.stats_manager is not None:
            metrics = {self.FRAME_SCORE_KEY: frame_score}
            metrics.update(score_components._asdict())
            self.stats_manager.set_metrics(frame_num, metrics)

        # Store all data required to calculate the next frame's score.
        self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
        return frame_score

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> List[int]:
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot scene_detect slow fades with this method).

        Arguments:
            frame_num: Frame number of frame that is being passed.
            frame_img: Decoded frame image (numpy.ndarray) to perform scene
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base SceneDetector class) returns True.

        Returns:
            List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        if frame_img is None:
            # TODO(0.6.3): Make frame_img a required argument in the interface. Log a warning
            # that passing None is deprecated and results will be incorrect if this is the case.
            return []

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self._last_scene_cut is None:
            self._last_scene_cut = frame_num

        self._frame_score = self._calculate_frame_score(frame_num, frame_img)
        if self._frame_score is None:
            return []

        # We consider any frame over the threshold a new scene, but only if
        # the minimum scene length has been reached (otherwise it is ignored).
        min_length_met = (frame_num - self._last_scene_cut) >= self._min_scene_len
        if self._frame_score >= self._threshold and min_length_met:
            self._last_scene_cut = frame_num
            return [frame_num]

        return []

    # TODO(#250): Based on the parameters passed to the ContentDetector constructor,
    # ensure that the last scene meets the minimum length requirement, otherwise it
    # should be merged with the previous scene. This can be done by caching the cuts
    # for the amount of time the minimum length is set to, returning any outstanding
    # final cuts in post_process.

    #def post_process(self, frame_num):
    #    """
    #    return []

    def _detect_edges(self, lum: numpy.ndarray) -> numpy.ndarray:
        """Detect edges using the luma channel of a frame.

        Arguments:
            lum: 2D 8-bit image representing the luma channel of a frame.

        Returns:
            2D 8-bit image of the same size as the input, where pixels with values of 255
            represent edges, and all other pixels are 0.
        """
        # Initialize kernel.
        if self._kernel is None:
            kernel_size = _estimated_kernel_size(lum.shape[1], lum.shape[0])
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)

        # Estimate levels for thresholding.
        # TODO(0.6.3): Add config file entries for sigma, aperture/kernel size, etc.
        sigma: float = 1.0 / 3.0
        median = numpy.median(lum)
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))

        # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
        # This increases edge overlap leading to improved robustness against noise and slow
        # camera movement. Note that very large kernel sizes can negatively affect accuracy.
        edges = cv2.Canny(lum, low, high)
        return cv2.dilate(edges, self._kernel)
