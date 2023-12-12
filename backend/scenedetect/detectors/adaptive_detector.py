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
""":class:`AdaptiveDetector` compares the difference in content between adjacent frames similar
to `ContentDetector` except the threshold isn't fixed, but is a rolling average of adjacent frame
changes. This can help mitigate false detections in situations such as fast camera motions.

This detector is available from the command-line as the `scene_detect-adaptive` command.
"""

from logging import getLogger
from typing import List, Optional

from numpy import ndarray

from backend.scenedetect.detectors import ContentDetector

logger = getLogger('pyscenedetect')


class AdaptiveDetector(ContentDetector):
    """Two-pass detector that calculates frame scores with ContentDetector, and then applies
    a rolling average when processing the result that can help mitigate false detections
    in situations such as camera movement.
    """

    ADAPTIVE_RATIO_KEY_TEMPLATE = "adaptive_ratio{luma_only} (w={window_width})"

    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_scene_len: int = 15,
        window_width: int = 2,
        min_content_val: float = 15.0,
        weights: ContentDetector.Components = ContentDetector.DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: Optional[int] = None,
        video_manager=None,
        min_delta_hsv: Optional[float] = None,
    ):
        """
        Arguments:
            adaptive_threshold: Threshold (float) that score ratio must exceed to trigger a
                new scene (see frame metric adaptive_ratio in stats file).
            min_scene_len: Minimum length of any scene.
            window_width: Size of window (number of frames) before and after each frame to
                average together in order to scene_detect deviations from the mean. Must be at least 1.
            min_content_val: Minimum threshold (float) that the content_val must exceed in order to
                register as a new scene. This is calculated the same way that `scene_detect-content`
                calculates frame score based on `weights`/`luma_only`/`kernel_size`.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
                If omitted, the default ContentDetector weights are used.
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel to use for post edge detection filtering. If None,
                automatically set based on video resolution.
            video_manager: [DEPRECATED] DO NOT USE. For backwards compatibility only.
            min_delta_hsv: [DEPRECATED] DO NOT USE. Use `min_content_val` instead.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `video_manager` and `min_delta_hsv` will
        # be removed in v0.8.
        if video_manager is not None:
            logger.error('video_manager is deprecated, use video instead.')
        if min_delta_hsv is not None:
            logger.error('min_delta_hsv is deprecated, use min_content_val instead.')
            min_content_val = min_delta_hsv
        if window_width < 1:
            raise ValueError('window_width must be at least 1.')

        super().__init__(
            threshold=255.0,
            min_scene_len=0,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
        )

        # TODO: Turn these options into properties.
        self.min_scene_len = min_scene_len
        self.adaptive_threshold = adaptive_threshold
        self.min_content_val = min_content_val
        self.window_width = window_width

        self._adaptive_ratio_key = AdaptiveDetector.ADAPTIVE_RATIO_KEY_TEMPLATE.format(
            window_width=window_width, luma_only='' if not luma_only else '_lum')
        self._first_frame_num = None
        self._last_frame_num = None

        self._last_cut: Optional[int] = None

        self._buffer = []

    @property
    def event_buffer_length(self) -> int:
        """Number of frames any detected cuts will be behind the current frame due to buffering."""
        return self.window_width

    def get_metrics(self) -> List[str]:
        """Combines base ContentDetector metric keys with the AdaptiveDetector one."""
        return super().get_metrics() + [self._adaptive_ratio_key]

    def stats_manager_required(self) -> bool:
        """Not required for AdaptiveDetector."""
        return False

    def process_frame(self, frame_num: int, frame_img: Optional[ndarray]) -> List[int]:
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot scene_detect slow fades with this method).

        Arguments:
            frame_num: Frame number of frame that is being passed.

            frame_img: Decoded frame image (numpy.ndarray) to perform scene
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base SceneDetector class) returns True.

        Returns:
            Empty list
        """

        # TODO(#283): Merge this with ContentDetector and turn it on by default.

        super().process_frame(frame_num=frame_num, frame_img=frame_img)

        required_frames = 1 + (2 * self.window_width)
        self._buffer.append((frame_num, self._frame_score))
        if not len(self._buffer) >= required_frames:
            return []
        self._buffer = self._buffer[-required_frames:]
        target = self._buffer[self.window_width]
        average_window_score = (
            sum(frame[1] for i, frame in enumerate(self._buffer) if i != self.window_width) /
            (2.0 * self.window_width))

        average_is_zero = abs(average_window_score) < 0.00001

        adaptive_ratio = 0.0
        if not average_is_zero:
            adaptive_ratio = min(target[1] / average_window_score, 255.0)
        elif average_is_zero and target[1] >= self.min_content_val:
            # if we would have divided by zero, set adaptive_ratio to the max (255.0)
            adaptive_ratio = 255.0
        if self.stats_manager is not None:
            self.stats_manager.set_metrics(target[0], {self._adaptive_ratio_key: adaptive_ratio})

        cut_list = []
        # Check to see if adaptive_ratio exceeds the adaptive_threshold as well as there
        # being a large enough content_val to trigger a cut
        if (adaptive_ratio >= self.adaptive_threshold and target[1] >= self.min_content_val):

            if self._last_cut is None:
                # No previously detected cuts
                cut_list.append(target[0])
                self._last_cut = target[0]
            elif (target[0] - self._last_cut) >= self.min_scene_len:
                # Respect the min_scene_len parameter
                cut_list.append(target[0])
                # TODO: Should this be updated every time the threshold is exceeded?
                # It might help with flash suppression for example.
                self._last_cut = target[0]

        return cut_list

    # TODO(0.6.3): Deprecate & remove this method.
    def get_content_val(self, frame_num: int) -> Optional[float]:
        """Returns the average content change for a frame."""
        if self.stats_manager is not None:
            return self.stats_manager.get_metrics(frame_num, [ContentDetector.FRAME_SCORE_KEY])[0]
        return 0.0

    def post_process(self, _unused_frame_num: int):
        """Not required for AdaptiveDetector."""
        return []
