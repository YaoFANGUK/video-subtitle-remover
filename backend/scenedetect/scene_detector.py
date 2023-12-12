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
"""``scenedetect.scene_detector`` Module

This module contains the :class:`SceneDetector` interface, from which all scene detectors in
:mod:`scenedetect.detectors` module are derived from.

The SceneDetector class represents the interface which detection algorithms are expected to provide
in order to be compatible with PySceneDetect.

.. warning::

    This API is still unstable, and changes and design improvements are planned for the v1.0
    release. Instead of just timecodes, detection algorithms will also provide a specific type of
    event (in, out, cut, etc...).
"""

from typing import List, Optional, Tuple

import numpy

from backend.scenedetect.stats_manager import StatsManager


# pylint: disable=unused-argument, no-self-use
class SceneDetector:
    """ Base class to inherit from when implementing a scene detection algorithm.

    This API is not yet stable and subject to change.

    This represents a "dense" scene detector, which returns a list of frames where
    the next scene/shot begins in a video.

    Also see the implemented scene detectors in the scenedetect.detectors module
    to get an idea of how a particular detector can be created.
    """
    # TODO(v0.7): Make this a proper abstract base class.

    stats_manager: Optional[StatsManager] = None
    """Optional :class:`StatsManager <scenedetect.stats_manager.StatsManager>` to
    use for caching frame metrics to and from."""

    # TODO(v1.0): Remove - this is a rarely used case for what is now a neglegible performance gain.
    def is_processing_required(self, frame_num: int) -> bool:
        """[DEPRECATED] DO NOT USE

        Test if all calculations for a given frame are already done.

        Returns:
            False if the SceneDetector has assigned _metric_keys, and the
            stats_manager property is set to a valid StatsManager object containing
            the required frame metrics/calculations for the given frame - thus, not
            needing the frame to perform scene detection.

            True otherwise (i.e. the frame_img passed to process_frame is required
            to be passed to process_frame for the given frame_num).
        """
        metric_keys = self.get_metrics()
        return not metric_keys or not (self.stats_manager is not None
                                       and self.stats_manager.metrics_exist(frame_num, metric_keys))

    def stats_manager_required(self) -> bool:
        """Stats Manager Required: Prototype indicating if detector requires stats.

        Returns:
            True if a StatsManager is required for the detector, False otherwise.
        """
        return False

    def get_metrics(self) -> List[str]:
        """Get Metrics:  Get a list of all metric names/keys used by the detector.

        Returns:
            List of strings of frame metric key names that will be used by
            the detector when a StatsManager is passed to process_frame.
        """
        return []

    def process_frame(self, frame_num: int, frame_img: Optional[numpy.ndarray]) -> List[int]:
        """Process Frame: Computes/stores metrics and detects any scene changes.

        Prototype method, no actual detection.

        Returns:
            List of frame numbers of cuts to be added to the cutting list.
        """
        return []

    def post_process(self, frame_num: int) -> List[int]:
        """Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List of frame numbers of cuts to be added to the cutting list.
        """
        return []

    @property
    def event_buffer_length(self) -> int:
        """The amount of frames a given event can be buffered for, in time. Represents maximum
        amount any event can be behind `frame_number` in the result of :meth:`process_frame`.
        """
        return 0


class SparseSceneDetector(SceneDetector):
    """Base class to inherit from when implementing a sparse scene detection algorithm.

    This class will be removed in v1.0 and should not be used.

    Unlike dense detectors, sparse detectors scene_detect "events" and return a *pair* of frames,
    as opposed to just a single cut.

    An example of a SparseSceneDetector is the MotionDetector.
    """

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> List[Tuple[int, int]]:
        """Process Frame: Computes/stores metrics and detects any scene changes.

        Prototype method, no actual detection.

        Returns:
            List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []

    def post_process(self, frame_num: int) -> List[Tuple[int, int]]:
        """Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []
