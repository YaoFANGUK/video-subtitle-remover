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
"""The ``scenedetect`` module comes with helper functions to simplify common use cases.
:func:`scene_detect` can be used to perform scene detection on a video by path.  :func:`open_video`
can be used to open a video for a
:class:`SceneManager <scenedetect.scene_manager.SceneManager>`.
"""

from logging import getLogger
from typing import List, Optional, Tuple, Union

# OpenCV is a required package, but we don't have it as an explicit dependency since we
# need to support both opencv-python and opencv-python-headless. Include some additional
# context with the exception if this is the case.
try:
    import cv2 as _
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "OpenCV could not be found, try installing opencv-python:\n\npip install opencv-python",
        name='cv2',
    ) from ex

# Commonly used classes/functions exported under the `scenedetect` namespace for brevity.
from backend.scenedetect.platform import init_logger
from backend.scenedetect.frame_timecode import FrameTimecode
from backend.scenedetect.video_stream import VideoStream, VideoOpenFailure
from backend.scenedetect.scene_detector import SceneDetector
from backend.scenedetect.backends import (AVAILABLE_BACKENDS, VideoStreamCv2, VideoStreamAv,
                                  VideoStreamMoviePy, VideoCaptureAdapter)
from backend.scenedetect.stats_manager import StatsManager, StatsFileCorrupt
from backend.scenedetect.scene_manager import SceneManager, save_images

# Used for module identification and when printing version & about info
# (e.g. calling `scenedetect version` or `scenedetect about`).
__version__ = '0.6.2'

init_logger()
logger = getLogger('pyscenedetect')


def open_video(
    path: str,
    framerate: Optional[float] = None,
    backend: str = 'opencv',
    **kwargs,
) -> VideoStream:
    """Open a video at the given path. If `backend` is specified but not available on the current
    system, OpenCV (`VideoStreamCv2`) will be used as a fallback.

    Arguments:
        path: Path to video file to open.
        framerate: Overrides detected framerate if set.
        backend: Name of specific backend to use, if possible. See
            :data:`scenedetect.backends.AVAILABLE_BACKENDS` for backends available on the current
            system. If the backend fails to open the video, OpenCV will be used as a fallback.
        kwargs: Optional named arguments to pass to the specified `backend` constructor for
            overriding backend-specific options.

    Returns:
        Backend object created with the specified video path.

    Raises:
        :class:`VideoOpenFailure`: Constructing the VideoStream fails. If multiple backends have
            been attempted, the error from the first backend will be returned.
    """
    last_error: Exception = None
    # If `backend` is available, try to open the video at `path` using it.
    if backend in AVAILABLE_BACKENDS:
        backend_type = AVAILABLE_BACKENDS[backend]
        try:
            logger.debug('Opening video with %s...', backend_type.BACKEND_NAME)
            return backend_type(path, framerate, **kwargs)
        except VideoOpenFailure as ex:
            logger.warning('Failed to open video with %s: %s', backend_type.BACKEND_NAME, str(ex))
            if backend == VideoStreamCv2.BACKEND_NAME:
                raise
            last_error = ex
    else:
        logger.warning('Backend %s not available.', backend)
    # Fallback to OpenCV if `backend` is unavailable, or specified backend failed to open `path`.
    backend_type = VideoStreamCv2
    logger.warning('Trying another backend: %s', backend_type.BACKEND_NAME)
    try:
        return backend_type(path, framerate)
    except VideoOpenFailure as ex:
        logger.debug('Failed to open video: %s', str(ex))
        if last_error is None:
            last_error = ex
    # Propagate any exceptions raised from specified backend, instead of errors from the fallback.
    assert last_error is not None
    raise last_error


def scene_detect(
    video_path: str,
    detector: SceneDetector,
    stats_file_path: Optional[str] = None,
    show_progress: bool = False,
    start_time: Optional[Union[str, float, int]] = None,
    end_time: Optional[Union[str, float, int]] = None,
    start_in_scene: bool = False,
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Perform scene detection on a given video `path` using the specified `detector`.

    Arguments:
        video_path: Path to input video (absolute or relative to working directory).
        detector: A `SceneDetector` instance (see :mod:`scenedetect.detectors` for a full list
            of detectors).
        stats_file_path: Path to save per-frame metrics to for statistical analysis or to
            determine a better threshold value.
        show_progress: Show a progress bar with estimated time remaining. Default is False.
        start_time: Starting point in video, in the form of a timecode ``HH:MM:SS[.nnn]`` (`str`),
            number of seconds ``123.45`` (`float`), or number of frames ``200`` (`int`).
        end_time: Starting point in video, in the form of a timecode ``HH:MM:SS[.nnn]`` (`str`),
            number of seconds ``123.45`` (`float`), or number of frames ``200`` (`int`).
        start_in_scene: Assume the video begins in a scene. This means that when detecting
            fast cuts with `ContentDetector`, if no cuts are found, the resulting scene list
            will contain a single scene spanning the entire video (instead of no scenes).
            When detecting fades with `ThresholdDetector`, the beginning portion of the video
            will always be included until the first fade-out event is detected.

    Returns:
        List of scenes (pairs of :class:`FrameTimecode` objects).

    Raises:
        :class:`VideoOpenFailure`: `video_path` could not be opened.
        :class:`StatsFileCorrupt`: `stats_file_path` is an invalid stats file
        ValueError: `start_time` or `end_time` are incorrectly formatted.
        TypeError: `start_time` or `end_time` are invalid types.
    """
    video = open_video(video_path)
    if start_time is not None:
        start_time = video.base_timecode + start_time
        video.seek(start_time)
    if end_time is not None:
        end_time = video.base_timecode + end_time
    # To reduce memory consumption when not required, we only add a StatsManager if we
    # need to save frame metrics to disk.
    scene_manager = SceneManager(StatsManager() if stats_file_path else None)
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(
        video=video,
        show_progress=show_progress,
        end_time=end_time,
    )
    if not scene_manager.stats_manager is None:
        scene_manager.stats_manager.save_to_csv(csv_file=stats_file_path)
    return scene_manager.get_scene_list(start_in_scene=start_in_scene)
