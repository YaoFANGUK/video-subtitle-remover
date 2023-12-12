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
"""``scenedetect.backends`` Module

This module contains :class:`VideoStream <scenedetect.video_stream.VideoStream>` implementations
backed by various Python multimedia libraries. In addition to creating backend objects directly,
:func:`scenedetect.open_video` can be used to open a video with a specified backend, falling
back to OpenCV if not available.

All backends available on the current system can be found via :data:`AVAILABLE_BACKENDS`.

If you already have a `cv2.VideoCapture` object you want to use for scene detection, you can
use a :class:`VideoCaptureAdapter <scenedetect.backends.opencv.VideoCaptureAdapter>` instead
of a backend. This is useful when working with devices or streams, for example.

===============================================================
Video Files
===============================================================

Assuming we have a file `video.mp4` in our working directory, we can load it and perform scene
detection on it using :func:`open_video`:

.. code:: python

    from scenedetect import open_video
    video = open_video('video.mp4')

An optional backend from :data:`AVAILABLE_BACKENDS` can be passed to :func:`open_video`
(e.g. `backend='opencv'`). Additional keyword arguments passed to :func:`open_video`
will be forwarded to the backend constructor. If the specified backend is unavailable, or
loading the video fails, ``opencv`` will be tried as a fallback.

Lastly, to use a specific backend directly:

.. code:: python

    # Manually importing and constructing a backend:
    from scenedetect.backends.opencv import VideoStreamCv2
    video = VideoStreamCv2('video.mp4')

In both examples above, the resulting ``video`` can be used with
:meth:`SceneManager.detect_scenes() <scenedetect.scene_manager.SceneManager.detect_scenes>`.

===============================================================
Devices / Cameras / Pipes
===============================================================

You can use an existing `cv2.VideoCapture` object with the PySceneDetect API using a
:class:`VideoCaptureAdapter <scenedetect.backends.opencv.VideoCaptureAdapter>`. For example,
to use a :class:`SceneManager <scenedetect.scene_manager.SceneManager>` with a webcam device:

.. code:: python

    from scenedetect import SceneManager, ContentDetector
    from scenedetect.backends import VideoCaptureAdapter
    # Open device ID 2.
    cap = cv2.VideoCapture(2)
    video = VideoCaptureAdapter(cap)
    total_frames = 1000
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video, duration=total_frames)

When working with live inputs, note that you can pass a callback to
:meth:`detect_scenes() <scenedetect.scene_manager.SceneManager.detect_scenes>` to be
called on every scene detection event. See the :mod:`SceneManager <scenedetect.scene_manager>`
examples for details.
"""

# TODO(v1.0): Consider removing and making this a namespace package so that additional backends can
# be dynamically added. The preferred approach for this should probably be:
# https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages

# TODO: Future VideoStream implementations under consideration:
#  - Nvidia VPF: https://developer.nvidia.com/blog/vpf-hardware-accelerated-video-processing-framework-in-python/

from typing import Dict, Type

# OpenCV must be available at minimum.
from backend.scenedetect.backends.opencv import VideoStreamCv2, VideoCaptureAdapter

try:
    from scenedetect.backends.pyav import VideoStreamAv
except ImportError:
    VideoStreamAv = None

try:
    from scenedetect.backends.moviepy import VideoStreamMoviePy
except ImportError:
    VideoStreamMoviePy = None

# TODO(0.6.3): Replace this with a function named `get_available_backends`.
AVAILABLE_BACKENDS: Dict[str, Type] = {
    backend.BACKEND_NAME: backend for backend in filter(None, [
        VideoStreamCv2,
        VideoStreamAv,
        VideoStreamMoviePy,
    ])
}
"""All available backends that :func:`scenedetect.open_video` can consider for the `backend`
parameter. These backends must support construction with the following signature:

    BackendType(path: str, framerate: Optional[float])
"""
