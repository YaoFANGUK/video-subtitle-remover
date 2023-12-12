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
"""``scenedetect.video_stream`` Module

This module contains the :class:`VideoStream` class, which provides a library agnostic
interface for video input. To open a video by path, use :func:`scenedetect.open_video`:

.. code:: python

    from scenedetect import open_video
    video = open_video('video.mp4')
    while True:
        frame = video.read()
        if frame is False:
            break
    print("Read %d frames" % video.frame_number)

You can also optionally specify a framerate and a specific backend library to use. Unless specified,
OpenCV will be used as the video backend. See :mod:`scenedetect.backends` for a detailed example.

New :class:`VideoStream <scenedetect.video_stream.VideoStream>` implementations can be
tested by adding it to the test suite in `tests/test_video_stream.py`.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Tuple, Optional, Union

from numpy import ndarray

from backend.scenedetect.frame_timecode import FrameTimecode

##
## VideoStream Exceptions
##


class SeekError(Exception):
    """Either an unrecoverable error happened while attempting to seek, or the underlying
    stream is not seekable (additional information will be provided when possible).

    The stream is guaranteed to be left in a valid state, but the position may be reset."""


class VideoOpenFailure(Exception):
    """Raised by a backend if opening a video fails."""

    # pylint: disable=useless-super-delegation
    def __init__(self, message: str = "Unknown backend error."):
        """
        Arguments:
            message: Additional context the backend can provide for the open failure.
        """
        super().__init__(message)

    # pylint: enable=useless-super-delegation


class FrameRateUnavailable(VideoOpenFailure):
    """Exception instance to provide consistent error messaging across backends when the video frame
    rate is unavailable or cannot be calculated. Subclass of VideoOpenFailure."""

    def __init__(self):
        super().__init__('Unable to obtain video framerate! Specify `framerate` manually, or'
                         ' re-encode/re-mux the video and try again.')


##
## VideoStream Interface (Base Class)
##


class VideoStream(ABC):
    """ Interface which all video backends must implement. """

    #
    # Default Implementations
    #

    @property
    def base_timecode(self) -> FrameTimecode:
        """FrameTimecode object to use as a time base."""
        return FrameTimecode(timecode=0, fps=self.frame_rate)

    #
    # Abstract Static Methods
    #

    @staticmethod
    @abstractmethod
    def BACKEND_NAME() -> str:
        """Unique name used to identify this backend. Should be a static property in derived
        classes (`BACKEND_NAME = 'backend_identifier'`)."""
        raise NotImplementedError

    #
    # Abstract Properties
    #

    @property
    @abstractmethod
    def path(self) -> Union[bytes, str]:
        """Video or device path."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> Union[bytes, str]:
        """Name of the video, without extension, or device."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        """Frame rate in frames/sec."""
        raise NotImplementedError

    @property
    @abstractmethod
    def duration(self) -> Optional[FrameTimecode]:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def aspect_ratio(self) -> float:
        """Pixel aspect ratio as a float (1.0 represents square pixels)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp, thus frame 1 corresponds
        to the presentation time 0.  Returns 0 even if `frame_number` is 1."""
        raise NotImplementedError

    @property
    @abstractmethod
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in
        milliseconds. The first frame has a PTS of 0."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_number(self) -> int:
        """Current position within stream as the frame number.

        Will return 0 until the first frame is `read`."""
        raise NotImplementedError

    #
    # Abstract Methods
    #

    @abstractmethod
    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """Read and decode the next frame as a numpy.ndarray. Returns False when video ends.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will return the current (last) frame.

        Returns:
            If decode = True, the decoded frame (numpy.ndarray), or False (bool) if end of video.
            If decode = False, a bool indicating if advancing to the the next frame succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """ Close and re-open the VideoStream (equivalent to seeking back to beginning). """
        raise NotImplementedError

    @abstractmethod
    def seek(self, target: Union[FrameTimecode, float, int]) -> None:
        """Seek to the given timecode. If given as a frame number, represents the current seek
        pointer (e.g. if seeking to 0, the next frame decoded will be the first frame of the video).

        For 1-based indices (first frame is frame #1), the target frame number needs to be converted
        to 0-based by subtracting one. For example, if we want to seek to the first frame, we call
        seek(0) followed by read(). If we want to seek to the 5th frame, we call seek(4) followed
        by read(), at which point frame_number will be 5.

        May not be supported on all backend types or inputs (e.g. cameras).

        Arguments:
            target: Target position in video stream to seek to.
                If float, interpreted as time in seconds.
                If int, interpreted as frame number.
        Raises:
            SeekError: An error occurs while seeking, or seeking is not supported.
            ValueError: `target` is not a valid value (i.e. it is negative).
        """
        raise NotImplementedError


# TODO(0.6.3): Add a StreamJoiner class to concatenate multiple videos using a specified backend.
