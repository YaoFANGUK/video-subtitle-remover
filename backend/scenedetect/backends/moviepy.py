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
""":class:`VideoStreamMoviePy` provides an adapter for MoviePy's `FFMPEG_VideoReader`.

Uses string identifier ``'moviepy'``.

.. warning::

    The MoviePy backend is still under development. Some features are not yet supported.
"""

from logging import getLogger
from typing import AnyStr, Tuple, Union, Optional

import cv2
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from numpy import ndarray

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.platform import get_file_name
from scenedetect.video_stream import VideoStream, SeekError, VideoOpenFailure
from scenedetect.backends.opencv import VideoStreamCv2

logger = getLogger('pyscenedetect')


class VideoStreamMoviePy(VideoStream):
    """MoviePy `FFMPEG_VideoReader` backend."""

    def __init__(self, path: AnyStr, framerate: Optional[float] = None, print_infos: bool = False):
        """Open a video or device.

        Arguments:
            path: Path to video,.
            framerate: If set, overrides the detected framerate.
            print_infos: If True, prints information about the opened video to stdout.

        Raises:
            OSError: file could not be found, access was denied, or the video is corrupt
            VideoOpenFailure: video could not be opened (may be corrupted)
        """
        super().__init__()

        # TODO(0.6.3) - Investigate how MoviePy handles ffmpeg not being on PATH.
        # TODO(0.6.3): Add framerate override.
        if framerate is not None:
            raise NotImplementedError("TODO(0.6.3)")

        self._path = path
        # TODO(0.6.3): Need to map errors based on the strings, since several failure
        # cases return IOErrors (e.g. could not read duration/video resolution). These
        # should be mapped to specific errors, e.g. write a function to map MoviePy
        # exceptions to a new set of equivalents.
        self._reader = FFMPEG_VideoReader(path, print_infos=print_infos)
        # This will always be one behind self._reader.lastread when we finally call read()
        # as MoviePy caches the first frame when opening the video. Thus self._last_frame
        # will always be the current frame, and self._reader.lastread will be the next.
        self._last_frame: Union[bool, ndarray] = False
        self._last_frame_rgb: Optional[ndarray] = None
        # Older versions don't track the video position when calling read_frame so we need
        # to keep track of the current frame number.
        self._frame_number = 0
        # We need to manually keep track of EOF as duration may not be accurate.
        self._eof = False
        # MoviePy doesn't support extracting the aspect ratio yet, so for now we just fall
        # back to using OpenCV to determine it.
        try:
            self._aspect_ratio = VideoStreamCv2(self._path).aspect_ratio
        except VideoOpenFailure as ex:
            logger.warning("Unable to determine aspect ratio: %s", str(ex))
            self._aspect_ratio = 1.0

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = 'moviepy'
    """Unique name used to identify this backend."""

    @property
    def frame_rate(self) -> float:
        """Framerate in frames/sec."""
        return self._reader.fps

    @property
    def path(self) -> Union[bytes, str]:
        """Video path."""
        return self._path

    @property
    def name(self) -> str:
        """Name of the video, without extension, or device."""
        return get_file_name(self.path, include_extension=False)

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise."""
        return True

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return tuple(self._reader.infos['video_size'])

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        assert isinstance(self._reader.infos['duration'], float)
        return self.base_timecode + self._reader.infos['duration']

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return self._aspect_ratio

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp of the last frame which was
        decoded by calling `read` with advance=True.

        This method will always return 0 (e.g. be equal to `base_timecode`) if no frames
        have been `read`."""
        frame_number = max(self._frame_number - 1, 0)
        return FrameTimecode(frame_number, self.frame_rate)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in milliseconds.
        The first frame has a time of 0.0 ms.

        This method will always return 0.0 if no frames have been `read`."""
        return self.position.get_seconds() * 1000.0

    @property
    def frame_number(self) -> int:
        """Current position within stream in frames as an int.

        1 indicates the first frame was just decoded by the last call to `read` with advance=True,
        whereas 0 indicates that no frames have been `read`.

        This method will always return 0 if no frames have been `read`."""
        return self._frame_number

    def seek(self, target: Union[FrameTimecode, float, int]):
        """Seek to the given timecode. If given as a frame number, represents the current seek
        pointer (e.g. if seeking to 0, the next frame decoded will be the first frame of the video).

        For 1-based indices (first frame is frame #1), the target frame number needs to be converted
        to 0-based by subtracting one. For example, if we want to seek to the first frame, we call
        seek(0) followed by read(). If we want to seek to the 5th frame, we call seek(4) followed
        by read(), at which point frame_number will be 5.

        Not supported if the VideoStream is a device/camera. Untested with web streams.

        Arguments:
            target: Target position in video stream to seek to.
                If float, interpreted as time in seconds.
                If int, interpreted as frame number.
        Raises:
            SeekError: An error occurs while seeking, or seeking is not supported.
            ValueError: `target` is not a valid value (i.e. it is negative).
        """
        if not isinstance(target, FrameTimecode):
            target = FrameTimecode(target, self.frame_rate)
        try:
            self._reader.get_frame(target.get_seconds())
        except IOError as ex:
            # Leave the object in a valid state.
            self.reset()
            if target >= self.duration:
                raise SeekError("Target frame is beyond end of video!") from ex
            raise
        self._last_frame = self._reader.lastread
        self._frame_number = target.frame_num

    def reset(self):
        """ Close and re-open the VideoStream (should be equivalent to calling `seek(0)`). """
        self._reader.initialize()
        self._last_frame = self._reader.read_frame()
        self._frame_number = 0
        self._eof = False

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """Read and decode the next frame as a numpy.ndarray. Returns False when video ends.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will return the current (last) frame.

        Returns:
            If decode = True, the decoded frame (numpy.ndarray), or False (bool) if end of video.
            If decode = False, a bool indicating if advancing to the the next frame succeeded.
        """
        if not advance:
            if self._last_frame_rgb is None:
                self._last_frame_rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
            return self._last_frame_rgb
        if not hasattr(self._reader, 'lastread'):
            return False
        self._last_frame = self._reader.lastread
        self._reader.read_frame()
        if self._last_frame is self._reader.lastread:
            # Didn't decode a new frame, must have hit EOF.
            if self._eof:
                return False
            self._eof = True
        self._frame_number += 1
        if decode:
            if self._last_frame is not None:
                self._last_frame_rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
            return self._last_frame_rgb
        return True
