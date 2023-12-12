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
""":class:`VideoStreamAv` provides an adapter for the PyAV av.InputContainer object.

Uses string identifier ``'pyav'``.
"""

from logging import getLogger
from typing import AnyStr, BinaryIO, Optional, Tuple, Union

# pylint: disable=c-extension-no-member
import av
from numpy import ndarray

from scenedetect.frame_timecode import FrameTimecode, MAX_FPS_DELTA
from scenedetect.platform import get_file_name
from scenedetect.video_stream import VideoStream, VideoOpenFailure, FrameRateUnavailable

logger = getLogger('pyscenedetect')

VALID_THREAD_MODES = [
    av.codec.context.ThreadType.NONE,
    av.codec.context.ThreadType.SLICE,
    av.codec.context.ThreadType.FRAME,
    av.codec.context.ThreadType.AUTO,
]


class VideoStreamAv(VideoStream):
    """PyAV `av.InputContainer` backend."""

    # TODO: Investigate adding an accurate_duration option to backends to calculate the duration
    # with higher precision. Sometimes it doesn't exactly match what the codec or VLC reports,
    # but we can try to seek to the end of the video first to determine it. Investigate how VLC
    # calculates the end time.
    def __init__(
        self,
        path_or_io: Union[AnyStr, BinaryIO],
        framerate: Optional[float] = None,
        name: Optional[str] = None,
        threading_mode: Optional[str] = None,
        suppress_output: bool = False,
    ):
        """Open a video by path.

        .. warning::

            Using `threading_mode` with `suppress_output = True` can cause lockups in your
            application. See the PyAV documentation for details:
            https://pyav.org/docs/stable/overview/caveats.html#sub-interpeters

        Arguments:
            path_or_io: Path to the video, or a file-like object.
            framerate: If set, overrides the detected framerate.
            name: Overrides the `name` property derived from the video path. Should be set if
                `path_or_io` is a file-like object.
            threading_mode: The PyAV video stream `thread_type`. See av.codec.context.ThreadType
                for valid threading modes ('AUTO', 'FRAME', 'NONE', and 'SLICE'). If this mode is
                'AUTO' or 'FRAME' and not all frames have been decoded, the video will be reopened
                if seekable, and the remaining frames decoded in single-threaded mode.
            suppress_output: If False, ffmpeg output will be sent to stdout/stderr by calling
                `av.logging.restore_default_callback()` before any other library calls. If True
                the application may deadlock if threading_mode is set. See the PyAV documentation
                for details: https://pyav.org/docs/stable/overview/caveats.html#sub-interpeters

        Raises:
            OSError: file could not be found or access was denied
            VideoOpenFailure: video could not be opened (may be corrupted)
            ValueError: specified framerate is invalid
        """
        self._container = None

        # TODO(#258): See what self._container.discard_corrupt = True does with corrupt videos.
        super().__init__()

        # Ensure specified framerate is valid if set.
        if framerate is not None and framerate < MAX_FPS_DELTA:
            raise ValueError('Specified framerate (%f) is invalid!' % framerate)

        self._name = '' if name is None else name
        self._path = ''
        self._frame = None
        self._reopened = True

        if threading_mode:
            threading_mode = threading_mode.upper()
            if not threading_mode in VALID_THREAD_MODES:
                raise ValueError('Invalid threading mode! Must be one of: %s' % VALID_THREAD_MODES)

        if not suppress_output:
            logger.debug('Restoring default ffmpeg log callbacks.')
            av.logging.restore_default_callback()

        try:
            if isinstance(path_or_io, (str, bytes)):
                self._path = path_or_io
                self._io = open(path_or_io, 'rb')
                if not self._name:
                    self._name = get_file_name(self.path, include_extension=False)
            else:
                self._io = path_or_io

            self._container = av.open(self._io)
            if threading_mode is not None:
                self._video_stream.thread_type = threading_mode
                self._reopened = False
                logger.debug('Threading mode set: %s', threading_mode)
        except OSError:
            raise
        except Exception as ex:
            raise VideoOpenFailure(str(ex)) from ex

        if framerate is None:
            # Calculate framerate from video container. `guessed_rate` below appears in PyAV 9.
            frame_rate = self._video_stream.guessed_rate if hasattr(
                self._video_stream, 'guessed_rate') else self._codec_context.framerate
            if frame_rate is None or frame_rate == 0:
                raise FrameRateUnavailable()
            # TODO: Refactor FrameTimecode to support raw timing rather than framerate based calculations.
            # See https://pyav.org/docs/develop/api/stream.html for details.
            frame_rate = frame_rate.numerator / float(frame_rate.denominator)
            if frame_rate < MAX_FPS_DELTA:
                raise FrameRateUnavailable()
            self._frame_rate: float = frame_rate
        else:
            assert framerate >= MAX_FPS_DELTA
            self._frame_rate: float = framerate

        # Calculate duration after we have set the framerate.
        self._duration_frames = self._get_duration()

    def __del__(self):
        if self._container is not None:
            self._container.close()

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = 'pyav'
    """Unique name used to identify this backend."""

    @property
    def path(self) -> Union[bytes, str]:
        """Video path."""
        return self._path

    @property
    def name(self) -> Union[bytes, str]:
        """Name of the video, without extension."""
        return self._name

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise."""
        return self._io.seekable()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return (self._codec_context.width, self._codec_context.height)

    @property
    def duration(self) -> FrameTimecode:
        """Duration of the video as a FrameTimecode."""
        return self.base_timecode + self._duration_frames

    @property
    def frame_rate(self) -> float:
        """Frame rate in frames/sec."""
        return self._frame_rate

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp, thus frame 1 corresponds
        to the presentation time 0.  Returns 0 even if `frame_number` is 1."""
        if self._frame is None:
            return self.base_timecode
        return FrameTimecode(round(self._frame.time * self.frame_rate), self.frame_rate)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in
        milliseconds. The first frame has a PTS of 0."""
        if self._frame is None:
            return 0.0
        return self._frame.time * 1000.0

    @property
    def frame_number(self) -> int:
        """Current position within stream as the frame number.

        Will return 0 until the first frame is `read`."""
        if self._frame:
            return self.position.frame_num + 1
        return 0

    @property
    def aspect_ratio(self) -> float:
        """Pixel aspect ratio as a float (1.0 represents square pixels)."""
        display_aspect_ratio = (
            self._codec_context.display_aspect_ratio.numerator /
            self._codec_context.display_aspect_ratio.denominator)
        frame_aspect_ratio = self.frame_size[0] / self.frame_size[1]
        return display_aspect_ratio / frame_aspect_ratio

    def seek(self, target: Union[FrameTimecode, float, int]) -> None:
        """Seek to the given timecode. If given as a frame number, represents the current seek
        pointer (e.g. if seeking to 0, the next frame decoded will be the first frame of the video).

        For 1-based indices (first frame is frame #1), the target frame number needs to be converted
        to 0-based by subtracting one. For example, if we want to seek to the first frame, we call
        seek(0) followed by read(). If we want to seek to the 5th frame, we call seek(4) followed
        by read(), at which point frame_number will be 5.

        May not be supported on all input codecs (see `is_seekable`).

        Arguments:
            target: Target position in video stream to seek to.
                If float, interpreted as time in seconds.
                If int, interpreted as frame number.
        Raises:
            ValueError: `target` is not a valid value (i.e. it is negative).
        """
        if target < 0:
            raise ValueError("Target cannot be negative!")
        beginning = (target == 0)
        target = (self.base_timecode + target)
        if target >= 1:
            target = target - 1
        target_pts = self._video_stream.start_time + int(
            (self.base_timecode + target).get_seconds() / self._video_stream.time_base)
        self._frame = None
        self._container.seek(target_pts, stream=self._video_stream)
        if not beginning:
            self.read(decode=False, advance=True)
        while self.position < target:
            if self.read(decode=False, advance=True) is False:
                break

    def reset(self):
        """ Close and re-open the VideoStream (should be equivalent to calling `seek(0)`). """
        self._container.close()
        self._frame = None
        try:
            self._container = av.open(self._path if self._path else self._io)
        except Exception as ex:
            raise VideoOpenFailure() from ex

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """Read and decode the next frame as a numpy.ndarray. Returns False when video ends.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will return the current (last) frame.

        Returns:
            If decode = True, the decoded frame (numpy.ndarray), or False (bool) if end of video.
            If decode = False, a bool indicating if advancing to the the next frame succeeded.
        """
        has_advanced = False
        if advance:
            try:
                last_frame = self._frame
                self._frame = next(self._container.decode(video=0))
            except av.error.EOFError:
                self._frame = last_frame
                if self._handle_eof():
                    return self.read(decode, advance=True)
                return False
            except StopIteration:
                return False
            has_advanced = True
        if decode:
            return self._frame.to_ndarray(format='bgr24')
        return has_advanced

    #
    # Private Methods/Properties
    #

    @property
    def _video_stream(self):
        """PyAV `av.video.stream.VideoStream` being used."""
        return self._container.streams.video[0]

    @property
    def _codec_context(self):
        """PyAV `av.codec.context.CodecContext` being used."""
        return self._video_stream.codec_context

    def _get_duration(self) -> int:
        """Get video duration as number of frames based on the video and set framerate."""
        # See https://pyav.org/docs/develop/api/time.html for details on how ffmpeg/PyAV
        # handle time calculations internally and which time base to use.
        assert self.frame_rate is not None, "Frame rate must be set before calling _get_duration!"
        # See if we can obtain the number of frames directly from the stream itself.
        if self._video_stream.frames > 0:
            return self._video_stream.frames
        # Calculate based on the reported container duration.
        duration_sec = None
        container = self._video_stream.container
        if container.duration is not None and container.duration > 0:
            # Containers use AV_TIME_BASE as the time base.
            duration_sec = float(self._video_stream.container.duration / av.time_base)
        # Lastly, if that calculation fails, try to calculate it based on the stream duration.
        if duration_sec is None or duration_sec < MAX_FPS_DELTA:
            if self._video_stream.duration is None:
                logger.warning('Video duration unavailable.')
                return 0
            # Streams use stream `time_base` as the time base.
            time_base = self._video_stream.time_base
            if time_base.denominator == 0:
                logger.warning(
                    'Unable to calculate video duration: time_base (%s) has zero denominator!',
                    str(time_base))
                return 0
            duration_sec = float(self._video_stream.duration / time_base)
        return round(duration_sec * self.frame_rate)

    def _handle_eof(self):
        """Fix for issue where if thread_type is 'AUTO' the whole video is not decoded.

        Re-open video if the threading mode is AUTO and we didn't decode all of the frames."""
        # Don't re-open the video if we already did, or if we already decoded all the frames.
        if self._reopened or self.frame_number >= self.duration:
            return False
        self._reopened = True
        # Don't re-open the video if we can't seek or aren't in AUTO/FRAME thread_type mode.
        if not self.is_seekable or not self._video_stream.thread_type in ('AUTO', 'FRAME'):
            return False
        last_frame = self.frame_number
        orig_pos = self._io.tell()
        try:
            self._io.seek(0)
            container = av.open(self._io)
        except:
            self._io.seek(orig_pos)
            raise
        self._container.close()
        self._container = container
        self.seek(last_frame)
        return True
