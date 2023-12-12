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
""":class:`VideoStreamCv2` is backed by the OpenCV `VideoCapture` object. This is the default
backend. Works with video files, image sequences, and network streams/URLs.

For wrapping input devices or pipes, there is also :class:`VideoCaptureAdapter` which can be
constructed from an existing `cv2.VideoCapture`. This allows performing scene detection on inputs
which do not support seeking.
"""

from logging import getLogger
import math
from typing import AnyStr, Tuple, Union, Optional
import os.path

import cv2
from numpy import ndarray

from backend.scenedetect.frame_timecode import FrameTimecode, MAX_FPS_DELTA
from backend.scenedetect.platform import get_file_name
from backend.scenedetect.video_stream import VideoStream, SeekError, VideoOpenFailure, FrameRateUnavailable

logger = getLogger('pyscenedetect')

IMAGE_SEQUENCE_IDENTIFIER = '%'

NON_VIDEO_FILE_INPUT_IDENTIFIERS = (
    IMAGE_SEQUENCE_IDENTIFIER,       # image sequence
    '://',                           # URL/network stream
    ' ! ',                           # gstreamer pipe
)


def _get_aspect_ratio(cap: cv2.VideoCapture, epsilon: float = 0.0001) -> float:
    """Display/pixel aspect ratio of the VideoCapture as a float (1.0 represents square pixels)."""
    # Versions of OpenCV < 3.4.1 do not support this, so we fall back to 1.0.
    if not 'CAP_PROP_SAR_NUM' in dir(cv2):
        return 1.0
    num: float = cap.get(cv2.CAP_PROP_SAR_NUM)
    den: float = cap.get(cv2.CAP_PROP_SAR_DEN)
    # If numerator or denominator are close to zero, so we fall back to 1.0.
    if abs(num) < epsilon or abs(den) < epsilon:
        return 1.0
    return num / den


class VideoStreamCv2(VideoStream):
    """OpenCV `cv2.VideoCapture` backend."""

    def __init__(
        self,
        path: AnyStr = None,
        framerate: Optional[float] = None,
        max_decode_attempts: int = 5,
        path_or_device: Union[bytes, str, int] = None,
    ):
        """Open a video file, image sequence, or network stream.

        Arguments:
            path: Path to the video. Can be a file, image sequence (`'folder/DSC_%04d.jpg'`),
                or network stream.
            framerate: If set, overrides the detected framerate.
            max_decode_attempts: Number of attempts to continue decoding the video
                after a frame fails to decode. This allows processing videos that
                have a few corrupted frames or metadata (in which case accuracy
                of detection algorithms may be lower). Once this limit is passed,
                decoding will stop and emit an error.
            path_or_device: [DEPRECATED] Specify `path` for files, image sequences, or
                network streams/URLs.  Use `VideoCaptureAdapter` for devices/pipes.

        Raises:
            OSError: file could not be found or access was denied
            VideoOpenFailure: video could not be opened (may be corrupted)
            ValueError: specified framerate is invalid
        """
        super().__init__()
        # TODO(v0.7): Replace with DeprecationWarning that `path_or_device` will be removed in v0.8.
        if path_or_device is not None:
            logger.error('path_or_device is deprecated, use path or VideoCaptureAdapter instead.')
            path = path_or_device
        if path is None:
            raise ValueError('Path must be specified!')
        if framerate is not None and framerate < MAX_FPS_DELTA:
            raise ValueError('Specified framerate (%f) is invalid!' % framerate)
        if max_decode_attempts < 0:
            raise ValueError('Maximum decode attempts must be >= 0!')

        self._path_or_device = path
        self._is_device = isinstance(self._path_or_device, int)

        # Initialized in _open_capture:
        self._cap: Optional[
            cv2.VideoCapture] = None # Reference to underlying cv2.VideoCapture object.
        self._frame_rate: Optional[float] = None

        # VideoCapture state
        self._has_grabbed = False
        self._max_decode_attempts = max_decode_attempts
        self._decode_failures = 0
        self._warning_displayed = False

        self._open_capture(framerate)

    #
    # Backend-Specific Methods/Properties
    #

    @property
    def capture(self) -> cv2.VideoCapture:
        """Returns reference to underlying VideoCapture object. Use with caution.

        Prefer to use this property only to take ownership of the underlying cv2.VideoCapture object
        backing this object. Seeking or using the read/grab methods through this property are
        unsupported and will leave this object in an inconsistent state.
        """
        assert self._cap
        return self._cap

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = 'opencv'
    """Unique name used to identify this backend."""

    @property
    def frame_rate(self) -> float:
        """Framerate in frames/sec."""
        assert self._frame_rate
        return self._frame_rate

    @property
    def path(self) -> Union[bytes, str]:
        """Video or device path."""
        if self._is_device:
            assert isinstance(self._path_or_device, (int))
            return "Device %d" % self._path_or_device
        assert isinstance(self._path_or_device, (bytes, str))
        return self._path_or_device

    @property
    def name(self) -> str:
        """Name of the video, without extension, or device."""
        if self._is_device:
            return self.path
        file_name: str = get_file_name(self.path, include_extension=False)
        if IMAGE_SEQUENCE_IDENTIFIER in file_name:
            # file_name is an image sequence, trim everything including/after the %.
            # TODO: This excludes any suffix after the sequence identifier.
            file_name = file_name[:file_name.rfind(IMAGE_SEQUENCE_IDENTIFIER)]
        return file_name

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise.

        Always False if opening a device/webcam."""
        return not self._is_device

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return (math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        if self._is_device:
            return None
        return self.base_timecode + math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return _get_aspect_ratio(self._cap)

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp of the last frame which was
        decoded by calling `read` with advance=True.

        This method will always return 0 (e.g. be equal to `base_timecode`) if no frames
        have been `read`."""
        if self.frame_number < 1:
            return self.base_timecode
        return self.base_timecode + (self.frame_number - 1)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in milliseconds.
        The first frame has a time of 0.0 ms.

        This method will always return 0.0 if no frames have been `read`."""
        return self._cap.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def frame_number(self) -> int:
        """Current position within stream in frames as an int.

        1 indicates the first frame was just decoded by the last call to `read` with advance=True,
        whereas 0 indicates that no frames have been `read`.

        This method will always return 0 if no frames have been `read`."""
        return math.trunc(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

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
        if self._is_device:
            raise SeekError("Cannot seek if input is a device!")
        if target < 0:
            raise ValueError("Target seek position cannot be negative!")

        # Have to seek one behind and call grab() after to that the VideoCapture
        # returns a valid timestamp when using CAP_PROP_POS_MSEC.
        target_frame_cv2 = (self.base_timecode + target).get_frames()
        if target_frame_cv2 > 0:
            target_frame_cv2 -= 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_cv2)
        self._has_grabbed = False
        # Preemptively grab the frame behind the target position if possible.
        if target > 0:
            self._has_grabbed = self._cap.grab()
            # If we seeked past the end of the video, need to seek one frame backwards
            # from the current position and grab that frame instead.
            if not self._has_grabbed:
                seek_pos = round(self._cap.get(cv2.CAP_PROP_POS_FRAMES) - 1.0)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, seek_pos))
                self._has_grabbed = self._cap.grab()

    def reset(self):
        """ Close and re-open the VideoStream (should be equivalent to calling `seek(0)`). """
        self._cap.release()
        self._open_capture(self._frame_rate)

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """Read and decode the next frame as a numpy.ndarray. Returns False when video ends,
        or the maximum number of decode attempts has passed.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will return the current (last) frame.

        Returns:
            If decode = True, the decoded frame (numpy.ndarray), or False (bool) if end of video.
            If decode = False, a bool indicating if advancing to the the next frame succeeded.
        """
        if not self._cap.isOpened():
            return False
        # Grab the next frame if possible.
        if advance:
            has_grabbed = self._cap.grab()
            # If we failed to grab the frame, retry a few times if required.
            if not has_grabbed:
                if self.duration > 0 and self.position < (self.duration - 1):
                    for _ in range(self._max_decode_attempts):
                        has_grabbed = self._cap.grab()
                        if has_grabbed:
                            break
                # Report previous failure in debug mode.
                if has_grabbed:
                    self._decode_failures += 1
                    logger.debug('Frame failed to decode.')
                    if not self._warning_displayed and self._decode_failures > 1:
                        logger.warning('Failed to decode some frames, results may be inaccurate.')
            # We didn't manage to grab a frame even after retrying, so just return.
            if not has_grabbed:
                return False
            self._has_grabbed = True
        # Need to make sure we actually grabbed a frame before calling retrieve.
        if decode and self._has_grabbed:
            _, frame = self._cap.retrieve()
            return frame
        return self._has_grabbed

    #
    # Private Methods
    #

    def _open_capture(self, framerate: Optional[float] = None):
        """Opens capture referenced by this object and resets internal state."""
        if self._is_device and self._path_or_device < 0:
            raise ValueError('Invalid/negative device ID specified.')
        input_is_video_file = not self._is_device and not any(
            identifier in self._path_or_device for identifier in NON_VIDEO_FILE_INPUT_IDENTIFIERS)
        # We don't have a way of querying why opening a video fails (errors are logged at least),
        # so provide a better error message if we try to open a file that doesn't exist.
        if input_is_video_file:
            if not os.path.exists(self._path_or_device):
                raise OSError('Video file not found.')

        cap = cv2.VideoCapture(self._path_or_device)
        if not cap.isOpened():
            raise VideoOpenFailure(
                'Ensure file is valid video and system dependencies are up to date.\n')

        # Display an error if the video codec type seems unsupported (#86) as this indicates
        # potential video corruption, or may explain missing frames. We only perform this check
        # for video files on-disk (skipped for devices, image sequences, streams, etc...).
        codec_unsupported: bool = (int(abs(cap.get(cv2.CAP_PROP_FOURCC))) == 0)
        if codec_unsupported and input_is_video_file:
            logger.error('Video codec detection failed. If output is incorrect:\n'
                         '  - Re-encode the input video with ffmpeg\n'
                         '  - Update OpenCV (pip install --upgrade opencv-python)\n'
                         '  - Use the PyAV backend (--backend pyav)\n'
                         'For details, see https://github.com/Breakthrough/PySceneDetect/issues/86')

        # Ensure the framerate is correct to avoid potential divide by zero errors. This can be
        # addressed in the PyAV backend if required since it supports integer timebases.
        assert framerate is None or framerate > MAX_FPS_DELTA, "Framerate must be validated if set!"
        if framerate is None:
            framerate = cap.get(cv2.CAP_PROP_FPS)
            if framerate < MAX_FPS_DELTA:
                raise FrameRateUnavailable()

        self._cap = cap
        self._frame_rate = framerate
        self._has_grabbed = False


# TODO(#168): Support non-monotonic timing for `position`. VFR timecode support is a
# prerequisite for this. Timecodes are currently calculated by multiplying the framerate
# by number of frames. Actual elapsed time can be obtained via `position_ms` for now.
class VideoCaptureAdapter(VideoStream):
    """Adapter for existing VideoCapture objects. Unlike VideoStreamCv2, this class supports
    VideoCaptures which may not support seeking.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        framerate: Optional[float] = None,
        max_read_attempts: int = 5,
    ):
        """Create from an existing OpenCV VideoCapture object. Used for webcams, live streams,
        pipes, or other inputs which may not support seeking.

        Arguments:
            cap: The `cv2.VideoCapture` object to wrap. Must already be opened and ready to
                have `cap.read()` called on it.
            framerate: If set, overrides the detected framerate.
            max_read_attempts: Number of attempts to continue decoding the video
                after a frame fails to decode. This allows processing videos that
                have a few corrupted frames or metadata (in which case accuracy
                of detection algorithms may be lower). Once this limit is passed,
                decoding will stop and emit an error.

        Raises:
            ValueError: capture is not open, framerate or max_read_attempts is invalid
        """
        super().__init__()

        if framerate is not None and framerate < MAX_FPS_DELTA:
            raise ValueError('Specified framerate (%f) is invalid!' % framerate)
        if max_read_attempts < 0:
            raise ValueError('Maximum decode attempts must be >= 0!')
        if not cap.isOpened():
            raise ValueError('Specified VideoCapture must already be opened!')
        if framerate is None:
            framerate = cap.get(cv2.CAP_PROP_FPS)
            if framerate < MAX_FPS_DELTA:
                raise FrameRateUnavailable()

        self._cap = cap
        self._frame_rate: float = framerate
        self._num_frames = 0
        self._max_read_attempts = max_read_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        self._time_base: float = 0.0

    #
    # Backend-Specific Methods/Properties
    #

    @property
    def capture(self) -> cv2.VideoCapture:
        """Returns reference to underlying VideoCapture object. Use with caution.

        Prefer to use this property only to take ownership of the underlying cv2.VideoCapture object
        backing this object. Using the read/grab methods through this property are unsupported and
        will leave this object in an inconsistent state.
        """
        assert self._cap
        return self._cap

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = 'opencv_adapter'
    """Unique name used to identify this backend."""

    @property
    def frame_rate(self) -> float:
        """Framerate in frames/sec."""
        assert self._frame_rate
        return self._frame_rate

    @property
    def path(self) -> str:
        """Always 'CAP_ADAPTER'."""
        return 'CAP_ADAPTER'

    @property
    def name(self) -> str:
        """Always 'CAP_ADAPTER'."""
        return 'CAP_ADAPTER'

    @property
    def is_seekable(self) -> bool:
        """Always False, as the underlying VideoCapture is assumed to not support seeking."""
        return False

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Reported size of each video frame in pixels as a tuple of (width, height)."""
        return (math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Always None, as the underlying VideoCapture is assumed to not have a known duration."""
        None

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return _get_aspect_ratio(self._cap)

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode. Use the :meth:`position_ms`
        if an accurate duration of elapsed time is required, as `position` is currently
        based off of the number of frames, and may not be accurate for devicesor live streams.

        This method will always return 0 (e.g. be equal to `base_timecode`) if no frames
        have been `read`."""
        if self.frame_number < 1:
            return self.base_timecode
        return self.base_timecode + (self.frame_number - 1)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in milliseconds.
        The first frame has a time of 0.0 ms.

        This method will always return 0.0 if no frames have been `read`."""
        if self._num_frames == 0:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) - self._time_base

    @property
    def frame_number(self) -> int:
        """Current position within stream in frames as an int.

        1 indicates the first frame was just decoded by the last call to `read` with advance=True,
        whereas 0 indicates that no frames have been `read`.

        This method will always return 0 if no frames have been `read`."""
        return self._num_frames

    def seek(self, target: Union[FrameTimecode, float, int]):
        """The underlying VideoCapture is assumed to not support seeking."""
        raise NotImplementedError("Seeking is not supported.")

    def reset(self):
        """Not supported."""
        raise NotImplementedError("Reset is not supported.")

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """Read and decode the next frame as a numpy.ndarray. Returns False when video ends,
        or the maximum number of decode attempts has passed.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will return the current (last) frame.

        Returns:
            If decode = True, the decoded frame (numpy.ndarray), or False (bool) if end of video.
            If decode = False, a bool indicating if advancing to the the next frame succeeded.
        """
        if not self._cap.isOpened():
            return False
        # Grab the next frame if possible.
        if advance:
            has_grabbed = self._cap.grab()
            # If we failed to grab the frame, retry a few times if required.
            if not has_grabbed:
                for _ in range(self._max_read_attempts):
                    has_grabbed = self._cap.grab()
                    if has_grabbed:
                        break
                # Report previous failure in debug mode.
                if has_grabbed:
                    self._decode_failures += 1
                    logger.debug('Frame failed to decode.')
                    if not self._warning_displayed and self._decode_failures > 1:
                        logger.warning('Failed to decode some frames, results may be inaccurate.')
            # We didn't manage to grab a frame even after retrying, so just return.
            if not has_grabbed:
                return False
            if self._num_frames == 0:
                self._time_base = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self._num_frames += 1
        # Need to make sure we actually grabbed a frame before calling retrieve.
        if decode and self._num_frames > 0:
            _, frame = self._cap.retrieve()
            return frame
        return True
