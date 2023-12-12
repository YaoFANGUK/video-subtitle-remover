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
"""``scenedetect.video_manager`` Module

[DEPRECATED] DO NOT USE. Use `open_video` from `scenedetect.backends` or create a
VideoStreamCv2 object (`scenedetect.backends.opencv`) instead.

This module exists for *some* backwards compatibility with v0.5, and will be removed
in a future release.
"""

import os
import math
from logging import getLogger

from typing import Iterable, List, Optional, Tuple, Union
from numpy import ndarray
import cv2

from scenedetect.platform import get_file_name
from scenedetect.frame_timecode import FrameTimecode, MAX_FPS_DELTA
from scenedetect.video_stream import VideoStream, VideoOpenFailure, FrameRateUnavailable
from scenedetect.backends.opencv import _get_aspect_ratio

##
## VideoManager Exceptions
##


class VideoParameterMismatch(Exception):
    """ VideoParameterMismatch: Raised when opening multiple videos with a VideoManager, and some
    of the video parameters (frame height, frame width, and framerate/FPS) do not match. """

    def __init__(self,
                 file_list=None,
                 message="OpenCV VideoCapture object parameters do not match."):
        # type: (Iterable[Tuple[int, float, float, str, str]], str) -> None
        # Pass message string to base Exception class.
        super(VideoParameterMismatch, self).__init__(message)
        # list of (param_mismatch_type: int, parameter value, expected value,
        #          filename: str, filepath: str)
        # where param_mismatch_type is an OpenCV CAP_PROP (e.g. CAP_PROP_FPS).
        self.file_list = file_list


class VideoDecodingInProgress(RuntimeError):
    """ VideoDecodingInProgress: Raised when attempting to call certain VideoManager methods that
    must be called *before* start() has been called. """


class InvalidDownscaleFactor(ValueError):
    """ InvalidDownscaleFactor: Raised when trying to set invalid downscale factor,
    i.e. the supplied downscale factor was not a positive integer greater than zero. """


##
## VideoManager Helper Functions
##


def get_video_name(video_file: str) -> Tuple[str, str]:
    """Get the video file/device name.

    Returns:
        Tuple of the form [name, video_file].
    """
    if isinstance(video_file, int):
        return ('Device %d' % video_file, video_file)
    return (os.path.split(video_file)[1], video_file)


def get_num_frames(cap_list: Iterable[cv2.VideoCapture]) -> int:
    """ Get Number of Frames: Returns total number of frames in the cap_list.

    Calls get(CAP_PROP_FRAME_COUNT) and returns the sum for all VideoCaptures.
    """
    return sum([math.trunc(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in cap_list])


def open_captures(
    video_files: Iterable[str],
    framerate: Optional[float] = None,
    validate_parameters: bool = True,
) -> Tuple[List[cv2.VideoCapture], float, Tuple[int, int]]:
    """ Open Captures - helper function to open all capture objects, set the framerate,
    and ensure that all open captures have been opened and the framerates match on a list
    of video file paths, or a list containing a single device ID.

    Arguments:
        video_files: List of one or more paths (str), or a list
            of a single integer device ID, to open as an OpenCV VideoCapture object.
            A ValueError will be raised if the list does not conform to the above.
        framerate: Framerate to assume when opening the video_files.
            If not set, the first open video is used for deducing the framerate of
            all videos in the sequence.
        validate_parameters (bool, optional): If true, will ensure that the frame sizes
            (width, height) and frame rate (FPS) of all passed videos is the same.
            A VideoParameterMismatch is raised if the framerates do not match.

    Returns:
        A tuple of form (cap_list, framerate, framesize) where cap_list is a list of open
        OpenCV VideoCapture objects in the same order as the video_files list, framerate
        is a float of the video(s) framerate(s), and framesize is a tuple of (width, height)
        where width and height are integers representing the frame size in pixels.

    Raises:
        ValueError: No video file(s) specified, or invalid/multiple device IDs specified.
        TypeError: `framerate` must be type `float`.
        IOError: Video file(s) not found.
        FrameRateUnavailable: Video framerate could not be obtained and `framerate`
            was not set manually.
        VideoParameterMismatch: All videos in `video_files` do not have equal parameters.
            Set `validate_parameters=False` to skip this check.
        VideoOpenFailure: Video(s) could not be opened.
    """
    is_device = False
    if not video_files:
        raise ValueError("Expected at least 1 video file or device ID.")
    if isinstance(video_files[0], int):
        if len(video_files) > 1:
            raise ValueError("If device ID is specified, no video sources may be appended.")
        elif video_files[0] < 0:
            raise ValueError("Invalid/negative device ID specified.")
        is_device = True
    elif not all([isinstance(video_file, (str, bytes)) for video_file in video_files]):
        print(video_files)
        raise ValueError("Unexpected element type in video_files list (expected str(s)/int).")
    elif framerate is not None and not isinstance(framerate, float):
        raise TypeError("Expected type float for parameter framerate.")
    # Check if files exist if passed video file is not an image sequence
    # (checked with presence of % in filename) or not a URL (://).
    if not is_device and any([
            not os.path.exists(video_file)
            for video_file in video_files
            if not ('%' in video_file or '://' in video_file)
    ]):
        raise IOError("Video file(s) not found.")
    cap_list = []

    try:
        cap_list = [cv2.VideoCapture(video_file) for video_file in video_files]
        video_names = [get_video_name(video_file) for video_file in video_files]
        closed_caps = [video_names[i] for i, cap in enumerate(cap_list) if not cap.isOpened()]
        if closed_caps:
            raise VideoOpenFailure(str(closed_caps))

        cap_framerates = [cap.get(cv2.CAP_PROP_FPS) for cap in cap_list]
        cap_framerate, check_framerate = validate_capture_framerate(video_names, cap_framerates,
                                                                    framerate)
        # Store frame sizes as integers (VideoCapture.get() returns float).
        cap_frame_sizes = [(math.trunc(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            math.trunc(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cap in cap_list]
        cap_frame_size = cap_frame_sizes[0]

        # If we need to validate the parameters, we check that the FPS and width/height
        # of all open captures is identical (or almost identical in the case of FPS).
        if validate_parameters:
            validate_capture_parameters(
                video_names=video_names,
                cap_frame_sizes=cap_frame_sizes,
                check_framerate=check_framerate,
                cap_framerates=cap_framerates)

    except:
        for cap in cap_list:
            cap.release()
        raise

    return (cap_list, cap_framerate, cap_frame_size)


def validate_capture_framerate(
    video_names: Iterable[Tuple[str, str]],
    cap_framerates: List[float],
    framerate: Optional[float] = None,
) -> Tuple[float, bool]:
    """Ensure the passed capture framerates are valid and equal.

    Raises:
        ValueError: Invalid framerate (must be positive non-zero value).
        TypeError: Framerate must be of type float.
        FrameRateUnavailable: Framerate for video could not be obtained,
            and `framerate` was not set.
    """
    check_framerate = True
    cap_framerate = cap_framerates[0]
    if framerate is not None:
        if isinstance(framerate, float):
            if framerate < MAX_FPS_DELTA:
                raise ValueError("Invalid framerate (must be a positive non-zero value).")
            cap_framerate = framerate
            check_framerate = False
        else:
            raise TypeError("Expected float for framerate, got %s." % type(framerate).__name__)
    else:
        unavailable_framerates = [(video_names[i][0], video_names[i][1])
                                  for i, fps in enumerate(cap_framerates)
                                  if fps < MAX_FPS_DELTA]
        if unavailable_framerates:
            raise FrameRateUnavailable()
    return (cap_framerate, check_framerate)


def validate_capture_parameters(
    video_names: List[Tuple[str, str]],
    cap_frame_sizes: List[Tuple[int, int]],
    check_framerate: bool = False,
    cap_framerates: Optional[List[float]] = None,
) -> None:
    """ Validate Capture Parameters: Ensures that all passed capture frame sizes and (optionally)
    framerates are equal.  Raises VideoParameterMismatch if there is a mismatch.

    Raises:
        VideoParameterMismatch
    """
    bad_params = []
    max_framerate_delta = MAX_FPS_DELTA
    # Check heights/widths match.
    bad_params += [(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0], cap_frame_sizes[0][0],
                    video_names[i][0], video_names[i][1])
                   for i, frame_size in enumerate(cap_frame_sizes)
                   if abs(frame_size[0] - cap_frame_sizes[0][0]) > 0]
    bad_params += [(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1], cap_frame_sizes[0][1],
                    video_names[i][0], video_names[i][1])
                   for i, frame_size in enumerate(cap_frame_sizes)
                   if abs(frame_size[1] - cap_frame_sizes[0][1]) > 0]
    # Check framerates if required.
    if check_framerate:
        bad_params += [(cv2.CAP_PROP_FPS, fps, cap_framerates[0], video_names[i][0],
                        video_names[i][1])
                       for i, fps in enumerate(cap_framerates)
                       if math.fabs(fps - cap_framerates[0]) > max_framerate_delta]

    if bad_params:
        raise VideoParameterMismatch(bad_params)


##
## VideoManager Class Implementation
##


class VideoManager(VideoStream):
    """[DEPRECATED] DO NOT USE.

    Provides a cv2.VideoCapture-like interface to a set of one or more video files,
    or a single device ID. Supports seeking and setting end time/duration."""

    BACKEND_NAME = 'video_manager_do_not_use'

    def __init__(self,
                 video_files: List[str],
                 framerate: Optional[float] = None,
                 logger=getLogger('pyscenedetect')):
        """[DEPRECATED] DO NOT USE.

        Arguments:
            video_files (list of str(s)/int): A list of one or more paths (str), or a list
                of a single integer device ID, to open as an OpenCV VideoCapture object.
            framerate (float, optional): Framerate to assume when storing FrameTimecodes.
                If not set (i.e. is None), it will be deduced from the first open capture
                in video_files, else raises a FrameRateUnavailable exception.

        Raises:
            ValueError: No video file(s) specified, or invalid/multiple device IDs specified.
            TypeError: `framerate` must be type `float`.
            IOError: Video file(s) not found.
            FrameRateUnavailable: Video framerate could not be obtained and `framerate`
                was not set manually.
            VideoParameterMismatch: All videos in `video_files` do not have equal parameters.
                Set `validate_parameters=False` to skip this check.
            VideoOpenFailure: Video(s) could not be opened.
        """
        # TODO(v0.7): Add DeprecationWarning that this class will be removed in v0.8: 'VideoManager
        # will be removed in PySceneDetect v0.8. Use VideoStreamCv2 or VideoCaptureAdapter instead.'
        logger.error("VideoManager is deprecated and will be removed.")
        if not video_files:
            raise ValueError("At least one string/integer must be passed in the video_files list.")
        # Need to support video_files as a single str too for compatibility.
        if isinstance(video_files, str):
            video_files = [video_files]
        # These VideoCaptures are only open in this process.
        self._is_device = isinstance(video_files[0], int)
        self._cap_list, self._cap_framerate, self._cap_framesize = open_captures(
            video_files=video_files, framerate=framerate)
        self._path = video_files[0] if not self._is_device else video_files
        self._end_of_video = False
        self._start_time = self.get_base_timecode()
        self._end_time = None
        self._curr_time = self.get_base_timecode()
        self._last_frame = None
        self._curr_cap, self._curr_cap_idx = None, None
        self._video_file_paths = video_files
        self._logger = logger
        if self._logger is not None:
            self._logger.info('Loaded %d video%s, framerate: %.3f FPS, resolution: %d x %d',
                              len(self._cap_list), 's' if len(self._cap_list) > 1 else '',
                              self.get_framerate(), *self.get_framesize())
        self._started = False
        self._frame_length = self.get_base_timecode() + get_num_frames(self._cap_list)
        self._first_cap_len = self.get_base_timecode() + get_num_frames([self._cap_list[0]])
        self._aspect_ratio = _get_aspect_ratio(self._cap_list[0])

    def set_downscale_factor(self, downscale_factor=None):
        """No-op. Set downscale_factor in `SceneManager` instead."""
        _ = downscale_factor

    def get_num_videos(self) -> int:
        """Get the length of the internal capture list,
        representing the number of videos the VideoManager was constructed with.

        Returns:
            int: Number of videos, equal to length of capture list.
        """
        return len(self._cap_list)

    def get_video_paths(self) -> List[str]:
        """Get list of strings containing paths to the open video(s).

        Returns:
            List[str]: List of paths to the video files opened by the VideoManager.
        """
        return list(self._video_file_paths)

    def get_video_name(self) -> str:
        """Get name of the video based on the first video path.

        Returns:
            The base name of the video file, without extension.
        """
        video_paths = self.get_video_paths()
        if not video_paths:
            return ''
        video_name = os.path.basename(video_paths[0])
        if video_name.rfind('.') >= 0:
            video_name = video_name[:video_name.rfind('.')]
        return video_name

    def get_framerate(self) -> float:
        """Get the framerate the VideoManager is assuming for all
        open VideoCaptures.  Obtained from either the capture itself, or the passed
        framerate parameter when the VideoManager object was constructed.

        Returns:
            Framerate, in frames/sec.
        """
        return self._cap_framerate

    def get_base_timecode(self) -> FrameTimecode:
        """Get a FrameTimecode object at frame 0 / time 00:00:00.

        The timecode returned by this method can be used to perform arithmetic (e.g.
        addition), passing the resulting values back to the VideoManager (e.g. for the
        :meth:`set_duration()` method), as the framerate of the returned FrameTimecode
        object matches that of the VideoManager.

        As such, this method is equivalent to creating a FrameTimecode at frame 0 with
        the VideoManager framerate, for example, given a VideoManager called obj,
        the following expression will evaluate as True:

            obj.get_base_timecode() == FrameTimecode(0, obj.get_framerate())

        Furthermore, the base timecode object returned by a particular VideoManager
        should not be passed to another one, unless you first verify that their
        framerates are the same.

        Returns:
            FrameTimecode at frame 0/time 00:00:00 with the video(s) framerate.
        """
        return FrameTimecode(timecode=0, fps=self._cap_framerate)

    def get_current_timecode(self) -> FrameTimecode:
        """ Get Current Timecode - returns a FrameTimecode object at current VideoManager position.

        Returns:
            Timecode at the current VideoManager position.
        """
        return self._curr_time

    def get_framesize(self) -> Tuple[int, int]:
        """Get frame size of the video(s) open in the VideoManager's capture objects.

        Returns:
            Video frame size, in pixels, in the form (width, height).
        """
        return self._cap_framesize

    def get_framesize_effective(self) -> Tuple[int, int]:
        """ Get Frame Size - returns the frame size of the video(s) open in the
        VideoManager's capture objects.

        Returns:
            Video frame size, in pixels, in the form (width, height).
        """
        return self._cap_framesize

    def set_duration(self,
                     duration: Optional[FrameTimecode] = None,
                     start_time: Optional[FrameTimecode] = None,
                     end_time: Optional[FrameTimecode] = None) -> None:
        """ Set Duration - sets the duration/length of the video(s) to decode, as well as
        the start/end times.  Must be called before :meth:`start()` is called, otherwise
        a VideoDecodingInProgress exception will be thrown.  May be called after
        :meth:`reset()` as well.

        Arguments:
            duration (Optional[FrameTimecode]): The (maximum) duration in time to
                decode from the opened video(s). Mutually exclusive with end_time
                (i.e. if duration is set, end_time must be None).
            start_time (Optional[FrameTimecode]): The time/first frame at which to
                start decoding frames from. If set, the input video(s) will be
                seeked to when start() is called, at which point the frame at
                start_time can be obtained by calling retrieve().
            end_time (Optional[FrameTimecode]): The time at which to stop decoding
                frames from the opened video(s). Mutually exclusive with duration
                (i.e. if end_time is set, duration must be None).

        Raises:
            VideoDecodingInProgress: Must call before start().
        """
        if self._started:
            raise VideoDecodingInProgress()

        # Ensure any passed timecodes have the proper framerate.
        if ((duration is not None and not duration.equal_framerate(self._cap_framerate))
                or (start_time is not None and not start_time.equal_framerate(self._cap_framerate))
                or (end_time is not None and not end_time.equal_framerate(self._cap_framerate))):
            raise ValueError("FrameTimecode framerate does not match.")

        if duration is not None and end_time is not None:
            raise TypeError("Only one of duration and end_time may be specified, not both.")

        if start_time is not None:
            self._start_time = start_time

        if end_time is not None:
            if end_time < self._start_time:
                raise ValueError("end_time is before start_time in time.")
            self._end_time = end_time
        elif duration is not None:
            self._end_time = self._start_time + duration

        if self._end_time is not None:
            self._frame_length = min(self._frame_length, self._end_time + 1)
        self._frame_length -= self._start_time

        if self._logger is not None:
            self._logger.info('Duration set, start: %s, duration: %s, end: %s.',
                              start_time.get_timecode() if start_time is not None else start_time,
                              duration.get_timecode() if duration is not None else duration,
                              end_time.get_timecode() if end_time is not None else end_time)

    def get_duration(self) -> FrameTimecode:
        """ Get Duration - gets the duration/length of the video(s) to decode,
        as well as the start/end times.

        If the end time was not set by :meth:`set_duration()`, the end timecode
        is calculated as the start timecode + total duration.

        Returns:
            Tuple[FrameTimecode, FrameTimecode, FrameTimecode]: The current video(s)
                total duration, start timecode, and end timecode.
        """
        end_time = self._end_time
        if end_time is None:
            end_time = self.get_base_timecode() + self._frame_length
        return (self._frame_length, self._start_time, end_time)

    def start(self) -> None:
        """ Start - starts video decoding and seeks to start time.  Raises
        exception VideoDecodingInProgress if the method is called after the
        decoder process has already been started.

        Raises:
            VideoDecodingInProgress: Must call :meth:`stop()` before this
                method if :meth:`start()` has already been called after
                initial construction.
        """
        if self._started:
            raise VideoDecodingInProgress()

        self._started = True
        self._get_next_cap()
        if self._start_time != 0:
            self.seek(self._start_time)

    # This overrides the seek method from the VideoStream interface, but the name was changed
    # from `timecode` to `target`. For compatibility, we allow calling seek with the form
    # seek(0), seek(timecode=0), and seek(target=0). Specifying both arguments is an error.
    # pylint: disable=arguments-differ
    def seek(self, timecode: FrameTimecode = None, target: FrameTimecode = None) -> bool:
        """Seek forwards to the passed timecode.

        Only supports seeking forwards (i.e. timecode must be greater than the
        current position).  Can only be used after the :meth:`start()`
        method has been called.

        Arguments:
            timecode: Time in video to seek forwards to. Only one of timecode or target can be set.
            target: Same as timecode. Only one of timecode or target can be set.

        Returns:
            bool: True if seeking succeeded, False if no more frames / end of video.

        Raises:
            ValueError: Either none or both `timecode` and `target` were set.
        """
        if timecode is None and target is None:
            raise ValueError('`target` must be set.')
        if timecode is not None and target is not None:
            raise ValueError('Only one of `timecode` or `target` can be set.')
        if target is not None:
            timecode = target
        assert timecode is not None
        if timecode < 0:
            raise ValueError("Target seek position cannot be negative!")

        if not self._started:
            self.start()

        timecode = self.base_timecode + timecode
        if self._end_time is not None and timecode > self._end_time:
            timecode = self._end_time

        # TODO: Seeking only works for the first (or current) video in the VideoManager.
        # Warn the user there are multiple videos in the VideoManager, and the requested
        # seek time exceeds the length of the first video.
        if len(self._cap_list) > 1 and timecode > self._first_cap_len:
            # TODO: This should throw an exception instead of potentially failing silently
            # if no logger was provided.
            if self._logger is not None:
                self._logger.error('Seeking past the first input video is not currently supported.')
                self._logger.warning('Seeking to end of first input.')
            timecode = self._first_cap_len
        if self._curr_cap is not None and self._end_of_video is not True:
            self._curr_cap.set(cv2.CAP_PROP_POS_FRAMES, timecode.get_frames() - 1)
            self._curr_time = timecode - 1

        while self._curr_time < timecode:
            if not self.grab():
                return False
        return True

    # pylint: enable=arguments-differ

    def release(self) -> None:
        """ Release (cv2.VideoCapture method), releases all open capture(s). """
        for cap in self._cap_list:
            cap.release()
        self._cap_list = []
        self._started = False

    def reset(self) -> None:
        """ Reset - Reopens captures passed to the constructor of the VideoManager.

        Can only be called after the :meth:`release()` method has been called.

        Raises:
            VideoDecodingInProgress: Must call :meth:`release()` before this method.
        """
        if self._started:
            self.release()

        self._started = False
        self._end_of_video = False
        self._curr_time = self.get_base_timecode()
        self._cap_list, self._cap_framerate, self._cap_framesize = open_captures(
            video_files=self._video_file_paths, framerate=self._curr_time.get_framerate())
        self._curr_cap, self._curr_cap_idx = None, None

    def get(self, capture_prop: int, index: Optional[int] = None) -> Union[float, int]:
        """ Get (cv2.VideoCapture method) - obtains capture properties from the current
        VideoCapture object in use.  Index represents the same index as the original
        video_files list passed to the constructor.  Getting/setting the position (POS)
        properties has no effect; seeking is implemented using VideoDecoder methods.

        Note that getting the property CAP_PROP_FRAME_COUNT will return the integer sum of
        the frame count for all VideoCapture objects if index is not specified (or is None),
        otherwise the frame count for the given VideoCapture index is returned instead.

        Arguments:
            capture_prop: OpenCV VideoCapture property to get (i.e. CAP_PROP_FPS).
            index (int, optional): Index in file_list of capture to get property from (default
                is zero). Index is not checked and will raise exception if out of bounds.

        Returns:
            float: Return value from calling get(property) on the VideoCapture object.
        """
        if capture_prop == cv2.CAP_PROP_FRAME_COUNT and index is None:
            return self._frame_length.get_frames()
        elif capture_prop == cv2.CAP_PROP_POS_FRAMES:
            return self._curr_time
        elif capture_prop == cv2.CAP_PROP_FPS:
            return self._cap_framerate
        elif index is None:
            index = 0
        return self._cap_list[index].get(capture_prop)

    def grab(self) -> bool:
        """ Grab (cv2.VideoCapture method) - retrieves a frame but does not return it.

        Returns:
            bool: True if a frame was grabbed, False otherwise.
        """
        if not self._started:
            self.start()

        grabbed = False
        if self._curr_cap is not None and not self._end_of_video:
            while not grabbed:
                grabbed = self._curr_cap.grab()
                if not grabbed and not self._get_next_cap():
                    break
        if self._end_time is not None and self._curr_time > self._end_time:
            grabbed = False
            self._last_frame = None
        if grabbed:
            self._curr_time += 1
        else:
            self._correct_frame_length()
        return grabbed

    def retrieve(self) -> Tuple[bool, Optional[ndarray]]:
        """ Retrieve (cv2.VideoCapture method) - retrieves and returns a frame.

        Frame returned corresponds to last call to :meth:`grab()`.

        Returns:
            Tuple of (True, frame_image) if a frame was grabbed during the last call to grab(),
            and where frame_image is a numpy ndarray of the decoded frame. Otherwise (False, None).
        """
        if not self._started:
            self.start()

        retrieved = False
        if self._curr_cap is not None and not self._end_of_video:
            while not retrieved:
                retrieved, self._last_frame = self._curr_cap.retrieve()
                if not retrieved and not self._get_next_cap():
                    break
        if self._end_time is not None and self._curr_time > self._end_time:
            retrieved = False
            self._last_frame = None
        return (retrieved, self._last_frame)

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """ Return next frame (or current if advance = False), or False if end of video.

        Arguments:
            decode: Decode and return the frame.
            advance: Seek to the next frame. If False, will remain on the current frame.

        Returns:
            If decode = True, returns either the decoded frame, or False if end of video.
            If decode = False, a boolean indicating if the next frame was advanced to or not is
            returned.
        """
        if not self._started:
            self.start()
        has_grabbed = False
        if advance:
            has_grabbed = self.grab()
        if decode:
            retrieved, frame = self.retrieve()
            return frame if retrieved else False
        return has_grabbed

    def _get_next_cap(self) -> bool:
        self._curr_cap = None
        if self._curr_cap_idx is None:
            self._curr_cap_idx = 0
            self._curr_cap = self._cap_list[0]
            return True
        else:
            if not (self._curr_cap_idx + 1) < len(self._cap_list):
                self._end_of_video = True
                return False
            self._curr_cap_idx += 1
            self._curr_cap = self._cap_list[self._curr_cap_idx]
            return True

    def _correct_frame_length(self) -> None:
        """ Checks if the current frame position exceeds that originally calculated,
        and adjusts the internally calculated frame length accordingly.  Called after
        exhausting all input frames from the video source(s).
        """
        self._end_time = self._curr_time
        self._frame_length = self._curr_time - self._start_time

    # VideoStream Interface (Some Covered Above)

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return self._aspect_ratio

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        return self.get_duration()[0]

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp of the last frame which was
        decoded by calling `read` with advance=True.

        This method will always return 0 (e.g. be equal to `base_timecode`) if no frames
        have been `read`."""
        frames = self._curr_time.get_frames()
        if frames < 1:
            return self.base_timecode
        return self.base_timecode + (frames - 1)

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
        return self._curr_time.get_frames()

    @property
    def frame_rate(self) -> float:
        """Framerate in frames/sec."""
        return self._cap_framerate

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return (math.trunc(self._cap_list[0].get(cv2.CAP_PROP_FRAME_WIDTH)),
                math.trunc(self._cap_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def is_seekable(self) -> bool:
        """Just returns True."""
        return True

    @property
    def path(self) -> Union[bytes, str]:
        """Video or device path."""
        if self._is_device:
            return "Device %d" % self._path
        return self._path

    @property
    def name(self) -> Union[bytes, str]:
        """Name of the video, without extension, or device."""
        if self._is_device:
            return self.path
        return get_file_name(self.path, include_extension=False)
