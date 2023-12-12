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
"""``scenedetect.frame_timecode`` Module

This module implements :class:`FrameTimecode` which is used as a way for PySceneDetect to store
frame-accurate timestamps of each cut. This is done by also specifying the video framerate with the
timecode, allowing a frame number to be converted to/from a floating-point number of seconds, or
string in the form `"HH:MM:SS[.nnn]"` where the `[.nnn]` part is optional.

See the following examples, or the :class:`FrameTimecode constructor <FrameTimecode>`.

===============================================================
Usage Examples
===============================================================

A :class:`FrameTimecode` can be created by specifying a timecode (`int` for number of frames,
`float` for number of seconds, or `str` in the form "HH:MM:SS" or "HH:MM:SS.nnn") with a framerate:

.. code:: python

    frames = FrameTimecode(timecode = 29, fps = 29.97)
    seconds_float = FrameTimecode(timecode = 10.0, fps = 10.0)
    timecode_str = FrameTimecode(timecode = "00:00:10.000", fps = 10.0)


Arithmetic/comparison operations with :class:`FrameTimecode` objects is also possible, and the
other operand can also be of the above types:

.. code:: python

    x = FrameTimecode(timecode = "00:01:00.000", fps = 10.0)
    # Can add int (frames), float (seconds), or str (timecode).
    print(x + 10)
    print(x + 10.0)
    print(x + "00:10:00")
    # Same for all comparison operators.
    print((x + 10.0) == "00:01:10.000")


:class:`FrameTimecode` objects can be added and subtracted, however the current implementation
disallows negative values, and will clamp negative results to 0.

.. warning::

    Be careful when subtracting :class:`FrameTimecode` objects or adding negative
    amounts of frames/seconds. In the example below, ``c`` will be at frame 0 since
    ``b > a``, but ``d`` will be at frame 5:

    .. code:: python

        a = FrameTimecode(5, 10.0)
        b = FrameTimecode(10, 10.0)
        c = a - b   # b > a, so c == 0
        d = b - a
        assert(c == 0)
        assert(d == 5)

"""

import math
from typing import Union

MAX_FPS_DELTA: float = 1.0 / 100000
"""Maximum amount two framerates can differ by for equality testing."""

# TODO(0.6.3): Replace uses of Union[int, float, str] with TimecodeValue.
TimecodeValue = Union[int, float, str]
"""Named type for values representing timecodes. Must be in one of the following forms:

  1. Timecode as `str` in the form 'HH:MM:SS[.nnn]' (`'01:23:45'` or `'01:23:45.678'`)
  2. Number of seconds as `float`, or `str` in form 'Ss' or 'S.SSSs' (`'2s'` or `'2.3456s'`)
  3. Exact number of frames as `int`, or `str` in form NNNNN (`123` or `'123'`)
"""


class FrameTimecode:
    """Object for frame-based timecodes, using the video framerate to compute back and
    forth between frame number and seconds/timecode.

    A timecode is valid only if it complies with one of the following three types/formats:

      1. Timecode as `str` in the form 'HH:MM:SS[.nnn]' (`'01:23:45'` or `'01:23:45.678'`)
      2. Number of seconds as `float`, or `str` in form 'Ss' or 'S.SSSs' (`'2s'` or `'2.3456s'`)
      3. Exact number of frames as `int`, or `str` in form NNNNN (`123` or `'123'`)
    """

    def __init__(self,
                 timecode: Union[int, float, str, 'FrameTimecode'] = None,
                 fps: Union[int, float, str, 'FrameTimecode'] = None):
        """
        Arguments:
            timecode: A frame number (int), number of seconds (float), or timecode (str in
                the form `'HH:MM:SS'` or `'HH:MM:SS.nnn'`).
            fps: The framerate or FrameTimecode to use as a time base for all arithmetic.
        Raises:
            TypeError: Thrown if either `timecode` or `fps` are unsupported types.
            ValueError: Thrown when specifying a negative timecode or framerate.
        """
        # The following two properties are what is used to keep track of time
        # in a frame-specific manner.  Note that once the framerate is set,
        # the value should never be modified (only read if required).
        # TODO(v1.0): Make these actual @properties.
        self.framerate = None
        self.frame_num = None

        # Copy constructor.  Only the timecode argument is used in this case.
        if isinstance(timecode, FrameTimecode):
            self.framerate = timecode.framerate
            self.frame_num = timecode.frame_num
            if fps is not None:
                raise TypeError('Framerate cannot be overwritten when copying a FrameTimecode.')
        else:
            # Ensure other arguments are consistent with API.
            if fps is None:
                raise TypeError('Framerate (fps) is a required argument.')
            if isinstance(fps, FrameTimecode):
                fps = fps.framerate

            # Process the given framerate, if it was not already set.
            if not isinstance(fps, (int, float)):
                raise TypeError('Framerate must be of type int/float.')
            if (isinstance(fps, int) and not fps > 0) or (isinstance(fps, float)
                                                          and not fps >= MAX_FPS_DELTA):
                raise ValueError('Framerate must be positive and greater than zero.')
            self.framerate = float(fps)

        # Process the timecode value, storing it as an exact number of frames.
        if isinstance(timecode, str):
            self.frame_num = self._parse_timecode_string(timecode)
        else:
            self.frame_num = self._parse_timecode_number(timecode)

    # TODO(v1.0): Add a `frame` property to replace the existing one and deprecate this getter.
    def get_frames(self) -> int:
        """Get the current time/position in number of frames.  This is the
        equivalent of accessing the self.frame_num property (which, along
        with the specified framerate, forms the base for all of the other
        time measurement calculations, e.g. the :meth:`get_seconds` method).

        If using to compare a :class:`FrameTimecode` with a frame number,
        you can do so directly against the object (e.g. ``FrameTimecode(10, 10.0) <= 10``).

        Returns:
            int: The current time in frames (the current frame number).
        """
        return self.frame_num

    # TODO(v1.0): Add a `framerate` property to replace the existing one and deprecate this getter.
    def get_framerate(self) -> float:
        """Get Framerate: Returns the framerate used by the FrameTimecode object.

        Returns:
            float: Framerate of the current FrameTimecode object, in frames per second.
        """
        return self.framerate

    def equal_framerate(self, fps) -> bool:
        """Equal Framerate: Determines if the passed framerate is equal to that of this object.

        Arguments:
            fps: Framerate to compare against within the precision constant defined in this module
                (see :data:`MAX_FPS_DELTA`).
        Returns:
            bool: True if passed fps matches the FrameTimecode object's framerate, False otherwise.

        """
        return math.fabs(self.framerate - fps) < MAX_FPS_DELTA

    # TODO(v1.0): Add a `seconds` property to replace this and deprecate the existing one.
    def get_seconds(self) -> float:
        """Get the frame's position in number of seconds.

        If using to compare a :class:`FrameTimecode` with a frame number,
        you can do so directly against the object (e.g. ``FrameTimecode(10, 10.0) <= 1.0``).

        Returns:
            float: The current time/position in seconds.
        """
        return float(self.frame_num) / self.framerate

    # TODO(v1.0): Add a `timecode` property to replace this and deprecate the existing one.
    def get_timecode(self, precision: int = 3, use_rounding: bool = True) -> str:
        """Get a formatted timecode string of the form HH:MM:SS[.nnn].

        Args:
            precision: The number of decimal places to include in the output ``[.nnn]``.
            use_rounding: Rounds the output to the desired precision. If False, the value
                will be truncated to the specified precision.

        Returns:
            str: The current time in the form ``"HH:MM:SS[.nnn]"``.
        """
        # Compute hours and minutes based off of seconds, and update seconds.
        secs = self.get_seconds()
        base = 60.0 * 60.0
        hrs = int(secs / base)
        secs -= (hrs * base)
        base = 60.0
        mins = int(secs / base)
        secs -= (mins * base)
        # Convert seconds into string based on required precision.
        if precision > 0:
            if use_rounding:
                secs = round(secs, precision)
            msec = format(secs, '.%df' % precision)[-precision:]
            secs = '%02d.%s' % (int(secs), msec)
        else:
            secs = '%02d' % int(round(secs, 0)) if use_rounding else '%02d' % int(secs)
        # Return hours, minutes, and seconds as a formatted timecode string.
        return '%02d:%02d:%s' % (hrs, mins, secs)

    # TODO(v1.0): Add a `previous` property to replace the existing one and deprecate this getter.
    def previous_frame(self) -> 'FrameTimecode':
        """Return a new FrameTimecode for the previous frame (or 0 if on frame 0)."""
        new_timecode = FrameTimecode(self)
        new_timecode.frame_num = max(0, new_timecode.frame_num - 1)
        return new_timecode

    def _seconds_to_frames(self, seconds: float) -> int:
        """Convert the passed value seconds to the nearest number of frames using
        the current FrameTimecode object's FPS (self.framerate).

        Returns:
            Integer number of frames the passed number of seconds represents using
            the current FrameTimecode's framerate property.
        """
        return round(seconds * self.framerate)

    def _parse_timecode_number(self, timecode: Union[int, float]) -> int:
        """ Parse a timecode number, storing it as the exact number of frames.
        Can be passed as frame number (int), seconds (float)

        Raises:
            TypeError, ValueError
        """
        # Process the timecode value, storing it as an exact number of frames.
        # Exact number of frames N
        if isinstance(timecode, int):
            if timecode < 0:
                raise ValueError('Timecode frame number must be positive and greater than zero.')
            return timecode
        # Number of seconds S
        elif isinstance(timecode, float):
            if timecode < 0.0:
                raise ValueError('Timecode value must be positive and greater than zero.')
            return self._seconds_to_frames(timecode)
        # FrameTimecode
        elif isinstance(timecode, FrameTimecode):
            return timecode.frame_num
        elif timecode is None:
            raise TypeError('Timecode/frame number must be specified!')
        else:
            raise TypeError('Timecode format/type unrecognized.')

    def _parse_timecode_string(self, timecode_string: str) -> int:
        """Parses a string based on the three possible forms (in timecode format,
        as an integer number of frames, or floating-point seconds, ending with 's').

        Requires that the `framerate` property is set before calling this method.
        Assuming a framerate of 30.0 FPS, the strings '00:05:00.000', '00:05:00',
        '9000', '300s', and '300.0s' are all possible valid values, all representing
        a period of time equal to 5 minutes, 300 seconds, or 9000 frames (at 30 FPS).

        Raises:
            TypeError, ValueError
        """
        if self.framerate is None:
            raise TypeError('self.framerate must be set before calling _parse_timecode_string.')
        # Number of seconds S
        if timecode_string.endswith('s'):
            secs = timecode_string[:-1]
            if not secs.replace('.', '').isdigit():
                raise ValueError('All characters in timecode seconds string must be digits.')
            secs = float(secs)
            if secs < 0.0:
                raise ValueError('Timecode seconds value must be positive.')
            return self._seconds_to_frames(secs)
        # Exact number of frames N
        elif timecode_string.isdigit():
            timecode = int(timecode_string)
            if timecode < 0:
                raise ValueError('Timecode frame number must be positive.')
            return timecode
        # Standard timecode in string format 'HH:MM:SS[.nnn]'
        else:
            tc_val = timecode_string.split(':')
            if not (len(tc_val) == 3 and tc_val[0].isdigit() and tc_val[1].isdigit()
                    and tc_val[2].replace('.', '').isdigit()):
                raise ValueError('Unrecognized or improperly formatted timecode string.')
            hrs, mins = int(tc_val[0]), int(tc_val[1])
            secs = float(tc_val[2]) if '.' in tc_val[2] else int(tc_val[2])
            if not (hrs >= 0 and mins >= 0 and secs >= 0 and mins < 60 and secs < 60):
                raise ValueError('Invalid timecode range (values outside allowed range).')
            secs += (((hrs * 60.0) + mins) * 60.0)
            return self._seconds_to_frames(secs)

    def __iadd__(self, other: Union[int, float, str, 'FrameTimecode']) -> 'FrameTimecode':
        if isinstance(other, int):
            self.frame_num += other
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                self.frame_num += other.frame_num
            else:
                raise ValueError('FrameTimecode instances require equal framerate for addition.')
        # Check if value to add is in number of seconds.
        elif isinstance(other, float):
            self.frame_num += self._seconds_to_frames(other)
        elif isinstance(other, str):
            self.frame_num += self._parse_timecode_string(other)
        else:
            raise TypeError('Unsupported type for performing addition with FrameTimecode.')
        if self.frame_num < 0: # Required to allow adding negative seconds/frames.
            self.frame_num = 0
        return self

    def __add__(self, other: Union[int, float, str, 'FrameTimecode']) -> 'FrameTimecode':
        to_return = FrameTimecode(timecode=self)
        to_return += other
        return to_return

    def __isub__(self, other: Union[int, float, str, 'FrameTimecode']) -> 'FrameTimecode':
        if isinstance(other, int):
            self.frame_num -= other
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                self.frame_num -= other.frame_num
            else:
                raise ValueError('FrameTimecode instances require equal framerate for subtraction.')
        # Check if value to add is in number of seconds.
        elif isinstance(other, float):
            self.frame_num -= self._seconds_to_frames(other)
        elif isinstance(other, str):
            self.frame_num -= self._parse_timecode_string(other)
        else:
            raise TypeError('Unsupported type for performing subtraction with FrameTimecode: %s' %
                            type(other))
        if self.frame_num < 0:
            self.frame_num = 0
        return self

    def __sub__(self, other: Union[int, float, str, 'FrameTimecode']) -> 'FrameTimecode':
        to_return = FrameTimecode(timecode=self)
        to_return -= other
        return to_return

    def __eq__(self, other: Union[int, float, str, 'FrameTimecode']) -> 'FrameTimecode':
        if isinstance(other, int):
            return self.frame_num == other
        elif isinstance(other, float):
            return self.get_seconds() == other
        elif isinstance(other, str):
            return self.frame_num == self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num == other.frame_num
            else:
                raise TypeError(
                    'FrameTimecode objects must have the same framerate to be compared.')
        elif other is None:
            return False
        else:
            raise TypeError('Unsupported type for performing == with FrameTimecode: %s' %
                            type(other))

    def __ne__(self, other: Union[int, float, str, 'FrameTimecode']) -> bool:
        return not self == other

    def __lt__(self, other: Union[int, float, str, 'FrameTimecode']) -> bool:
        if isinstance(other, int):
            return self.frame_num < other
        elif isinstance(other, float):
            return self.get_seconds() < other
        elif isinstance(other, str):
            return self.frame_num < self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num < other.frame_num
            else:
                raise TypeError(
                    'FrameTimecode objects must have the same framerate to be compared.')
        else:
            raise TypeError('Unsupported type for performing < with FrameTimecode: %s' %
                            type(other))

    def __le__(self, other: Union[int, float, str, 'FrameTimecode']) -> bool:
        if isinstance(other, int):
            return self.frame_num <= other
        elif isinstance(other, float):
            return self.get_seconds() <= other
        elif isinstance(other, str):
            return self.frame_num <= self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num <= other.frame_num
            else:
                raise TypeError(
                    'FrameTimecode objects must have the same framerate to be compared.')
        else:
            raise TypeError('Unsupported type for performing <= with FrameTimecode: %s' %
                            type(other))

    def __gt__(self, other: Union[int, float, str, 'FrameTimecode']) -> bool:
        if isinstance(other, int):
            return self.frame_num > other
        elif isinstance(other, float):
            return self.get_seconds() > other
        elif isinstance(other, str):
            return self.frame_num > self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num > other.frame_num
            else:
                raise TypeError(
                    'FrameTimecode objects must have the same framerate to be compared.')
        else:
            raise TypeError('Unsupported type for performing > with FrameTimecode: %s' %
                            type(other))

    def __ge__(self, other: Union[int, float, str, 'FrameTimecode']) -> bool:
        if isinstance(other, int):
            return self.frame_num >= other
        elif isinstance(other, float):
            return self.get_seconds() >= other
        elif isinstance(other, str):
            return self.frame_num >= self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num >= other.frame_num
            else:
                raise TypeError(
                    'FrameTimecode objects must have the same framerate to be compared.')
        else:
            raise TypeError('Unsupported type for performing >= with FrameTimecode: %s' %
                            type(other))

    # TODO(v1.0): __int__ and __float__ should be removed. Mark as deprecated, and indicate
    # need to use relevant property instead.

    def __int__(self) -> int:
        return self.frame_num

    def __float__(self) -> float:
        return self.get_seconds()

    def __str__(self) -> str:
        return self.get_timecode()

    def __repr__(self) -> str:
        return '%s [frame=%d, fps=%.3f]' % (self.get_timecode(), self.frame_num, self.framerate)

    def __hash__(self) -> int:
        return self.frame_num
