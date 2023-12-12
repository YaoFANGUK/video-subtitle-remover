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
"""``scenedetect.platform`` Module

This moduke contains all platform/library specific compatibility fixes, as well as some utility
functions to handle logging and invoking external commands.
"""

import importlib
import logging
import os
import os.path
import platform
import re
import string
import subprocess
import sys
from typing import AnyStr, Dict, List, Optional, Union

import cv2

##
## tqdm Library
##


class FakeTqdmObject:
    """Provides a no-op tqdm-like object."""

    # pylint: disable=unused-argument
    def __init__(self, **kawrgs):
        """No-op."""

    def update(self, n=1):
        """No-op."""

    def close(self):
        """No-op."""

    def set_description(self, desc=None, refresh=True):
        """No-op."""

    # pylint: enable=unused-argument


class FakeTqdmLoggingRedirect:
    """Provides a no-op tqdm context manager for redirecting log messages."""

    # pylint: disable=redefined-builtin,unused-argument
    def __init__(self, **kawrgs):
        """No-op."""

    def __enter__(self):
        """No-op."""

    def __exit__(self, type, value, traceback):
        """No-op."""

    # pylint: enable=redefined-builtin,unused-argument


# Try to import tqdm and the logging redirect, otherwise provide fake implementations..
try:
    # pylint: disable=unused-import
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    # pylint: enable=unused-import
except ModuleNotFoundError:
    # pylint: disable=invalid-name
    tqdm = FakeTqdmObject
    logging_redirect_tqdm = FakeTqdmLoggingRedirect
    # pylint: enable=invalid-name

##
## OpenCV imwrite Supported Image Types & Quality/Compression Parameters
##


# TODO: Move this into scene_manager.
def get_cv2_imwrite_params() -> Dict[str, Union[int, None]]:
    """ Get OpenCV imwrite Params: Returns a dict of supported image formats and
    their associated quality/compression parameter index, or None if that format
    is not supported.

    Returns:
        Dictionary of supported image formats/extensions ('jpg', 'png', etc...) mapped to the
        respective OpenCV quality or compression parameter as {'jpg': cv2.IMWRITE_JPEG_QUALITY,
        'png': cv2.IMWRITE_PNG_COMPRESSION, ...}. Parameter will be None if not found on the
        current system library (e.g. {'jpg': None}).
    """

    def _get_cv2_param(param_name: str) -> Union[int, None]:
        if param_name.startswith('CV_'):
            param_name = param_name[3:]
        try:
            return getattr(cv2, param_name)
        except AttributeError:
            return None

    return {
        'jpg': _get_cv2_param('IMWRITE_JPEG_QUALITY'),
        'png': _get_cv2_param('IMWRITE_PNG_COMPRESSION'),
        'webp': _get_cv2_param('IMWRITE_WEBP_QUALITY')
    }


##
## File I/O
##


def get_file_name(file_path: AnyStr, include_extension=True) -> AnyStr:
    """Return the file name that `file_path` refers to, optionally removing the extension.

    If `include_extension` is False, the result will always be a str.

    E.g. /tmp/foo.bar -> foo"""
    file_name = os.path.basename(file_path)
    if not include_extension:
        file_name = str(file_name)
        last_dot_pos = file_name.rfind('.')
        if last_dot_pos >= 0:
            file_name = file_name[:last_dot_pos]
    return file_name


def get_and_create_path(file_path: AnyStr, output_directory: Optional[AnyStr] = None) -> AnyStr:
    """ Get & Create Path: Gets and returns the full/absolute path to file_path
    in the specified output_directory if set, creating any required directories
    along the way.

    If file_path is already an absolute path, then output_directory is ignored.

    Arguments:
        file_path: File name to get path for.  If file_path is an absolute
            path (e.g. starts at a drive/root), no modification of the path
            is performed, only ensuring that all output directories are created.
        output_dir: An optional output directory to override the
            directory of file_path if it is relative to the working directory.

    Returns:
        Full path to output file suitable for writing.

    """
    # If an output directory is defined and the file path is a relative path, open
    # the file handle in the output directory instead of the working directory.
    if output_directory is not None and not os.path.isabs(file_path):
        file_path = os.path.join(output_directory, file_path)
    # Now that file_path is an absolute path, let's make sure all the directories
    # exist for us to start writing files there.
    os.makedirs(os.path.split(os.path.abspath(file_path))[0], exist_ok=True)
    return file_path


##
## Logging
##


def init_logger(log_level: int = logging.INFO,
                show_stdout: bool = False,
                log_file: Optional[str] = None):
    """Initializes logging for PySceneDetect. The logger instance used is named 'pyscenedetect'.
    By default the logger has no handlers to suppress output. All existing log handlers are replaced
    every time this function is invoked.

    Arguments:
        log_level: Verbosity of log messages. Should be one of [logging.INFO, logging.DEBUG,
            logging.WARNING, logging.ERROR, logging.CRITICAL].
        show_stdout: If True, add handler to show log messages on stdout (default: False).
        log_file: If set, add handler to dump debug log messages to given file path.
    """
    # Format of log messages depends on verbosity.
    INFO_TEMPLATE = '[PySceneDetect] %(message)s'
    DEBUG_TEMPLATE = '%(levelname)s: %(module)s.%(funcName)s(): %(message)s'
    # Get the named logger and remove any existing handlers.
    logger_instance = logging.getLogger('pyscenedetect')
    logger_instance.handlers = []
    logger_instance.setLevel(log_level)
    # Add stdout handler if required.
    if show_stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter(fmt=DEBUG_TEMPLATE if log_level == logging.DEBUG else INFO_TEMPLATE))
        logger_instance.addHandler(handler)
    # Add debug log handler if required.
    if log_file:
        log_file = get_and_create_path(log_file)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(fmt=DEBUG_TEMPLATE))
        logger_instance.addHandler(handler)


##
## Running External Commands
##


class CommandTooLong(Exception):
    """Raised if the length of a command line argument exceeds the limit allowed on Windows."""


def invoke_command(args: List[str]) -> int:
    """Same as calling Python's subprocess.call() method, but explicitly
    raises a different exception when the command length is too long.

    See https://github.com/Breakthrough/PySceneDetect/issues/164 for details.

    Arguments:
        args: List of strings to pass to subprocess.call().

    Returns:
        Return code of command.

    Raises:
        CommandTooLong: `args` exceeds built in command line length limit on Windows.
    """
    try:
        return subprocess.call(args)
    except OSError as err:
        if os.name != 'nt':
            raise
        exception_string = str(err)
        # Error 206: The filename or extension is too long
        # Error 87:  The parameter is incorrect
        to_match = ('206', '87')
        if any([x in exception_string for x in to_match]):
            raise CommandTooLong() from err
        raise


def get_ffmpeg_path() -> Optional[str]:
    """Get path to ffmpeg if available on the current system. First looks at PATH, then checks if
    one is available from the `imageio_ffmpeg` package. Returns None if ffmpeg couldn't be found.
    """
    try:
        subprocess.call(['ffmpeg', '-v', 'quiet'])
        return 'ffmpeg'
    except OSError:
        pass
    # Failed to invoke ffmpeg from PATH, see if we have a copy from imageio_ffmpeg.
    try:
        # pylint: disable=import-outside-toplevel
        from imageio_ffmpeg import get_ffmpeg_exe
        # pylint: enable=import-outside-toplevel
        subprocess.call([get_ffmpeg_exe(), '-v', 'quiet'])
        return get_ffmpeg_exe()
    # Gracefully handle case where imageio_ffmpeg is not available.
    except ModuleNotFoundError:
        pass
    # Handle case where path might be wrong/non-existent.
    except OSError:
        pass
    # get_ffmpeg_exe may throw a RuntimeError if the executable is not available.
    except RuntimeError:
        pass
    return None


def get_ffmpeg_version() -> Optional[str]:
    """Get ffmpeg version identifier, or None if ffmpeg is not found. Uses `get_ffmpeg_path()`."""
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path is None:
        return None
    # If get_ffmpeg_path() returns a value, the path it returns should be invocable.
    output = subprocess.check_output(args=[ffmpeg_path, '-version'], text=True)
    output_split = output.split()
    if len(output_split) >= 3 and output_split[1] == 'version':
        return output_split[2]
    # If parsing the version fails, return the entire first line of output.
    return output.splitlines()[0]


def get_mkvmerge_version() -> Optional[str]:
    """Get mkvmerge version identifier, or None if mkvmerge is not found in PATH."""
    tool_name = 'mkvmerge'
    try:
        output = subprocess.check_output(args=[tool_name, '--version'], text=True)
    except FileNotFoundError:
        return None
    output_split = output.split()
    if len(output_split) >= 1 and output_split[0] == tool_name:
        return ' '.join(output_split[1:])
    # If parsing the version fails, return the entire first line of output.
    return output.splitlines()[0]


def get_system_version_info() -> str:
    """Get the system's operating system, Python, packages, and external tool versions.
    Useful for debugging or filing bug reports.

    Used for the `scenedetect version -a` command.
    """
    output_template = '{:<12} {}'
    line_separator = '-' * 60
    not_found_str = 'Not Installed'
    out_lines = []

    # System (Python, OS)
    out_lines += ['System Info', line_separator]
    out_lines += [
        output_template.format(name, version) for name, version in (
            ('OS', '%s' % platform.platform()),
            ('Python', '%d.%d.%d' % sys.version_info[0:3]),
        )
    ]

    # Third-Party Packages
    out_lines += ['', 'Packages', line_separator]
    third_party_packages = (
        'av',
        'click',
        'cv2',
        'moviepy',
        'numpy',
        'platformdirs',
        'scenedetect',
        'tqdm',
    )
    for module_name in third_party_packages:
        try:
            module = importlib.import_module(module_name)
            out_lines.append(output_template.format(module_name, module.__version__))
        except ModuleNotFoundError:
            out_lines.append(output_template.format(module_name, not_found_str))

    # External Tools
    out_lines += ['', 'Tools', line_separator]

    tool_version_info = (
        ('ffmpeg', get_ffmpeg_version()),
        ('mkvmerge', get_mkvmerge_version()),
    )

    for (tool_name, tool_version) in tool_version_info:
        out_lines.append(
            output_template.format(tool_name, tool_version if tool_version else not_found_str))

    return '\n'.join(out_lines)


class Template(string.Template):
    """Template matcher used to replace instances of $TEMPLATES in filenames."""
    idpattern = '[A-Z0-9_]+'
    flags = re.ASCII
