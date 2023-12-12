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
# This software may also invoke mkvmerge or FFmpeg, if available.
# FFmpeg is a trademark of Fabrice Bellard.
# mkvmerge is Copyright (C) 2005-2016, Matroska.
# Certain distributions of PySceneDetect may include the above software;
# see the included LICENSE-FFMPEG and LICENSE-MKVMERGE files.
#
"""``scenedetect.video_splitter`` Module

The `scenedetect.video_splitter` module contains functions to split existing videos into clips
using ffmpeg or mkvmerge.

These programs can be obtained from following URLs (note that mkvmerge is a part mkvtoolnix):

 * FFmpeg:   [ https://ffmpeg.org/download.html ]
 * mkvmerge: [ https://mkvtoolnix.download/downloads.html ]

If you are a Linux user, you can likely obtain the above programs from your package manager.

Once installed, ensure the program can be accessed system-wide by calling the `mkvmerge` or `ffmpeg`
command from a terminal/command prompt. PySceneDetect will automatically use whichever program is
available on the computer, depending on the specified command-line options.
"""

import logging
import subprocess
import math
import time
from typing import Iterable, Optional, Tuple

from scenedetect.platform import (tqdm, invoke_command, CommandTooLong, get_file_name,
                                  get_ffmpeg_path, Template)
from scenedetect.frame_timecode import FrameTimecode

logger = logging.getLogger('pyscenedetect')

TimecodePair = Tuple[FrameTimecode, FrameTimecode]
"""Named type for pairs of timecodes, which typically represents the start/end of a scene."""

COMMAND_TOO_LONG_STRING = """
Cannot split video due to too many scenes (resulting command
is too large to process). To work around this issue, you can
split the video manually by exporting a list of cuts with the
`list-scenes` command.
See https://github.com/Breakthrough/PySceneDetect/issues/164
for details.  Sorry about that!
"""

FFMPEG_PATH: Optional[str] = get_ffmpeg_path()
"""Relative path to the Ffmpeg binary on this system, if any (will be None if not available)."""

DEFAULT_FFMPEG_ARGS = '-map 0 -c:v libx264 -preset veryfast -crf 22 -c:a aac'
"""Default arguments passed to ffmpeg when invoking the `split_video_ffmpeg` function."""

##
## Command Availability Checking Functions
##


def is_mkvmerge_available() -> bool:
    """ Is mkvmerge Available: Gracefully checks if mkvmerge command is available.

    Returns:
        True if `mkvmerge` can be invoked, False otherwise.
    """
    ret_val = None
    try:
        ret_val = subprocess.call(['mkvmerge', '--quiet'])
    except OSError:
        return False
    if ret_val is not None and ret_val != 2:
        return False
    return True


def is_ffmpeg_available() -> bool:
    """ Is ffmpeg Available: Gracefully checks if ffmpeg command is available.

    Returns:
        True if `ffmpeg` can be invoked, False otherwise.
    """
    return FFMPEG_PATH is not None


##
## Split Video Functions
##


def split_video_mkvmerge(
    input_video_path: str,
    scene_list: Iterable[TimecodePair],
    output_file_template: str = '$VIDEO_NAME.mkv',
    video_name: Optional[str] = None,
    show_output: bool = False,
    suppress_output=None,
):
    """ Calls the mkvmerge command on the input video, splitting it at the
    passed timecodes, where each scene is written in sequence from 001.

    Arguments:
        input_video_path: Path to the video to be split.
        scene_list : List of scenes as pairs of FrameTimecodes denoting the start/end times.
        output_file_template: Template to use for output files. Mkvmerge always adds the suffix
            "-$SCENE_NUMBER". Can use $VIDEO_NAME as a template parameter (e.g. "$VIDEO_NAME.mkv").
        video_name (str): Name of the video to be substituted in output_file_template for
            $VIDEO_NAME. If not specified, will be obtained from the filename.
        show_output: If False, adds the --quiet flag when invoking `mkvmerge`..
        suppress_output: [DEPRECATED] DO NOT USE. For backwards compatibility only.
    Returns:
        Return code of invoking mkvmerge (0 on success). If scene_list is empty, will
        still return 0, but no commands will be invoked.
    """
    # Handle backwards compatibility with v0.5 API.
    if isinstance(input_video_path, list):
        logger.error('Using a list of paths is deprecated. Pass a single path instead.')
        if len(input_video_path) > 1:
            raise ValueError('Concatenating multiple input videos is not supported.')
        input_video_path = input_video_path[0]
    if suppress_output is not None:
        logger.error('suppress_output is deprecated, use show_output instead.')
        show_output = not suppress_output

    if not scene_list:
        return 0

    logger.info('Splitting input video using mkvmerge, output path template:\n  %s',
                output_file_template)

    if video_name is None:
        video_name = get_file_name(input_video_path, include_extension=False)

    ret_val = 0
    # mkvmerge automatically appends '-$SCENE_NUMBER', so we remove it if present.
    output_file_template = output_file_template.replace('-$SCENE_NUMBER',
                                                        '').replace('$SCENE_NUMBER', '')
    output_file_name = Template(output_file_template).safe_substitute(VIDEO_NAME=video_name)

    try:
        call_list = ['mkvmerge']
        if not show_output:
            call_list.append('--quiet')
        call_list += [
            '-o', output_file_name, '--split',
            'parts:%s' % ','.join([
                '%s-%s' % (start_time.get_timecode(), end_time.get_timecode())
                for start_time, end_time in scene_list
            ]), input_video_path
        ]
        total_frames = scene_list[-1][1].get_frames() - scene_list[0][0].get_frames()
        processing_start_time = time.time()
        # TODO(v0.6.2): Capture stdout/stderr and show that if the command fails.
        ret_val = invoke_command(call_list)
        if show_output:
            logger.info('Average processing speed %.2f frames/sec.',
                        float(total_frames) / (time.time() - processing_start_time))
    except CommandTooLong:
        logger.error(COMMAND_TOO_LONG_STRING)
    except OSError:
        logger.error('mkvmerge could not be found on the system.'
                     ' Please install mkvmerge to enable video output support.')
    if ret_val != 0:
        logger.error('Error splitting video (mkvmerge returned %d).', ret_val)
    return ret_val


def split_video_ffmpeg(
    input_video_path: str,
    scene_list: Iterable[TimecodePair],
    output_file_template: str = '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4',
    video_name: Optional[str] = None,
    arg_override: str = DEFAULT_FFMPEG_ARGS,
    show_progress: bool = False,
    show_output: bool = False,
    suppress_output=None,
    hide_progress=None,
):
    """ Calls the ffmpeg command on the input video, generating a new video for
    each scene based on the start/end timecodes.

    Arguments:
        input_video_path: Path to the video to be split.
        scene_list (List[Tuple[FrameTimecode, FrameTimecode]]): List of scenes
            (pairs of FrameTimecodes) denoting the start/end frames of each scene.
        output_file_template (str): Template to use for generating the output filenames.
            Can use $VIDEO_NAME and $SCENE_NUMBER in this format, for example:
            `$VIDEO_NAME - Scene $SCENE_NUMBER.mp4`
        video_name (str): Name of the video to be substituted in output_file_template. If not
            passed will be calculated from input_video_path automatically.
        arg_override (str): Allows overriding the arguments passed to ffmpeg for encoding.
        show_progress (bool): If True, will show progress bar provided by tqdm (if installed).
        show_output (bool): If True, will show output from ffmpeg for first split.
        suppress_output: [DEPRECATED] DO NOT USE. For backwards compatibility only.
        hide_progress: [DEPRECATED] DO NOT USE. For backwards compatibility only.

    Returns:
        Return code of invoking ffmpeg (0 on success). If scene_list is empty, will
        still return 0, but no commands will be invoked.
    """
    # Handle backwards compatibility with v0.5 API.
    if isinstance(input_video_path, list):
        logger.error('Using a list of paths is deprecated. Pass a single path instead.')
        if len(input_video_path) > 1:
            raise ValueError('Concatenating multiple input videos is not supported.')
        input_video_path = input_video_path[0]
    if suppress_output is not None:
        logger.error('suppress_output is deprecated, use show_output instead.')
        show_output = not suppress_output
    if hide_progress is not None:
        logger.error('hide_progress is deprecated, use show_progress instead.')
        show_progress = not hide_progress

    if not scene_list:
        return 0

    logger.info('Splitting input video using ffmpeg, output path template:\n  %s',
                output_file_template)

    if video_name is None:
        video_name = get_file_name(input_video_path, include_extension=False)

    arg_override = arg_override.replace('\\"', '"')

    ret_val = 0
    arg_override = arg_override.split(' ')
    scene_num_format = '%0'
    scene_num_format += str(max(3, math.floor(math.log(len(scene_list), 10)) + 1)) + 'd'

    try:
        progress_bar = None
        total_frames = scene_list[-1][1].get_frames() - scene_list[0][0].get_frames()
        if show_progress:
            progress_bar = tqdm(total=total_frames, unit='frame', miniters=1, dynamic_ncols=True)
        processing_start_time = time.time()
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = (end_time - start_time)
            # Format output filename with template variable
            output_file_template_iter = Template(output_file_template).safe_substitute(
                VIDEO_NAME=video_name,
                SCENE_NUMBER=scene_num_format % (i + 1),
                START_TIME=str(start_time.get_timecode().replace(":", ";")),
                END_TIME=str(end_time.get_timecode().replace(":", ";")),
                START_FRAME=str(start_time.get_frames()),
                END_FRAME=str(end_time.get_frames()))

            # Gracefully handle case where FFMPEG_PATH might be unset.
            call_list = [FFMPEG_PATH if FFMPEG_PATH is not None else 'ffmpeg']
            if not show_output:
                call_list += ['-v', 'quiet']
            elif i > 0:
                # Only show ffmpeg output for the first call, which will display any
                # errors if it fails, and then break the loop. We only show error messages
                # for the remaining calls.
                call_list += ['-v', 'error']
            call_list += [
                '-nostdin', '-y', '-ss',
                str(start_time.get_seconds()), '-i', input_video_path, '-t',
                str(duration.get_seconds())
            ]
            call_list += arg_override
            call_list += ['-sn']
            call_list += [output_file_template_iter]
            ret_val = invoke_command(call_list)
            if show_output and i == 0 and len(scene_list) > 1:
                logger.info(
                    'Output from ffmpeg for Scene 1 shown above, splitting remaining scenes...')
            if ret_val != 0:
                # TODO(v0.6.2): Capture stdout/stderr and display it on any failed calls.
                logger.error('Error splitting video (ffmpeg returned %d).', ret_val)
                break
            if progress_bar:
                progress_bar.update(duration.get_frames())

        if progress_bar:
            progress_bar.close()
        if show_output:
            logger.info('Average processing speed %.2f frames/sec.',
                        float(total_frames) / (time.time() - processing_start_time))

    except CommandTooLong:
        logger.error(COMMAND_TOO_LONG_STRING)
    except OSError:
        logger.error('ffmpeg could not be found on the system.'
                     ' Please install ffmpeg to enable video output support.')
    return ret_val
