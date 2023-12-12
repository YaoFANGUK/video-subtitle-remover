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
"""Implementation of the PySceneDetect application itself (the `scenedetect` command). The main CLI
entry-point function is :func:scenedetect_cli, which is a chained command group.

Commands are first parsed into a context (`CliContext`), which is then passed to a controller which
performs scene detection and other required actions (`run_scenedetect`).
"""

# Some parts of this file need word wrap to be displayed.
# pylint: disable=line-too-long

import inspect
import logging
from typing import AnyStr, Optional, Tuple

import click

import scenedetect
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.backends import AVAILABLE_BACKENDS
from scenedetect.platform import get_system_version_info

from scenedetect._cli.config import CHOICE_MAP, CONFIG_FILE_PATH, CONFIG_MAP
from scenedetect._cli.context import CliContext, USER_CONFIG

_PROGRAM_VERSION = scenedetect.__version__
"""Used to avoid name conflict with named `scenedetect` command below."""

logger = logging.getLogger('pyscenedetect')

_LINE_SEPARATOR = '-' * 72

# About & copyright message string shown for the 'about' CLI command (scenedetect about).
_ABOUT_STRING = """
Site: http://scenedetect.com/
Docs: http://manual.scenedetect.com/
Code: https://github.com/Breakthrough/PySceneDetect/

Copyright (C) 2014-2023 Brandon Castellano. All rights reserved.

PySceneDetect is released under the BSD 3-Clause license. See the
included LICENSE file or visit the PySceneDetect website for details.
This software uses the following third-party components:

  > NumPy [Copyright (C) 2018, Numpy Developers]
  > OpenCV [Copyright (C) 2018, OpenCV Team]
  > click [Copyright (C) 2018, Armin Ronacher]
  > simpletable [Copyright (C) 2014 Matheus Vieira Portela]

This software may also invoke the following third-party executables:

  > FFmpeg [Copyright (C) 2018, Fabrice Bellard]
  > mkvmerge [Copyright (C) 2005-2016, Matroska]

If included with your distribution of PySceneDetect, see the included
LICENSE-FFMPEG and LICENSE-MKVMERGE or visit:
  [ https://scenedetect.com/copyright/ ]

FFmpeg and mkvmerge are distributed only with certain PySceneDetect
releases, in order to allow for automatic video splitting capability.
If they were not included with your distribution, they can usually be
installed from your operating system's package manager, or downloaded
from the following URLs:

    FFmpeg:   [ https://ffmpeg.org/download.html ]
    mkvmerge: [ https://mkvtoolnix.download/downloads.html ]
        (Note that mkvmerge is a part of the mkvtoolnix package.)

Once installed, ensure the respective program can be accessed from the
same location running PySceneDetect by calling the `ffmpeg` or
`mkvmerge` command from a terminal/command prompt.

PySceneDetect will automatically use whichever program is available on
the computer, depending on the specified command-line options.

Additionally, certain Windows distributions may include a compiled
Python distribution. For license information regarding the distributed
version of Python, see the included LICENSE-PYTHON file for details,
or visit the following URL: [ https://docs.python.org/3/license.html ]

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED.
"""


class _Command(click.Command):
    """Custom formatting for commands."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Writes the help into the formatter if it exists."""
        if ctx.parent:
            formatter.write(click.style('`%s` Command' % ctx.command.name, fg='cyan'))
            formatter.write_paragraph()
            formatter.write(click.style(_LINE_SEPARATOR, fg='cyan'))
            formatter.write_paragraph()
        else:
            formatter.write(click.style(_LINE_SEPARATOR, fg='yellow'))
            formatter.write_paragraph()
            formatter.write(click.style('PySceneDetect Help', fg='yellow'))
            formatter.write_paragraph()
            formatter.write(click.style(_LINE_SEPARATOR, fg='yellow'))
            formatter.write_paragraph()

        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Writes the help text to the formatter if it exists."""
        if self.help:
            base_command = (ctx.parent.info_name if ctx.parent is not None else ctx.info_name)
            formatted_help = self.help.format(
                scenedetect=base_command, scenedetect_with_video='%s -i video.mp4' % base_command)
            text = inspect.cleandoc(formatted_help).partition("\f")[0]
            formatter.write_paragraph()
            formatter.write_text(text)

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Writes the epilog into the formatter if it exists."""
        if self.epilog:
            epilog = inspect.cleandoc(self.epilog)
            formatter.write_paragraph()
            formatter.write_text(epilog)


class _CommandGroup(_Command, click.Group):
    """Custom formatting for command groups."""
    pass


def _print_command_help(ctx: click.Context, command: click.Command):
    """Print help/usage for a given command. Modifies `ctx` in-place."""
    ctx.info_name = command.name
    ctx.command = command
    click.echo('')
    click.echo(command.get_help(ctx))


@click.group(
    cls=_CommandGroup,
    chain=True,
    context_settings=dict(help_option_names=['-h', '--help']),
    invoke_without_command=True,
    epilog="""Type "scenedetect [command] --help" for command usage. See https://scenedetect.com/docs/ for online docs."""
)
# We cannot make this a required argument otherwise we will reject commands of the form
# `scenedetect help scene_detect-content` or `scenedetect scene_detect-content --help`.
@click.option(
    '--input',
    '-i',
    multiple=False,
    required=False,
    metavar='VIDEO',
    type=click.STRING,
    help='[REQUIRED] Input video file. Image sequences and URLs are supported.',
)
@click.option(
    '--output',
    '-o',
    multiple=False,
    required=False,
    metavar='DIR',
    type=click.Path(exists=False, dir_okay=True, writable=True, resolve_path=True),
    help='Output directory for created files. If unset, working directory will be used. May be overridden by command options.%s'
    % (USER_CONFIG.get_help_string("global", "output", show_default=False)),
)
@click.option(
    '--config',
    '-c',
    metavar='FILE',
    type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=False),
    help='Path to config file. If unset, tries to load config from %s' % (CONFIG_FILE_PATH),
)
@click.option(
    '--stats',
    '-s',
    metavar='CSV',
    type=click.Path(exists=False, file_okay=True, writable=True, resolve_path=False),
    help='Stats file (.csv) to write frame metrics. Existing files will be overwritten. Used for tuning detection parameters and data analysis.',
)
@click.option(
    '--framerate',
    '-f',
    metavar='FPS',
    type=click.FLOAT,
    default=None,
    help='Override framerate with value as frames/sec.',
)
@click.option(
    '--min-scene-len',
    '-m',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Minimum length of any scene. TIMECODE can be specified as number of frames (-m=10), time in seconds followed by "s" (-m=2.5s), or timecode (-m=00:02:53.633).%s'
    % USER_CONFIG.get_help_string("global", "min-scene-len"),
)
@click.option(
    '--drop-short-scenes',
    is_flag=True,
    flag_value=True,
    help='Drop scenes shorter than -m/--min-scene-len, instead of combining with neighbors.%s' %
    (USER_CONFIG.get_help_string('global', 'drop-short-scenes')),
)
@click.option(
    '--merge-last-scene',
    is_flag=True,
    flag_value=True,
    help='Merge last scene with previous if shorter than -m/--min-scene-len.%s' %
    (USER_CONFIG.get_help_string('global', 'merge-last-scene')),
)
@click.option(
    '--backend',
    '-b',
    metavar='BACKEND',
    type=click.Choice(CHOICE_MAP["global"]["backend"]),
    default=None,
    help='Backend to use for video input. Backend options can be set using a config file (-c/--config). [available: %s]%s'
    % (', '.join(AVAILABLE_BACKENDS.keys()), USER_CONFIG.get_help_string("global", "backend")),
)
@click.option(
    '--downscale',
    '-d',
    metavar='N',
    type=click.INT,
    default=None,
    help='Integer factor to downscale video by before processing. If unset, value is selected based on resolution. Set -d=1 to disable downscaling.%s'
    % (USER_CONFIG.get_help_string("global", "downscale", show_default=False)),
)
@click.option(
    '--frame-skip',
    '-fs',
    metavar='N',
    type=click.INT,
    default=None,
    help='Skip N frames during processing. Reduces processing speed at expense of accuracy. -fs=1 skips every other frame processing 50%% of the video, -fs=2 processes 33%% of the video frames, -fs=3 processes 25%%, etc... %s'
    % USER_CONFIG.get_help_string("global", "frame-skip"),
)
@click.option(
    '--verbosity',
    '-v',
    metavar='LEVEL',
    type=click.Choice(CHOICE_MAP['global']['verbosity'], False),
    default=None,
    help='Amount of information to show. LEVEL must be one of: %s. Overrides -q/--quiet.%s' %
    (', '.join(CHOICE_MAP["global"]["verbosity"]), USER_CONFIG.get_help_string(
        "global", "verbosity")),
)
@click.option(
    '--logfile',
    '-l',
    metavar='FILE',
    type=click.Path(exists=False, file_okay=True, writable=True, resolve_path=False),
    help='Save debug log to FILE. Appends to existing file if present.',
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    flag_value=True,
    help='Suppress output to terminal/stdout. Equivalent to setting --verbosity=none.',
)
@click.pass_context
# pylint: disable=redefined-builtin
def scenedetect(
    ctx: click.Context,
    input: Optional[AnyStr],
    output: Optional[AnyStr],
    stats: Optional[AnyStr],
    config: Optional[AnyStr],
    framerate: Optional[float],
    min_scene_len: Optional[str],
    drop_short_scenes: bool,
    merge_last_scene: bool,
    backend: Optional[str],
    downscale: Optional[int],
    frame_skip: Optional[int],
    verbosity: Optional[str],
    logfile: Optional[AnyStr],
    quiet: bool,
):
    """PySceneDetect is a scene cut/transition detection program. PySceneDetect takes an input video, runs detection on it, and uses the resulting scene information to generate output. The syntax for using PySceneDetect is:

    {scenedetect_with_video} [detector] [commands]

For [detector] use `scene_detect-adaptive` or `scene_detect-content` to find fast cuts, and `scene_detect-threshold` for fades in/out. If [detector] is not specified, a default detector will be used.

Examples:

Split video wherever a new scene is detected:

    {scenedetect_with_video} split-video

Save scene list in CSV format with images at the start, middle, and end of each scene:

    {scenedetect_with_video} list-scenes save-images

Skip the first 10 seconds of the input video:

    {scenedetect_with_video} time --start 10s scene_detect-content

Show summary of all options and commands:

    {scenedetect} --help

Global options (e.g. -i/--input, -c/--config) must be specified before any commands and their options. The order of commands is not strict, but each command must only be specified once.
"""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_options(
        input_path=input,
        output=output,
        framerate=framerate,
        stats_file=stats,
        downscale=downscale,
        frame_skip=frame_skip,
        min_scene_len=min_scene_len,
        drop_short_scenes=drop_short_scenes,
        merge_last_scene=merge_last_scene,
        backend=backend,
        quiet=quiet,
        logfile=logfile,
        config=config,
        stats=stats,
        verbosity=verbosity,
    )


# pylint: enable=redefined-builtin


@click.command('help', cls=_Command)
@click.argument(
    'command_name',
    required=False,
    type=click.STRING,
)
@click.pass_context
def help_command(ctx: click.Context, command_name: str):
    """Print help for command (`help [command]`)."""
    assert isinstance(ctx.obj, CliContext)
    assert isinstance(ctx.parent.command, click.MultiCommand)
    parent_command = ctx.parent.command
    all_commands = set(parent_command.list_commands(ctx))
    if command_name is not None:
        if not command_name in all_commands:
            error_strs = [
                'unknown command. List of valid commands:',
                '  %s' % ', '.join(sorted(all_commands))
            ]
            raise click.BadParameter('\n'.join(error_strs), param_hint='command')
        click.echo('')
        _print_command_help(ctx, parent_command.get_command(ctx, command_name))
    else:
        click.echo(ctx.parent.get_help())
        for command in sorted(all_commands):
            _print_command_help(ctx, parent_command.get_command(ctx, command))
    ctx.exit()


@click.command('about', cls=_Command, add_help_option=False)
@click.pass_context
def about_command(ctx: click.Context):
    """Print license/copyright info."""
    assert isinstance(ctx.obj, CliContext)
    click.echo('')
    click.echo(click.style(_LINE_SEPARATOR, fg='cyan'))
    click.echo(click.style(' About PySceneDetect %s' % _PROGRAM_VERSION, fg='yellow'))
    click.echo(click.style(_LINE_SEPARATOR, fg='cyan'))
    click.echo(_ABOUT_STRING)
    ctx.exit()


@click.command('version', cls=_Command, add_help_option=False)
@click.pass_context
def version_command(ctx: click.Context):
    """Print PySceneDetect version."""
    assert isinstance(ctx.obj, CliContext)
    click.echo('')
    click.echo(get_system_version_info())
    ctx.exit()


@click.command('time', cls=_Command)
@click.option(
    '--start',
    '-s',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Time in video to start detection. TIMECODE can be specified as number of frames (--start=100 for frame 100), time in seconds followed by "s" (--start=100s for 100 seconds), or timecode (--start=00:01:40 for 1m40s).',
)
@click.option(
    '--duration',
    '-d',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Maximum time in video to process. TIMECODE format is the same as other arguments. Mutually exclusive with -e/--end.',
)
@click.option(
    '--end',
    '-e',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Time in video to end detecting scenes. TIMECODE format is the same as other arguments. Mutually exclusive with -d/--duration',
)
@click.pass_context
def time_command(
    ctx: click.Context,
    start: Optional[str],
    duration: Optional[str],
    end: Optional[str],
):
    """Set start/end/duration of input video.

Values can be specified as frames (NNNN), seconds (NNNN.NNs), or timecode (HH:MM:SS.nnn). For example, to process only the first minute of a video:

    {scenedetect_with_video} time --end 00:01:00

    {scenedetect_with_video} time --duration 60s

Note that --end and --duration are mutually exclusive (i.e. only one of the two can be set). Lastly, the following is an example using absolute frame numbers to process frames 0 through 1000:

    {scenedetect_with_video} time --start 0 --end 1000
"""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_time(
        start=start,
        duration=duration,
        end=end,
    )


@click.command('scene_detect-content', cls=_Command)
@click.option(
    '--threshold',
    '-t',
    metavar='VAL',
    type=click.FloatRange(CONFIG_MAP['scene_detect-content']['threshold'].min_val,
                          CONFIG_MAP['scene_detect-content']['threshold'].max_val),
    default=None,
    help='Threshold (float) that frame score must exceed to trigger a cut. Refers to "content_val" in stats file.%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "threshold")),
)
@click.option(
    '--weights',
    '-w',
    type=(float, float, float, float),
    default=None,
    metavar='HUE SAT LUM EDGE',
    help='Weights of 4 components used to calculate frame score from (delta_hue, delta_sat, delta_lum, delta_edges).%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "weights")),
)
@click.option(
    '--luma-only',
    '-l',
    is_flag=True,
    flag_value=True,
    help='Only use luma (brightness) channel. Useful for greyscale videos. Equivalent to setting "-w 0 0 1 0".%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "luma-only")),
)
@click.option(
    '--kernel-size',
    '-k',
    metavar='N',
    type=click.INT,
    default=None,
    help='Size of kernel for expanding detected edges. Must be odd integer greater than or equal to 3. If unset, kernel size is estimated using video resolution.%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "kernel-size")),
)
@click.option(
    '--min-scene-len',
    '-m',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Minimum length of any scene. Overrides global option -m/--min-scene-len. TIMECODE can be specified in frames (-m=100), in seconds with `s` suffix (-m=3.5s), or timecode (-m=00:01:52.778).%s'
    % ('' if USER_CONFIG.is_default('scene_detect-content', 'min-scene-len') else
       USER_CONFIG.get_help_string('scene_detect-content', 'min-scene-len')),
)
@click.pass_context
def detect_content_command(
    ctx: click.Context,
    threshold: Optional[float],
    weights: Optional[Tuple[float, float, float, float]],
    luma_only: bool,
    kernel_size: Optional[int],
    min_scene_len: Optional[str],
):
    """Perform content detection algorithm on input video.

For each frame, a score from 0 to 255.0 is calculated which represents the difference in content between the current and previous frame (higher = more different). A cut is generated when a frame score exceeds -t/--threshold. Frame scores are saved under the "content_val" column in a statsfile.

Scores are calculated from several components which are also recorded in the statsfile:

 - *delta_hue*: Difference between pixel hue values of adjacent frames.

 - *delta_sat*: Difference between pixel saturation values of adjacent frames.

 - *delta_lum*: Difference between pixel luma (brightness) values of adjacent frames.

 - *delta_edges*: Difference between calculated edges of adjacent frames. Typically larger than other components, so threshold may need to be increased to compensate.

Once calculated, these components are multiplied by the specified -w/--weights to calculate the final frame score ("content_val").  Weights are set as a set of 4 numbers in the form (*delta_hue*, *delta_sat*, *delta_lum*, *delta_edges*). For example, "--weights 1.0 0.5 1.0 0.2 --threshold 32" is a good starting point for trying edge detection. The final sum is normalized by the weight of all components, so they need not equal 100%. Edge detection is disabled by default to improve performance.

Examples:

    {scenedetect_with_video} scene_detect-content

    {scenedetect_with_video} scene_detect-content --threshold 27.5
"""
    assert isinstance(ctx.obj, CliContext)
    detector_args = ctx.obj.get_detect_content_params(
        threshold=threshold,
        luma_only=luma_only,
        min_scene_len=min_scene_len,
        weights=weights,
        kernel_size=kernel_size)
    logger.debug('Adding detector: ContentDetector(%s)', detector_args)
    ctx.obj.add_detector(ContentDetector(**detector_args))


@click.command('scene_detect-adaptive', cls=_Command)
@click.option(
    '--threshold',
    '-t',
    metavar='VAL',
    type=click.FLOAT,
    default=None,
    help='Threshold (float) that frame score must exceed to trigger a cut. Refers to "adaptive_ratio" in stats file.%s'
    % (USER_CONFIG.get_help_string('scene_detect-adaptive', 'threshold')),
)
@click.option(
    '--min-content-val',
    '-c',
    metavar='VAL',
    type=click.FLOAT,
    default=None,
    help='Minimum threshold (float) that "content_val" must exceed to trigger a cut.%s' %
    (USER_CONFIG.get_help_string('scene_detect-adaptive', 'min-content-val')),
)
@click.option(
    '--min-delta-hsv',
    '-d',
    metavar='VAL',
    type=click.FLOAT,
    default=None,
    help='[DEPRECATED] Use -c/--min-content-val instead.%s' %
    (USER_CONFIG.get_help_string('scene_detect-adaptive', 'min-delta-hsv')),
    hidden=True,
)
@click.option(
    '--frame-window',
    '-f',
    metavar='VAL',
    type=click.INT,
    default=None,
    help='Size of window to scene_detect deviations from mean. Represents how many frames before/after the current one to use for mean.%s'
    % (USER_CONFIG.get_help_string('scene_detect-adaptive', 'frame-window')),
)
@click.option(
    '--weights',
    '-w',
    type=(float, float, float, float),
    default=None,
    help='Weights of 4 components ("delta_hue", "delta_sat", "delta_lum", "delta_edges") used to calculate "content_val".%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "weights")),
)
@click.option(
    '--luma-only',
    '-l',
    is_flag=True,
    flag_value=True,
    help='Only use luma (brightness) channel. Useful for greyscale videos. Equivalent to "--weights 0 0 1 0".%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "luma-only")),
)
@click.option(
    '--kernel-size',
    '-k',
    metavar='N',
    type=click.INT,
    default=None,
    help='Size of kernel for expanding detected edges. Must be odd number >= 3. If unset, size is estimated using video resolution.%s'
    % (USER_CONFIG.get_help_string("scene_detect-content", "kernel-size")),
)
@click.option(
    '--min-scene-len',
    '-m',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Minimum length of any scene. Overrides global option -m/--min-scene-len. TIMECODE can be specified in frames (-m=100), in seconds with `s` suffix (-m=3.5s), or timecode (-m=00:01:52.778).%s'
    % ('' if USER_CONFIG.is_default('scene_detect-adaptive', 'min-scene-len') else
       USER_CONFIG.get_help_string('scene_detect-adaptive', 'min-scene-len')),
)
@click.pass_context
def detect_adaptive_command(
    ctx: click.Context,
    threshold: Optional[float],
    min_content_val: Optional[float],
    min_delta_hsv: Optional[float],
    frame_window: Optional[int],
    weights: Optional[Tuple[float, float, float, float]],
    luma_only: bool,
    kernel_size: Optional[int],
    min_scene_len: Optional[str],
):
    """Perform adaptive detection algorithm on input video.

Two-pass algorithm that first calculates frame scores with `scene_detect-content`, and then applies a rolling average when processing the result. This can help mitigate false detections in situations such as camera movement.

Examples:

    {scenedetect_with_video} scene_detect-adaptive

    {scenedetect_with_video} scene_detect-adaptive --threshold 3.2
"""
    assert isinstance(ctx.obj, CliContext)
    detector_args = ctx.obj.get_detect_adaptive_params(
        threshold=threshold,
        min_content_val=min_content_val,
        min_delta_hsv=min_delta_hsv,
        frame_window=frame_window,
        luma_only=luma_only,
        min_scene_len=min_scene_len,
        weights=weights,
        kernel_size=kernel_size,
    )
    logger.debug('Adding detector: AdaptiveDetector(%s)', detector_args)
    ctx.obj.add_detector(AdaptiveDetector(**detector_args))


@click.command('scene_detect-threshold', cls=_Command)
@click.option(
    '--threshold',
    '-t',
    metavar='VAL',
    type=click.FloatRange(CONFIG_MAP['scene_detect-threshold']['threshold'].min_val,
                          CONFIG_MAP['scene_detect-threshold']['threshold'].max_val),
    default=None,
    help='Threshold (integer) that frame score must exceed to start a new scene. Refers to "delta_rgb" in stats file.%s'
    % (USER_CONFIG.get_help_string('scene_detect-threshold', 'threshold')),
)
@click.option(
    '--fade-bias',
    '-f',
    metavar='PERCENT',
    type=click.FloatRange(CONFIG_MAP['scene_detect-threshold']['fade-bias'].min_val,
                          CONFIG_MAP['scene_detect-threshold']['fade-bias'].max_val),
    default=None,
    help='Percent (%%) from -100 to 100 of timecode skew of cut placement. -100 indicates the start frame, +100 indicates the end frame, and 0 is the middle of both.%s'
    % (USER_CONFIG.get_help_string('scene_detect-threshold', 'fade-bias')),
)
@click.option(
    '--add-last-scene',
    '-l',
    is_flag=True,
    flag_value=True,
    help='If set and video ends after a fade-out event, generate a final cut at the last fade-out position.%s'
    % (USER_CONFIG.get_help_string('scene_detect-threshold', 'add-last-scene')),
)
@click.option(
    '--min-scene-len',
    '-m',
    metavar='TIMECODE',
    type=click.STRING,
    default=None,
    help='Minimum length of any scene. Overrides global option -m/--min-scene-len. TIMECODE can be specified in frames (-m=100), in seconds with `s` suffix (-m=3.5s), or timecode (-m=00:01:52.778).%s'
    % ('' if USER_CONFIG.is_default('scene_detect-threshold', 'min-scene-len') else
       USER_CONFIG.get_help_string('scene_detect-threshold', 'min-scene-len')),
)
@click.pass_context
def detect_threshold_command(
    ctx: click.Context,
    threshold: Optional[float],
    fade_bias: Optional[float],
    add_last_scene: bool,
    min_scene_len: Optional[str],
):
    """Perform threshold detection algorithm on input video.

Detects fade-in and fade-out events using average pixel values. Resulting cuts are placed between adjacent fade-out and fade-in events.

Examples:

    {scenedetect_with_video} scene_detect-threshold

    {scenedetect_with_video} scene_detect-threshold --threshold 15
"""
    assert isinstance(ctx.obj, CliContext)
    detector_args = ctx.obj.get_detect_threshold_params(
        threshold=threshold,
        fade_bias=fade_bias,
        add_last_scene=add_last_scene,
        min_scene_len=min_scene_len,
    )
    logger.debug('Adding detector: ThresholdDetector(%s)', detector_args)
    ctx.obj.add_detector(ThresholdDetector(**detector_args))


@click.command('load-scenes', cls=_Command)
@click.option(
    '--input',
    '-i',
    multiple=False,
    metavar='FILE',
    required=True,
    type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=True),
    help='Scene list to read cut information from.')
@click.option(
    '--start-col-name',
    '-c',
    metavar='STRING',
    type=click.STRING,
    default=None,
    help='Name of column used to mark scene cuts.%s' %
    (USER_CONFIG.get_help_string('load-scenes', 'start-col-name')))
@click.pass_context
def load_scenes_command(ctx: click.Context, input: Optional[str], start_col_name: Optional[str]):
    """Load scenes from CSV instead of detecting. Can be used with CSV generated by `list-scenes`. Scenes are loaded using the specified column as cut locations (frame number or timecode).

Examples:

    {scenedetect_with_video} load-scenes -i scenes.csv

    {scenedetect_with_video} load-scenes -i scenes.csv --start-col-name "Start Timecode"
"""
    assert isinstance(ctx.obj, CliContext)
    logger.debug('Loading scenes from %s (start_col_name = %s)', input, start_col_name)
    ctx.obj.handle_load_scenes(input=input, start_col_name=start_col_name)


@click.command('export-html', cls=_Command)
@click.option(
    '--filename',
    '-f',
    metavar='NAME',
    default='$VIDEO_NAME-Scenes.html',
    type=click.STRING,
    help='Filename format to use for the scene list HTML file. You can use the $VIDEO_NAME macro in the file name. Note that you may have to wrap the format name using single quotes.%s'
    % (USER_CONFIG.get_help_string('export-html', 'filename')),
)
@click.option(
    '--no-images',
    is_flag=True,
    flag_value=True,
    help='Export the scene list including or excluding the saved images.%s' %
    (USER_CONFIG.get_help_string('export-html', 'no-images')),
)
@click.option(
    '--image-width',
    '-w',
    metavar='pixels',
    type=click.INT,
    help='Width in pixels of the images in the resulting HTML table.%s' %
    (USER_CONFIG.get_help_string('export-html', 'image-width', show_default=False)),
)
@click.option(
    '--image-height',
    '-h',
    metavar='pixels',
    type=click.INT,
    help='Height in pixels of the images in the resulting HTML table.%s' %
    (USER_CONFIG.get_help_string('export-html', 'image-height', show_default=False)),
)
@click.pass_context
def export_html_command(
    ctx: click.Context,
    filename: Optional[AnyStr],
    no_images: bool,
    image_width: Optional[int],
    image_height: Optional[int],
):
    """Export scene list to HTML file. Requires save-images unless --no-images is specified."""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_export_html(
        filename=filename,
        no_images=no_images,
        image_width=image_width,
        image_height=image_height,
    )


@click.command('list-scenes', cls=_Command)
@click.option(
    '--output',
    '-o',
    metavar='DIR',
    type=click.Path(exists=False, dir_okay=True, writable=True, resolve_path=False),
    help='Output directory to save videos to. Overrides global option -o/--output if set.%s' %
    (USER_CONFIG.get_help_string('list-scenes', 'output', show_default=False)),
)
@click.option(
    '--filename',
    '-f',
    metavar='NAME',
    default='$VIDEO_NAME-Scenes.csv',
    type=click.STRING,
    help='Filename format to use for the scene list CSV file. You can use the $VIDEO_NAME macro in the file name. Note that you may have to wrap the name using single quotes or use escape characters (e.g. -f=\$VIDEO_NAME-Scenes.csv).%s'
    % (USER_CONFIG.get_help_string('list-scenes', 'filename')),
)
@click.option(
    '--no-output-file',
    '-n',
    is_flag=True,
    flag_value=True,
    help='Only print scene list.%s' %
    (USER_CONFIG.get_help_string('list-scenes', 'no-output-file')),
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    flag_value=True,
    help='Suppress printing scene list.%s' % (USER_CONFIG.get_help_string('list-scenes', 'quiet')),
)
@click.option(
    '--skip-cuts',
    '-s',
    is_flag=True,
    flag_value=True,
    help='Skip cutting list as first row in the CSV file. Set for RFC 4180 compliant output.%s' %
    (USER_CONFIG.get_help_string('list-scenes', 'skip-cuts')),
)
@click.pass_context
def list_scenes_command(
    ctx: click.Context,
    output: Optional[AnyStr],
    filename: Optional[AnyStr],
    no_output_file: bool,
    quiet: bool,
    skip_cuts: bool,
):
    """Create scene list CSV file (will be named $VIDEO_NAME-Scenes.csv by default)."""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_list_scenes(
        output=output,
        filename=filename,
        no_output_file=no_output_file,
        quiet=quiet,
        skip_cuts=skip_cuts,
    )


@click.command('split-video', cls=_Command)
@click.option(
    '--output',
    '-o',
    metavar='DIR',
    type=click.Path(exists=False, dir_okay=True, writable=True, resolve_path=False),
    help='Output directory to save videos to. Overrides global option -o/--output if set.%s' %
    (USER_CONFIG.get_help_string('split-video', 'output', show_default=False)),
)
@click.option(
    '--filename',
    '-f',
    metavar='NAME',
    default=None,
    type=click.STRING,
    help='File name format to use when saving videos, with or without extension. You can use $VIDEO_NAME and $SCENE_NUMBER macros in the filename. You may have to wrap the format in single quotes or use escape characters to avoid variable expansion (e.g. -f=\\$VIDEO_NAME-Scene-\\$SCENE_NUMBER).%s'
    % (USER_CONFIG.get_help_string('split-video', 'filename')),
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    flag_value=True,
    help='Hide output from external video splitting tool.%s' %
    (USER_CONFIG.get_help_string('split-video', 'quiet')),
)
@click.option(
    '--copy',
    '-c',
    is_flag=True,
    flag_value=True,
    help='Copy instead of re-encode. Faster but less precise. Equivalent to: --args="-map 0 -c:v copy -c:a copy"%s'
    % (USER_CONFIG.get_help_string('split-video', 'copy')),
)
@click.option(
    '--high-quality',
    '-hq',
    is_flag=True,
    flag_value=True,
    help='Encode video with higher quality, overrides -f option if present. Equivalent to: --rate-factor=17 --preset=slow%s'
    % (USER_CONFIG.get_help_string('split-video', 'high-quality')),
)
@click.option(
    '--rate-factor',
    '-crf',
    metavar='RATE',
    default=None,
    type=click.IntRange(CONFIG_MAP['split-video']['rate-factor'].min_val,
                        CONFIG_MAP['split-video']['rate-factor'].max_val),
    help='Video encoding quality (x264 constant rate factor), from 0-100, where lower is higher quality (larger output). 0 indicates lossless.%s'
    % (USER_CONFIG.get_help_string('split-video', 'rate-factor')),
)
@click.option(
    '--preset',
    '-p',
    metavar='LEVEL',
    default=None,
    type=click.Choice(CHOICE_MAP['split-video']['preset']),
    help='Video compression quality (x264 preset). Can be one of: %s. Faster modes take less time but output may be larger.%s'
    % (', '.join(
        CHOICE_MAP['split-video']['preset']), USER_CONFIG.get_help_string('split-video', 'preset')),
)
@click.option(
    '--args',
    '-a',
    metavar='ARGS',
    type=click.STRING,
    default=None,
    help='Override codec arguments passed to FFmpeg when splitting scenes. Use double quotes (") around arguments. Must specify at least audio/video codec.%s'
    % (USER_CONFIG.get_help_string('split-video', 'args')),
)
@click.option(
    '--mkvmerge',
    '-m',
    is_flag=True,
    flag_value=True,
    help='Split video using mkvmerge. Faster than re-encoding, but less precise. If set, options other than -f/--filename, -q/--quiet and -o/--output will be ignored. Note that mkvmerge automatically appends the $SCENE_NUMBER suffix.%s'
    % (USER_CONFIG.get_help_string('split-video', 'mkvmerge')),
)
@click.pass_context
def split_video_command(
    ctx: click.Context,
    output: Optional[AnyStr],
    filename: Optional[AnyStr],
    quiet: bool,
    copy: bool,
    high_quality: bool,
    rate_factor: Optional[int],
    preset: Optional[str],
    args: Optional[str],
    mkvmerge: bool,
):
    """Split input video using ffmpeg or mkvmerge.

Examples:

    {scenedetect_with_video} split-video

    {scenedetect_with_video} split-video --copy

    {scenedetect_with_video} split-video --filename \$VIDEO_NAME-Clip-\$SCENE_NUMBER
"""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_split_video(
        output=output,
        filename=filename,
        quiet=quiet,
        copy=copy,
        high_quality=high_quality,
        rate_factor=rate_factor,
        preset=preset,
        args=args,
        mkvmerge=mkvmerge,
    )


@click.command('save-images', cls=_Command)
@click.option(
    '--output',
    '-o',
    metavar='DIR',
    type=click.Path(exists=False, dir_okay=True, writable=True, resolve_path=False),
    help='Output directory for images. Overrides global option -o/--output if set.%s' %
    (USER_CONFIG.get_help_string('save-images', 'output', show_default=False)),
)
@click.option(
    '--filename',
    '-f',
    metavar='NAME',
    default=None,
    type=click.STRING,
    help='Filename format *without* extension to use when saving images. You can use the $VIDEO_NAME, $SCENE_NUMBER, $IMAGE_NUMBER, and $FRAME_NUMBER macros in the file name. You may have to use escape characters (e.g. -f=\\$SCENE_NUMBER-Image-\\$IMAGE_NUMBER) or single quotes.%s'
    % (USER_CONFIG.get_help_string('save-images', 'filename')),
)
@click.option(
    '--num-images',
    '-n',
    metavar='N',
    default=None,
    type=click.INT,
    help='Number of images to generate per scene. Will always include start/end frame, unless -n=1, in which case the image will be the frame at the mid-point of the scene.%s'
    % (USER_CONFIG.get_help_string('save-images', 'num-images')),
)
@click.option(
    '--jpeg',
    '-j',
    is_flag=True,
    flag_value=True,
    help='Set output format to JPEG (default).%s' %
    (USER_CONFIG.get_help_string('save-images', 'format', show_default=False)),
)
@click.option(
    '--webp',
    '-w',
    is_flag=True,
    flag_value=True,
    help='Set output format to WebP',
)
@click.option(
    '--quality',
    '-q',
    metavar='Q',
    default=None,
    type=click.IntRange(0, 100),
    help='JPEG/WebP encoding quality, from 0-100 (higher indicates better quality). For WebP, 100 indicates lossless. [default: JPEG: 95, WebP: 100]%s'
    % (USER_CONFIG.get_help_string('save-images', 'quality', show_default=False)),
)
@click.option(
    '--png',
    '-p',
    is_flag=True,
    flag_value=True,
    help='Set output format to PNG.',
)
@click.option(
    '--compression',
    '-c',
    metavar='C',
    default=None,
    type=click.IntRange(0, 9),
    help='PNG compression rate, from 0-9. Higher values produce smaller files but result in longer compression time. This setting does not affect image quality, only file size.%s'
    % (USER_CONFIG.get_help_string('save-images', 'compression')),
)
@click.option(
    '-m',
    '--frame-margin',
    metavar='N',
    default=None,
    type=click.INT,
    help='Number of frames to ignore at beginning/end of scenes when saving images. Controls temporal padding on scene boundaries.%s'
    % (USER_CONFIG.get_help_string('save-images', 'num-images')),
)
@click.option(
    '--scale',
    '-s',
    metavar='S',
    default=None,
    type=click.FLOAT,
    help='Factor to scale images by. Ignored if -W/--width or -H/--height is set.%s' %
    (USER_CONFIG.get_help_string('save-images', 'scale', show_default=False)),
)
@click.option(
    '--height',
    '-H',
    metavar='H',
    default=None,
    type=click.INT,
    help='Height (pixels) of images.%s' %
    (USER_CONFIG.get_help_string('save-images', 'height', show_default=False)),
)
@click.option(
    '--width',
    '-W',
    metavar='W',
    default=None,
    type=click.INT,
    help='Width (pixels) of images.%s' %
    (USER_CONFIG.get_help_string('save-images', 'width', show_default=False)),
)
@click.pass_context
def save_images_command(
    ctx: click.Context,
    output: Optional[AnyStr],
    filename: Optional[AnyStr],
    num_images: Optional[int],
    jpeg: bool,
    webp: bool,
    quality: Optional[int],
    png: bool,
    compression: Optional[int],
    frame_margin: Optional[int],
    scale: Optional[float],
    height: Optional[int],
    width: Optional[int],
):
    """Create images for each detected scene.

Images can be resized

Examples:

    {scenedetect_with_video} save-images

    {scenedetect_with_video} save-images --width 1024

    {scenedetect_with_video} save-images --filename \$SCENE_NUMBER-img\$IMAGE_NUMBER
"""
    assert isinstance(ctx.obj, CliContext)
    ctx.obj.handle_save_images(
        num_images=num_images,
        output=output,
        filename=filename,
        jpeg=jpeg,
        webp=webp,
        quality=quality,
        png=png,
        compression=compression,
        frame_margin=frame_margin,
        scale=scale,
        height=height,
        width=width,
    )


# ----------------------------------------------------------------------
# Commands Omitted From Help List
# ----------------------------------------------------------------------

# Info Commands
scenedetect.add_command(help_command)
scenedetect.add_command(version_command)
scenedetect.add_command(about_command)

# ----------------------------------------------------------------------
# Commands Added To Help List
# ----------------------------------------------------------------------

# Input / Output
scenedetect.add_command(time_command)
scenedetect.add_command(export_html_command)
scenedetect.add_command(list_scenes_command)
scenedetect.add_command(save_images_command)
scenedetect.add_command(split_video_command)

# Detection Algorithms
scenedetect.add_command(detect_content_command)
scenedetect.add_command(detect_threshold_command)
scenedetect.add_command(detect_adaptive_command)
scenedetect.add_command(load_scenes_command)
