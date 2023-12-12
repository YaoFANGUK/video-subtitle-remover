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
"""Context of which command-line options and config settings the user provided."""

import logging
import os
from typing import Any, AnyStr, Dict, Optional, Tuple, Type

import click

import scenedetect

from scenedetect import open_video, AVAILABLE_BACKENDS
from scenedetect._scene_loader import SceneLoader

from scenedetect.scene_detector import SceneDetector
from scenedetect.platform import get_and_create_path, get_cv2_imwrite_params, init_logger
from scenedetect.frame_timecode import FrameTimecode, MAX_FPS_DELTA
from scenedetect.video_stream import VideoStream, VideoOpenFailure, FrameRateUnavailable
from scenedetect.video_splitter import is_mkvmerge_available, is_ffmpeg_available
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.stats_manager import StatsManager
from scenedetect.scene_manager import SceneManager, Interpolation

from scenedetect._cli.config import ConfigRegistry, ConfigLoadFailure, CHOICE_MAP

logger = logging.getLogger('pyscenedetect')

USER_CONFIG = ConfigRegistry(throw_exception=False)


def parse_timecode(value: str,
                   frame_rate: float,
                   first_index_is_one: bool = False) -> FrameTimecode:
    """Parses a user input string into a FrameTimecode assuming the given framerate.

    If value is None, None will be returned instead of processing the value.

    Raises:
        click.BadParameter
     """
    if value is None:
        return None
    try:
        if first_index_is_one and value.isdigit():
            value = int(value)
            if value >= 1:
                value -= 1
        return FrameTimecode(timecode=value, fps=frame_rate)
    except ValueError as ex:
        raise click.BadParameter(
            'timecode must be in frames (1234), seconds (123.4s), or HH:MM:SS (00:02:03.400)'
        ) from ex


def contains_sequence_or_url(video_path: str) -> bool:
    """Checks if the video path is a URL or image sequence."""
    return '%' in video_path or '://' in video_path


def check_split_video_requirements(use_mkvmerge: bool) -> None:
    """ Validates that the proper tool is available on the system to perform the
    `split-video` command.

    Arguments:
        use_mkvmerge: True if mkvmerge (-m), False otherwise.

    Raises: click.BadParameter if the proper video splitting tool cannot be found.
    """

    if (use_mkvmerge and not is_mkvmerge_available()) or not is_ffmpeg_available():
        error_strs = [
            "{EXTERN_TOOL} is required for split-video{EXTRA_ARGS}.".format(
                EXTERN_TOOL='mkvmerge' if use_mkvmerge else 'ffmpeg',
                EXTRA_ARGS=' when mkvmerge (-m) is set' if use_mkvmerge else '')
        ]
        error_strs += ['Ensure the program is available on your system and try again.']
        if not use_mkvmerge and is_mkvmerge_available():
            error_strs += ['You can specify mkvmerge (-m) to use mkvmerge for splitting.']
        elif use_mkvmerge and is_ffmpeg_available():
            error_strs += ['You can specify copy (-c) to use ffmpeg stream copying.']
        error_str = '\n'.join(error_strs)
        raise click.BadParameter(error_str, param_hint='split-video')


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
class CliContext:
    """Context of the command-line interface and config file parameters passed between sub-commands.

    Handles validation of options taken in from the CLI *and* configuration files.

    After processing the main program options via `handle_options`, the CLI will then call
    the respective `handle_*` method for each command. Once all commands have been
    processed, the main program actions are executed by passing this object to the
    `run_scenedetect` function in `scenedetect.cli.controller`.
    """

    def __init__(self):
        self.config = USER_CONFIG
        self.video_stream: VideoStream = None
        self.scene_manager: SceneManager = None
        self.stats_manager: StatsManager = None

        # Global `scenedetect` Options
        self.output_directory: str = None                   # -o/--output
        self.quiet_mode: bool = None                        # -q/--quiet or -v/--verbosity quiet
        self.stats_file_path: str = None                    # -s/--stats
        self.drop_short_scenes: bool = None                 # --drop-short-scenes
        self.merge_last_scene: bool = None                  # --merge-last-scene
        self.min_scene_len: FrameTimecode = None            # -m/--min-scene-len
        self.frame_skip: int = None                         # -fs/--frame-skip
        self.default_detector: Tuple[Type[SceneDetector],
                                     Dict[str, Any]] = None # [global] default-detector

        # `time` Command Options
        self.time: bool = False
        self.start_time: FrameTimecode = None # time -s/--start
        self.end_time: FrameTimecode = None   # time -e/--end
        self.duration: FrameTimecode = None   # time -d/--duration

        # `save-images` Command Options
        self.save_images: bool = False
        self.image_extension: str = None        # save-images -j/--jpeg, -w/--webp, -p/--png
        self.image_directory: str = None        # save-images -o/--output
        self.image_param: int = None            # save-images -q/--quality if -j/-w,
                                                #   otherwise -c/--compression if -p
        self.image_name_format: str = None      # save-images -f/--name-format
        self.num_images: int = None             # save-images -n/--num-images
        self.frame_margin: int = 1              # save-images -m/--frame-margin
        self.scale: float = None                # save-images -s/--scale
        self.height: int = None                 # save-images -h/--height
        self.width: int = None                  # save-images -w/--width
        self.scale_method: Interpolation = None # [save-images] scale-method

        # `split-video` Command Options
        self.split_video: bool = False
        self.split_mkvmerge: bool = None   # split-video -m/--mkvmerge
        self.split_args: str = None        # split-video -a/--args, -c/--copy
        self.split_directory: str = None   # split-video -o/--output
        self.split_name_format: str = None # split-video -f/--filename
        self.split_quiet: bool = None      # split-video -q/--quiet

        # `list-scenes` Command Options
        self.list_scenes: bool = False
        self.print_scene_list: bool = None      # list-scenes -q/--quiet
        self.scene_list_directory: str = None   # list-scenes -o/--output
        self.scene_list_name_format: str = None # list-scenes -f/--filename
        self.scene_list_output: bool = None     # list-scenes -n/--no-output
        self.skip_cuts: bool = None             # list-scenes -s/--skip-cuts

        # `export-html` Command Options
        self.export_html: bool = False
        self.html_name_format: str = None     # export-html -f/--filename
        self.html_include_images: bool = None # export-html --no-images
        self.image_width: int = None          # export-html -w/--image-width
        self.image_height: int = None         # export-html -h/--image-height

    #
    # Command Handlers
    #

    def handle_options(
        self,
        input_path: AnyStr,
        output: Optional[AnyStr],
        framerate: float,
        stats_file: Optional[AnyStr],
        downscale: Optional[int],
        frame_skip: int,
        min_scene_len: str,
        drop_short_scenes: bool,
        merge_last_scene: bool,
        backend: Optional[str],
        quiet: bool,
        logfile: Optional[AnyStr],
        config: Optional[AnyStr],
        stats: Optional[AnyStr],
        verbosity: Optional[str],
    ):
        """Parse all global options/arguments passed to the main scenedetect command,
        before other sub-commands (e.g. this function processes the [options] when calling
        `scenedetect [options] [commands [command options]]`).

        Raises:
            click.BadParameter: One of the given options/parameters is invalid.
            click.Abort: Fatal initialization failure.
        """

        # TODO(v1.0): Make the stats value optional (e.g. allow -s only), and allow use of
        # $VIDEO_NAME macro in the name.  Default to $VIDEO_NAME.csv.

        try:
            init_failure = not self.config.initialized
            init_log = self.config.get_init_log()
            quiet = not init_failure and quiet
            self._initialize_logging(quiet=quiet, verbosity=verbosity, logfile=logfile)

            # Configuration file was specified via CLI argument -c/--config.
            if config and not init_failure:
                self.config = ConfigRegistry(config)
                init_log += self.config.get_init_log()
                # Re-initialize logger with the correct verbosity.
                if verbosity is None and not self.config.is_default('global', 'verbosity'):
                    verbosity_str = self.config.get_value('global', 'verbosity')
                    assert verbosity_str in CHOICE_MAP['global']['verbosity']
                    self.quiet_mode = False
                    self._initialize_logging(verbosity=verbosity_str, logfile=logfile)

        except ConfigLoadFailure as ex:
            init_failure = True
            init_log += ex.init_log
            if ex.reason is not None:
                init_log += [(logging.ERROR, 'Error: %s' % str(ex.reason).replace('\t', '  '))]
        finally:
            # Make sure we print the version number even on any kind of init failure.
            logger.info('PySceneDetect %s', scenedetect.__version__)
            for (log_level, log_str) in init_log:
                logger.log(log_level, log_str)
            if init_failure:
                logger.critical("Error processing configuration file.")
                raise click.Abort()

        if self.config.config_dict:
            logger.debug("Current configuration:\n%s", str(self.config.config_dict))

        logger.debug('Parsing program options.')
        if stats is not None and frame_skip:
            error_strs = [
                'Unable to scene_detect scenes with stats file if frame skip is not 0.',
                '  Either remove the -fs/--frame-skip option, or the -s/--stats file.\n'
            ]
            logger.error('\n'.join(error_strs))
            raise click.BadParameter(
                'Combining the -s/--stats and -fs/--frame-skip options is not supported.',
                param_hint='frame skip + stats file')

        # Handle the case where -i/--input was not specified (e.g. for the `help` command).
        if input_path is None:
            return

        # Have to load the input video to obtain a time base before parsing timecodes.
        self._open_video_stream(
            input_path=input_path,
            framerate=framerate,
            backend=self.config.get_value("global", "backend", backend, ignore_default=True))

        self.output_directory = output if output else self.config.get_value("global", "output")
        if self.output_directory:
            logger.info('Output directory set:\n  %s', self.output_directory)

        self.min_scene_len = parse_timecode(
            min_scene_len if min_scene_len is not None else self.config.get_value(
                "global", "min-scene-len"), self.video_stream.frame_rate)
        self.drop_short_scenes = drop_short_scenes or self.config.get_value(
            "global", "drop-short-scenes")
        self.merge_last_scene = merge_last_scene or self.config.get_value(
            "global", "merge-last-scene")
        self.frame_skip = self.config.get_value("global", "frame-skip", frame_skip)

        # Create StatsManager if --stats is specified.
        if stats_file:
            self.stats_file_path = get_and_create_path(stats_file, self.output_directory)
            self.stats_manager = StatsManager()

        # Initialize default detector with values in the config file.
        default_detector = self.config.get_value("global", "default-detector")
        if default_detector == 'scene_detect-adaptive':
            self.default_detector = (AdaptiveDetector, self.get_detect_adaptive_params())
        elif default_detector == 'scene_detect-content':
            self.default_detector = (ContentDetector, self.get_detect_content_params())
        elif default_detector == 'scene_detect-threshold':
            self.default_detector = (ThresholdDetector, self.get_detect_threshold_params())
        else:
            raise click.BadParameter("Unknown detector type!", param_hint='default-detector')

        logger.debug('Initializing SceneManager.')
        scene_manager = SceneManager(self.stats_manager)

        if downscale is None and self.config.is_default("global", "downscale"):
            scene_manager.auto_downscale = True
        else:
            scene_manager.auto_downscale = False
            downscale = self.config.get_value("global", "downscale", downscale)
            try:
                scene_manager.downscale = downscale
            except ValueError as ex:
                logger.debug(str(ex))
                raise click.BadParameter(str(ex), param_hint='downscale factor')
        scene_manager.interpolation = Interpolation[self.config.get_value(
            'global', 'downscale-method').upper()]
        self.scene_manager = scene_manager

    def get_detect_content_params(
        self,
        threshold: Optional[float] = None,
        luma_only: bool = None,
        min_scene_len: Optional[str] = None,
        weights: Optional[Tuple[float, float, float, float]] = None,
        kernel_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Handle scene_detect-content command options and return dict to construct one with."""
        self._ensure_input_open()

        if self.drop_short_scenes:
            min_scene_len = 0
        else:
            if min_scene_len is None:
                if self.config.is_default('scene_detect-content', 'min-scene-len'):
                    min_scene_len = self.min_scene_len.frame_num
                else:
                    min_scene_len = self.config.get_value('scene_detect-content', 'min-scene-len')
            min_scene_len = parse_timecode(min_scene_len, self.video_stream.frame_rate).frame_num

        if weights is not None:
            try:
                weights = ContentDetector.Components(*weights)
            except ValueError as ex:
                logger.debug(str(ex))
                raise click.BadParameter(str(ex), param_hint='weights')
        return {
            'weights': self.config.get_value('scene_detect-content', 'weights', weights),
            'kernel_size': self.config.get_value('scene_detect-content', 'kernel-size', kernel_size),
            'luma_only': luma_only or self.config.get_value('scene_detect-content', 'luma-only'),
            'min_scene_len': min_scene_len,
            'threshold': self.config.get_value('scene_detect-content', 'threshold', threshold),
        }

    def get_detect_adaptive_params(
        self,
        threshold: Optional[float] = None,
        min_content_val: Optional[float] = None,
        frame_window: Optional[int] = None,
        luma_only: bool = None,
        min_scene_len: Optional[str] = None,
        weights: Optional[Tuple[float, float, float, float]] = None,
        kernel_size: Optional[int] = None,
        min_delta_hsv: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Handle scene_detect-adaptive command options and return dict to construct one with."""
        self._ensure_input_open()

        # TODO(v0.7): Remove these branches when removing -d/--min-delta-hsv.
        if min_delta_hsv is not None:
            logger.error('-d/--min-delta-hsv is deprecated, use -c/--min-content-val instead.')
            if min_content_val is None:
                min_content_val = min_delta_hsv
        # Handle case where deprecated min-delta-hsv is set, and use it to set min-content-val.
        if not self.config.is_default("scene_detect-adaptive", "min-delta-hsv"):
            logger.error('[scene_detect-adaptive] config file option `min-delta-hsv` is deprecated'
                         ', use `min-delta-hsv` instead.')
            if self.config.is_default("scene_detect-adaptive", "min-content-val"):
                self.config.config_dict["scene_detect-adaptive"]["min-content-val"] = (
                    self.config.config_dict["scene_detect-adaptive"]["min-deleta-hsv"])

        if self.drop_short_scenes:
            min_scene_len = 0
        else:
            if min_scene_len is None:
                if self.config.is_default("scene_detect-adaptive", "min-scene-len"):
                    min_scene_len = self.min_scene_len.frame_num
                else:
                    min_scene_len = self.config.get_value("scene_detect-adaptive", "min-scene-len")
            min_scene_len = parse_timecode(min_scene_len, self.video_stream.frame_rate).frame_num

        if weights is not None:
            try:
                weights = ContentDetector.Components(*weights)
            except ValueError as ex:
                logger.debug(str(ex))
                raise click.BadParameter(str(ex), param_hint='weights')
        return {
            'adaptive_threshold':
                self.config.get_value("scene_detect-adaptive", "threshold", threshold),
            'weights':
                self.config.get_value("scene_detect-adaptive", "weights", weights),
            'kernel_size':
                self.config.get_value("scene_detect-adaptive", "kernel-size", kernel_size),
            'luma_only':
                luma_only or self.config.get_value("scene_detect-adaptive", "luma-only"),
            'min_content_val':
                self.config.get_value("scene_detect-adaptive", "min-content-val", min_content_val),
            'min_scene_len':
                min_scene_len,
            'window_width':
                self.config.get_value("scene_detect-adaptive", "frame-window", frame_window),
        }

    def get_detect_threshold_params(
        self,
        threshold: Optional[float] = None,
        fade_bias: Optional[float] = None,
        add_last_scene: bool = None,
        min_scene_len: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle scene_detect-threshold command options and return dict to construct one with."""
        self._ensure_input_open()

        if self.drop_short_scenes:
            min_scene_len = 0
        else:
            if min_scene_len is None:
                if self.config.is_default("scene_detect-threshold", "min-scene-len"):
                    min_scene_len = self.min_scene_len.frame_num
                else:
                    min_scene_len = self.config.get_value("scene_detect-threshold", "min-scene-len")
            min_scene_len = parse_timecode(min_scene_len, self.video_stream.frame_rate).frame_num

        return {
                                                                                                # TODO(v1.0): add_last_scene cannot be disabled right now.
            'add_final_scene':
                add_last_scene or self.config.get_value("scene_detect-threshold", "add-last-scene"),
            'fade_bias':
                self.config.get_value("scene_detect-threshold", "fade-bias", fade_bias),
            'min_scene_len':
                min_scene_len,
            'threshold':
                self.config.get_value("scene_detect-threshold", "threshold", threshold),
        }

    def handle_load_scenes(self, input: AnyStr, start_col_name: Optional[str]):
        """Handle `load-scenes` command options."""
        self._ensure_input_open()
        start_col_name = self.config.get_value("load-scenes", "start-col-name", start_col_name)
        self.add_detector(
            SceneLoader(
                file=input, framerate=self.video_stream.frame_rate, start_col_name=start_col_name))

    def handle_export_html(
        self,
        filename: Optional[AnyStr],
        no_images: bool,
        image_width: Optional[int],
        image_height: Optional[int],
    ):
        """Handle `export-html` command options."""
        self._ensure_input_open()
        if self.export_html:
            self._on_duplicate_command('export_html')

        no_images = no_images or self.config.get_value('export-html', 'no-images')
        self.html_include_images = not no_images

        self.html_name_format = self.config.get_value('export-html', 'filename', filename)
        self.image_width = self.config.get_value('export-html', 'image-width', image_width)
        self.image_height = self.config.get_value('export-html', 'image-height', image_height)

        if not self.save_images and not no_images:
            raise click.BadArgumentUsage(
                'The export-html command requires that the save-images command\n'
                'is specified before it, unless --no-images is specified.')
        logger.info('HTML file name format:\n %s', filename)

        self.export_html = True

    def handle_list_scenes(
        self,
        output: Optional[AnyStr],
        filename: Optional[AnyStr],
        no_output_file: bool,
        quiet: bool,
        skip_cuts: bool,
    ):
        """Handle `list-scenes` command options."""
        self._ensure_input_open()
        if self.list_scenes:
            self._on_duplicate_command('list-scenes')

        self.skip_cuts = skip_cuts or self.config.get_value('list-scenes', 'skip-cuts')
        self.print_scene_list = not (quiet or self.config.get_value('list-scenes', 'quiet'))
        no_output_file = no_output_file or self.config.get_value('list-scenes', 'no-output-file')

        self.scene_list_directory = self.config.get_value(
            'list-scenes', 'output', output, ignore_default=True)
        self.scene_list_name_format = self.config.get_value('list-scenes', 'filename', filename)
        if self.scene_list_name_format is not None and not no_output_file:
            logger.info('Scene list filename format:\n  %s', self.scene_list_name_format)
        self.scene_list_output = not no_output_file
        if self.scene_list_directory is not None:
            logger.info('Scene list output directory:\n  %s', self.scene_list_directory)

        self.list_scenes = True

    def handle_split_video(
        self,
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
        """Handle `split-video` command options."""
        self._ensure_input_open()
        if self.split_video:
            self._on_duplicate_command('split-video')

        check_split_video_requirements(use_mkvmerge=mkvmerge)

        if contains_sequence_or_url(self.video_stream.path):
            error_str = 'The split-video command is incompatible with image sequences/URLs.'
            raise click.BadParameter(error_str, param_hint='split-video')

        ##
        ## Common Arguments/Options
        ##

        self.split_video = True
        self.split_quiet = quiet or self.config.get_value('split-video', 'quiet')
        self.split_directory = self.config.get_value(
            'split-video', 'output', output, ignore_default=True)
        if self.split_directory is not None:
            logger.info('Video output path set:  \n%s', self.split_directory)
        self.split_name_format = self.config.get_value('split-video', 'filename', filename)

        # We only load the config values for these flags/options if none of the other
        # encoder flags/options were set via the CLI to avoid any conflicting options
        # (e.g. if the config file sets `high-quality = yes` but `--copy` is specified).
        if not (mkvmerge or copy or high_quality or args or rate_factor or preset):
            mkvmerge = self.config.get_value('split-video', 'mkvmerge')
            copy = self.config.get_value('split-video', 'copy')
            high_quality = self.config.get_value('split-video', 'high-quality')
            rate_factor = self.config.get_value('split-video', 'rate-factor')
            preset = self.config.get_value('split-video', 'preset')
            args = self.config.get_value('split-video', 'args')

        # Disallow certain combinations of flags/options.
        if mkvmerge or copy:
            command = 'mkvmerge (-m)' if mkvmerge else 'copy (-c)'
            if high_quality:
                raise click.BadParameter(
                    'high-quality (-hq) cannot be used with %s' % (command),
                    param_hint='split-video')
            if args:
                raise click.BadParameter(
                    'args (-a) cannot be used with %s' % (command), param_hint='split-video')
            if rate_factor:
                raise click.BadParameter(
                    'rate-factor (crf) cannot be used with %s' % (command),
                    param_hint='split-video')
            if preset:
                raise click.BadParameter(
                    'preset (-p) cannot be used with %s' % (command), param_hint='split-video')

        ##
        ## mkvmerge-Specific Arguments/Options
        ##
        if mkvmerge:
            if copy:
                logger.warning('copy mode (-c) ignored due to mkvmerge mode (-m).')
            self.split_mkvmerge = True
            logger.info('Using mkvmerge for video splitting.')
            return

        ##
        ## ffmpeg-Specific Arguments/Options
        ##
        if copy:
            args = '-map 0 -c:v copy -c:a copy'
        elif not args:
            if rate_factor is None:
                rate_factor = 22 if not high_quality else 17
            if preset is None:
                preset = 'veryfast' if not high_quality else 'slow'
            args = ('-map 0 -c:v libx264 -preset {PRESET} -crf {RATE_FACTOR} -c:a aac'.format(
                PRESET=preset, RATE_FACTOR=rate_factor))

        logger.info('ffmpeg arguments: %s', args)
        self.split_args = args
        if filename:
            logger.info('Output file name format: %s', filename)

    def handle_save_images(
        self,
        num_images: Optional[int],
        output: Optional[AnyStr],
        filename: Optional[AnyStr],
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
        """Handle `save-images` command options."""
        self._ensure_input_open()
        if self.save_images:
            self._on_duplicate_command('save-images')

        if '://' in self.video_stream.path:
            error_str = '\nThe save-images command is incompatible with URLs.'
            logger.error(error_str)
            raise click.BadParameter(error_str, param_hint='save-images')

        num_flags = sum([1 if flag else 0 for flag in [jpeg, webp, png]])
        if num_flags > 1:
            logger.error('Multiple image type flags set for save-images command.')
            raise click.BadParameter(
                'Only one image type (JPG/PNG/WEBP) can be specified.', param_hint='save-images')
        # Only use config params for image format if one wasn't specified.
        elif num_flags == 0:
            image_format = self.config.get_value('save-images', 'format').lower()
            jpeg = image_format == 'jpeg'
            webp = image_format == 'webp'
            png = image_format == 'png'

        # Only use config params for scale/height/width if none of them are specified explicitly.
        if scale is None and height is None and width is None:
            self.scale = self.config.get_value('save-images', 'scale')
            self.height = self.config.get_value('save-images', 'height')
            self.width = self.config.get_value('save-images', 'width')
        else:
            self.scale = scale
            self.height = height
            self.width = width

        self.scale_method = Interpolation[self.config.get_value('save-images',
                                                                'scale-method').upper()]

        default_quality = 100 if webp else 95
        quality = (
            default_quality if self.config.is_default('save-images', 'quality') else
            self.config.get_value('save-images', 'quality'))

        compression = self.config.get_value('save-images', 'compression', compression)
        self.image_param = compression if png else quality

        self.image_extension = 'jpg' if jpeg else 'png' if png else 'webp'
        valid_params = get_cv2_imwrite_params()
        if not self.image_extension in valid_params or valid_params[self.image_extension] is None:
            error_strs = [
                'Image encoder type `%s` not supported.' % self.image_extension.upper(),
                'The specified encoder type could not be found in the current OpenCV module.',
                'To enable this output format, please update the installed version of OpenCV.',
                'If you build OpenCV, ensure the the proper dependencies are enabled. '
            ]
            logger.debug('\n'.join(error_strs))
            raise click.BadParameter('\n'.join(error_strs), param_hint='save-images')

        self.image_directory = self.config.get_value(
            'save-images', 'output', output, ignore_default=True)

        self.image_name_format = self.config.get_value('save-images', 'filename', filename)
        self.num_images = self.config.get_value('save-images', 'num-images', num_images)
        self.frame_margin = self.config.get_value('save-images', 'frame-margin', frame_margin)

        image_type = ('jpeg' if jpeg else self.image_extension).upper()
        image_param_type = 'Compression' if png else 'Quality'
        image_param_type = ' [%s: %d]' % (image_param_type, self.image_param)
        logger.info('Image output format set: %s%s', image_type, image_param_type)
        if self.image_directory is not None:
            logger.info('Image output directory set:\n  %s', os.path.abspath(self.image_directory))

        self.save_images = True

    def handle_time(self, start, duration, end):
        """Handle `time` command options."""
        self._ensure_input_open()
        if self.time:
            self._on_duplicate_command('time')

        if duration is not None and end is not None:
            raise click.BadParameter(
                'Only one of --duration/-d or --end/-e can be specified, not both.',
                param_hint='time')

        logger.debug('Setting video time:\n    start: %s, duration: %s, end: %s', start, duration,
                     end)

        self.start_time = parse_timecode(
            start, self.video_stream.frame_rate, first_index_is_one=True)
        self.end_time = parse_timecode(end, self.video_stream.frame_rate, first_index_is_one=True)
        self.duration = parse_timecode(
            duration, self.video_stream.frame_rate, first_index_is_one=True)
        self.time = True

    #
    # Private Methods
    #

    def _initialize_logging(
        self,
        quiet: Optional[bool] = None,
        verbosity: Optional[str] = None,
        logfile: Optional[AnyStr] = None,
    ):
        """Setup logging based on CLI args and user configuration settings."""
        if quiet is not None:
            self.quiet_mode = bool(quiet)
        curr_verbosity = logging.INFO
        # Convert verbosity into it's log level enum, and override quiet mode if set.
        if verbosity is not None:
            assert verbosity in CHOICE_MAP['global']['verbosity']
            if verbosity.lower() == 'none':
                self.quiet_mode = True
                verbosity = 'info'
            else:
                # Override quiet mode if verbosity is set.
                self.quiet_mode = False
            curr_verbosity = getattr(logging, verbosity.upper())
        else:
            verbosity_str = USER_CONFIG.get_value('global', 'verbosity')
            assert verbosity_str in CHOICE_MAP['global']['verbosity']
            if verbosity_str.lower() == 'none':
                self.quiet_mode = True
            else:
                curr_verbosity = getattr(logging, verbosity_str.upper())
                # Override quiet mode if verbosity is set.
                if not USER_CONFIG.is_default('global', 'verbosity'):
                    self.quiet_mode = False
        # Initialize logger with the set CLI args / user configuration.
        init_logger(log_level=curr_verbosity, show_stdout=not self.quiet_mode, log_file=logfile)

    def add_detector(self, detector):
        """ Add Detector: Adds a detection algorithm to the CliContext's SceneManager. """
        self._ensure_input_open()
        try:
            self.scene_manager.add_detector(detector)
        except scenedetect.stats_manager.FrameMetricRegistered as ex:
            raise click.BadParameter(
                message='Cannot specify detection algorithm twice.',
                param_hint=detector.cli_name) from ex

    def _ensure_input_open(self) -> None:
        """Ensure self.video_stream was initialized (i.e. -i/--input was specified),
        otherwise raises an exception. Should only be used from commands that require an
        input video to process the options (e.g. those that require a timecode).

        Raises:
            click.BadParameter: self.video_stream was not initialized.
        """
        if self.video_stream is None:
            raise click.ClickException('No input video (-i/--input) was specified.')

    def _open_video_stream(self, input_path: AnyStr, framerate: Optional[float],
                           backend: Optional[str]):
        if '%' in input_path and backend != 'opencv':
            raise click.BadParameter(
                'The OpenCV backend (`--backend opencv`) must be used to process image sequences.',
                param_hint='-i/--input')
        if framerate is not None and framerate < MAX_FPS_DELTA:
            raise click.BadParameter('Invalid framerate specified!', param_hint='-f/--framerate')
        try:
            if backend is None:
                backend = self.config.get_value('global', 'backend')
            else:
                if not backend in AVAILABLE_BACKENDS:
                    raise click.BadParameter(
                        'Specified backend %s is not available on this system!' % backend,
                        param_hint='-b/--backend')
            # Open the video with the specified backend, loading any required config settings.
            if backend == 'pyav':
                self.video_stream = open_video(
                    path=input_path,
                    framerate=framerate,
                    backend=backend,
                    threading_mode=self.config.get_value('backend-pyav', 'threading-mode'),
                    suppress_output=self.config.get_value('backend-pyav', 'suppress-output'),
                )
            elif backend == 'opencv':
                self.video_stream = open_video(
                    path=input_path,
                    framerate=framerate,
                    backend=backend,
                    max_decode_attempts=self.config.get_value('backend-opencv',
                                                              'max-decode-attempts'),
                )
            # Handle backends without any config options.
            else:
                self.video_stream = open_video(
                    path=input_path,
                    framerate=framerate,
                    backend=backend,
                )
            logger.debug('Video opened using backend %s', type(self.video_stream).__name__)
        except FrameRateUnavailable as ex:
            raise click.BadParameter(
                'Failed to obtain framerate for input video. Manually specify framerate with the'
                ' -f/--framerate option, or try re-encoding the file.',
                param_hint='-i/--input') from ex
        except VideoOpenFailure as ex:
            raise click.BadParameter(
                'Failed to open input video%s: %s' %
                (' using %s backend' % backend if backend else '', str(ex)),
                param_hint='-i/--input') from ex
        except OSError as ex:
            raise click.BadParameter('Input error:\n\n\t%s\n' % str(ex), param_hint='-i/--input')

    def _on_duplicate_command(self, command: str) -> None:
        """Called when a command is duplicated to stop parsing and raise an error.

        Arguments:
            command: Command that was duplicated for error context.

        Raises:
            click.BadParameter
        """
        error_strs = []
        error_strs.append('Error: Command %s specified multiple times.' % command)
        error_strs.append('The %s command may appear only one time.')

        logger.error('\n'.join(error_strs))
        raise click.BadParameter(
            '\n  Command %s may only be specified once.' % command,
            param_hint='%s command' % command)
