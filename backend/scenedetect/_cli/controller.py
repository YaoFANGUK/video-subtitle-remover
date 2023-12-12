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
"""Logic for the PySceneDetect command."""

import logging
import os
from string import Template
import time
from typing import Dict, List, Tuple, Optional
from string import Template

from scenedetect.detectors import AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.platform import get_and_create_path, get_file_name
from scenedetect.scene_manager import save_images, write_scene_list, write_scene_list_html
from scenedetect.video_splitter import split_video_mkvmerge, split_video_ffmpeg
from scenedetect.video_stream import SeekError

from scenedetect._cli.context import CliContext, check_split_video_requirements

logger = logging.getLogger('pyscenedetect')


def run_scenedetect(context: CliContext):
    """Perform main CLI application control logic. Run once all command-line options and
    configuration file options have been validated.

    Arguments:
        context: Prevalidated command-line option context to use for processing.
    """
    # No input may have been specified depending on the commands/args that were used.
    logger.debug("Running controller.")
    if context.scene_manager is None:
        logger.debug("No input specified.")
        return
    # Use default detector if one was not specified.
    if context.scene_manager.get_num_detectors() == 0:
        detector_type, detector_args = context.default_detector
        logger.debug('Using default detector: %s(%s)' % (detector_type.__name__, detector_args))
        context.scene_manager.add_detector(detector_type(**detector_args))

    perf_start_time = time.time()
    if context.start_time is not None:
        logger.debug('Seeking to start time...')
        try:
            context.video_stream.seek(target=context.start_time)
        except SeekError as ex:
            logging.critical('Failed to seek to %s / frame %d: %s',
                             context.start_time.get_timecode(), context.start_time.get_frames(),
                             str(ex))
            return

    num_frames = context.scene_manager.detect_scenes(
        video=context.video_stream,
        duration=context.duration,
        end_time=context.end_time,
        frame_skip=context.frame_skip,
        show_progress=not context.quiet_mode)

    # Handle case where video failure is most likely due to multiple audio tracks (#179).
    if num_frames <= 0 and context.video_stream.BACKEND_NAME == 'opencv':
        logger.critical(
            'Failed to read any frames from video file. This could be caused by the video'
            ' having multiple audio tracks. If so, try installing the PyAV backend:\n'
            '      pip install av\n'
            'Or remove the audio tracks by running either:\n'
            '      ffmpeg -i input.mp4 -c copy -an output.mp4\n'
            '      mkvmerge -o output.mkv input.mp4\n'
            'For details, see https://scenedetect.com/faq/')
        return

    perf_duration = time.time() - perf_start_time
    logger.info('Processed %d frames in %.1f seconds (average %.2f FPS).', num_frames,
                perf_duration,
                float(num_frames) / perf_duration)

    # Handle -s/--stats option.
    _save_stats(context)

    # Get list of detected cuts/scenes from the SceneManager to generate the required output
    # files, based on the given commands (list-scenes, split-video, save-images, etc...).
    cut_list = context.scene_manager.get_cut_list(show_warning=False)
    scene_list = context.scene_manager.get_scene_list(start_in_scene=True)

    # Handle --merge-last-scene.
    if context.merge_last_scene and context.min_scene_len is not None and context.min_scene_len > 0:
        if len(scene_list) > 1 and (scene_list[-1][1] - scene_list[-1][0]) < context.min_scene_len:
            new_last_scene = (scene_list[-2][0], scene_list[-1][1])
            scene_list = scene_list[:-2] + [new_last_scene]

    # Handle --drop-short-scenes.
    if context.drop_short_scenes and context.min_scene_len > 0:
        scene_list = [s for s in scene_list if (s[1] - s[0]) >= context.min_scene_len]

    # Ensure we don't divide by zero.
    if scene_list:
        logger.info(
            'Detected %d scenes, average shot length %.1f seconds.', len(scene_list),
            sum([(end_time - start_time).get_seconds() for start_time, end_time in scene_list]) /
            float(len(scene_list)))
    else:
        logger.info('No scenes detected.')

    # Handle list-scenes command.
    _list_scenes(context, scene_list, cut_list)

    # Handle save-images command.
    image_filenames = _save_images(context, scene_list)

    # Handle export-html command.
    _export_html(context, scene_list, cut_list, image_filenames)

    # Handle split-video command.
    _split_video(context, scene_list)


def _save_stats(context: CliContext) -> None:
    """Handles saving the statsfile if -s/--stats was specified."""
    if context.stats_file_path is not None:
        # We check if the save is required in order to reduce unnecessary log messages.
        if context.stats_manager.is_save_required():
            logger.info('Saving frame metrics to stats file: %s',
                        os.path.basename(context.stats_file_path))
            context.stats_manager.save_to_csv(csv_file=context.stats_file_path)
        else:
            logger.debug('No frame metrics updated, skipping update of the stats file.')


def _list_scenes(context: CliContext, scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                 cut_list: List[FrameTimecode]) -> None:
    """Handles the `list-scenes` command."""
    if context.scene_list_output:
        scene_list_filename = Template(
            context.scene_list_name_format).safe_substitute(VIDEO_NAME=context.video_stream.name)
        if not scene_list_filename.lower().endswith('.csv'):
            scene_list_filename += '.csv'
        scene_list_path = get_and_create_path(
            scene_list_filename, context.scene_list_directory
            if context.scene_list_directory is not None else context.output_directory)
        logger.info('Writing scene list to CSV file:\n  %s', scene_list_path)
        with open(scene_list_path, 'wt') as scene_list_file:
            write_scene_list(
                output_csv_file=scene_list_file,
                scene_list=scene_list,
                include_cut_list=not context.skip_cuts,
                cut_list=cut_list)

    if context.print_scene_list:
        logger.info(
            """Scene List:
-----------------------------------------------------------------------
 | Scene # | Start Frame |  Start Time  |  End Frame  |   End Time   |
-----------------------------------------------------------------------
%s
-----------------------------------------------------------------------
""", '\n'.join([
                ' |  %5d  | %11d | %s | %11d | %s |' %
                (i + 1, start_time.get_frames() + 1, start_time.get_timecode(),
                 end_time.get_frames(), end_time.get_timecode())
                for i, (start_time, end_time) in enumerate(scene_list)
            ]))

    if cut_list:
        logger.info('Comma-separated timecode list:\n  %s',
                    ','.join([cut.get_timecode() for cut in cut_list]))


def _save_images(
        context: CliContext,
        scene_list: List[Tuple[FrameTimecode, FrameTimecode]]) -> Optional[Dict[int, List[str]]]:
    """Handles the `save-images` command."""
    if not context.save_images:
        return None

    image_output_dir = context.output_directory
    if context.image_directory is not None:
        image_output_dir = context.image_directory

    return save_images(
        scene_list=scene_list,
        video=context.video_stream,
        num_images=context.num_images,
        frame_margin=context.frame_margin,
        image_extension=context.image_extension,
        encoder_param=context.image_param,
        image_name_template=context.image_name_format,
        output_dir=image_output_dir,
        show_progress=not context.quiet_mode,
        scale=context.scale,
        height=context.height,
        width=context.width,
        interpolation=context.scale_method)


def _export_html(context: CliContext, scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                 cut_list: List[FrameTimecode], image_filenames: Optional[Dict[int,
                                                                               List[str]]]) -> None:
    """Handles the `export-html` command."""
    if not context.export_html:
        return

    html_filename = Template(
        context.html_name_format).safe_substitute(VIDEO_NAME=context.video_stream.name)
    if not html_filename.lower().endswith('.html'):
        html_filename += '.html'
    html_path = get_and_create_path(
        html_filename, context.image_directory
        if context.image_directory is not None else context.output_directory)
    logger.info('Exporting to html file:\n %s:', html_path)
    if not context.html_include_images:
        image_filenames = None
    write_scene_list_html(
        html_path,
        scene_list,
        cut_list,
        image_filenames=image_filenames,
        image_width=context.image_width,
        image_height=context.image_height)


def _split_video(context: CliContext, scene_list: List[Tuple[FrameTimecode,
                                                             FrameTimecode]]) -> None:
    """Handles the `split-video` command."""
    if not context.split_video:
        return

    output_path_template = context.split_name_format
    # Add proper extension to filename template if required.
    dot_pos = output_path_template.rfind('.')
    extension_length = 0 if dot_pos < 0 else len(output_path_template) - (dot_pos + 1)
    # If using mkvmerge, force extension to .mkv.
    if context.split_mkvmerge and not output_path_template.endswith('.mkv'):
        output_path_template += '.mkv'
    # Otherwise, if using ffmpeg, only add an extension if one doesn't exist.
    elif not 2 <= extension_length <= 4:
        output_path_template += '.mp4'
    # Pre-expand $VIDEO_NAME so it can be used for a directory.
    # TODO: Do this elsewhere in a future version for all output options.
    output_path_template = Template(output_path_template).safe_substitute(
        VIDEO_NAME=get_file_name(context.video_stream.path, include_extension=False))
    output_path_template = get_and_create_path(
        output_path_template, context.split_directory
        if context.split_directory is not None else context.output_directory)
    # Ensure the appropriate tool is available before handling split-video.
    check_split_video_requirements(context.split_mkvmerge)
    if context.split_mkvmerge:
        split_video_mkvmerge(
            input_video_path=context.video_stream.path,
            scene_list=scene_list,
            output_file_template=output_path_template,
            show_output=not (context.quiet_mode or context.split_quiet),
        )
    else:
        split_video_ffmpeg(
            input_video_path=context.video_stream.path,
            scene_list=scene_list,
            output_file_template=output_path_template,
            arg_override=context.split_args,
            show_progress=not context.quiet_mode,
            show_output=not (context.quiet_mode or context.split_quiet),
        )
    if scene_list:
        logger.info('Video splitting completed, scenes written to disk.')
