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
"""``scenedetect.scene_manager`` Module

This module implements :class:`SceneManager`, coordinates running a
:mod:`SceneDetector <scenedetect.detectors>` over the frames of a video
(:mod:`VideoStream <scenedetect.video_stream>`). Video decoding is done in a separate thread to
improve performance.

This module also contains other helper functions (e.g. :func:`save_images`) which can be used to
process the resulting scene list.

===============================================================
Usage
===============================================================

The following example shows basic usage of a :class:`SceneManager`:

.. code:: python

    from scenedetect import open_video, SceneManager, ContentDetector
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    scenes = scene_manager.get_scene_list()

An optional callback can also be invoked on each detected scene, for example:

.. code:: python

    from scenedetect import open_video, SceneManager, ContentDetector

    # Callback to invoke on the first frame of every new scene detection.
    def on_new_scene(frame_img: numpy.ndarray, frame_num: int):
        print("New scene found at frame %d." % frame_num)

    video = open_video(test_video_file)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video, callback=on_new_scene)

To use a `SceneManager` with a webcam/device or existing `cv2.VideoCapture` device, use the
:class:`VideoCaptureAdapter <scenedetect.backends.opencv.VideoCaptureAdapter>` instead of
`open_video`.

=======================================================================
Storing Per-Frame Statistics
=======================================================================

`SceneManager` can use an optional
:class:`StatsManager <scenedetect.stats_manager.StatsManager>` to save frame statistics to disk:

.. code:: python

    from scenedetect import open_video, ContentDetector, SceneManager, StatsManager
    video = open_video(test_video_file)
    scene_manager = SceneManager(stats_manager=StatsManager())
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    print_scenes(scene_list=scene_list)
    # Save per-frame statistics to disk.
    scene_manager.stats_manager.save_to_csv(csv_file=STATS_FILE_PATH)

The statsfile can be used to find a better threshold for certain inputs, or perform statistical
analysis of the video.
"""

import csv
from enum import Enum
from typing import Iterable, List, Tuple, Optional, Dict, Callable, Union, TextIO
import threading
import queue
import logging
import math
import sys

import cv2
import numpy as np
from backend.scenedetect._thirdparty.simpletable import (SimpleTableCell, SimpleTableImage, SimpleTableRow,
                                                 SimpleTable, HTMLPage)

from backend.scenedetect.platform import (tqdm, get_and_create_path, get_cv2_imwrite_params, Template)
from backend.scenedetect.frame_timecode import FrameTimecode
from backend.scenedetect.video_stream import VideoStream
from backend.scenedetect.scene_detector import SceneDetector, SparseSceneDetector
from backend.scenedetect.stats_manager import StatsManager, FrameMetricRegistered

logger = logging.getLogger('pyscenedetect')

# TODO: This value can and should be tuned for performance improvements as much as possible,
# until accuracy falls, on a large enough dataset. This has yet to be done, but the current
# value doesn't seem to have caused any issues at least.
DEFAULT_MIN_WIDTH: int = 256
"""The default minimum width a frame will be downscaled to when calculating a downscale factor."""

MAX_FRAME_QUEUE_LENGTH: int = 4
"""Maximum number of decoded frames which can be buffered while waiting to be processed."""

PROGRESS_BAR_DESCRIPTION = 'Detected: %d | Progress'
"""Template to use for progress bar."""


class Interpolation(Enum):
    """Interpolation method used for image resizing. Based on constants defined in OpenCV."""
    NEAREST = cv2.INTER_NEAREST
    """Nearest neighbor interpolation."""
    LINEAR = cv2.INTER_LINEAR
    """Bilinear interpolation."""
    CUBIC = cv2.INTER_CUBIC
    """Bicubic interpolation."""
    AREA = cv2.INTER_AREA
    """Pixel area relation resampling. Provides moire'-free downscaling."""
    LANCZOS4 = cv2.INTER_LANCZOS4
    """Lanczos interpolation over 8x8 neighborhood."""


def compute_downscale_factor(frame_width: int, effective_width: int = DEFAULT_MIN_WIDTH) -> int:
    """Get the optimal default downscale factor based on a video's resolution (currently only
    the width in pixels is considered).

    The resulting effective width of the video will be between frame_width and 1.5 * frame_width
    pixels (e.g. if frame_width is 200, the range of effective widths will be between 200 and 300).

    Arguments:
        frame_width: Actual width of the video frame in pixels.
        effective_width: Desired minimum width in pixels.

    Returns:
        int: The default downscale factor to use to achieve at least the target effective_width.
    """
    assert not (frame_width < 1 or effective_width < 1)
    if frame_width < effective_width:
        return 1
    return frame_width // effective_width


def get_scenes_from_cuts(
    cut_list: Iterable[FrameTimecode],
    start_pos: Union[int, FrameTimecode],
    end_pos: Union[int, FrameTimecode],
    base_timecode: Optional[FrameTimecode] = None,
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Returns a list of tuples of start/end FrameTimecodes for each scene based on a
    list of detected scene cuts/breaks.

    This function is called when using the :meth:`SceneManager.get_scene_list` method.
    The scene list is generated from a cutting list (:meth:`SceneManager.get_cut_list`),
    noting that each scene is contiguous, starting from the first to last frame of the input.
    If `cut_list` is empty, the resulting scene will span from `start_pos` to `end_pos`.

    Arguments:
        cut_list: List of FrameTimecode objects where scene cuts/breaks occur.
        base_timecode: The base_timecode of which all FrameTimecodes in the cut_list are based on.
        num_frames: The number of frames, or FrameTimecode representing duration, of the video that
            was processed (used to generate last scene's end time).
        start_frame: The start frame or FrameTimecode of the cut list. Used to generate the first
            scene's start time.
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.
    Returns:
        List of tuples in the form (start_time, end_time), where both start_time and
        end_time are FrameTimecode objects representing the exact time/frame where each
        scene occupies based on the input cut_list.
    """
    # TODO(v0.7): Use the warnings module to turn this into a warning.
    if base_timecode is not None:
        logger.error('`base_timecode` argument is deprecated has no effect.')

    # Scene list, where scenes are tuples of (Start FrameTimecode, End FrameTimecode).
    scene_list = []
    if not cut_list:
        scene_list.append((start_pos, end_pos))
        return scene_list
    # Initialize last_cut to the first frame we processed,as it will be
    # the start timecode for the first scene in the list.
    last_cut = start_pos
    for cut in cut_list:
        scene_list.append((last_cut, cut))
        last_cut = cut
    # Last scene is from last cut to end of video.
    scene_list.append((last_cut, end_pos))

    return scene_list


def write_scene_list(output_csv_file: TextIO,
                     scene_list: Iterable[Tuple[FrameTimecode, FrameTimecode]],
                     include_cut_list: bool = True,
                     cut_list: Optional[Iterable[FrameTimecode]] = None) -> None:
    """Writes the given list of scenes to an output file handle in CSV format.

    Arguments:
        output_csv_file: Handle to open file in write mode.
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        include_cut_list: Bool indicating if the first row should include the timecodes where
            each scene starts. Should be set to False if RFC 4180 compliant CSV output is required.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not specified,
            the cut list is generated using the start times of each scene following the first one.
    """
    csv_writer = csv.writer(output_csv_file, lineterminator='\n')
    # If required, output the cutting list as the first row (i.e. before the header row).
    if include_cut_list:
        csv_writer.writerow(
            ["Timecode List:"] +
            cut_list if cut_list else [start.get_timecode() for start, _ in scene_list[1:]])
    csv_writer.writerow([
        "Scene Number", "Start Frame", "Start Timecode", "Start Time (seconds)", "End Frame",
        "End Timecode", "End Time (seconds)", "Length (frames)", "Length (timecode)",
        "Length (seconds)"
    ])
    for i, (start, end) in enumerate(scene_list):
        duration = end - start
        csv_writer.writerow([
            '%d' % (i + 1),
            '%d' % (start.get_frames() + 1),
            start.get_timecode(),
            '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(),
            end.get_timecode(),
            '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(),
            duration.get_timecode(),
            '%.3f' % duration.get_seconds()
        ])


def write_scene_list_html(output_html_filename,
                          scene_list,
                          cut_list=None,
                          css=None,
                          css_class='mytable',
                          image_filenames=None,
                          image_width=None,
                          image_height=None):
    """Writes the given list of scenes to an output file handle in html format.

    Arguments:
        output_html_filename: filename of output html file
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not passed,
            the start times of each scene (besides the 0th scene) is used instead.
        css: String containing all the css information for the resulting html page.
        css_class: String containing the named css class
        image_filenames: dict where key i contains a list with n elements (filenames of
            the n saved images from that scene)
        image_width: Optional desired width of images in table in pixels
        image_height: Optional desired height of images in table in pixels
    """
    if not css:
        css = """
        table.mytable {
            font-family: times;
            font-size:12px;
            color:#000000;
            border-width: 1px;
            border-color: #eeeeee;
            border-collapse: collapse;
            background-color: #ffffff;
            width=100%;
            max-width:550px;
            table-layout:fixed;
        }
        table.mytable th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
            background-color: #e6eed6;
            color:#000000;
        }
        table.mytable td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
        }
        #code {
            display:inline;
            font-family: courier;
            color: #3d9400;
        }
        #string {
            display:inline;
            font-weight: bold;
        }
        """

    # Output Timecode list
    timecode_table = SimpleTable(
        [["Timecode List:"] +
         (cut_list if cut_list else [start.get_timecode() for start, _ in scene_list[1:]])],
        css_class=css_class)

    # Output list of scenes
    header_row = [
        "Scene Number", "Start Frame", "Start Timecode", "Start Time (seconds)", "End Frame",
        "End Timecode", "End Time (seconds)", "Length (frames)", "Length (timecode)",
        "Length (seconds)"
    ]
    for i, (start, end) in enumerate(scene_list):
        duration = end - start

        row = SimpleTableRow([
            '%d' % (i + 1),
            '%d' % (start.get_frames() + 1),
            start.get_timecode(),
            '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(),
            end.get_timecode(),
            '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(),
            duration.get_timecode(),
            '%.3f' % duration.get_seconds()
        ])

        if image_filenames:
            for image in image_filenames[i]:
                row.add_cell(
                    SimpleTableCell(
                        SimpleTableImage(image, width=image_width, height=image_height)))

        if i == 0:
            scene_table = SimpleTable(rows=[row], header_row=header_row, css_class=css_class)
        else:
            scene_table.add_row(row=row)

    # Write html file
    page = HTMLPage()
    page.add_table(timecode_table)
    page.add_table(scene_table)
    page.css = css
    page.save(output_html_filename)


#
# TODO(v1.0): Refactor to take a SceneList object; consider moving this and save scene list
# to a better spot, or just move them to scene_list.py.
#
def save_images(scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                video: VideoStream,
                num_images: int = 3,
                frame_margin: int = 1,
                image_extension: str = 'jpg',
                encoder_param: int = 95,
                image_name_template: str = '$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
                output_dir: Optional[str] = None,
                show_progress: Optional[bool] = False,
                scale: Optional[float] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                interpolation: Interpolation = Interpolation.CUBIC,
                video_manager=None) -> Dict[int, List[str]]:
    """Save a set number of images from each scene, given a list of scenes
    and the associated video/frame source.

    Arguments:
        scene_list: A list of scenes (pairs of FrameTimecode objects) returned
            from calling a SceneManager's detect_scenes() method.
        video: A VideoStream object corresponding to the scene list.
            Note that the video will be closed/re-opened and seeked through.
        num_images: Number of images to generate for each scene.  Minimum is 1.
        frame_margin: Number of frames to pad each scene around the beginning
            and end (e.g. moves the first/last image into the scene by N frames).
            Can set to 0, but will result in some video files failing to extract
            the very last frame.
        image_extension: Type of image to save (must be one of 'jpg', 'png', or 'webp').
        encoder_param: Quality/compression efficiency, based on type of image:
            'jpg' / 'webp':  Quality 0-100, higher is better quality.  100 is lossless for webp.
            'png': Compression from 1-9, where 9 achieves best filesize but is slower to encode.
        image_name_template: Template to use when creating the images on disk. Can
            use the macros $VIDEO_NAME, $SCENE_NUMBER, and $IMAGE_NUMBER. The image
            extension is applied automatically as per the argument image_extension.
        output_dir: Directory to output the images into.  If not set, the output
            is created in the working directory.
        show_progress: If True, shows a progress bar if tqdm is installed.
        scale: Optional factor by which to rescale saved images. A scaling factor of 1 would
            not result in rescaling. A value < 1 results in a smaller saved image, while a
            value > 1 results in an image larger than the original. This value is ignored if
            either the height or width values are specified.
        height: Optional value for the height of the saved images. Specifying both the height
            and width will resize images to an exact size, regardless of aspect ratio.
            Specifying only height will rescale the image to that number of pixels in height
            while preserving the aspect ratio.
        width: Optional value for the width of the saved images. Specifying both the width
            and height will resize images to an exact size, regardless of aspect ratio.
            Specifying only width will rescale the image to that number of pixels wide
            while preserving the aspect ratio.
        interpolation: Type of interpolation to use when resizing images.
        video_manager: [DEPRECATED] DO NOT USE. For backwards compatibility only.

    Returns:
        Dictionary of the format { scene_num : [image_paths] }, where scene_num is the
        number of the scene in scene_list (starting from 1), and image_paths is a list of
        the paths to the newly saved/created images.

    Raises:
        ValueError: Raised if any arguments are invalid or out of range (e.g.
        if num_images is negative).
    """
    # TODO(v0.7): Add DeprecationWarning that `video_manager` will be removed in v0.8.
    if video_manager is not None:
        logger.error('`video_manager` argument is deprecated, use `video` instead.')
        video = video_manager

    if not scene_list:
        return {}
    if num_images <= 0 or frame_margin < 0:
        raise ValueError()

    # TODO: Validate that encoder_param is within the proper range.
    # Should be between 0 and 100 (inclusive) for jpg/webp, and 1-9 for png.
    imwrite_param = [get_cv2_imwrite_params()[image_extension], encoder_param
                    ] if encoder_param is not None else []

    video.reset()

    # Setup flags and init progress bar if available.
    completed = True
    logger.info('Generating output images (%d per scene)...', num_images)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=len(scene_list) * num_images, unit='images', dynamic_ncols=True)

    filename_template = Template(image_name_template)

    scene_num_format = '%0'
    scene_num_format += str(max(3, math.floor(math.log(len(scene_list), 10)) + 1)) + 'd'
    image_num_format = '%0'
    image_num_format += str(math.floor(math.log(num_images, 10)) + 2) + 'd'

    framerate = scene_list[0][0].framerate

    # TODO(v1.0): Split up into multiple sub-expressions so auto-formatter works correctly.
    timecode_list = [
        [
            FrameTimecode(int(f), fps=framerate) for f in [
                                                                                               # middle frames
                a[len(a) // 2] if (0 < j < num_images - 1) or num_images == 1

                                                                                               # first frame
                else min(a[0] + frame_margin, a[-1]) if j == 0

                                                                                               # last frame
                else max(a[-1] - frame_margin, a[0])

                                                                                               # for each evenly-split array of frames in the scene list
                for j, a in enumerate(np.array_split(r, num_images))
            ]
        ] for i, r in enumerate([
                                                                                               # pad ranges to number of images
            r if 1 + r[-1] - r[0] >= num_images else list(r) + [r[-1]] * (num_images - len(r))
                                                                                               # create range of frames in scene
            for r in (
                range(
                    start.get_frames(),
                    start.get_frames() + max(
                        1,                                                                     # guard against zero length scenes
                        end.get_frames() - start.get_frames()))
                                                                                               # for each scene in scene list
                for start, end in scene_list)
        ])
    ]

    image_filenames = {i: [] for i in range(len(timecode_list))}
    aspect_ratio = video.aspect_ratio
    if abs(aspect_ratio - 1.0) < 0.01:
        aspect_ratio = None

    logger.debug('Writing images with template %s', filename_template.template)
    for i, scene_timecodes in enumerate(timecode_list):
        for j, image_timecode in enumerate(scene_timecodes):
            video.seek(image_timecode)
            frame_im = video.read()
            if frame_im is not None:
                # TODO: Allow NUM to be a valid suffix in addition to NUMBER.
                file_path = '%s.%s' % (filename_template.safe_substitute(
                    VIDEO_NAME=video.name,
                    SCENE_NUMBER=scene_num_format % (i + 1),
                    IMAGE_NUMBER=image_num_format % (j + 1),
                    FRAME_NUMBER=image_timecode.get_frames()), image_extension)
                image_filenames[i].append(file_path)
                # TODO(0.6.3): Combine this resize with the ones below.
                if aspect_ratio is not None:
                    frame_im = cv2.resize(
                        frame_im, (0, 0),
                        fx=aspect_ratio,
                        fy=1.0,
                        interpolation=interpolation.value)
                frame_height = frame_im.shape[0]
                frame_width = frame_im.shape[1]

                # Figure out what kind of resizing needs to be done
                if height or width:
                    if height and not width:
                        factor = height / float(frame_height)
                        width = int(factor * frame_width)
                    if width and not height:
                        factor = width / float(frame_width)
                        height = int(factor * frame_height)
                    assert height > 0 and width > 0
                    frame_im = cv2.resize(
                        frame_im, (width, height), interpolation=interpolation.value)
                elif scale:
                    frame_im = cv2.resize(
                        frame_im, (0, 0), fx=scale, fy=scale, interpolation=interpolation.value)

                cv2.imwrite(get_and_create_path(file_path, output_dir), frame_im, imwrite_param)
            else:
                completed = False
                break
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    if not completed:
        logger.error('Could not generate all output images.')

    return image_filenames


##
## SceneManager Class Implementation
##


class SceneManager:
    """The SceneManager facilitates detection of scenes (:meth:`detect_scenes`) on a video
    (:class:`VideoStream <scenedetect.video_stream.VideoStream>`) using a detector
    (:meth:`add_detector`). Video decoding is done in parallel in a background thread.
    """

    def __init__(
        self,
        stats_manager: Optional[StatsManager] = None,
    ):
        """
        Arguments:
            stats_manager: :class:`StatsManager` to bind to this `SceneManager`. Can be
                accessed via the `stats_manager` property of the resulting object to save to disk.
        """
        self._cutting_list = []
        self._event_list = []
        self._detector_list = []
        self._sparse_detector_list = []
        # TODO(v1.0): This class should own a StatsManager instead of taking an optional one.
        # Expose a new `stats_manager` @property from the SceneManager, and either change the
        # `stats_manager` argument to to `store_stats: bool=False`, or lazy-init one.

        # TODO(v1.0): This class should own a VideoStream as well, instead of passing one
        # to the detect_scenes method. If concatenation is required, it can be implemented as
        # a generic VideoStream wrapper.
        self._stats_manager: Optional[StatsManager] = stats_manager

        # Position of video that was first passed to detect_scenes.
        self._start_pos: FrameTimecode = None
        # Position of video on the last frame processed by detect_scenes.
        self._last_pos: FrameTimecode = None
        self._base_timecode: Optional[FrameTimecode] = None
        self._downscale: int = 1
        self._auto_downscale: bool = True
        # Interpolation method to use when downscaling. Defaults to linear interpolation
        # as a good balance between quality and performance.
        self._interpolation: Interpolation = Interpolation.LINEAR
        # Boolean indicating if we have only seen EventType.CUT events so far.
        self._only_cuts: bool = True
        # Set by decode thread when an exception occurs.
        self._exception_info = None
        self._stop = threading.Event()

        self._frame_buffer = []
        self._frame_buffer_size = 0

    @property
    def interpolation(self) -> Interpolation:
        """Interpolation method to use when downscaling frames. Must be one of cv2.INTER_*."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: Interpolation):
        self._interpolation = value

    @property
    def stats_manager(self) -> Optional[StatsManager]:
        """Getter for the StatsManager associated with this SceneManager, if any."""
        return self._stats_manager

    @property
    def downscale(self) -> int:
        """Factor to downscale each frame by. Will always be >= 1, where 1
        indicates no scaling. Will be ignored if auto_downscale=True."""
        return self._downscale

    @downscale.setter
    def downscale(self, value: int):
        """Set to 1 for no downscaling, 2 for 2x downscaling, 3 for 3x, etc..."""
        if value < 1:
            raise ValueError("Downscale factor must be a positive integer >= 1!")
        if self.auto_downscale:
            logger.warning("Downscale factor will be ignored because auto_downscale=True!")
        if value is not None and not isinstance(value, int):
            logger.warning("Downscale factor will be truncated to integer!")
            value = int(value)
        self._downscale = value

    @property
    def auto_downscale(self) -> bool:
        """If set to True, will automatically downscale based on video frame size.

        Overrides `downscale` if set."""
        return self._auto_downscale

    @auto_downscale.setter
    def auto_downscale(self, value: bool):
        self._auto_downscale = value

    def add_detector(self, detector: SceneDetector) -> None:
        """Add/register a SceneDetector (e.g. ContentDetector, ThresholdDetector) to
        run when detect_scenes is called. The SceneManager owns the detector object,
        so a temporary may be passed.

        Arguments:
            detector (SceneDetector): Scene detector to add to the SceneManager.
        """
        if self._stats_manager is None and detector.stats_manager_required():
            # Make sure the lists are empty so that the detectors don't get
            # out of sync (require an explicit statsmanager instead)
            assert not self._detector_list and not self._sparse_detector_list
            self._stats_manager = StatsManager()

        detector.stats_manager = self._stats_manager
        if self._stats_manager is not None:
            try:
                self._stats_manager.register_metrics(detector.get_metrics())
            except FrameMetricRegistered:
                # Allow multiple detection algorithms of the same type to be added
                # by suppressing any FrameMetricRegistered exceptions due to attempts
                # to re-register the same frame metric keys.
                # TODO(#334): Fix this, this should not be part of regular control flow.
                pass

        if not issubclass(type(detector), SparseSceneDetector):
            self._detector_list.append(detector)
        else:
            self._sparse_detector_list.append(detector)

        self._frame_buffer_size = max(detector.event_buffer_length, self._frame_buffer_size)

    def get_num_detectors(self) -> int:
        """Get number of registered scene detectors added via add_detector. """
        return len(self._detector_list)

    def clear(self) -> None:
        """Clear all cuts/scenes and resets the SceneManager's position.

        Any statistics generated are still saved in the StatsManager object passed to the
        SceneManager's constructor, and thus, subsequent calls to detect_scenes, using the same
        frame source seeked back to the original time (or beginning of the video) will use the
        cached frame metrics that were computed and saved in the previous call to detect_scenes.
        """
        self._cutting_list.clear()
        self._event_list.clear()
        self._last_pos = None
        self._start_pos = None
        self.clear_detectors()

    def clear_detectors(self) -> None:
        """Remove all scene detectors added to the SceneManager via add_detector(). """
        self._detector_list.clear()
        self._sparse_detector_list.clear()

    def get_scene_list(self,
                       base_timecode: Optional[FrameTimecode] = None,
                       start_in_scene: bool = False) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """Return a list of tuples of start/end FrameTimecodes for each detected scene.

        Arguments:
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility.
            start_in_scene: Assume the video begins in a scene. This means that when detecting
                fast cuts with `ContentDetector`, if no cuts are found, the resulting scene list
                will contain a single scene spanning the entire video (instead of no scenes).
                When detecting fades with `ThresholdDetector`, the beginning portion of the video
                will always be included until the first fade-out event is detected.

        Returns:
            List of tuples in the form (start_time, end_time), where both start_time and
            end_time are FrameTimecode objects representing the exact time/frame where each
            detected scene in the video begins and ends.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `base_timecode` will be removed in v0.8.
        if base_timecode is not None:
            logger.error('`base_timecode` argument is deprecated and has no effect.')
        if self._base_timecode is None:
            return []
        cut_list = self._get_cutting_list()
        scene_list = get_scenes_from_cuts(
            cut_list=cut_list, start_pos=self._start_pos, end_pos=self._last_pos + 1)
        # If we didn't actually scene_detect any cuts, make sure the resulting scene_list is empty
        # unless start_in_scene is True.
        if not cut_list and not start_in_scene:
            scene_list = []
        return sorted(self._get_event_list() + scene_list)

    def _get_cutting_list(self) -> List[int]:
        """Return a sorted list of unique frame numbers of any detected scene cuts."""
        if not self._cutting_list:
            return []
        assert self._base_timecode is not None
        # Ensure all cuts are unique by using a set to remove all duplicates.
        return [self._base_timecode + cut for cut in sorted(set(self._cutting_list))]

    def _get_event_list(self) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        if not self._event_list:
            return []
        assert self._base_timecode is not None
        return [(self._base_timecode + start, self._base_timecode + end)
                for start, end in self._event_list]

    def _process_frame(self,
                       frame_num: int,
                       frame_im: np.ndarray,
                       callback: Optional[Callable[[np.ndarray, int], None]] = None) -> bool:
        """Add any cuts detected with the current frame to the cutting list. Returns True if any new
        cuts were detected, False otherwise."""
        new_cuts = False
        # TODO(#283): This breaks with AdaptiveDetector as cuts differ from the frame number
        # being processed. Allow detectors to specify the max frame lookahead they require
        # (i.e. any event will never be more than N frames behind the current one).
        self._frame_buffer.append(frame_im)
        # frame_buffer[-1] is current frame, -2 is one behind, etc
        # so index based on cut frame should be [event_frame - (frame_num + 1)]
        self._frame_buffer = self._frame_buffer[-(self._frame_buffer_size + 1):]
        for detector in self._detector_list:
            cuts = detector.process_frame(frame_num, frame_im)
            self._cutting_list += cuts
            new_cuts = True if cuts else False
            if callback:
                for cut_frame_num in cuts:
                    buffer_index = cut_frame_num - (frame_num + 1)
                    callback(self._frame_buffer[buffer_index], cut_frame_num)
        for detector in self._sparse_detector_list:
            events = detector.process_frame(frame_num, frame_im)
            self._event_list += events
            if callback:
                for event_start, _ in events:
                    buffer_index = event_start - (frame_num + 1)
                    callback(self._frame_buffer[buffer_index], event_start)
        return new_cuts

    def _post_process(self, frame_num: int) -> None:
        """Add remaining cuts to the cutting list, after processing the last frame."""
        for detector in self._detector_list:
            self._cutting_list += detector.post_process(frame_num)

    def stop(self) -> None:
        """Stop the current :meth:`detect_scenes` call, if any. Thread-safe."""
        self._stop.set()

    def detect_scenes(self,
                      video: VideoStream = None,
                      duration: Optional[FrameTimecode] = None,
                      end_time: Optional[FrameTimecode] = None,
                      frame_skip: int = 0,
                      show_progress: bool = False,
                      callback: Optional[Callable[[np.ndarray, int], None]] = None,
                      frame_source: Optional[VideoStream] = None) -> int:
        """Perform scene detection on the given video using the added SceneDetectors, returning the
        number of frames processed. Results can be obtained by calling :meth:`get_scene_list` or
        :meth:`get_cut_list`.

        Video decoding is performed in a background thread to allow scene detection and frame
        decoding to happen in parallel. Detection will continue until no more frames are left,
        the specified duration or end time has been reached, or :meth:`stop` was called.

        Arguments:
            video: VideoStream obtained from either `scenedetect.open_video`, or by creating
                one directly (e.g. `scenedetect.backends.opencv.VideoStreamCv2`).
            duration: Amount of time to scene_detect from current video position. Cannot be
                specified if `end_time` is set.
            end_time: Time to stop processing at. Cannot be specified if `duration` is set.
            frame_skip: Not recommended except for extremely high framerate videos.
                Number of frames to skip (i.e. process every 1 in N+1 frames,
                where N is frame_skip, processing only 1/N+1 percent of the video,
                speeding up the detection time at the expense of accuracy).
                `frame_skip` **must** be 0 (the default) when using a StatsManager.
            show_progress: If True, and the ``tqdm`` module is available, displays
                a progress bar with the progress, framerate, and expected time to
                complete processing the video frame source.
            callback: If set, called after each scene/event detected.
            frame_source: [DEPRECATED] DO NOT USE. For compatibility with previous version.
        Returns:
            int: Number of frames read and processed from the frame source.
        Raises:
            ValueError: `frame_skip` **must** be 0 (the default) if the SceneManager
                was constructed with a StatsManager object.
        """
        # TODO(v0.7): Add DeprecationWarning that `frame_source` will be removed in v0.8.
        # TODO(v0.8): Remove default value for `video`` when removing `frame_source`.
        if frame_source is not None:
            video = frame_source
        if video is None:
            raise TypeError("detect_scenes() missing 1 required positional argument: 'video'")

        if frame_skip > 0 and self.stats_manager is not None:
            raise ValueError('frame_skip must be 0 when using a StatsManager.')
        if duration is not None and end_time is not None:
            raise ValueError('duration and end_time cannot be set at the same time!')
        if duration is not None and duration < 0:
            raise ValueError('duration must be greater than or equal to 0!')
        if end_time is not None and end_time < 0:
            raise ValueError('end_time must be greater than or equal to 0!')

        self._base_timecode = video.base_timecode
        # TODO(v1.0): Fix this properly by making SceneManager create and own a StatsManager,
        # and requiring the framerate to be passed to the StatsManager the constructor.
        if self._stats_manager is not None:
            self._stats_manager._base_timecode = self._base_timecode
        start_frame_num: int = video.frame_number

        if duration is not None:
            end_time: Union[int, FrameTimecode] = duration + start_frame_num

        if end_time is not None:
            end_time: FrameTimecode = self._base_timecode + end_time

        # Can only calculate total number of frames we expect to process if the duration of
        # the video is available.
        total_frames = 0
        if video.duration is not None:
            if end_time is not None and end_time < video.duration:
                total_frames = (end_time - start_frame_num) + 1
            else:
                total_frames = (video.duration.get_frames() - start_frame_num)

        # Calculate the desired downscale factor and log the effective resolution.
        if self.auto_downscale:
            downscale_factor = compute_downscale_factor(frame_width=video.frame_size[0])
        else:
            downscale_factor = self.downscale
        if downscale_factor > 1:
            logger.info('Downscale factor set to %d, effective resolution: %d x %d',
                        downscale_factor, video.frame_size[0] // downscale_factor,
                        video.frame_size[1] // downscale_factor)

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=int(total_frames),
                unit='frames',
                desc=PROGRESS_BAR_DESCRIPTION % 0,
                dynamic_ncols=True,
            )

        frame_queue = queue.Queue(MAX_FRAME_QUEUE_LENGTH)
        self._stop.clear()
        decode_thread = threading.Thread(
            target=SceneManager._decode_thread,
            args=(self, video, frame_skip, downscale_factor, end_time, frame_queue),
            daemon=True)
        decode_thread.start()
        frame_im = None

        logger.info('Detecting scenes...')
        while not self._stop.is_set():
            next_frame, position = frame_queue.get()
            if next_frame is None and position is None:
                break
            if not next_frame is None:
                frame_im = next_frame
            new_cuts = self._process_frame(position.frame_num, frame_im, callback)
            if progress_bar is not None:
                if new_cuts:
                    progress_bar.set_description(
                        PROGRESS_BAR_DESCRIPTION % len(self._cutting_list), refresh=False)
                progress_bar.update(1 + frame_skip)

        if progress_bar is not None:
            progress_bar.close()
        # Unblock any puts in the decode thread before joining. This can happen if the main
        # processing thread stops before the decode thread.
        while not frame_queue.empty():
            frame_queue.get_nowait()
        decode_thread.join()

        if self._exception_info is not None:
            raise self._exception_info[1].with_traceback(self._exception_info[2])

        self._last_pos = video.position
        self._post_process(video.position.frame_num)
        return video.frame_number - start_frame_num

    def _decode_thread(
        self,
        video: VideoStream,
        frame_skip: int,
        downscale_factor: int,
        end_time: FrameTimecode,
        out_queue: queue.Queue,
    ):
        try:
            while not self._stop.is_set():
                frame_im = None
                # We don't do any kind of locking here since the worst-case of this being wrong
                # is that we do some extra work, and this function should never mutate any data
                # (all of which should be modified under the GIL).
                # TODO(v1.0): This optimization should be removed as it is an uncommon use case and
                # greatly increases the complexity of detection algorithms using it.
                if self._is_processing_required(video.position.frame_num):
                    frame_im = video.read()
                    if frame_im is False:
                        break
                    if downscale_factor > 1:
                        frame_im = cv2.resize(
                            frame_im, (round(frame_im.shape[1] / downscale_factor),
                                       round(frame_im.shape[0] / downscale_factor)),
                            interpolation=self._interpolation.value)
                else:
                    if video.read(decode=False) is False:
                        break

                # Set the start position now that we decoded at least the first frame.
                if self._start_pos is None:
                    self._start_pos = video.position

                out_queue.put((frame_im, video.position))

                if frame_skip > 0:
                    for _ in range(frame_skip):
                        if not video.read(decode=False):
                            break
                # End time includes the presentation time of the frame, but the `position`
                # property of a VideoStream references the beginning of the frame in time.
                if end_time is not None and not (video.position + 1) < end_time:
                    break

        # If *any* exceptions occur, we re-raise them in the main thread so that the caller of
        # detect_scenes can handle it.
        except KeyboardInterrupt:
            logger.debug("Received KeyboardInterrupt.")
            self._stop.set()
        except BaseException:
            logger.critical('Fatal error: Exception raised in decode thread.')
            self._exception_info = sys.exc_info()
            self._stop.set()

        finally:
            # Handle case where start position was never set if we did not decode any frames.
            if self._start_pos is None:
                self._start_pos = video.position
            # Make sure main thread stops processing loop.
            out_queue.put((None, None))

        # pylint: enable=bare-except

    #
    # Deprecated Methods
    #

    # pylint: disable=unused-argument

    def get_cut_list(self,
                     base_timecode: Optional[FrameTimecode] = None,
                     show_warning: bool = True) -> List[FrameTimecode]:
        """[DEPRECATED] Return a list of FrameTimecodes of the detected scene changes/cuts.

        Unlike get_scene_list, the cutting list returns a list of FrameTimecodes representing
        the point in the input video where a new scene was detected, and thus the frame
        where the input should be cut/split. The cutting list, in turn, is used to generate
        the scene list, noting that each scene is contiguous starting from the first frame
        and ending at the last frame detected.

        If only sparse detectors are used (e.g. MotionDetector), this will always be empty.

        Arguments:
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.
            show_warning: If set to False, suppresses the error from being warned. In v0.7,
                this will have no effect and the error will become a Python warning.

        Returns:
            List of FrameTimecode objects denoting the points in time where a scene change
            was detected in the input video, which can also be passed to external tools
            for automated splitting of the input into individual scenes.
        """
        # TODO(v0.7): Use the warnings module to turn this into a warning.
        if show_warning:
            logger.error('`get_cut_list()` is deprecated and will be removed in a future release.')
        return self._get_cutting_list()

    def get_event_list(
            self,
            base_timecode: Optional[FrameTimecode] = None
    ) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """[DEPRECATED] DO NOT USE.

        Get a list of start/end timecodes of sparse detection events.

        Unlike get_scene_list, the event list returns a list of FrameTimecodes representing
        the point in the input video where a new scene was detected only by sparse detectors,
        otherwise it is the same.

        Arguments:
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.

        Returns:
            List of pairs of FrameTimecode objects denoting the detected scenes.
        """
        # TODO(v0.7): Use the warnings module to turn this into a warning.
        logger.error('`get_event_list()` is deprecated and will be removed in a future release.')
        return self._get_event_list()

    # pylint: enable=unused-argument

    def _is_processing_required(self, frame_num: int) -> bool:
        """True if frame metrics not in StatsManager, False otherwise."""
        if self.stats_manager is None:
            return True
        return all([detector.is_processing_required(frame_num) for detector in self._detector_list])
