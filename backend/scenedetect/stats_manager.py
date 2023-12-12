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
"""``scenedetect.stats_manager`` Module

This module contains the :class:`StatsManager` class, which provides a key-value store for each
:class:`SceneDetector <scenedetect.scene_detector.SceneDetector>` to write the metrics calculated
for each frame. The :class:`StatsManager` must be registered to a
:class:`SceneManager <scenedetect.scene_manager.SceneManager>` upon construction.

The entire :class:`StatsManager` can be :meth:`saved to <StatsManager.save_to_csv>` a
human-readable CSV file, allowing for precise determination of the ideal threshold (or other
detection parameters) for the given input.
"""

import csv
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Union
import os.path

from backend.scenedetect.frame_timecode import FrameTimecode

logger = getLogger('pyscenedetect')

##
## StatsManager CSV File Column Names (Header Row)
##

COLUMN_NAME_FRAME_NUMBER = "Frame Number"
"""Name of column containing frame numbers in the statsfile CSV."""

COLUMN_NAME_TIMECODE = "Timecode"
"""Name of column containing timecodes in the statsfile CSV."""

##
## StatsManager Exceptions
##


class FrameMetricRegistered(Exception):
    """ Raised when attempting to register a frame metric key which has
    already been registered. """

    def __init__(self,
                 metric_key: str,
                 message: str = "Attempted to re-register frame metric key."):
        super().__init__(message)
        self.metric_key = metric_key


class FrameMetricNotRegistered(Exception):
    """ Raised when attempting to call get_metrics(...)/set_metrics(...) with a
    frame metric that does not exist, or has not been registered. """

    def __init__(self,
                 metric_key: str,
                 message: str = "Attempted to get/set frame metrics for unregistered metric key."):
        super().__init__(message)
        self.metric_key = metric_key


class StatsFileCorrupt(Exception):
    """Raised when frame metrics/stats could not be loaded from a provided CSV file."""

    def __init__(self,
                 message: str = "Could not load frame metric data data from passed CSV file."):
        super().__init__(message)


##
## StatsManager Class Implementation
##


# TODO(v1.0): Relax restriction on metric types only being float or int when loading from disk
# is fully deprecated.
class StatsManager:
    """Provides a key-value store for frame metrics/calculations which can be used
    for two-pass detection algorithms, as well as saving stats to a CSV file.

    Analyzing a statistics CSV file is also very useful for finding the optimal
    algorithm parameters for certain detection methods. Additionally, the data
    may be plotted by a graphing module (e.g. matplotlib) by obtaining the
    metric of interest for a series of frames by iteratively calling get_metrics(),
    after having called the detect_scenes(...) method on the SceneManager object
    which owns the given StatsManager instance.

    Only metrics consisting of `float` or `int` should be used currently.
    """

    def __init__(self, base_timecode: FrameTimecode = None):
        """Initialize a new StatsManager.

        Arguments:
            base_timecode: Timecode associated with this object. Must not be None (default value
                will be removed in a future release).
        """
        # Frame metrics is a dict of frame (int): metric_dict (Dict[str, float])
        # of each frame metric key and the value it represents (usually float).
        self._frame_metrics: Dict[FrameTimecode, Dict[str, float]] = dict()
        self._registered_metrics: Set[str] = set()                   # Set of frame metric keys.
        self._loaded_metrics: Set[str] = set()                       # Metric keys loaded from stats file.
        self._metrics_updated: bool = False                          # Flag indicating if metrics require saving.
        self._base_timecode: Optional[FrameTimecode] = base_timecode # Used for timing calculations.

    def register_metrics(self, metric_keys: Iterable[str]) -> None:
        """Register a list of metric keys that will be used by the detector.

        Used to ensure that multiple detector keys don't overlap.

        Raises:
            FrameMetricRegistered: A particular metric_key has already been registered/added
                to the StatsManager. Only if the StatsManager is being used for read-only
                access (i.e. all frames in the video have already been processed for the given
                metric_key in the exception) is this behavior desirable.
        """
        for metric_key in metric_keys:
            if metric_key not in self._registered_metrics:
                self._registered_metrics.add(metric_key)
            else:
                raise FrameMetricRegistered(metric_key)

    # TODO(v1.0): Change frame_number to a FrameTimecode now that it is just a hash and will
    # be required for VFR support.
    def get_metrics(self, frame_number: int, metric_keys: Iterable[str]) -> List[Any]:
        """Return the requested statistics/metrics for a given frame.

        Arguments:
            frame_number (int): Frame number to retrieve metrics for.
            metric_keys (List[str]): A list of metric keys to look up.

        Returns:
            A list containing the requested frame metrics for the given frame number
            in the same order as the input list of metric keys. If a metric could
            not be found, None is returned for that particular metric.
        """
        return [self._get_metric(frame_number, metric_key) for metric_key in metric_keys]

    def set_metrics(self, frame_number: int, metric_kv_dict: Dict[str, Any]) -> None:
        """ Set Metrics: Sets the provided statistics/metrics for a given frame.

        Arguments:
            frame_number: Frame number to retrieve metrics for.
            metric_kv_dict: A dict mapping metric keys to the
                respective integer/floating-point metric values to set.
        """
        for metric_key in metric_kv_dict:
            self._set_metric(frame_number, metric_key, metric_kv_dict[metric_key])

    def metrics_exist(self, frame_number: int, metric_keys: Iterable[str]) -> bool:
        """ Metrics Exist: Checks if the given metrics/stats exist for the given frame.

        Returns:
            bool: True if the given metric keys exist for the frame, False otherwise.
        """
        return all([self._metric_exists(frame_number, metric_key) for metric_key in metric_keys])

    def is_save_required(self) -> bool:
        """ Is Save Required: Checks if the stats have been updated since loading.

        Returns:
            bool: True if there are frame metrics/statistics not yet written to disk,
            False otherwise.
        """
        return self._metrics_updated

    def save_to_csv(self,
                    csv_file: Union[str, bytes, TextIO],
                    base_timecode: Optional[FrameTimecode] = None,
                    force_save=True) -> None:
        """ Save To CSV: Saves all frame metrics stored in the StatsManager to a CSV file.

        Arguments:
            csv_file: A file handle opened in write mode (e.g. open('...', 'w')) or a path as str.
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility.
            force_save: If True, writes metrics out even if an update is not required.

        Raises:
            OSError: If `path` cannot be opened or a write failure occurs.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `base_timecode` will be removed in v0.8.
        if base_timecode is not None:
            logger.error('base_timecode is deprecated.')

        # Ensure we need to write to the file, and that we have data to do so with.
        if not ((self.is_save_required() or force_save) and self._registered_metrics
                and self._frame_metrics):
            logger.info("No metrics to save.")
            return

        assert self._base_timecode is not None

        # If we get a path instead of an open file handle, recursively call ourselves
        # again but with file handle instead of path.
        if isinstance(csv_file, (str, bytes)):
            with open(csv_file, 'w') as file:
                self.save_to_csv(csv_file=file, force_save=force_save)
                return

        csv_writer = csv.writer(csv_file, lineterminator='\n')
        metric_keys = sorted(list(self._registered_metrics.union(self._loaded_metrics)))
        csv_writer.writerow([COLUMN_NAME_FRAME_NUMBER, COLUMN_NAME_TIMECODE] + metric_keys)
        frame_keys = sorted(self._frame_metrics.keys())
        logger.info("Writing %d frames to CSV...", len(frame_keys))
        for frame_key in frame_keys:
            frame_timecode = self._base_timecode + frame_key
            csv_writer.writerow(
                [frame_timecode.get_frames() +
                 1, frame_timecode.get_timecode()] +
                [str(metric) for metric in self.get_metrics(frame_key, metric_keys)])

    @staticmethod
    def valid_header(row: List[str]) -> bool:
        """Check that the given CSV row is a valid header for a statsfile.

        Arguments:
            row: A row decoded from the CSV reader.

        Returns:
            True if `row` is a valid statsfile header, False otherwise.
        """
        if not row or not len(row) >= 2:
            return False
        if row[0] != COLUMN_NAME_FRAME_NUMBER or row[1] != COLUMN_NAME_TIMECODE:
            return False
        return True

    # TODO(v1.0): Remove.
    def load_from_csv(self, csv_file: Union[str, bytes, TextIO]) -> Optional[int]:
        """[DEPRECATED] DO NOT USE

        Load all metrics stored in a CSV file into the StatsManager instance. Will be removed in a
        future release after becoming a no-op.

        Arguments:
            csv_file: A file handle opened in read mode (e.g. open('...', 'r')) or a path as str.

        Returns:
            int or None: Number of frames/rows read from the CSV file, or None if the
            input file was blank or could not be found.

        Raises:
            StatsFileCorrupt: Stats file is corrupt and can't be loaded, or wrong file
                was specified.
        """
        # TODO: Make this an error, then make load_from_csv() a no-op, and finally, remove it.
        logger.warning("load_from_csv() is deprecated and will be removed in a future release.")

        # If we get a path instead of an open file handle, check that it exists, and if so,
        # recursively call ourselves again but with file set instead of path.
        if isinstance(csv_file, (str, bytes)):
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as file:
                    return self.load_from_csv(csv_file=file)
            # Path doesn't exist.
            return None

        # If we get here, file is a valid file handle in read-only text mode.
        csv_reader = csv.reader(csv_file, lineterminator='\n')
        num_cols = None
        num_metrics = None
        num_frames = None
        # First Row: Frame Num, Timecode, [metrics...]
        try:
            row = next(csv_reader)
            # Backwards compatibility for previous versions of statsfile
            # which included an additional header row.
            if not self.valid_header(row):
                row = next(csv_reader)
        except StopIteration:
            # If the file is blank or we couldn't decode anything, assume the file was empty.
            return None
        if not self.valid_header(row):
            raise StatsFileCorrupt()
        num_cols = len(row)
        num_metrics = num_cols - 2
        if not num_metrics > 0:
            raise StatsFileCorrupt('No metrics defined in CSV file.')
        self._loaded_metrics = row[2:]
        num_frames = 0
        for row in csv_reader:
            metric_dict = {}
            if not len(row) == num_cols:
                raise StatsFileCorrupt('Wrong number of columns detected in stats file row.')
            for i, metric_str in enumerate(row[2:]):
                if metric_str and metric_str != 'None':
                    try:
                        metric_dict[self._loaded_metrics[i]] = float(metric_str)
                    except ValueError:
                        raise StatsFileCorrupt('Corrupted value in stats file: %s' %
                                               metric_str) from ValueError
            frame_number = int(row[0])
            # Switch from 1-based to 0-based frame numbers.
            if frame_number > 0:
                frame_number -= 1
            self.set_metrics(frame_number, metric_dict)
            num_frames += 1
        logger.info('Loaded %d metrics for %d frames.', num_metrics, num_frames)
        self._metrics_updated = False
        return num_frames

    def _get_metric(self, frame_number: int, metric_key: str) -> Optional[Any]:
        if self._metric_exists(frame_number, metric_key):
            return self._frame_metrics[frame_number][metric_key]
        return None

    def _set_metric(self, frame_number: int, metric_key: str, metric_value: Any) -> None:
        self._metrics_updated = True
        if not frame_number in self._frame_metrics:
            self._frame_metrics[frame_number] = dict()
        self._frame_metrics[frame_number][metric_key] = metric_value

    def _metric_exists(self, frame_number: int, metric_key: str) -> bool:
        return (frame_number in self._frame_metrics
                and metric_key in self._frame_metrics[frame_number])
