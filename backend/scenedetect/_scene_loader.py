# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site:   http://www.scenedetect.scenedetect.com/         ]
#     [  Docs:   http://manual.scenedetect.scenedetect.com/      ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#
# Copyright (C) 2014-2023 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
""":class:`SceneLoader` is a class designed for use cases in which a list of
scenes is read from a csv file and actual detection of scene boundaries does not
need to occur.

This is available from the command-line as the `load-scenes` command.
"""

import os
import csv

import typing as ty

import numpy

from scenedetect.scene_detector import SceneDetector
from scenedetect.frame_timecode import FrameTimecode


class SceneLoader(SceneDetector):
    """Detector which load a list of predefined cuts from a CSV file. Used by the CLI to implement
    the `load-scenes` functionality. Incompatible with other detectors.
    """

    def __init__(self, file: ty.TextIO, framerate: float, start_col_name: str = "Start Frame"):
        """
        Arguments:
            file: Path to csv file containing scene data for video
            framerate: Framerate used to construct `FrameTimecode` for parsing input.
            start_col_name: Header for column containing the frame/timecode where cuts occur.
        """
        super().__init__()

        # Check to make specified csv file exists
        if not file:
            raise ValueError('file path to csv file must be specified')
        if not os.path.exists(file):
            raise ValueError('specified csv file does not exist')

        self.csv_file = file

        # Open csv and check and read first row for column headers
        (self.file_reader, csv_headers) = self._open_csv(self.csv_file, start_col_name)

        # Check to make sure column headers are present
        if start_col_name not in csv_headers:
            raise ValueError('specified column header for scene start is not present')

        self._col_idx = csv_headers.index(start_col_name)
        self._last_scene_row = None
        self._scene_start = None

        # `SceneDetector` works on cuts, so we have to skip the first scene and use the first frame
        # of the next scene as the cut point. This can be fixed if we used `SparseSceneDetector`
        # but this part of the API is being reworked and hasn't been used by any detectors yet.
        self._cut_list = sorted(
            FrameTimecode(row[self._col_idx], fps=framerate).frame_num - 1
            for row in self.file_reader)
        if self._cut_list:
            self._cut_list = self._cut_list[1:]

    def _open_csv(self, csv_file, start_col_name):
        """Opens the specified csv file for reading.

        Arguments:
            csv_file:       Path to csv file containing scene data for video

        Returns:
            (reader, headers):    csv.reader object and headers
        """
        input_file = open(csv_file, 'r')
        file_reader = csv.reader(input_file)
        csv_headers = next(file_reader)
        if not start_col_name in csv_headers:
            csv_headers = next(file_reader)
        return (file_reader, csv_headers)

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> ty.List[int]:
        """Simply reads cut data from a given csv file. Video is not analyzed. Therefore this
        detector is incompatible with other detectors or a StatsManager.

        Arguments:
            frame_num:  Frame number of frame that is being passed.
            frame_img:  Decoded frame image (numpy.ndarray) to perform scene detection on. This is
                unused for this detector as the video is not analyzed, but is allowed for
                compatibility.

        Returns:
            cut_list:   List of cuts (as provided by input csv file)
        """
        if frame_num in self._cut_list:
            return [frame_num]
        return []

    def is_processing_required(self, frame_num):
        return False
