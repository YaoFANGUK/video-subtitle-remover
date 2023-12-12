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
"""Handles loading configuration files from disk and validating each section. Only validation of the
config file schema and data types are performed. Constants/defaults are also defined here where
possible and re-used by the CLI so that there is one source of truth.
"""

from abc import ABC, abstractmethod
import logging
import os
import os.path
from configparser import ConfigParser, ParsingError
from typing import Any, AnyStr, Dict, List, Optional, Tuple, Union

from platformdirs import user_config_dir

from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import Interpolation
from scenedetect.video_splitter import DEFAULT_FFMPEG_ARGS

VALID_PYAV_THREAD_MODES = ['NONE', 'SLICE', 'FRAME', 'AUTO']


class OptionParseFailure(Exception):
    """Raised when a value provided in a user config file fails validation."""

    def __init__(self, error):
        super().__init__()
        self.error = error


class ValidatedValue(ABC):
    """Used to represent configuration values that must be validated against constraints."""

    @property
    @abstractmethod
    def value(self) -> Any:
        """Get the value after validation."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_config(config_value: str, default: 'ValidatedValue') -> 'ValidatedValue':
        """Validate and get the user-specified configuration option.

        Raises:
            OptionParseFailure: Value from config file did not meet validation constraints.
        """
        raise NotImplementedError()


class TimecodeValue(ValidatedValue):
    """Validator for timecode values in frames (1234), seconds (123.4s), or HH:MM:SS.

    Stores value in original representation."""

    def __init__(self, value: Union[int, float, str]):
        # Ensure value is a valid timecode.
        FrameTimecode(timecode=value, fps=100.0)
        self._value = value

    @property
    def value(self) -> Union[int, float, str]:
        return self._value

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def from_config(config_value: str, default: 'TimecodeValue') -> 'TimecodeValue':
        try:
            return TimecodeValue(config_value)
        except ValueError as ex:
            raise OptionParseFailure(
                'Timecodes must be in frames (1234), seconds (123.4s), or HH:MM:SS (00:02:03.400).'
            ) from ex


class RangeValue(ValidatedValue):
    """Validator for int/float ranges. `min_val` and `max_val` are inclusive."""

    def __init__(
        self,
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
    ):
        if value < min_val or value > max_val:
            # min and max are inclusive.
            raise ValueError()
        self._value = value
        self._min_val = min_val
        self._max_val = max_val

    @property
    def value(self) -> Union[int, float]:
        return self._value

    @property
    def min_val(self) -> Union[int, float]:
        """Minimum value of the range."""
        return self._min_val

    @property
    def max_val(self) -> Union[int, float]:
        """Maximum value of the range."""
        return self._max_val

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def from_config(config_value: str, default: 'RangeValue') -> 'RangeValue':
        try:
            return RangeValue(
                value=int(config_value) if isinstance(default.value, int) else float(config_value),
                min_val=default.min_val,
                max_val=default.max_val,
            )
        except ValueError as ex:
            raise OptionParseFailure('Value must be between %s and %s.' %
                                     (default.min_val, default.max_val)) from ex


class ScoreWeightsValue(ValidatedValue):
    """Validator for score weight values (currently a tuple of four numbers)."""

    _IGNORE_CHARS = [',', '/', '(', ')']
    """Characters to ignore."""

    def __init__(self, value: Union[str, ContentDetector.Components]):
        if isinstance(value, ContentDetector.Components):
            self._value = value
        else:
            translation_table = str.maketrans(
                {char: ' ' for char in ScoreWeightsValue._IGNORE_CHARS})
            values = value.translate(translation_table).split()
            if not len(values) == 4:
                raise ValueError("Score weights must be specified as four numbers!")
            self._value = ContentDetector.Components(*(float(val) for val in values))

    @property
    def value(self) -> Tuple[float, float, float, float]:
        return self._value

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return '%.3f, %.3f, %.3f, %.3f' % self.value

    @staticmethod
    def from_config(config_value: str, default: 'ScoreWeightsValue') -> 'ScoreWeightsValue':
        try:
            return ScoreWeightsValue(config_value)
        except ValueError as ex:
            raise OptionParseFailure(
                'Score weights must be specified as four numbers in the form (H,S,L,E),'
                ' e.g. (0.9, 0.2, 2.0, 0.5). Commas/brackets/slashes are ignored.') from ex


class KernelSizeValue(ValidatedValue):
    """Validator for kernel sizes (odd integer > 1, or -1 for auto size)."""

    def __init__(self, value: int):
        if value == -1:
            # Downscale factor of -1 maps to None internally for auto downscale.
            value = None
        elif value < 0:
            # Disallow other negative values.
            raise ValueError()
        elif value % 2 == 0:
            # Disallow even values.
            raise ValueError()
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        if self.value is None:
            return 'auto'
        return str(self.value)

    @staticmethod
    def from_config(config_value: str, default: 'KernelSizeValue') -> 'KernelSizeValue':
        try:
            return KernelSizeValue(int(config_value))
        except ValueError as ex:
            raise OptionParseFailure(
                'Value must be an odd integer greater than 1, or set to -1 for auto kernel size.'
            ) from ex


ConfigValue = Union[bool, int, float, str]
ConfigDict = Dict[str, Dict[str, ConfigValue]]

_CONFIG_FILE_NAME: AnyStr = 'scenedetect.cfg'
_CONFIG_FILE_DIR: AnyStr = user_config_dir("PySceneDetect", False)

CONFIG_FILE_PATH: AnyStr = os.path.join(_CONFIG_FILE_DIR, _CONFIG_FILE_NAME)

CONFIG_MAP: ConfigDict = {
    'backend-opencv': {
        'max-decode-attempts': 5,
    },
    'backend-pyav': {
        'suppress-output': False,
        'threading-mode': 'auto',
    },
    'scene_detect-adaptive': {
        'frame-window': 2,
        'kernel-size': KernelSizeValue(-1),
        'luma-only': False,
        'min-content-val': RangeValue(15.0, min_val=0.0, max_val=255.0),
        'min-scene-len': TimecodeValue(0),
        'threshold': RangeValue(3.0, min_val=0.0, max_val=255.0),
        'weights': ScoreWeightsValue(ContentDetector.DEFAULT_COMPONENT_WEIGHTS),
                                                                                   # TODO(v0.7): Remove `min-delta-hsv``.
        'min-delta-hsv': RangeValue(15.0, min_val=0.0, max_val=255.0),
    },
    'scene_detect-content': {
        'kernel-size': KernelSizeValue(-1),
        'luma-only': False,
        'min-scene-len': TimecodeValue(0),
        'threshold': RangeValue(27.0, min_val=0.0, max_val=255.0),
        'weights': ScoreWeightsValue(ContentDetector.DEFAULT_COMPONENT_WEIGHTS),
    },
    'scene_detect-threshold': {
        'add-last-scene': True,
        'fade-bias': RangeValue(0, min_val=-100.0, max_val=100.0),
        'min-scene-len': TimecodeValue(0),
        'threshold': RangeValue(12.0, min_val=0.0, max_val=255.0),
    },
    'load-scenes': {
        'start-col-name': 'Start Frame',
    },
    'export-html': {
        'filename': '$VIDEO_NAME-Scenes.html',
        'image-height': 0,
        'image-width': 0,
        'no-images': False,
    },
    'list-scenes': {
        'output': '',
        'filename': '$VIDEO_NAME-Scenes.csv',
        'no-output-file': False,
        'quiet': False,
        'skip-cuts': False,
    },
    'global': {
        'backend': 'opencv',
        'default-detector': 'scene_detect-adaptive',
        'downscale': 0,
        'downscale-method': 'linear',
        'drop-short-scenes': False,
        'frame-skip': 0,
        'merge-last-scene': False,
        'min-scene-len': TimecodeValue('0.6s'),
        'output': '',
        'verbosity': 'info',
    },
    'save-images': {
        'compression': RangeValue(3, min_val=0, max_val=9),
        'filename': '$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
        'format': 'jpeg',
        'frame-margin': 1,
        'height': 0,
        'num-images': 3,
        'output': '',
        'quality': RangeValue(0, min_val=0, max_val=100),                        # Default depends on format
        'scale': 1.0,
        'scale-method': 'linear',
        'width': 0,
    },
    'split-video': {
        'args': DEFAULT_FFMPEG_ARGS,
        'copy': False,
        'filename': '$VIDEO_NAME-Scene-$SCENE_NUMBER',
        'high-quality': False,
        'mkvmerge': False,
        'output': '',
        'preset': 'veryfast',
        'quiet': False,
        'rate-factor': RangeValue(22, min_val=0, max_val=100),
    },
}
"""Mapping of valid configuration file parameters and their default values or placeholders.
The types of these values are used when decoding the configuration file. Valid choices for
certain string options are stored in `CHOICE_MAP`."""

CHOICE_MAP: Dict[str, Dict[str, List[str]]] = {
    'global': {
        'backend': ['opencv', 'pyav', 'moviepy'],
        'default-detector': ['scene_detect-adaptive', 'scene_detect-content', 'scene_detect-threshold'],
        'downscale-method': [value.name.lower() for value in Interpolation],
        'verbosity': ['debug', 'info', 'warning', 'error', 'none'],
    },
    'split-video': {
        'preset': [
            'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower',
            'veryslow'
        ],
    },
    'save-images': {
        'format': ['jpeg', 'png', 'webp'],
        'scale-method': [value.name.lower() for value in Interpolation],
    },
    'backend-pyav': {
        'threading_mode': [str(mode).lower() for mode in VALID_PYAV_THREAD_MODES],
    },
}
"""Mapping of string options which can only be of a particular set of values. We use a list instead
of a set to preserve order when generating error contexts. Values are case-insensitive, and must be
in lowercase in this map."""


def _validate_structure(config: ConfigParser) -> List[str]:
    """Validates the layout of the section/option mapping.

    Returns:
        List of any parsing errors in human-readable form.
    """
    errors: List[str] = []
    for section in config.sections():
        if not section in CONFIG_MAP.keys():
            errors.append('Unsupported config section: [%s]' % (section))
            continue
        for (option_name, _) in config.items(section):
            if not option_name in CONFIG_MAP[section].keys():
                errors.append('Unsupported config option in [%s]: %s' % (section, option_name))
    return errors


def _parse_config(config: ConfigParser) -> Tuple[ConfigDict, List[str]]:
    """Process the given configuration into a key-value mapping.

    Returns:
        Configuration mapping and list of any processing errors in human readable form.
    """
    out_map: ConfigDict = {}
    errors: List[str] = []
    for command in CONFIG_MAP:
        out_map[command] = {}
        for option in CONFIG_MAP[command]:
            if command in config and option in config[command]:
                try:
                    value_type = None
                    if isinstance(CONFIG_MAP[command][option], bool):
                        value_type = 'yes/no value'
                        out_map[command][option] = config.getboolean(command, option)
                        continue
                    elif isinstance(CONFIG_MAP[command][option], int):
                        value_type = 'integer'
                        out_map[command][option] = config.getint(command, option)
                        continue
                    elif isinstance(CONFIG_MAP[command][option], float):
                        value_type = 'number'
                        out_map[command][option] = config.getfloat(command, option)
                        continue
                except ValueError as _:
                    errors.append('Invalid [%s] value for %s: %s is not a valid %s.' %
                                  (command, option, config.get(command, option), value_type))
                    continue

                # Handle custom validation types.
                config_value = config.get(command, option)
                default = CONFIG_MAP[command][option]
                option_type = type(default)
                if issubclass(option_type, ValidatedValue):
                    try:
                        out_map[command][option] = option_type.from_config(
                            config_value=config_value, default=default)
                    except OptionParseFailure as ex:
                        errors.append('Invalid [%s] value for %s:\n  %s\n%s' %
                                      (command, option, config_value, ex.error))
                    continue

                # If we didn't process the value as a given type, handle it as a string. We also
                # replace newlines with spaces, and strip any remaining leading/trailing whitespace.
                if value_type is None:
                    config_value = config.get(command, option).replace('\n', ' ').strip()
                    if command in CHOICE_MAP and option in CHOICE_MAP[command]:
                        if config_value.lower() not in CHOICE_MAP[command][option]:
                            errors.append('Invalid [%s] value for %s: %s. Must be one of: %s.' %
                                          (command, option, config.get(command, option), ', '.join(
                                              choice for choice in CHOICE_MAP[command][option])))
                            continue
                    out_map[command][option] = config_value
                    continue

    return (out_map, errors)


class ConfigLoadFailure(Exception):
    """Raised when a user-specified configuration file fails to be loaded or validated."""

    def __init__(self, init_log: Tuple[int, str], reason: Optional[Exception] = None):
        super().__init__()
        self.init_log = init_log
        self.reason = reason


class ConfigRegistry:

    def __init__(self, path: Optional[str] = None, throw_exception: bool = True):
        self._config: ConfigDict = {} # Options set in the loaded config file.
        self._init_log: List[Tuple[int, str]] = []
        self._initialized = False

        try:
            self._load_from_disk(path)
            self._initialized = True

        except ConfigLoadFailure as ex:
            if throw_exception:
                raise
            # If we fail to load the user config file, ensure the object is flagged as
            # uninitialized, and log the error so it can be dealt with if necessary.
            self._init_log = ex.init_log
            if ex.reason is not None:
                self._init_log += [
                    (logging.ERROR, 'Error: %s' % str(ex.reason).replace('\t', '  ')),
                ]
            self._initialized = False

    @property
    def config_dict(self) -> ConfigDict:
        """Current configuration options that are set for each command."""
        return self._config

    @property
    def initialized(self) -> bool:
        """True if the ConfigRegistry was constructed without errors, False otherwise."""
        return self._initialized

    def get_init_log(self):
        """Get initialization log. Consumes the log, so subsequent calls will return None."""
        init_log = self._init_log
        self._init_log = []
        return init_log

    def _log(self, log_level, log_str):
        self._init_log.append((log_level, log_str))

    def _load_from_disk(self, path=None):
        # Validate `path`, or if not provided, use CONFIG_FILE_PATH if it exists.
        if path:
            self._init_log.append((logging.INFO, "Loading config from file:\n  %s" % path))
            if not os.path.exists(path):
                self._init_log.append((logging.ERROR, "File not found: %s" % (path)))
                raise ConfigLoadFailure(self._init_log)
        else:
            # Gracefully handle the case where there isn't a user config file.
            if not os.path.exists(CONFIG_FILE_PATH):
                self._init_log.append((logging.DEBUG, "User config file not found."))
                return
            path = CONFIG_FILE_PATH
            self._init_log.append((logging.INFO, "Loading user config file:\n  %s" % path))
        # Try to load and parse the config file at `path`.
        config = ConfigParser()
        try:
            with open(path, 'r') as config_file:
                config_file_contents = config_file.read()
            config.read_string(config_file_contents, source=path)
        except ParsingError as ex:
            raise ConfigLoadFailure(self._init_log, reason=ex)
        except OSError as ex:
            raise ConfigLoadFailure(self._init_log, reason=ex)
        # At this point the config file syntax is correct, but we need to still validate
        # the parsed options (i.e. that the options have valid values).
        errors = _validate_structure(config)
        if not errors:
            self._config, errors = _parse_config(config)
        if errors:
            for log_str in errors:
                self._init_log.append((logging.ERROR, log_str))
            raise ConfigLoadFailure(self._init_log)

    def is_default(self, command: str, option: str) -> bool:
        """True if specified config option is unset (i.e. the default), False otherwise."""
        return not (command in self._config and option in self._config[command])

    def get_value(self,
                  command: str,
                  option: str,
                  override: Optional[ConfigValue] = None,
                  ignore_default: bool = False) -> ConfigValue:
        """Get the current setting or default value of the specified command option."""
        assert command in CONFIG_MAP and option in CONFIG_MAP[command]
        if override is not None:
            return override
        if command in self._config and option in self._config[command]:
            value = self._config[command][option]
        else:
            value = CONFIG_MAP[command][option]
            if ignore_default:
                return None
        if issubclass(type(value), ValidatedValue):
            return value.value
        return value

    def get_help_string(self,
                        command: str,
                        option: str,
                        show_default: Optional[bool] = None) -> str:
        """Get a string to specify for the help text indicating the current command option value,
        if set, or the default.

        Arguments:
            command: A command name or, "global" for global options.
            option: Command-line option to set within `command`.
            show_default: Always show default value. Default is False for flag/bool values,
                True otherwise.
        """
        assert command in CONFIG_MAP and option in CONFIG_MAP[command]
        is_flag = isinstance(CONFIG_MAP[command][option], bool)
        if command in self._config and option in self._config[command]:
            if is_flag:
                value_str = 'on' if self._config[command][option] else 'off'
            else:
                value_str = str(self._config[command][option])
            return ' [setting: %s]' % (value_str)
        if show_default is False or (show_default is None and is_flag
                                     and CONFIG_MAP[command][option] is False):
            return ''
        return ' [default: %s]' % (str(CONFIG_MAP[command][option]))
