from PySide6 import QtWidgets
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard)
from backend.config import config, tr, HARDWARD_ACCELERATION_OPTION
from backend.tools.constant import InpaintMode, SubtitleDetectMode

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        
        # 界面语言设置
        self.interface_combo = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["SubtitleExtractorGUI"]["InterfaceLanguage"],
            content="",
            parent=parent,
            texts=config.intefaceTexts.keys(),
        )
        self.addWidget(self.interface_combo)
        
        # 处理模式设置
        self.inpaint_mode_combo = ComboBoxSettingCard(
            configItem=config.inpaintMode,
            icon=FluentIcon.GLOBE,
            title=tr["SubtitleExtractorGUI"]["InpaintMode"],
            content="",
            parent=parent,
            texts=[list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)],
        )
        self.inpaint_mode_combo.setToolTip(tr["SubtitleExtractorGUI"]["InpaintModeDesc"])
        self.addWidget(self.inpaint_mode_combo)

        self.subtitle_detect_model_combo = ComboBoxSettingCard(
            configItem=config.subtitleDetectMode,
            icon=FluentIcon.SEARCH,
            title=tr["SubtitleExtractorGUI"]["SubtitleDetectMode"],
            content="",
            parent=parent,
            texts=[mode.value for mode in config.subtitleDetectMode.validator.options],
        )
        self.addWidget(self.subtitle_detect_model_combo)

        # 是否启用硬件加速
        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=parent
        )
        self.addWidget(self.hardware_acceleration)
        # 如果硬件加速选项被禁用, 设置硬件加速为False并只读
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
            config.set(config.hardwareAcceleration, False)
        # 添加一些空间
        self.addStretch(1)
    
    def reset_setting(self):
        """重置所有设置为默认值"""
        # 这里需要实现重置逻辑
        pass