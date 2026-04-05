"""
@desc: 高级设置页面
"""

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout, CardWidget, SubtitleLabel,
                           FluentIcon, NavigationWidget, NavigationItemPosition,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           HyperlinkCard, PrimaryPushSettingCard, PushSettingCard,
                           MessageBox)
from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor

class AdvancedSettingInterface(ScrollArea):
    """高级设置页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__init_widgets()

    def __init_widgets(self):
        # 创建滚动内容的容器
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)
        
        # 设置滚动区域属性
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # 设置滚动区域样式以适应主题
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        
        # 设置UI
        self.setup_ui()
        self.setup_layout()

    def setup_layout(self):
        self.subtitle_detection_group.addSettingCard(self.subtitle_yx_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_deviation_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_y_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_y_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_x_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_backward_frame_count)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_forward_frame_count)
        self.expandLayout.addWidget(self.subtitle_detection_group)

        self.sttn_group.addSettingCard(self.sttn_neighbor_stride)
        self.sttn_group.addSettingCard(self.sttn_reference_length)
        self.sttn_group.addSettingCard(self.sttn_max_load_num)
        self.expandLayout.addWidget(self.sttn_group)

        self.propainter_group.addSettingCard(self.propainter_max_load_num)
        self.expandLayout.addWidget(self.propainter_group)

        self.advanced_group.addSettingCard(self.save_directory)
        self.advanced_group.addSettingCard(self.check_update_on_startup)
        self.expandLayout.addWidget(self.advanced_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        
        self.expandLayout.addWidget(self.about_group)
        self.expandLayout.setSpacing(16)
        self.expandLayout.setContentsMargins(16, 16, 16, 48)
        
    def setup_ui(self):
        """设置UI"""
        # 字幕检测设置组
        self.subtitle_detection_group = SettingCardGroup(tr["Setting"]["SubtitleDetectionSetting"], self.scrollWidget)
        # STTN设置组
        self.sttn_group = SettingCardGroup(tr["Setting"]["SttnSetting"], self.scrollWidget)
        # Propainter设置组
        self.propainter_group = SettingCardGroup(tr["Setting"]["ProPainterSetting"], self.scrollWidget)
        # 高级设置组
        self.advanced_group = SettingCardGroup(tr["Setting"]["AdvancedSetting"], self.scrollWidget)
        # 关于设置组
        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        
        self.subtitle_yx_axis_difference_pixel = RangeSettingCard(
            configItem=config.subtitleYXAxisDifferencePixel,
            icon=FluentIcon.ZOOM,
            title=tr["Setting"]["SubtitleYXAxisDifferencePixel"],
            content=tr["Setting"]["SubtitleYXAxisDifferencePixelDesc"],
            parent=self.subtitle_detection_group
        )
        
        self.subtitle_area_deviation_pixel = RangeSettingCard(
            configItem=config.subtitleAreaDeviationPixel,
            icon=FluentIcon.ZOOM_IN,
            title=tr["Setting"]["SubtitleAreaDeviationPixel"],
            content=tr["Setting"]["SubtitleAreaDeviationPixelDesc"],
            parent=self.subtitle_detection_group
        )
        
        self.subtitle_area_y_axis_difference_pixel = RangeSettingCard(
            configItem=config.subtitleAreaYAxisDifferencePixel,
            icon=FluentIcon.ALIGNMENT,
            title=tr["Setting"]["SubtitleAreaYAxisDifferencePixel"],
            content=tr["Setting"]["SubtitleAreaYAxisDifferencePixelDesc"],
            parent=self.subtitle_detection_group
        )

        self.subtitle_area_pixel_tolerance_y_pixel = RangeSettingCard(
            configItem=config.subtitleAreaPixelToleranceYPixel,
            icon=FluentIcon.UP,
            title=tr["Setting"]["SubtitleAreaPixelToleranceYPixel"],
            content=tr["Setting"]["SubtitleAreaPixelToleranceYPixelDesc"],
            parent=self.subtitle_detection_group
        )

        self.subtitle_area_pixel_tolerance_x_pixel = RangeSettingCard(
            configItem=config.subtitleAreaPixelToleranceXPixel,
            icon=FluentIcon.RIGHT_ARROW,
            title=tr["Setting"]["SubtitleAreaPixelToleranceXPixel"],
            content=tr["Setting"]["SubtitleAreaPixelToleranceXPixelDesc"],
            parent=self.subtitle_detection_group
        )

        self.subtitle_timeline_backward_frame_count = RangeSettingCard(
            configItem=config.subtitleTimelineBackwardFrameCount,
            icon=FluentIcon.PAGE_LEFT,
            title=tr["Setting"]["SubtitleTimelineBackwardFrameCount"],
            content=tr["Setting"]["SubtitleTimelineBackwardFrameCountDesc"],
            parent=self.subtitle_detection_group
        )

        self.subtitle_timeline_forward_frame_count = RangeSettingCard(
            configItem=config.subtitleTimelineForwardFrameCount,
            icon=FluentIcon.PAGE_RIGHT,
            title=tr["Setting"]["subtitleTimelineForwardFrameCount"],
            content=tr["Setting"]["subtitleTimelineForwardFrameCountDesc"],
            parent=self.subtitle_detection_group
        )

        self.sttn_neighbor_stride = RangeSettingCard(
            configItem=config.sttnNeighborStride,
            icon=FluentIcon.UNIT,
            title=tr["Setting"]["SttnNeighborStride"],
            content=tr["Setting"]["SttnNeighborStrideDesc"],
            parent=self.sttn_group
        )

        self.sttn_reference_length = RangeSettingCard(
            configItem=config.sttnReferenceLength,
            icon=FluentIcon.MORE,
            title=tr["Setting"]["SttnReferenceLength"],
            content=tr["Setting"]["SttnReferenceLengthDesc"],
            parent=self.sttn_group
        )

        self.sttn_max_load_num = RangeSettingCard(
            configItem=config.sttnMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["SttnMaxLoadNum"],
            content=tr["Setting"]["SttnMaxLoadNumDesc"],
            parent=self.sttn_group
        )

        self.propainter_max_load_num = RangeSettingCard(
            configItem=config.propainterMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["PropainterMaxLoadNum"],
            content=tr["Setting"]["PropainterMaxLoadNumDesc"],
            parent=self.propainter_group
        )

        # 视频保存路径
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.advanced_group
        )
        self.save_directory.clicked.connect(self.choose_save_directory)

        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.advanced_group
        )

        # 添加反馈链接
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(PROJECT_ISSUES_URL)
        ))
        # 添加版权信息
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())
        # 添加项目链接
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """ show message box """
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """ check software update

        Parameters
        ----------
        ignore: bool
            ignore message box when no updates are available
        """
        TaskExecutor.runTask(self.version_manager.has_new_version).then(
            lambda success: self.on_version_info_fetched(success, ignore))

    def on_version_info_fetched(self, success, ignore=False):
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True,
                lambda: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(PROJECT_RELEASES_URL)
                )
            )
        elif not ignore:
            self.show_message_box(
                tr["Setting"]["NoUpdatesAvailableTitle"],
                tr["Setting"]["NoUpdatesAvailableDesc"],
            )
    
    def choose_save_directory(self):
        """选择保存目录"""
        last_save_directory = "./" if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], last_save_directory)
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)