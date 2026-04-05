# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao（原作者） / 改写：Jason Eric
@Time    : 2023/4/1 6:07 下午（原始时间）
@FileName: gui.py
@desc: 字幕去除器图形化界面（由 PySimpleGUI 改写为 PySide6）
"""

import sys
import os
import configparser
import cv2
import multiprocessing
from PySide6.QtCore import Qt, QTranslator
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QLabel
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, setThemeColor, OptionsConfigItem,
                          OptionsValidator, SubtitleLabel, HollowHandleStyle, qconfig, ConfigItem, QConfig,
                          NavigationWidget, NavigationItemPosition, isDarkTheme, InfoBar)

from qframelesswindow.utils import getSystemAccentColor
from backend.config import config, tr, VERSION
from backend.tools.theme_listener import SystemThemeListener
from backend.tools.process_manager import ProcessManager
from ui.advanced_setting_interface import AdvancedSettingInterface
from ui.home_interface import HomeInterface


class SubtitleExtractorGUI(FluentWindow): 
    def __init__(self):
        super().__init__()
        # 禁用云母效果
        self.setMicaEffectEnabled(False)
        # 设置深色主题并跟随系统主题色
        # setTheme(Theme.LIGHT)
        # setThemeColor(getSystemAccentColor(), save=True)

        # 初始化系统主题监听器并连接信号
        # self.themeListener = SystemThemeListener(self)
        # self.themeListener.start()
 
        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon("design/vsr.ico"))
        self.setWindowTitle(tr['SubtitleExtractorGUI']['Title'] + " v" + VERSION)
        # 创建界面布局
        self._create_layout()
        self._connectSignalToSlot()
        self._lazy_check_update()

    def _lazy_check_update(self):
        """ 延迟检查更新 """
        if not config.checkUpdateOnStartup.value:
            return
        self.check_update_timer = QtCore.QTimer(self)
        self.check_update_timer.setSingleShot(True)
        self.check_update_timer.timeout.connect(lambda: self.advancedSettingInterface.check_update(ignore=True))
        self.check_update_timer.start(2000)

    def _connectSignalToSlot(self):
        config.appRestartSig.connect(self._showRestartTooltip)

    def _showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            'Updated successfully',
            'Configuration takes effect after restart',
            duration=5000,
            parent=self
        )

    def _create_layout(self):
        # 创建主页面和高级设置页面
        self.homeInterface = HomeInterface(self)
        self.homeInterface.setObjectName("HomeInterface")
        self.advancedSettingInterface = AdvancedSettingInterface(self)
        self.advancedSettingInterface.setObjectName("AdvancedSettingInterface")
        
        # 添加到主窗口作为子界面
        self.addSubInterface(self.homeInterface,FluentIcon.HOME, tr['SubtitleExtractorGUI']['Title'])
        self.addSubInterface(self.advancedSettingInterface, FluentIcon.SETTING, tr['Setting']['AdvancedSetting'], NavigationItemPosition.BOTTOM)

    def on_navigation_item_changed(self, key):
        """导航项变更时的处理函数"""
        if key == 'main':
            self.stackWidget.setCurrentIndex(0)
        elif key == 'advanced':
            self.stackWidget.setCurrentIndex(1)

    def closeEvent(self, event):
        """程序关闭时保存窗口位置并恢复标准输出和标准错误"""
        self.save_window_position()
        # 断开信号连接
        # self.themeListener.terminate()
        # self.themeListener.deleteLater()
        ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

    def save_window_position(self):
        """保存窗口位置到配置文件"""
        # 保存窗口位置和大小
        config.set(config.windowX, self.x())
        config.set(config.windowY, self.y())
        config.set(config.windowW, self.width())
        config.set(config.windowH, self.height())

    def update_progress(self):
        # 定时器轮询更新进度（现在更新到视频滑块上）
        if self.se is not None:
            try:
                pos = min(self.frame_count - 1, int(self.se.progress_total / 100 * self.frame_count))
                if pos != self.video_slider.value():
                    self.video_slider.setValue(pos)
                # 检查是否完成
                if self.se.isFinished:
                    self.processing_finished()
            except Exception as e:
                # 捕获任何异常，防止崩溃
                print(f"更新进度时出错: {str(e)}")

    def load_window_position(self):
        # 尝试读取窗口位置
        try:
            x = config.windowX.value
            y = config.windowY.value
            width = config.windowW.value
            height = config.windowH.value

            if not x or not y:
                self.center_window()
                return

            # 确保窗口在屏幕内
            screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
            if (x >= 0 and y >= 0 and 
                x + width <= screen_rect.width() and 
                y + height <= screen_rect.height()):
                self.setGeometry(x, y, width, height)
            else:
                self.center_window()
        except Exception as e:
            print(e)
            self.center_window()
    
    def center_window(self):
        """将窗口居中显示"""
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        window_rect = self.frameGeometry()
        center_point = screen_rect.center()
        window_rect.moveCenter(center_point)
        self.move(window_rect.topLeft())

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 检测Ctrl+C组合键
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            print("\n程序被用户中断(Ctrl+C)，正在退出...")
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    window = SubtitleExtractorGUI()
    # 先设置透明, 再显示, 否则会有闪烁的效果
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()
    # 使用动画效果逐渐显示窗口
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)  # 300毫秒的动画
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    app.exec()