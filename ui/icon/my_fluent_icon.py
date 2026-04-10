import os
import sys
from enum import Enum

from qfluentwidgets import getIconColor, Theme, FluentIconBase


class MyFluentIcon(FluentIconBase, Enum):
    Stop = "stop"

    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        # 支持打包环境和开发环境
        if getattr(sys, 'frozen', False):
            # 打包环境：使用 sys._MEIPASS 作为基础路径
            base_path = sys._MEIPASS
        else:
            # 开发环境：使用相对路径
            base_path = os.path.join(os.path.dirname(__file__), '..')

        return os.path.join(base_path, 'ui', 'icon', f'{self.value}_{getIconColor(theme)}.svg')
