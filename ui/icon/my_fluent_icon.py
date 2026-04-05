from enum import Enum

from qfluentwidgets import getIconColor, Theme, FluentIconBase


class MyFluentIcon(FluentIconBase, Enum):
    Stop = "stop"

    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        return f'./ui/icon/{self.value}_{getIconColor(theme)}.svg'
