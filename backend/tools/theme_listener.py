from PySide6 import QtCore, QtWidgets, QtGui

from qfluentwidgets import setTheme, qconfig, Theme
import darkdetect


class SystemThemeListener(QtCore.QThread):
    """ System theme listener """

    systemThemeChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def run(self):
        darkdetect.listener(self._onThemeChanged)

    def _onThemeChanged(self, theme: str):
        theme = Theme.DARK if theme.lower() == "dark" else Theme.LIGHT
        setTheme(theme)
        if qconfig.themeMode.value != Theme.AUTO or theme == qconfig.theme:
            return

        qconfig.theme = Theme.AUTO
        qconfig._cfg.themeChanged.emit(Theme.AUTO)
        self.systemThemeChanged.emit()