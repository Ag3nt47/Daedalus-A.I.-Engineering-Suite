"""Desktop application entry point."""

from __future__ import annotations

import os
import sys
from importlib.resources import files

_WINDOWS_APP_USER_MODEL_ID = "Daedalus.AIEngineeringSuite"
_APP_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _set_windows_app_user_model_id() -> None:
    """Give the Qt process a stable taskbar identity instead of Python's."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            _WINDOWS_APP_USER_MODEL_ID
        )
    except (AttributeError, OSError):
        # Older or restricted Windows environments can omit the shell API. The
        # embedded executable icon and Qt window icon still provide branding.
        pass


def _application_icon():
    """Build a multi-size QIcon from the package-owned master artwork."""

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon, QPixmap

    icon = QIcon()
    try:
        icon_bytes = (
            files("daedalus")
            .joinpath("assets", "daedalus-app-icon.png")
            .read_bytes()
        )
    except (FileNotFoundError, OSError):
        return icon

    master = QPixmap()
    if not master.loadFromData(icon_bytes, "PNG"):
        return icon
    for dimension in _APP_ICON_SIZES:
        icon.addPixmap(
            master.scaled(
                dimension,
                dimension,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    return icon


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    _set_windows_app_user_model_id()
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from daedalus.gui.main_window import MainWindow
    from daedalus.workspace.manager import WorkspaceManager

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationDisplayName("Daedalus AI Engineering Suite")
    app.setApplicationName("DaedalusAI")
    app.setOrganizationName("Daedalus Contributors")
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    app_icon = _application_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    manager = WorkspaceManager.from_environment()
    manager.bootstrap()
    window = MainWindow(manager)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
