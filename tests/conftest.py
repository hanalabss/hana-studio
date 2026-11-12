import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
