"""디자인(카드 꾸미기) 화면.

    page = DesignPage(workspace)
    shell.register_page(PAGE_DESIGN, page, "디자인", "design", step_label="카드 꾸미기")
"""

from .background import BackgroundEngine, RembgEngine
from .page import DesignPage, FileDialogs

__all__ = ["BackgroundEngine", "DesignPage", "FileDialogs", "RembgEngine"]
