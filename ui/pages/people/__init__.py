"""명단 화면.

    page = PeoplePage(workspace)
    shell.register_page(PAGE_LIST, page, "명단", "people", step_label="명단 넣기")

보는 조건(workspace.view_filter, view_order)과 체크한 사람(workspace.checked_ids)은 화면이 알아서 맞춰 둔다.
명단 파일(.hsdata)은 카드 파일이 저장돼 있으면 그 옆에, 아니면 사용자 폴더에 만든다(storage.py).
"""

from .importing import ImportOutcome, ImportPlan, ImportWizard, run_import
from .model import IdsFilter, PeopleTableModel, PhotoIndex
from .page import PeoplePage

__all__ = ["IdsFilter", "ImportOutcome", "ImportPlan", "ImportWizard", "PeoplePage", "PeopleTableModel",
           "PhotoIndex", "run_import"]
