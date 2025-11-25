"""
HanaStudioMainWindowV2 - Canva 스타일 메인 윈도우
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QToolBar, QLabel,
    QPushButton, QMessageBox, QVBoxLayout, QFileDialog, QProgressDialog
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction

from canvas import CardCanvas
from canvas.layers import ImageLayer, TextLayer
from .panels import ElementPanel, PropertyPanel, LayerPanel
from .styles_v2 import get_canva_style


class HanaStudioMainWindowV2(QMainWindow):
    """Hana Studio v2 메인 윈도우"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hana Studio v2 - RTAI 카드 디자인 툴")
        self.setGeometry(100, 100, 1400, 900)

        # 스타일 적용
        self.setStyleSheet(get_canva_style())

        # 캔버스 (앞면/뒷면)
        self.front_canvas = CardCanvas()
        self.back_canvas_view = CardCanvas()
        self.canvas = self.front_canvas  # 현재 활성 캔버스
        self.current_side = "front"  # front 또는 back

        # 선택된 레이어
        self.selected_layer = None

        # 프린터 스레드
        self.current_printer_thread = None

        # 현재 프로젝트 파일
        self.current_project_path = None

        # 마스크 이미지 경로 (레이아웃 인쇄용)
        self.mask_image_path = None
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None

        # 프린터 관련 속성
        self.printer_dll_path = None
        self.printer_available = False
        self.selected_printer_info = None

        # UI 설정
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._connect_signals()

        # AI 모델 로더 초기화 (백그라운드에서 로드)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, self._init_ai_model)

        # 프린터 초기화 (백그라운드에서 지연 실행)
        QTimer.singleShot(2000, self._lazy_init_printer)

    def _setup_ui(self):
        """UI 레이아웃 설정"""
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃 (3단 구성)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 왼쪽 패널 (요소 추가)
        self.element_panel = ElementPanel()
        main_layout.addWidget(self.element_panel)

        # 중앙 캔버스 영역
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(8, 8, 8, 8)

        # 앞면/뒷면 캔버스를 나란히 표시
        canvas_row = QHBoxLayout()

        # 앞면 캔버스
        front_group = QVBoxLayout()

        # 앞면 라벨과 선택 버튼
        front_header = QHBoxLayout()
        front_label = QLabel("앞면")
        front_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        front_label.setStyleSheet("font-weight: bold; padding: 4px;")
        front_header.addWidget(front_label)

        self.front_select_btn = QPushButton("앞면 편집")
        self.front_select_btn.setObjectName("primary_btn")
        self.front_select_btn.clicked.connect(self._switch_to_front_canvas)
        front_header.addWidget(self.front_select_btn)
        front_group.addLayout(front_header)

        front_group.addWidget(self.canvas)
        canvas_row.addLayout(front_group)

        # 뒷면 캔버스
        back_group = QVBoxLayout()

        # 뒷면 라벨과 선택 버튼
        back_header = QHBoxLayout()
        back_label = QLabel("뒷면")
        back_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        back_label.setStyleSheet("font-weight: bold; padding: 4px;")
        back_header.addWidget(back_label)

        self.back_select_btn = QPushButton("뒷면 편집")
        self.back_select_btn.setObjectName("secondary_btn")
        self.back_select_btn.clicked.connect(self._switch_to_back_canvas)
        back_header.addWidget(self.back_select_btn)
        back_group.addLayout(back_header)

        back_group.addWidget(self.back_canvas_view)
        canvas_row.addLayout(back_group)

        canvas_layout.addLayout(canvas_row)

        # 줌 컨트롤
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(0, 4, 0, 4)

        self.zoom_label = QLabel("줌: 100%")
        zoom_layout.addWidget(self.zoom_label)

        zoom_50_btn = QPushButton("50%")
        zoom_50_btn.clicked.connect(lambda: self.canvas.set_zoom(0.5))
        zoom_layout.addWidget(zoom_50_btn)

        zoom_100_btn = QPushButton("100%")
        zoom_100_btn.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        zoom_layout.addWidget(zoom_100_btn)

        zoom_200_btn = QPushButton("200%")
        zoom_200_btn.clicked.connect(lambda: self.canvas.set_zoom(2.0))
        zoom_layout.addWidget(zoom_200_btn)

        zoom_fit_btn = QPushButton("화면 맞춤")
        zoom_fit_btn.clicked.connect(self.canvas.fit_in_view)
        zoom_layout.addWidget(zoom_fit_btn)

        zoom_layout.addStretch()

        canvas_layout.addWidget(zoom_widget)

        main_layout.addWidget(canvas_container, 1)  # 가변 너비

        # 오른쪽 패널 컨테이너
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 레이어 패널
        self.layer_panel = LayerPanel()
        right_layout.addWidget(self.layer_panel)

        # 속성 패널
        self.property_panel = PropertyPanel()
        right_layout.addWidget(self.property_panel)

        main_layout.addWidget(right_container)

    def _setup_menubar(self):
        """메뉴바 설정"""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일")

        # 새 프로젝트
        new_action = QAction("새 프로젝트", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_design)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        # 프로젝트 저장
        save_action = QAction("프로젝트 저장", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        # 다른 이름으로 저장
        save_as_action = QAction("다른 이름으로 저장...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)

        # 프로젝트 열기
        open_action = QAction("프로젝트 열기...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # 이미지로 내보내기
        export_action = QAction("이미지로 내보내기...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

    def _setup_toolbar(self):
        """툴바 설정"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 툴바 액션
        new_action = QAction("새로운 디자인", self)
        new_action.triggered.connect(self.new_design)
        toolbar.addAction(new_action)

        toolbar.addSeparator()

        undo_action = QAction("실행취소", self)
        undo_action.setShortcut("Ctrl+Z")
        toolbar.addAction(undo_action)

        redo_action = QAction("다시실행", self)
        redo_action.setShortcut("Ctrl+Y")
        toolbar.addAction(redo_action)

        toolbar.addSeparator()

        delete_action = QAction("삭제", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_action)

        toolbar.addSeparator()

        # 수동 마스킹 버튼
        manual_mask_action = QAction("수동 마스킹", self)
        manual_mask_action.triggered.connect(self._open_manual_masking_dialog)
        toolbar.addAction(manual_mask_action)

    def _connect_signals(self):
        """시그널 연결"""
        # 요소 패널
        self.element_panel.image_add_requested.connect(self.add_image_layer)
        self.element_panel.text_add_requested.connect(self.add_text_layer)
        self.element_panel.bg_remove_requested.connect(self.remove_background)

        # 속성 패널
        self.property_panel.print_requested.connect(self.print_card)
        self.property_panel.fit_to_canvas_requested.connect(self.fit_layer_to_canvas)
        self.property_panel.orientation_changed.connect(self._on_canvas_orientation_changed)
        self.property_panel.print_mode_changed.connect(self._on_print_mode_changed)

        # 레이어 패널
        self.layer_panel.layer_selected.connect(self._on_layer_panel_selection)
        self.layer_panel.layer_order_changed.connect(self._on_layer_order_changed)

        # 앞면 캔버스 시그널
        self.front_canvas.zoom_changed.connect(self._on_zoom_changed)
        self.front_canvas.scene.selectionChanged.connect(self._on_selection_changed)

        # 뒷면 캔버스 시그널
        self.back_canvas_view.zoom_changed.connect(self._on_zoom_changed)
        self.back_canvas_view.scene.selectionChanged.connect(self._on_selection_changed)

    def _init_ai_model(self):
        """AI 모델 백그라운드 로딩 시작"""
        try:
            from core.model_loader import get_model_loader

            model_loader = get_model_loader()
            model_loader.set_parent_widget(self)

            # 백그라운드 로딩 시작
            if not model_loader.is_loaded and not model_loader.is_loading:
                print("[AI] AI 모델 백그라운드 로딩 시작...")
                model_loader.start_background_loading()
            else:
                print("[AI] AI 모델이 이미 로드되었거나 로딩 중입니다.")
        except Exception as e:
            print(f"[ERROR] AI 모델 초기화 실패: {e}")

    @Slot(str)
    def _on_print_mode_changed(self, mode: str):
        """인쇄 모드 변경 이벤트 핸들러

        Args:
            mode: "normal" (일반 인쇄) 또는 "layered" (레이아웃 인쇄)
        """
        print(f"[MODE] 인쇄 모드 변경: {mode}")

        if mode == "normal":
            # 일반 인쇄 모드: ORIGINAL 표시
            self.update_canvas_for_normal_print()

        elif mode == "layered":
            # 레이아웃 인쇄 모드: 마스크 여부에 따라 다르게 표시
            if self.mask_image_path and self.selected_layer and self.selected_layer.mask_pixmap:
                # 마스크 있음: 시뮬레이션 표시
                self.update_canvas_for_layered_print_with_mask()
            else:
                # 마스크 없음: ORIGINAL 표시
                self.update_canvas_for_layered_print_no_mask()

    @Slot()
    def _switch_to_front_canvas(self):
        """앞면 캔버스로 전환"""
        if self.current_side == "front":
            return

        self.current_side = "front"
        self.canvas = self.front_canvas

        # 버튼 스타일 업데이트
        self.front_select_btn.setObjectName("primary_btn")
        self.back_select_btn.setObjectName("secondary_btn")
        self.front_select_btn.style().unpolish(self.front_select_btn)
        self.front_select_btn.style().polish(self.front_select_btn)
        self.back_select_btn.style().unpolish(self.back_select_btn)
        self.back_select_btn.style().polish(self.back_select_btn)

        # 레이어 패널 업데이트
        self.layer_panel.update_layers(self.canvas.scene)

        # 선택 상태 초기화
        self.selected_layer = None
        self.property_panel.set_layer(None)

        print("[INFO] 앞면 캔버스로 전환")

    @Slot()
    def _switch_to_back_canvas(self):
        """뒷면 캔버스로 전환"""
        if self.current_side == "back":
            return

        self.current_side = "back"
        self.canvas = self.back_canvas_view

        # 버튼 스타일 업데이트
        self.front_select_btn.setObjectName("secondary_btn")
        self.back_select_btn.setObjectName("primary_btn")
        self.front_select_btn.style().unpolish(self.front_select_btn)
        self.front_select_btn.style().polish(self.front_select_btn)
        self.back_select_btn.style().unpolish(self.back_select_btn)
        self.back_select_btn.style().polish(self.back_select_btn)

        # 레이어 패널 업데이트
        self.layer_panel.update_layers(self.canvas.scene)

        # 선택 상태 초기화
        self.selected_layer = None
        self.property_panel.set_layer(None)

        print("[INFO] 뒷면 캔버스로 전환")

    @Slot(str)
    def add_image_layer(self, image_path: str):
        """이미지 레이어 추가"""
        try:
            layer = ImageLayer(image_path)

            # 캔버스 크기
            card_rect = self.canvas.scene.get_card_rect()
            card_width = card_rect.width()
            card_height = card_rect.height()

            # 이미지가 캔버스보다 크면 축소
            if layer.image_width > card_width or layer.image_height > card_height:
                # 캔버스에 맞게 비율 유지하며 축소
                scale_w = card_width / layer.image_width
                scale_h = card_height / layer.image_height
                scale = min(scale_w, scale_h) * 0.9  # 90%로 여유 공간 확보

                new_width = int(layer.image_width * scale)
                new_height = int(layer.image_height * scale)

                layer.set_size(new_width, new_height)
                print(f"[INFO] 이미지 축소: {layer.image_width}x{layer.image_height} → {new_width}x{new_height}")

            # 캔버스 중앙에 배치
            layer.setPos(
                card_rect.center().x() - layer.boundingRect().width() / 2,
                card_rect.center().y() - layer.boundingRect().height() / 2
            )

            self.canvas.scene.addItem(layer)
            layer.setSelected(True)

            # 레이어 패널 업데이트
            self.layer_panel.update_layers(self.canvas.scene)

            # 양면인쇄 체크박스 업데이트
            self._update_dual_side_checkbox()

            print(f"[OK] 이미지 레이어 추가: {image_path}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 추가 실패:\n{e}")
            print(f"[ERROR] 이미지 추가 실패: {e}")

    @Slot()
    def add_text_layer(self):
        """텍스트 레이어 추가"""
        try:
            layer = TextLayer("텍스트를 입력하세요")

            # 캔버스 중앙에 배치
            card_rect = self.canvas.scene.get_card_rect()
            layer.setPos(
                card_rect.center().x() - layer.boundingRect().width() / 2,
                card_rect.center().y() - layer.boundingRect().height() / 2
            )

            self.canvas.scene.addItem(layer)
            layer.setSelected(True)

            # 레이어 패널 업데이트
            self.layer_panel.update_layers(self.canvas.scene)

            # 양면인쇄 체크박스 업데이트
            self._update_dual_side_checkbox()

            print("[OK] 텍스트 레이어 추가")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"텍스트 추가 실패:\n{e}")
            print(f"[ERROR] 텍스트 추가 실패: {e}")

    def _update_dual_side_checkbox(self):
        """뒷면 캔버스 상태에 따라 양면인쇄 체크박스 자동 업데이트"""
        # 뒷면 캔버스에 레이어가 있는지 확인
        has_back_layers = False
        for item in self.back_canvas_view.scene.items():
            if isinstance(item, (ImageLayer, TextLayer)):
                has_back_layers = True
                break

        # 양면인쇄 체크박스 자동 업데이트
        self.property_panel.dual_side_checkbox.setChecked(has_back_layers)

        if has_back_layers:
            print("[INFO] 뒷면 레이어 감지 → 양면인쇄 자동 선택")
        else:
            print("[INFO] 뒷면 레이어 없음 → 양면인쇄 자동 해제")

    @Slot()
    def delete_selected(self):
        """선택된 레이어 삭제"""
        selected_items = self.canvas.scene.selectedItems()

        if not selected_items:
            return

        for item in selected_items:
            # 레이어 아이템만 삭제 (핸들 등은 제외)
            if isinstance(item, (ImageLayer, TextLayer)):
                self.canvas.scene.removeItem(item)

        # 레이어 패널 업데이트
        self.layer_panel.update_layers(self.canvas.scene)

        # 양면인쇄 체크박스 업데이트
        self._update_dual_side_checkbox()

        print(f"[OK] {len(selected_items)}개 레이어 삭제")

    @Slot()
    def _on_selection_changed(self):
        """선택 변경"""
        selected_items = self.canvas.scene.selectedItems()

        # 레이어만 필터링
        selected_layers = [item for item in selected_items if isinstance(item, (ImageLayer, TextLayer))]

        if selected_layers:
            self.selected_layer = selected_layers[0]
            self.property_panel.set_layer(self.selected_layer)
            # 레이어 패널도 동기화
            self.layer_panel.set_selected_layer(self.selected_layer)
        else:
            self.selected_layer = None
            self.property_panel.set_layer(None)

    @Slot(float)
    def _on_zoom_changed(self, zoom_level):
        """줌 레벨 변경"""
        percentage = int(zoom_level * 100)
        self.zoom_label.setText(f"줌: {percentage}%")

    @Slot()
    def new_design(self):
        """새 디자인"""
        reply = QMessageBox.question(
            self,
            "새 디자인",
            "현재 작업을 초기화하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 앞면 모든 레이어 삭제
            for item in self.front_canvas.scene.items():
                if isinstance(item, (ImageLayer, TextLayer)):
                    self.front_canvas.scene.removeItem(item)

            # 뒷면 모든 레이어 삭제
            for item in self.back_canvas_view.scene.items():
                if isinstance(item, (ImageLayer, TextLayer)):
                    self.back_canvas_view.scene.removeItem(item)

            # 레이어 패널 업데이트
            self.layer_panel.update_layers(self.canvas.scene)

            # 양면인쇄 체크박스 업데이트
            self._update_dual_side_checkbox()

            self.canvas.reset_zoom()
            print("[OK] 새 디자인 시작")

    @Slot()
    def print_card(self):
        """카드 인쇄 - 600DPI 렌더링 + 프린터 통합"""
        try:
            # 인쇄 방향, 모드, 양면인쇄, 매수 가져오기
            orientation = self.property_panel.get_print_orientation()
            orientation_text = "가로" if orientation == "landscape" else "세로"

            print_mode = self.property_panel.get_print_mode()
            print_mode_text = "일반 인쇄" if print_mode == "normal" else "레이아웃 인쇄 (YMCW)"

            is_dual_side = self.property_panel.get_is_dual_side()
            side_text = "양면" if is_dual_side else "단면 (앞면만)"

            print_quantity = self.property_panel.get_print_quantity()

            # 레이어 변수 초기화 (일반 인쇄 모드에서도 사용)
            front_layer_with_mask = None
            back_layer_with_mask = None

            # 레이어 모드 검증 및 마스크 저장
            if print_mode == "layered":
                from core.file_manager import FileManager
                from canvas.layers import ImageLayer
                file_mgr = FileManager()

                # 앞면 캔버스에서 마스크가 있는 ImageLayer 검색 (선택 상태와 무관)
                front_layer_with_mask = None
                for item in self.front_canvas.scene.items():
                    if isinstance(item, ImageLayer) and hasattr(item, 'mask_pixmap') and item.mask_pixmap:
                        front_layer_with_mask = item
                        break

                if front_layer_with_mask:
                    print("[PRINT] 레이어 모드: 앞면 마스크 저장 중...")
                    self.front_saved_mask_path = file_mgr.save_mask_for_printing(
                        front_layer_with_mask.mask_pixmap,
                        front_layer_with_mask.image_path,
                        "front"
                    )
                    print(f"[PRINT] 앞면 마스크 저장: {self.front_saved_mask_path}")
                else:
                    QMessageBox.warning(self, "경고", "레이어 인쇄를 위해서는 마스킹 이미지가 필요합니다.\n배경 제거를 먼저 수행해주세요.")
                    return

                # 뒷면 마스크 확인 및 저장 (양면인쇄인 경우)
                if is_dual_side:
                    # 뒷면 캔버스에서 마스크가 있는 ImageLayer 검색 (선택 상태와 무관)
                    back_layer_with_mask = None
                    for item in self.back_canvas_view.scene.items():
                        if isinstance(item, ImageLayer) and hasattr(item, 'mask_pixmap') and item.mask_pixmap:
                            back_layer_with_mask = item
                            break

                    if back_layer_with_mask:
                        self.back_saved_mask_path = file_mgr.save_mask_for_printing(
                            back_layer_with_mask.mask_pixmap,
                            back_layer_with_mask.image_path,
                            "back"
                        )
                        print(f"[PRINT] 뒷면 마스크 저장: {self.back_saved_mask_path}")
                    else:
                        print("[PRINT] 뒷면 마스크 없음 (일반 인쇄로 진행)")
                        self.back_saved_mask_path = None

            # 인쇄 이미지 준비
            import tempfile
            import os
            from utils.safe_temp_path import get_cached_safe_temp_dir

            temp_dir = get_cached_safe_temp_dir()

            # 레이어 인쇄: 원본 이미지 경로 사용 (V1과 동일)
            # 일반 인쇄: 캔버스 렌더링 사용
            if print_mode == "layered" and front_layer_with_mask:
                # 레이어 인쇄: 원본 이미지를 프린터에 전달 (마스크는 별도)
                temp_image_path = front_layer_with_mask.image_path
                print(f"[PRINT] 레이어 인쇄 - 원본 이미지 사용: {temp_image_path}")
            else:
                # 일반 인쇄: 캔버스 렌더링
                from renderer import PrintRenderer
                renderer = PrintRenderer(self.front_canvas.scene)
                temp_image_path = os.path.join(temp_dir, "hana_studio_v2_print_front.png")
                success = renderer.save_to_file(temp_image_path, dpi=600)

                if not success:
                    QMessageBox.critical(self, "오류", "앞면 이미지 렌더링에 실패했습니다.")
                    return

                print(f"[OK] 앞면 600DPI 렌더링 완료: {temp_image_path}")

            # 뒷면 이미지 준비
            temp_back_image_path = None
            if is_dual_side:
                # 레이어 인쇄 + 뒷면 마스크 있으면: 원본 이미지 경로 사용
                if print_mode == "layered" and back_layer_with_mask:
                    temp_back_image_path = back_layer_with_mask.image_path
                    print(f"[PRINT] 레이어 인쇄 - 뒷면 원본 이미지 사용: {temp_back_image_path}")
                else:
                    # 일반 인쇄 또는 뒷면 마스크 없음: 캔버스 렌더링
                    from renderer import PrintRenderer
                    back_renderer = PrintRenderer(self.back_canvas_view.scene)
                    temp_back_render_path = os.path.join(temp_dir, "hana_studio_v2_print_back_temp.png")
                    back_success = back_renderer.save_to_file(temp_back_render_path, dpi=600)

                    if back_success:
                        temp_back_image_path = temp_back_render_path
                        print(f"[OK] 뒷면 캔버스 렌더링 완료")
                    else:
                        QMessageBox.warning(self, "경고", "뒷면 이미지 렌더링에 실패했습니다. 단면으로 인쇄됩니다.")
                        is_dual_side = False

            # 프린터 모듈 체크
            try:
                from printer import find_printer_dll
                from printer.printer_thread import print_manager

                printer_dll_path = find_printer_dll()

                if not printer_dll_path:
                    # DLL 없으면 이미지만 저장
                    reply = QMessageBox.question(
                        self,
                        "프린터 DLL 없음",
                        f"프린터 DLL을 찾을 수 없습니다.\n\n"
                        f"600DPI 이미지가 생성되었습니다:\n{temp_image_path}\n\n"
                        f"이미지를 데스크톱에 저장하시겠습니까?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )

                    if reply == QMessageBox.StandardButton.Yes:
                        import shutil
                        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                        save_path = os.path.join(desktop, "hana_studio_v2_print.png")
                        shutil.copy(temp_image_path, save_path)
                        QMessageBox.information(self, "저장 완료", f"이미지가 저장되었습니다:\n{save_path}")
                        print(f"[OK] 이미지 저장: {save_path}")

                    return

                # 이미 인쇄 중인지 확인
                if print_manager.get_print_status()['is_printing']:
                    QMessageBox.warning(self, "경고", "이미 인쇄가 진행 중입니다.")
                    return

                # 인쇄 확인 다이얼로그
                reply = QMessageBox.question(
                    self,
                    "카드 인쇄",
                    f"카드를 인쇄하시겠습니까?\n\n"
                    f"• 해상도: 600 DPI\n"
                    f"• 방향: {orientation_text}\n"
                    f"• 모드: {print_mode_text}\n"
                    f"• 매수: {print_quantity}장\n"
                    f"• 출력: {side_text}\n\n"
                    f"프린터에 카드가 준비되어 있는지 확인해주세요.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    return

                # 프린터 스레드 시작
                self._start_printing(printer_dll_path, temp_image_path, temp_back_image_path, orientation, print_mode, is_dual_side, print_quantity)

            except ImportError as e:
                # 프린터 모듈 없음
                QMessageBox.information(
                    self,
                    "프린터 모듈 없음",
                    f"600DPI 이미지가 생성되었습니다:\n{temp_image_path}\n\n"
                    f"프린터 모듈을 찾을 수 없습니다.\n{e}"
                )
                print(f"[INFO] 프린터 모듈 없음: {e}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"인쇄 중 오류가 발생했습니다:\n{e}")
            print(f"[ERROR] 인쇄 오류: {e}")

    def _start_printing(self, dll_path: str, image_path: str, back_image_path: str = None, orientation: str = "portrait", print_mode: str = "normal", is_dual_side: bool = False, quantity: int = 1):
        """프린터 스레드 시작"""
        try:
            from printer.printer_thread import print_manager

            side_text = "양면" if is_dual_side else "단면"
            print(f"[PRINTER] 프린터 인쇄 시작 (방향: {orientation}, 모드: {print_mode}, {side_text}, 매수: {quantity})...")

            # 마스크 경로는 레이아웃 인쇄 모드일 때만 전달
            front_mask = self.front_saved_mask_path if print_mode == "layered" else None
            back_mask = self.back_saved_mask_path if (is_dual_side and print_mode == "layered") else None

            if print_mode == "layered" and front_mask:
                print(f"[PRINTER] 레이아웃 인쇄 - 앞면 마스크: {front_mask}")
                if back_mask:
                    print(f"[PRINTER] 레이아웃 인쇄 - 뒷면 마스크: {back_mask}")

            # 뒷면 이미지 설정
            back_image = back_image_path if back_image_path else None
            if back_image:
                print(f"[PRINTER] 뒷면 이미지: {back_image}")

            # 프린터 스레드 시작 (V1과 동일하게 selected_printer=None으로 전달)
            # Note: selected_printer를 전달하면 auto_select_printer()가 enum_printers() 전에 호출되어 오류 발생
            self.current_printer_thread = print_manager.start_multi_print(
                dll_path=dll_path,
                front_image_path=image_path,
                back_image_path=back_image,
                front_mask_path=front_mask,
                back_mask_path=back_mask,
                print_mode=print_mode,
                is_dual_side=is_dual_side,
                quantity=quantity,
                front_orientation=orientation,
                back_orientation=orientation,
                adjusted_x=0.0,
                adjusted_y=0.0,
                selected_printer=None  # V1과 동일: enum_printers() 후 첫 번째 프린터 자동 선택
            )

            # 시그널 연결
            self.current_printer_thread.progress.connect(self._on_printer_progress)
            self.current_printer_thread.finished.connect(self._on_printer_finished)
            self.current_printer_thread.error.connect(self._on_printer_error)
            self.current_printer_thread.print_progress.connect(self._on_print_progress)
            self.current_printer_thread.card_completed.connect(self._on_card_completed)

            # 스레드 시작
            self.current_printer_thread.start()

        except Exception as e:
            error_msg = f"인쇄 시작 실패: {e}"
            print(f"[ERROR] {error_msg}")
            QMessageBox.critical(self, "인쇄 오류", error_msg)

    @Slot(str)
    def _on_printer_progress(self, message: str):
        """프린터 진행 상황"""
        print(f"[PRINTER] {message}")
        # TODO: 진행 상태를 UI에 표시

    @Slot(bool)
    def _on_printer_finished(self, success: bool):
        """프린터 작업 완료"""
        if success:
            print("[OK] 인쇄 완료!")
            QMessageBox.information(self, "인쇄 완료", "카드 인쇄가 완료되었습니다.")
        else:
            print("[ERROR] 인쇄 실패")
            QMessageBox.warning(self, "인쇄 실패", "카드 인쇄에 실패했습니다.")

    @Slot(str)
    def _on_printer_error(self, error_msg: str):
        """프린터 오류"""
        print(f"[ERROR] 프린터 오류: {error_msg}")
        QMessageBox.critical(self, "프린터 오류", f"인쇄 중 오류가 발생했습니다:\n{error_msg}")

    @Slot(int, int)
    def _on_print_progress(self, current: int, total: int):
        """인쇄 진행률"""
        print(f"[PRINTER] 인쇄 진행: {current}/{total}")

    @Slot(int)
    def _on_card_completed(self, card_num: int):
        """카드 완료"""
        print(f"[PRINTER] 카드 {card_num} 완료")

    @Slot()
    def remove_background(self):
        """배경 제거 (마스킹 처리) - 임계값 조정 가능"""
        from canvas.layers import ImageLayer

        if not self.selected_layer or not isinstance(self.selected_layer, ImageLayer):
            QMessageBox.warning(self, "경고", "이미지 레이어를 선택해주세요.")
            return

        try:
            # 이미지 경로 확인
            if not self.selected_layer.image_path:
                QMessageBox.warning(self, "경고", "이미지 경로를 찾을 수 없습니다.")
                return

            # AI 모델 로드
            from core.model_loader import get_ai_session

            session = get_ai_session()
            if not session:
                QMessageBox.critical(self, "오류", "AI 모델을 로드할 수 없습니다.")
                return

            # 마스킹 다이얼로그 표시
            from ui_v2.dialogs import MaskingDialog

            dialog = MaskingDialog(self.selected_layer.image_path, session, self)
            dialog.threshold_applied.connect(self._on_masking_applied)

            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"배경제거 실패:\n{e}")
            print(f"[ERROR] 배경제거 오류: {e}")

    # ============================================================
    # 캔버스 업데이트 함수 (인쇄 모드별 미리보기)
    # ============================================================

    def update_canvas_for_normal_print(self):
        """일반 인쇄 모드: ORIGINAL 이미지 표시"""
        if not self.selected_layer:
            return

        print("[CANVAS] 일반 인쇄 모드 → 원본 이미지 표시")
        self.selected_layer.pixmap = self.selected_layer.original_pixmap.copy()
        self.selected_layer.update()

    def update_canvas_for_layered_print_no_mask(self):
        """레이아웃 인쇄 모드 (마스크 없음): ORIGINAL 이미지 표시"""
        if not self.selected_layer:
            return

        print("[CANVAS] 레이아웃 인쇄 모드 (마스크 없음) → 원본 이미지 표시")
        self.selected_layer.pixmap = self.selected_layer.original_pixmap.copy()
        self.selected_layer.update()

    def update_canvas_for_layered_print_with_mask(self):
        """레이아웃 인쇄 모드 (마스크 있음): 시뮬레이션 표시"""
        if not self.selected_layer or not self.selected_layer.mask_pixmap:
            return

        print("[CANVAS] 레이아웃 인쇄 모드 (마스크 있음) → 합성 결과 표시 (원본 + 녹색 테두리)")

        # 합성 결과 생성 (원본 + 녹색 테두리)
        composite = self._create_layered_print_simulation(
            self.selected_layer.original_pixmap,
            self.selected_layer.mask_pixmap
        )

        if composite:
            self.selected_layer.pixmap = composite
            self.selected_layer.update()

    def _create_layered_print_simulation(self, original_pixmap, mask_pixmap):
        """합성 결과 생성 (원본 + 녹색 테두리)

        Args:
            original_pixmap: 원본 QPixmap
            mask_pixmap: 마스크 QPixmap

        Returns:
            합성 결과 QPixmap (원본 + 마스킹 영역 녹색 테두리)
        """
        try:
            import cv2
            import numpy as np
            from PySide6.QtGui import QImage, QPixmap
            import tempfile
            import os

            # QPixmap을 numpy 배열로 변환
            original_img = original_pixmap.toImage()
            mask_img = mask_pixmap.toImage()

            # QImage → numpy 배열
            w, h = original_img.width(), original_img.height()
            ptr = original_img.constBits()
            original = np.array(ptr).reshape(h, w, 4)  # RGBA
            original = cv2.cvtColor(original, cv2.COLOR_RGBA2BGR)

            w, h = mask_img.width(), mask_img.height()
            ptr = mask_img.constBits()
            mask = np.array(ptr).reshape(h, w, 4)  # RGBA
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)

            # 크기가 다르면 마스크 리사이즈
            if original.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

            # === 합성 결과: 원본 + 녹색 테두리 ===
            result = original.copy()

            # 마스크의 검은색 영역 = W 레이어 인쇄 영역
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            object_mask = (gray_mask < 128).astype(np.uint8)  # 검은색 = 객체 영역

            # 윤곽선 찾기
            contours, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 녹색 테두리 그리기 (레이어 분리 표시)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 3)

            # BGR을 RGB로 변환 (Qt는 RGB 형식 사용)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            print(f"[CANVAS] 합성 결과 생성: 원본 + {len(contours)}개 테두리")

            # numpy 배열 → QPixmap
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "composite_temp.jpg")

            # 한글 경로 대응
            success, encoded_image = cv2.imencode('.jpg', result_rgb)
            if success:
                with open(temp_path, 'wb') as f:
                    f.write(encoded_image.tobytes())

                composite_pixmap = QPixmap(temp_path)
                return composite_pixmap
            else:
                print("[ERROR] 합성 이미지 인코딩 실패")
                return None

        except Exception as e:
            print(f"[ERROR] 합성 결과 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    @Slot(int, str)
    def _on_masking_applied(self, threshold: int, mask_path: str):
        """배경제거 적용 - 새로운 UX 설계"""
        try:
            from PySide6.QtGui import QPixmap
            import cv2
            import numpy as np

            if not self.selected_layer:
                raise ValueError("선택된 레이어가 없습니다")

            # 1. 마스크 경로 저장 (인쇄 시 사용)
            self.mask_image_path = mask_path
            print(f"[INFO] 마스킹 적용: 원본={self.selected_layer.image_path}, 마스크={mask_path}")

            # 2. ImageLayer에 마스크 QPixmap 저장
            self.selected_layer.mask_pixmap = QPixmap(mask_path)
            print("[INFO] 마스크 이미지를 ImageLayer에 저장")

            # 3. FINAL_RESULT 생성 (비교용 - 마스킹 영역 녹색 강조)
            from core.file_manager import FileManager
            file_mgr = FileManager()
            original = file_mgr._safe_imread(self.selected_layer.image_path)
            mask = file_mgr._safe_imread(mask_path)

            if original is None or mask is None:
                raise ValueError("이미지 로드 실패")

            # 비교용 최종 결과: 마스킹 영역을 녹색으로 강조
            final_result = original.copy()
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masked_area = (gray_mask < 128)  # 검은색 = 마스킹 영역
            final_result[masked_area, 1] = np.clip(final_result[masked_area, 1] + 50, 0, 255)

            # numpy → QPixmap (비교용)
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            final_result_path = os.path.join(temp_dir, f"final_result_{os.path.basename(self.selected_layer.image_path)}")

            success, encoded_image = cv2.imencode('.jpg', final_result)
            if success:
                with open(final_result_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
                self.selected_layer.final_result_pixmap = QPixmap(final_result_path)
                print("[INFO] 비교용 FINAL_RESULT 생성 완료")

            # 4. 자동으로 레이아웃 인쇄 모드로 전환
            current_mode = self.property_panel.get_print_mode()
            if current_mode != "layered":
                print("[AUTO] 배경제거 완료 → 레이아웃 인쇄 모드로 자동 전환")
                self.property_panel.set_print_mode("layered")
                # set_print_mode가 _on_print_mode_changed를 트리거하여 캔버스 업데이트

            # 5. 이미 레이아웃 모드인 경우 수동으로 캔버스 업데이트
            else:
                print("[CANVAS] 합성 결과 업데이트")
                self.update_canvas_for_layered_print_with_mask()

            # 6. 캔버스 크기에 자동 맞춤
            self.fit_layer_to_canvas()

            print(f"[OK] 배경제거 적용 완료 - 임계값: {threshold}")
            print("[OK] 캔버스에 합성 결과 표시 (원본 + 녹색 테두리)")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"마스크 적용 실패:\n{e}")
            print(f"[ERROR] 마스크 적용 오류: {e}")
            import traceback
            traceback.print_exc()

    @Slot()
    def _open_manual_masking_dialog(self):
        """수동 마스킹 다이얼로그 열기"""
        try:
            from ui_v2.dialogs import ManualMaskingDialog

            dialog = ManualMaskingDialog(self)
            dialog.masking_applied.connect(self._on_manual_masking_applied)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"수동 마스킹 다이얼로그 실행 실패:\n{e}")
            print(f"[ERROR] 수동 마스킹 다이얼로그 오류: {e}")
            import traceback
            traceback.print_exc()

    @Slot(int, str, str, int, int, int, int)
    def _on_manual_masking_applied(self, threshold, mask_path, original_path,
                                   offset_x_orig, offset_y_orig, offset_x_mask, offset_y_mask):
        """수동 마스킹 적용 - 새로운 UX 설계"""
        try:
            from PySide6.QtGui import QPixmap
            import cv2
            import numpy as np
            import tempfile
            import os

            if not self.selected_layer:
                raise ValueError("선택된 레이어가 없습니다")

            # 1. 마스크 및 오프셋 저장 (인쇄 시 사용)
            self.mask_image_path = mask_path
            self.manual_masking_offsets = {
                'offset_x_original': offset_x_orig,
                'offset_y_original': offset_y_orig,
                'offset_x_mask': offset_x_mask,
                'offset_y_mask': offset_y_mask
            }
            print(f"[INFO] 수동 마스킹 적용: 원본={original_path}, 마스크={mask_path}")
            print(f"[INFO] 오프셋: 원본({offset_x_orig}, {offset_y_orig}), 마스크({offset_x_mask}, {offset_y_mask})")

            # 2. ImageLayer에 마스크 QPixmap 저장
            self.selected_layer.mask_pixmap = QPixmap(mask_path)
            print("[INFO] 마스크 이미지를 ImageLayer에 저장")

            # 3. FINAL_RESULT 생성 (비교용 - 오프셋 적용된 합성)
            from core.file_manager import FileManager
            file_mgr = FileManager()
            original = file_mgr._safe_imread(original_path)
            mask = file_mgr._safe_imread(mask_path)

            if original is None or mask is None:
                raise ValueError("이미지 로드 실패")

            # 크기 조정
            if original.shape[:2] != mask.shape[:2]:
                h, w = original.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # 오프셋 적용된 합성 (비교용)
            final_result = self._create_manual_composite(original, mask, offset_x_orig, offset_y_orig,
                                                         offset_x_mask, offset_y_mask)

            # numpy → QPixmap (비교용)
            temp_dir = tempfile.gettempdir()
            final_result_path = os.path.join(temp_dir, f"final_result_manual_{os.path.basename(original_path)}")

            success, encoded_image = cv2.imencode('.jpg', final_result)
            if success:
                with open(final_result_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
                self.selected_layer.final_result_pixmap = QPixmap(final_result_path)
                print("[INFO] 비교용 FINAL_RESULT 생성 완료 (수동 마스킹)")

            # 4. 자동으로 레이아웃 인쇄 모드로 전환
            current_mode = self.property_panel.get_print_mode()
            if current_mode != "layered":
                print("[AUTO] 수동 마스킹 완료 → 레이아웃 인쇄 모드로 자동 전환")
                self.property_panel.set_print_mode("layered")
                # set_print_mode가 _on_print_mode_changed를 트리거하여 캔버스 업데이트

            # 5. 이미 레이아웃 모드인 경우 수동으로 캔버스 업데이트
            else:
                print("[CANVAS] 합성 결과 업데이트 (수동 마스킹)")
                self.update_canvas_for_layered_print_with_mask()

            # 6. 캔버스 크기에 자동 맞춤
            self.fit_layer_to_canvas()

            print(f"[OK] 수동 마스킹 적용 완료 - 임계값: {threshold}")
            print("[OK] 캔버스에 합성 결과 표시 (원본 + 녹색 테두리)")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"수동 마스크 적용 실패:\n{e}")
            print(f"[ERROR] 수동 마스크 적용 오류: {e}")
            import traceback
            traceback.print_exc()

    def _create_manual_composite(self, original, mask, offset_x_orig, offset_y_orig,
                                 offset_x_mask, offset_y_mask):
        """오프셋이 적용된 수동 마스킹 합성 이미지 생성"""
        import numpy as np
        import cv2

        h, w = original.shape[:2]

        # W 레이어 + YMC 레이어 효과
        result = original.copy().astype(np.float32)

        # 마스크의 검은색/흰색 영역 구분
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 오프셋 적용을 위한 변환 행렬
        offset_diff_x = offset_x_orig - offset_x_mask
        offset_diff_y = offset_y_orig - offset_y_mask

        # 마스크를 오프셋만큼 이동
        if offset_diff_x != 0 or offset_diff_y != 0:
            M = np.float32([[1, 0, offset_diff_x], [0, 1, offset_diff_y]])
            gray_mask = cv2.warpAffine(gray_mask, M, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)  # 빈 영역은 배경(흰색)으로

        background_mask = (gray_mask >= 128)  # 흰색 = 배경 = W 레이어 없음

        # 배경 영역(W 레이어 없는 곳)은 반투명 효과
        white_bg = np.ones_like(result) * 255
        result[background_mask] = cv2.addWeighted(
            result[background_mask], 0.3,  # 원본 30%
            white_bg[background_mask], 0.7,  # 흰색 배경 70%
            0
        )

        return result.astype(np.uint8)

    @Slot()
    def fit_layer_to_canvas(self):
        """선택된 레이어를 캔버스 크기에 맞춤"""
        from canvas.layers import ImageLayer

        if not self.selected_layer or not isinstance(self.selected_layer, ImageLayer):
            QMessageBox.warning(self, "경고", "이미지 레이어를 선택해주세요.")
            return

        try:
            # 캔버스 크기
            card_rect = self.canvas.scene.get_card_rect()
            card_width = card_rect.width()
            card_height = card_rect.height()

            # 이미지를 캔버스 크기에 맞춤
            self.selected_layer.set_size(card_width, card_height)

            # 캔버스 중앙에 배치
            self.selected_layer.setPos(card_rect.x(), card_rect.y())

            print(f"[OK] 레이어를 캔버스에 맞춤: {card_width}x{card_height}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"캔버스 맞추기 실패:\n{e}")
            print(f"[ERROR] 캔버스 맞추기 오류: {e}")


    def _on_canvas_orientation_changed(self, orientation: str):
        """캔버스 방향 변경 (앞면 + 뒷면 모두)"""
        try:
            # 앞면 캔버스 방향 변경
            self.front_canvas.scene.set_orientation(orientation)
            self.front_canvas.fitInView(self.front_canvas.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            # 뒷면 캔버스 방향 변경
            self.back_canvas_view.scene.set_orientation(orientation)
            self.back_canvas_view.fitInView(self.back_canvas_view.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            print(f"[OK] 캔버스 방향 변경 (앞면 + 뒷면): {orientation}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"방향 변경 실패:\n{e}")
            print(f"[ERROR] 캔버스 방향 변경 오류: {e}")

    @Slot()
    def save_project(self):
        """프로젝트 저장"""
        if self.current_project_path:
            self._save_to_file(self.current_project_path)
        else:
            self.save_project_as()

    @Slot()
    def save_project_as(self):
        """다른 이름으로 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "프로젝트 저장",
            "",
            "Hana Studio 프로젝트 (*.hana)"
        )

        if file_path:
            if not file_path.endswith('.hana'):
                file_path += '.hana'
            self._save_to_file(file_path)

    def _save_to_file(self, file_path: str):
        """파일로 저장"""
        import json

        try:
            # 모든 레이어 데이터 수집
            layers_data = []
            for item in self.canvas.scene.items():
                if isinstance(item, (ImageLayer, TextLayer)):
                    layers_data.append(item.get_layer_data())

            # 프로젝트 데이터
            project_data = {
                'version': '2.0',
                'layers': layers_data
            }

            # JSON으로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)

            self.current_project_path = file_path
            print(f"[OK] 프로젝트 저장: {file_path}")
            QMessageBox.information(self, "저장 완료", "프로젝트가 저장되었습니다.")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 실패:\n{e}")
            print(f"[ERROR] 저장 오류: {e}")

    @Slot()
    def load_project(self):
        """프로젝트 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "프로젝트 열기",
            "",
            "Hana Studio 프로젝트 (*.hana)"
        )

        if file_path:
            self._load_from_file(file_path)

    def _load_from_file(self, file_path: str):
        """파일에서 불러오기"""
        import json

        try:
            # JSON 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            # 기존 레이어 제거
            self.canvas.scene.clear()
            self.selected_layer = None
            self.property_panel.set_layer(None)

            # 레이어 복원
            for layer_data in project_data.get('layers', []):
                layer_type = layer_data.get('type')

                if layer_type == 'ImageLayer':
                    layer = ImageLayer.from_data(layer_data)
                    self.canvas.scene.addItem(layer)
                    layer.signals.selected.connect(self._on_layer_selected)

                elif layer_type == 'TextLayer':
                    layer = TextLayer.from_data(layer_data)
                    self.canvas.scene.addItem(layer)
                    layer.signals.selected.connect(self._on_layer_selected)

            self.current_project_path = file_path

            # 레이어 패널 업데이트
            self.layer_panel.update_layers(self.canvas.scene)

            # 양면인쇄 체크박스 업데이트
            self._update_dual_side_checkbox()

            print(f"[OK] 프로젝트 불러오기: {file_path}")
            QMessageBox.information(self, "불러오기 완료", "프로젝트를 불러왔습니다.")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"불러오기 실패:\n{e}")
            print(f"[ERROR] 불러오기 오류: {e}")

    @Slot()
    def export_image(self):
        """이미지로 내보내기"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "이미지 내보내기",
            "",
            "PNG 이미지 (*.png)"
        )

        if file_path:
            if not file_path.endswith('.png'):
                file_path += '.png'

            try:
                # 렌더링
                from renderer import PrintRenderer
                renderer = PrintRenderer()
                image = renderer.render_scene(self.canvas.scene)

                # 저장
                image.save(file_path, "PNG", dpiX=600, dpiY=600)
                print(f"[OK] 이미지 내보내기: {file_path}")
                QMessageBox.information(self, "내보내기 완료", f"이미지를 저장했습니다:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "오류", f"내보내기 실패:\n{e}")
                print(f"[ERROR] 내보내기 오류: {e}")

    @Slot(object)
    def _on_layer_panel_selection(self, layer):
        """레이어 패널에서 레이어 선택"""
        # 씬에서도 선택 상태 변경
        self.canvas.scene.clearSelection()
        layer.setSelected(True)

    @Slot()
    def _on_layer_order_changed(self):
        """레이어 순서 변경"""
        # 레이어 패널의 순서대로 씬의 Z-order 업데이트
        z_order = 0
        for i in range(self.layer_panel.layer_list.count()):
            item = self.layer_panel.layer_list.item(i)
            widget = self.layer_panel.layer_list.itemWidget(item)
            if widget:
                widget.layer.setZValue(self.layer_panel.layer_list.count() - z_order)
                z_order += 1
        print("[OK] 레이어 순서 변경")

    def _lazy_init_printer(self):
        """프린터 관련 지연 초기화"""
        try:
            from printer import find_printer_dll

            self.printer_dll_path = find_printer_dll()

            if self.printer_dll_path:
                print(f"[PRINTER] DLL 발견: {self.printer_dll_path}")
                self.printer_available = True

                # PropertyPanel에 프린터 상태 업데이트
                self.property_panel.update_printer_status("프린터: 준비됨")

                # 프린터 선택 대화상자 자동 표시 (지연 실행)
                from PySide6.QtCore import QTimer
                QTimer.singleShot(500, self._auto_show_printer_dialog)
            else:
                print("[PRINTER] DLL을 찾을 수 없음")
                self.printer_available = False
                self.property_panel.update_printer_status("프린터: DLL 없음")

        except ImportError as e:
            print(f"[PRINTER] 프린터 모듈 없음: {e}")
            self.printer_available = False
            self.property_panel.update_printer_status("프린터: 모듈 없음")
        except Exception as e:
            print(f"[ERROR] 프린터 초기화 실패: {e}")
            self.printer_available = False
            self.property_panel.update_printer_status("프린터: 초기화 실패")

    def _auto_show_printer_dialog(self):
        """프린터 선택 대화상자 자동 표시"""
        try:
            from printer.printer_discovery import discover_available_printers
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QPushButton, QHBoxLayout

            # 프린터 검색
            printers, summary = discover_available_printers(self.printer_dll_path)

            if not printers:
                print("[PRINTER] 검색된 프린터 없음")
                self.property_panel.update_printer_status("프린터: 검색 안됨")
                return

            # 프린터 선택 다이얼로그
            dialog = QDialog(self)
            dialog.setWindowTitle("프린터 선택")
            dialog.setMinimumWidth(400)

            layout = QVBoxLayout(dialog)

            # 안내 문구
            info_label = QLabel(f"검색된 프린터: {len(printers)}대")
            layout.addWidget(info_label)

            # 프린터 목록
            printer_list = QListWidget()
            for printer in printers:
                printer_list.addItem(f"{printer.name} ({printer.connection_type})")
            printer_list.setCurrentRow(0)
            layout.addWidget(printer_list)

            # 버튼
            button_layout = QHBoxLayout()
            select_btn = QPushButton("선택")
            cancel_btn = QPushButton("취소")
            button_layout.addWidget(select_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)

            # 시그널 연결
            def on_select():
                selected_row = printer_list.currentRow()
                if selected_row >= 0:
                    self.selected_printer_info = printers[selected_row]
                    print(f"[PRINTER] 선택됨: {self.selected_printer_info.name}")
                    self.property_panel.update_printer_status(f"프린터: {self.selected_printer_info.name}")
                dialog.accept()

            select_btn.clicked.connect(on_select)
            cancel_btn.clicked.connect(dialog.reject)

            # 다이얼로그 표시
            dialog.exec()

        except Exception as e:
            print(f"[ERROR] 프린터 선택 대화상자 실패: {e}")
            import traceback
            traceback.print_exc()
