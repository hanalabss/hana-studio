# 하나스튜디오 확장: 모듈 계약

여러 에이전트가 병렬로 구현할 때 지켜야 하는 공통 약속이다. 배경은 `card-software-comparison-and-roadmap.md`, SDK 인자는 `r600-sdk-reference.md`에 있다.

## 공통 규칙

- Python 3.12, PySide6 6.9. 새 의존성은 `requirements.txt`에 추가하고 최종 보고에 적는다. 이미 설치됨: Pillow, numpy, opencv, openpyxl, qrcode, python-barcode, pytest.
- 단위는 mm(float). 인쇄 해상도 기본 300dpi. 카드 기본 크기 CR80 = 85.6 × 53.98mm(가로 기준). 면마다 `orientation`("landscape" | "portrait")을 가진다.
- 테스트는 `tests/<모듈>/` 아래 pytest. Qt가 필요하면 `QT_QPA_PLATFORM=offscreen`(tests/conftest.py에서 설정). 소스 옆에 테스트 파일을 두지 않는다.
- 자기 담당 디렉터리 밖의 파일은 고치지 않는다. 필요하면 최종 보고에 "통합 시 필요한 변경"으로 적는다.
- 커밋하지 않는다. PM이 통합 후 커밋한다.
- 사용자에게 보이는 문구는 한국어. 코드 주석은 꼭 필요한 곳에만.
- 다눌·카드프레소의 코드, 폰트, 템플릿, DLL은 가져오지 않는다.
- 기능마다 테스트를 쓰고, 실패하면 원인을 고쳐 다시 돌린다. 테스트를 건너뛰거나 비활성화하지 않는다.

## 모듈과 담당

| 디렉터리 | 내용 | 담당 |
|---|---|---|
| `design/` | 문서 모델, 바인딩 해석, 렌더러, 파일 저장 | 디자인 코어 |
| `data/` | 레코드 저장소, 가져오기, 필터, 인쇄 대상 선택, 인쇄 상태 기록 | 데이터 코어 |
| `printer/` | 프린터 추상화, RTAI 확장, SMART 드라이버, 인쇄 큐, 작업 이력 | 프린터 |
| `services/` | 프로젝트 파일, 프린터 선택·상태, 인쇄 전 점검, 인쇄 실행(문서+명단 → 큐) | 서비스 |
| `ui/` | 화면 전체 (`ui/editor`, `ui/pages/*`, `ui/shell`, `ui/theme`, `ui/widgets`) | UI |

의존 방향: `ui → services → design / data / printer`. `design`·`data`·`printer`는 서로를 모른다.

## design/

```python
# design/model.py  (Qt에 의존하지 않는 dataclass, 단위 mm)
Document(version, name, card_width_mm, card_height_mm,          # 가로 기준 캔버스 크기, 도련 포함
         sides: dict["front"|"back", Side], dual_side: bool, bleed_mm: float = 0.0,
         base_dir: str|None)                                    # base_dir: assets/ 가 풀린 작업 폴더(저장 안 됨)
  side_size_mm(side_key) -> (w, h)            # 면 방향 반영, 도련 포함
  side_bounds(side_key, inside_bleed=False)   # inside_bleed=True면 칼선 안쪽
  active_side_keys() / iter_items() / find_item(id) / asset_refs()
  reflow_side(side_key, *, orientation=None, card_size_mm=None, mode="reposition") -> {item.id: ItemGeometry}
Side(orientation, items, background)          # background: "#rrggbb" 또는 assets/ 상대 경로
Item 공통: id, kind, x, y, w, h (mm), rotation(도, 가운데 기준 시계 방향), z, locked, visible, printable,
           layer:"color"|"white"|"uv", name, source: Source|None
종류: TextItem, ImageItem(asset, fit, crop, record_crops, mask_asset), RectItem, EllipseItem, LineItem,
      BarcodeItem, QRItem
Crop(offset_x, offset_y, zoom)                # offset: 남는 폭의 절반을 1로 본 -1..1 (+ = 오른쪽·아래)

# 방향·크기 바꾸기(순수 함수: 항목은 그대로, 새 위치만 돌려준다)
ItemGeometry(x, y, w, h, rotation)  .of(item) / .bounds() / .apply_to(item)
reflow_items(items, old_size_mm, new_size_mm, mode="reposition"|"scale"|"fit") -> {item.id: ItemGeometry}
rotate_items(items, size_mm, quarter_turns=1) -> {item.id: ItemGeometry}     # 면 전체를 시계 방향 90°씩

# design/binding.py
Source 종류: FixedSource, ColumnSource(column), ConcatSource(parts), CounterSource, DateSource
BindingContext(counter_index=0, now=datetime.now, photo_base_dir=None, record_key=None)
resolve(item, record: Mapping|None, context=None) -> Any     # record None이면 디자인 화면용 자리표시
columns_used(document) -> set[str]
```

- `record`는 열 이름 → 값의 평범한 dict다. 사진 열의 값은 파일 경로 문자열이다(`data.photos.PhotoResolver.binding_record`가 만들어 준다).
- `design/renderer.py` (QPainter 기반, 편집 화면과 인쇄가 같은 코드를 쓴다):
  - `value_for(document, item, record=None, context=None) -> Any` : 바인딩을 풀고 문서 안 이미지 경로를 절대 경로로 바꾼, `paint_item`에 넘길 값.
  - `paint_item(painter, item, value, mm_to_px, *, crop=None, design_mode=True, mask_mode=False, problems=None)` : 면 좌표(원점 = 면 왼쪽 위)에 그린다.
  - `paint_item_local(painter, item, value, mm_to_px, *, ...)` : 항목 원점 기준 (0,0)–(w·mm_to_px, h·mm_to_px)에 그린다. 위치·회전은 부른 쪽(QGraphicsItem)이 건다. painter의 확대 배율은 자유다.
  - `text_extent(item, value, mm_to_px=1.0) -> (left, top, right, bottom)` : 글자가 상자 밖으로 넘치는 길이(글자 항목·글자 있는 바코드, 그 밖은 0).
  - `fitted_font_size_pt(item, text)`, `image_target_rect(...)`, `resolve_font_family(name)`, `load_image(path)`.
  - `render_side(document, side_key, record=None, dpi=300, context=None, *, design_mode=None, crop_bleed=False, strict=False) -> RenderedSide`
  - `RenderedSide(color: PIL(RGB), white_mask: PIL(L)|None, uv: PIL(L)|None, orientation, dpi, warnings)`; `.save(directory, stem) -> {"color_path", "white_mask_path", "uv_path", "orientation"}` (그대로 `SideImages`에 넣는다).
  - 도련: `side_pixel_size`(도련 포함), `cut_pixel_size(document, side_key, dpi)` = round(칼선 mm × dpi ÷ 25.4), `trim_bleed(rendered, document, side_key)`. `crop_bleed=True`의 결과는 정확히 `cut_pixel_size`다(가운데 기준).
- `design/storage.py`: `.hsd` = ZIP(`document.json` + `assets/` + `thumbnail.png`). `save(document, path)`, `load(path, workdir=None) -> Document`, `import_asset(workdir, file) -> 상대 경로`, `read_thumbnail(path)`. 버전 번호를 넣고, 모르는 필드는 버리지 않고 보존한다.
- ImageItem은 배경 제거 마스크를 `mask_asset`로 가질 수 있고, 렌더러는 이를 화이트 레이어에 합성한다. 배경 제거 실행 자체는 기존 `core/image_processor.py`를 UI 쪽에서 호출한다.

## data/

```python
# data/store.py
DataStore(path)                      # SQLite 파일 하나. 문서와 별도 파일(.hsdata)
  columns() -> list[Column]          # Column(name, type:"text"|"number"|"date"|"time"|"list"|"photo", options)
  add_column / rename_column / remove_column / move_column / set_column_type / set_column_options
  count(filter=None) / ids(filter, order) / get(_id) / rows(filter=None, order=None, offset=0, limit=None)
  Record: {"_id": int, <열 이름>: 값, "_printed": bool, "_print_count": int, "_printed_at": str|None}
  insert(values) / insert_many(rows) / update(_id, values) / delete(ids) / transaction()
  mark_printed(ids, printed=True) / reset_print_status(ids|None)
  get_photo_adjustment(_id, column, default=PhotoAdjustment()) -> PhotoAdjustment|None   # default=None이면 "저장된 값 없음"을 구별
  has_photo_adjustment(_id, column) / set_photo_adjustment / apply_photo_adjustment_to_all / clear_photo_adjustments
PhotoAdjustment(zoom=1.0, offset_x=0.0, offset_y=0.0)
  # zoom: 틀에 맞춘 크기 대비 배율. offset: 옮기는 거리 ÷ 틀 크기(+ = 오른쪽·아래). design.Crop과 단위가 다르다.
# data/importers.py : preview_csv / preview_excel / preview_file -> ImportPreview ; import_into(store, preview, column_map)
# data/filters.py   : 조건 빌더(열, 연산자, 값, AND/OR) → 매개변수 SQL. 문자열 이어 붙이기 금지.
# data/selection.py : parse_ranges("1,3,4-9", total) ; PrintSelection(전체·현재·범위·미인쇄).resolve(store, filter, order)
# data/photos.py    : PhotoResolver(store).path(record, column) / .binding_record(record), 누락 목록
# data/export.py    : export_csv / export_xlsx
```

- 사용자의 원본 CSV/Excel 파일은 절대 고치지 않는다. 가져온 사본을 `.hsdata`에 둔다.
- 스레드: `DataStore`는 연결 하나를 재진입 잠금으로 감싸므로 어느 스레드에서 불러도 되고 호출은 차례로 실행된다. `transaction()` 블록은 끝날 때까지 다른 스레드를 기다리게 한다. 변경 알림은 없다(자세한 규칙은 클래스 설명서).

## printer/

- 기존 `printer/r600_printer.py`의 공개 메서드와 SDK 호출은 이름·동작을 유지한다. 추가만 한다.

```python
# printer/base.py  (Qt에 의존하지 않는다)
class CardPrinter(ABC):
    family: "rtai" | "smart"
    capabilities() -> set[str]   # "dual_side","white","uv","magnetic","status","ribbon_level",...
    connect(device: PrinterDevice) / close() / is_connected / info()
    status() -> PrinterStatus
    print_card(job: CardJob, progress=None) -> None   # 실패 시 PrinterError(사용자용 한국어 message, code, recoverable)
    write_magnetic(tracks) / read_magnetic() / inject_card() / eject_card()   # 지원하지 않으면 NotSupported
CardJob(front: SideImages|None, back: SideImages|None, copies=1, options: dict, record_id=None)
  # options: offset_x_mm, offset_y_mm, magnetic, k_split, precheck, bleed_handled(이미 칼선 크기 → 드라이버가 도련 제거 안 함) ...
SideImages(color_path, white_mask_path=None, uv_path=None, orientation="portrait")
PrinterStatus(ready, message, code, ribbon_remaining, film_remaining,
              ribbon_unit, film_unit,        # "cards" | "percent" | "unknown" - 숫자의 단위
              ribbon_capacity, ribbon_type, feeder_empty, card_inside, busy, warnings, details)
  .ribbon_cards_left / .film_cards_left      # 단위가 "cards"일 때만 숫자, 아니면 None
  # RTAI: SDK 값 그대로라 "unknown". SMART: (최대, 남은) 횟수가 말이 될 때만 "cards".
remaining_text(value, unit) -> "약 310장" | "62 %" | "310 (단위 확인 안 됨)" | "확인 불가"

# 진행 상황: print_card(progress=...)에는 callback(percent, text) 또는 StructuredProgress(handler)를 넘긴다
ProgressEvent(percent, text, stage="", panel=None, fraction=None, side=None, copy=1, copies=1)
  # stage: STAGE_* ("prepare","feed","magnetic","setup","front_image","back_image","print",
  #        "print_front","print_back","wait","eject","done"). panel·fraction은 장치가 알려 줄 때만(지금은 항상 None)

# 카드 공급자: 카드를 찍기 직전에 만들어 준다
CardProvider: __len__() + __getitem__(index) -> CardJob  [+ is_dual_side, record_id(index)]
PrintCancelled(PrinterError)     # 공급자가 던지면 그 카드는 "cancelled"로 남고 작업이 멈춘다

# printer/registry.py : discover(families) -> DiscoveryResult(list[PrinterDevice]) ; open(device) -> CardPrinter ; use_fakes()
# printer/history.py  : JobHistory - 작업·카드별 결과(success | failed | cancelled)를 SQLite에 기록

# printer/print_queue.py  (QObject, 전역 인스턴스 print_queue)
add_job(**legacy)                     # 기존 방식(이미지 한 벌 × quantity)
add_card_job(device, cards, name="", stop_on_error=True, dual_side=None) -> job_id   # cards: list[CardJob] | CardProvider
cancel_job(job_id) / remove_job / stop_current_job
cancel_all()                          # 기다리지 않는다. 작업마다 job_cancelled. 스레드 종료 확인은 wait_idle(ms)
get_job(job_id) -> PrintJob|None ; history_id_for(job_id) -> int|None ; PrintJob.history_id
시그널: job_added, job_started(job_id, index, total), job_progress(percent, text), job_completed(job_id, ok),
        job_cancelled(job_id), job_error(job_id, message),            # 연결·초기화 실패
        card_result(job_id, index, ok, message),
        card_finished(job_id, index, "success"|"failed"|"cancelled", message),
        card_progress(job_id, index, ProgressEvent), all_jobs_completed(...), queue_updated, log_message
```

- 실제 DLL 없이 돌아가는 가짜 백엔드(`printer/fakes.py`)를 두고 테스트는 그것으로 한다. 실기 확인용 스크립트는 `scripts/hw_*.py`에 둔다(카드를 소모하는 동작은 `--yes` 없이는 실행하지 않는다).
- 도련: `utils.bleed_crop`의 비율 기반 도련 제거는 기존 방식 호출의 기본 동작으로 남아 있다. `RtaiAdapter`·`SmartAdapter`는 `job.options["bleed_handled"]`가 참이면 건너뛴다.

## services/

```python
# services/render_job.py
PrintOptions(copies, dpi, stop_on_error, offset_x_mm, offset_y_mm, magnetic, name_column, counter_offset, ...)
RenderJob.for_records(document, store, record_ids, options, capabilities) / .for_copies(document, copies, ...)
  .render(index) -> RenderedCard ; .card_job(index) -> CardJob ; .release(index) ; .close()
  # 카드는 요청받을 때 렌더링하고(도련은 칼선 크기로 잘라 options["bleed_handled"]=True), 결과가 나오면 파일을 지운다
CardFeeder(render_job).cards() -> CardProvider      # 인쇄 스레드의 요청을 UI 스레드에서 렌더링해 넘긴다
crop_from_adjustment(PhotoAdjustment, image_size, frame_size, fit) -> Crop ; adjustment_from_crop(...)

# services/print_service.py  (QObject, UI 스레드에서 쓴다)
PrintService(queue=None, history=None)
  start(document, store, selection, device, options=None, *, filter=None, order=None) -> PrintJobHandle
  cancel(job, mode="after_current"|"now") / cancel_all / retry_failed(job, include_unprinted=False) / shutdown()
  시그널: job_queued(handle), job_started(handle), progress(PrintProgress), card_result(CardResult),
          finished(PrintSummary), error(handle, message)
PrintJobHandle: id, name, device, total, state, results, summary, history_id, failed_indices, unprinted_indices
PrintProgress(job_id, done, total, index, record_id, record_name, stage(문구), panel(모르면 ""), percent,
              stage_key, fraction, card_percent, side)     # stage_key: STAGE_* + "connect" | "next_card"
CardResult(job_id, index, record_id, record_name, ok, status:"success"|"failed"|"cancelled", message, warnings)

# services/preflight.py : check(document, store, record_ids, device, printer_status, *, capabilities, options) -> PreflightReport
#   Finding(code, severity:"error"|"warning", message, record_ids, item_ids, side)
#   ribbon_low는 잔량 단위가 "cards"일 때만 낸다.
# services/printers.py  : PrinterService - 탐지·선택·상태 폴링(devices_changed, current_changed, status_changed)
# services/project.py   : Project(.hsproj = 디자인 + 명단 + 인쇄 옵션), RecentProjects, 자동 저장·복구
```

## 실행 환경

- 테스트: 저장소 루트에서 `python -m pytest` (`pytest.ini`: `testpaths = tests`, 임포트 방식 prepend). 테스트 파일·도우미 이름은 저장소 전체에서 하나뿐이어야 한다(`test_<폴더>_*.py`, `<폴더>_helpers.py`).
- 콘솔: 기존 인쇄 코드의 이모지 `print()`가 cp949 콘솔·리다이렉트에서 죽지 않도록 프로세스 시작 때 `utils.console.ensure_utf8_console()`을 한 번 부른다.
- 빌드: `HanaStudio.spec`이 프로젝트 패키지 폴더를 훑어 모든 모듈을 넣고, `openpyxl`·`qrcode`·`barcode`(글꼴 포함)·`dll/smart`를 함께 넣는다.
