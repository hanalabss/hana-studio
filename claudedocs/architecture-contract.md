# 하나스튜디오 확장: 모듈 계약

여러 에이전트가 병렬로 구현할 때 지켜야 하는 공통 약속이다. 배경은 `card-software-comparison-and-roadmap.md`, SDK 인자는 `r600-sdk-reference.md`에 있다.

## 공통 규칙

- Python 3.12, PySide6 6.9. 새 의존성은 `requirements.txt`에 추가하고 최종 보고에 적는다. 이미 설치됨: Pillow, numpy, opencv, openpyxl, qrcode, python-barcode, pytest.
- 단위는 mm(float). 인쇄 해상도 기본 300dpi. 카드 기본 크기 CR80 = 85.6 × 53.98mm(가로 기준). 면마다 `orientation`("landscape" | "portrait")을 가진다.
- 테스트는 `tests/` 아래 pytest. Qt가 필요하면 `QT_QPA_PLATFORM=offscreen`(tests/conftest.py에서 설정). 소스 옆에 테스트 파일을 두지 않는다.
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
| `printer/` | 프린터 추상화, RTAI 확장, SMART 드라이버, 작업 이력 | 프린터 |
| `ui/` | 화면 전체 (시안 승인 후 착수) | UI |

## design/

```python
# design/model.py  (Qt 위젯에 의존하지 않는 dataclass)
Document(version:int, name:str, card_width_mm:float, card_height_mm:float,
         sides: dict["front"|"back", Side], dual_side: bool)
Side(orientation:str, items: list[Item], background: str|None)
Item 공통: id:str, kind:str, x,y,w,h (mm), rotation:float(도), z:int, locked:bool,
           visible:bool, printable:bool, layer:"color"|"white"|"uv", name:str,
           source: Source|None
종류: TextItem, ImageItem, RectItem, EllipseItem, LineItem, BarcodeItem, QRItem
Source(type:"fixed"|"column"|"concat"|"counter"|"date", ...)   # design/binding.py
```

- `design/binding.py`: `resolve(item, record: Mapping[str, Any] | None, context: BindingContext) -> Any`. `record`는 열 이름 → 값의 평범한 dict다. 사진 열의 값은 파일 경로 문자열이다. `record`가 None이면 디자인 화면용 자리표시 값을 돌려준다.
- `design/renderer.py`: QPainter 기반.
  - `paint_item(painter, item, value, mm_to_px: float)` : 편집기 캔버스와 인쇄 렌더러가 같이 쓴다(화면과 인쇄 결과 일치).
  - `render_side(document, side_key, record=None, dpi=300, context=None) -> RenderedSide`
  - `RenderedSide(color: PIL.Image(RGB), white_mask: PIL.Image(L)|None, uv: PIL.Image(L)|None)`
- `design/storage.py`: `.hsd` = ZIP(`document.json` + `assets/` + `thumbnail.png`). `save(document, path)`, `load(path) -> Document`. 버전 번호를 넣고, 모르는 필드는 버리지 않고 보존한다. 이미지 경로는 문서 안 `assets/` 상대 경로.
- ImageItem은 `remove_background` 결과 마스크를 `mask_asset`로 가질 수 있고, 렌더러는 이를 화이트 레이어에 합성한다. 배경 제거 실행 자체는 기존 `core/image_processor.py`를 UI 쪽에서 호출한다.

## data/

```python
# data/store.py
DataStore(path)                      # SQLite 파일 하나. 문서와 별도 파일(.hsdata)
  columns() -> list[Column]          # Column(name, type:"text"|"number"|"date"|"time"|"list"|"photo", options)
  add_column / rename_column / remove_column
  count(filter=None) / rows(filter=None, order=None, offset=0, limit=None) -> list[Record]
  Record: {"_id": int, <열 이름>: 값, "_printed": bool, "_print_count": int, "_printed_at": str|None}
  insert(values) / update(_id, values) / delete(ids)
  mark_printed(ids, printed=True) / reset_print_status(ids|None)
# data/importers.py
  preview_csv(path) -> ImportPreview(encoding, delimiter, has_header, columns, sample_rows)
  preview_excel(path, sheet=None) ; import_into(store, preview, column_map) -> ImportResult
# data/filters.py   : 조건 빌더(열, 연산자, 값, AND/OR) → 매개변수 SQL. 문자열 이어 붙이기 금지.
# data/selection.py : parse_ranges("1,3,4-9", total) -> list[int]
# data/photos.py    : 폴더 + 열 값 + 확장자 후보로 사진 파일 찾기, 누락 목록
```

- 사용자의 원본 CSV/Excel 파일은 절대 고치지 않는다. 가져온 사본을 `.hsdata`에 둔다.
- `rows()`가 주는 Record에서 `_`로 시작하지 않는 키만 뽑으면 `design.binding.resolve`의 `record`가 된다.

## printer/

- 기존 `printer/r600_printer.py`의 공개 메서드와 SDK 호출은 이름·동작을 유지한다. 추가만 한다.
```python
# printer/base.py
class CardPrinter(ABC):
    family: "rtai" | "smart"
    capabilities() -> set[str]   # "dual_side","white","uv","magnetic","status","ribbon_level",...
    connect(device: PrinterDevice) / close()
    status() -> PrinterStatus(ready:bool, message:str, code:int|None, ribbon_remaining:int|None, ...)
    print_card(job: CardJob) -> None     # 실패 시 PrinterError(사용자용 한국어 message, code)
    write_magnetic(tracks) / read_magnetic()   # 지원하지 않으면 NotSupported
CardJob(front: SideImages|None, back: SideImages|None, copies:int=1, options: dict)
SideImages(color_path:str, white_mask_path:str|None, uv_path:str|None, orientation:str)
# printer/registry.py : discover() -> list[PrinterDevice(family, name, connection, address)]
#                       open(device) -> CardPrinter
# printer/history.py  : 작업·카드별 결과를 SQLite에 기록 (앱 재시작 후에도 남음)
```
- 실제 DLL 없이 돌아가는 가짜 백엔드(`printer/fakes.py`)를 두고 테스트는 그것으로 한다. 실기 확인용 스크립트는 `scripts/hw_*.py`에 둔다(카드를 소모하는 동작은 `--yes` 없이는 실행하지 않는다).
