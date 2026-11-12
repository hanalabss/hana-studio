"""편집기 캔버스가 쓰는 색 모음. 테마 토큰에서 채워 넣을 수 있게 한 곳에 둔다."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EditorPalette:
    """색은 "#RRGGBB" 또는 "#AARRGGBB" 문자열."""

    name: str = "light"
    canvas_bg: str = "#F6F7F9"
    card_fill: str = "#FFFFFF"
    card_shadow: str = "#1B2430"
    shadow_opacity: float = 0.16
    outside_opacity: float = 0.74
    selection: str = "#2456E6"
    handle_fill: str = "#FFFFFF"
    hover: str = "#802456E6"
    rubber_fill: str = "#1F2456E6"
    guide: str = "#E0457B"
    cut_line: str = "#E5484D"
    bleed_tint: str = "#17E5484D"
    grid: str = "#121B2430"
    grid_major: str = "#301B2430"
    warning: str = "#E8890C"
    locked: str = "#8A93A1"
    tag_bg: str = "#2456E6"
    tag_text: str = "#FFFFFF"
    field_tint: str = "#FFF7D6"          # 명단에서 가져오는 항목에 형광펜처럼 곱해 얹는 옅은 노랑(화면에만 보인다)
    field_tint_line: str = "#80E0A800"
    readout_bg: str = "#1B2430"
    readout_text: str = "#FFFFFF"
    ruler_bg: str = "#FFFFFF"
    ruler_text: str = "#5B6676"
    ruler_tick: str = "#AEB6C2"
    ruler_border: str = "#DDE1E7"
    ruler_marker: str = "#2456E6"
    ruler_span: str = "#262456E6"

    @classmethod
    def light(cls) -> "EditorPalette":
        return cls()

    @classmethod
    def dark(cls) -> "EditorPalette":
        return cls(
            name="dark",
            canvas_bg="#15171A",
            card_fill="#FFFFFF",
            card_shadow="#000000",
            shadow_opacity=0.55,
            outside_opacity=0.78,
            selection="#FF4D8D",
            handle_fill="#FFFFFF",
            hover="#80FF4D8D",
            rubber_fill="#26FF4D8D",
            guide="#35C2F0",
            cut_line="#FF4D8D",
            bleed_tint="#1FFF4D8D",
            grid="#121B2430",
            grid_major="#301B2430",
            warning="#FFC857",
            locked="#8A919C",
            tag_bg="#FF4D8D",
            tag_text="#15171A",
            field_tint="#FFF5CC",
            field_tint_line="#99E0A800",
            readout_bg="#F2F3F5",
            readout_text="#15171A",
            ruler_bg="#1E2126",
            ruler_text="#B4BAC3",
            ruler_tick="#5A616B",
            ruler_border="#30353C",
            ruler_marker="#FF4D8D",
            ruler_span="#33FF4D8D",
        )

    def with_changes(self, **changes) -> "EditorPalette":
        return replace(self, **changes)
