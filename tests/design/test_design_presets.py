import json

import numpy as np
import pytest

from design.binding import columns_used
from design.model import Document
from design.presets import (
    CR80, CUT_54X86, CUT_54X86_BLEED, PRESETS, TEMPLATES, build_template, get_preset, new_document,
    with_bleed,
)
from design.renderer import render_side


def test_preset_sizes():
    assert (CR80.width_mm, CR80.height_mm, CR80.bleed_mm) == (85.6, 53.98, 0.0)
    assert (CUT_54X86.width_mm, CUT_54X86.height_mm, CUT_54X86.bleed_mm) == (86.0, 54.0, 0.0)
    assert (CUT_54X86_BLEED.width_mm, CUT_54X86_BLEED.height_mm, CUT_54X86_BLEED.bleed_mm) == (90.0, 58.0, 2.0)
    assert CUT_54X86_BLEED.cut_size_mm == CUT_54X86.cut_size_mm == (86.0, 54.0)
    assert len({p.key for p in PRESETS}) == len(PRESETS) == 3
    assert get_preset("cr80") is CR80
    with pytest.raises(KeyError):
        get_preset("a4")


def test_with_bleed_matches_existing_convention():
    derived = with_bleed(CUT_54X86)
    assert (derived.width_mm, derived.height_mm, derived.bleed_mm) == (90.0, 58.0, 2.0)
    assert derived.key == CUT_54X86_BLEED.key
    wider = with_bleed(CUT_54X86_BLEED, 3.0)
    assert (wider.width_mm, wider.height_mm, wider.cut_size_mm) == (92.0, 60.0, (86.0, 54.0))
    cr80 = with_bleed(CR80, 1.5)
    assert cr80.width_mm == pytest.approx(88.6) and cr80.cut_size_mm == pytest.approx((85.6, 53.98))


def test_existing_config_still_describes_the_bleed_preset():
    with open("config.json", encoding="utf-8") as handle:
        printer = json.load(handle)["printer"]
    short, long_edge, bleed = printer["design_width_mm"], printer["design_height_mm"], printer["bleed_mm"]
    assert (long_edge, short, bleed) == (
        CUT_54X86_BLEED.width_mm, CUT_54X86_BLEED.height_mm, CUT_54X86_BLEED.bleed_mm
    )


def test_new_document():
    document = new_document(CUT_54X86_BLEED, "portrait", dual_side=True, name="새 카드")
    assert document.side_size_mm("front") == (58.0, 90.0) == document.side_size_mm("back")
    assert document.side_bounds("front", inside_bleed=True) == (2.0, 2.0, 56.0, 88.0)
    assert (document.bleed_mm, document.dual_side, document.name) == (2.0, True, "새 카드")
    assert new_document().side_size_mm("front") == (85.6, 53.98)


def test_template_catalog():
    keys = [t.key for t in TEMPLATES]
    assert keys[0] == "blank" and len(keys) == len(set(keys)) >= 4
    assert all(t.label and t.description for t in TEMPLATES)
    blank = build_template("blank")
    assert blank.side("front").items == [] and blank.name == "빈 카드"
    with pytest.raises(KeyError):
        build_template("없는템플릿")


@pytest.mark.parametrize("template", [t.key for t in TEMPLATES if t.key != "blank"])
@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
def test_templates_fit_the_card_and_render(template, preset, orientation):
    document = build_template(template, preset, orientation)
    side = document.side("front")
    width, height = document.side_size_mm("front")
    assert side.orientation == orientation and len(side.items) >= 4
    assert len({item.id for item in side.items}) == len(side.items)
    assert [item.z for item in side.sorted_items()] == list(range(len(side.items)))

    cut_left, cut_top, cut_right, cut_bottom = document.side_bounds("front", inside_bleed=True)
    for item in side.items:
        left, top, right, bottom = item.bounds()
        assert left >= -1e-6 and top >= -1e-6 and right <= width + 1e-6 and bottom <= height + 1e-6, item.name
        if item.kind != "rect":
            assert left >= cut_left - 1e-6 and right <= cut_right + 1e-6, f"{item.name}: 칼선 밖"
            assert top >= cut_top - 1e-6 and bottom <= cut_bottom + 1e-6, f"{item.name}: 칼선 밖"
        assert item.name

    assert Document.from_dict(json.loads(json.dumps(document.to_dict()))) == document

    preview = render_side(document, "front", dpi=150)
    assert preview.warnings == []
    assert len(np.unique(np.array(preview.color).reshape(-1, 3), axis=0)) > 2

    record = {name: "1234567" for name in columns_used(document)}
    record.pop("사진", None)
    printed = render_side(document, "front", record=record, context=None, strict=True)
    assert printed.color.size == (round(width * 300 / 25.4), round(height * 300 / 25.4))


def test_template_columns():
    assert columns_used(build_template("employee")) == {"사진", "이름", "부서", "직급", "사번"}
    assert columns_used(build_template("student")) == {"사진", "이름", "학과", "학번"}
    assert columns_used(build_template("membership")) == {"이름", "회원번호"}
    assert columns_used(build_template("visitor")) == {"이름", "방문처"}


def test_templates_are_independent_instances():
    first, second = build_template("employee"), build_template("employee")
    first.side("front").items[0].x += 5
    assert first.side("front").items[0].x != second.side("front").items[0].x
    assert {i.id for i in first.side("front").items}.isdisjoint({i.id for i in second.side("front").items})
