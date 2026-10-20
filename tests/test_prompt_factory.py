"""Prompt generation and judging.

Only the pure parts are tested here: parsing, blind judging, selection. The
kiro-cli invocation is a thin shell around them.
"""

import pytest

from tools import prompt_factory as pf


def test_scene_labels_are_the_six_from_the_dataset_plan():
    assert set(pf.SCENES) == {
        "static_talking_head",
        "cinematic_slowpan",
        "high_motion_action",
        "fine_texture_nature",
        "graphics_text",
        "animation_flat",
    }


def test_parse_prompts_strips_numbering_and_bullets():
    text = """1. A medium shot of a woman speaking to camera in a dim kitchen.
2) Slow push in on a mountain ridge at sunrise, gentle camera drift.
- Close up of a hummingbird hovering beside a red feeder, wings blurred.
* Wide shot of a city crosswalk as people walk toward the camera.
"""
    prompts = pf.parse_prompts(text)
    assert prompts[0].startswith("A medium shot of a woman")
    assert all(not p.startswith(("1", "2", "-", "*")) for p in prompts)
    assert len(prompts) == 4


def test_parse_prompts_drops_short_lines_duplicates_and_commentary():
    text = """Here are the prompts you asked for:
A wide shot of a harbour at dawn with boats leaving the marina slowly.
A wide shot of a harbour at dawn with boats leaving the marina slowly.
Too short.
That is all.
"""
    prompts = pf.parse_prompts(text)
    assert prompts == ["A wide shot of a harbour at dawn with boats leaving the marina slowly."]


def test_parse_judgement_reads_fenced_json_lines():
    text = """```json
{"id": 1, "fit": 5, "specificity": 4, "motion": 3}
{"id": 2, "fit": 2, "specificity": 5, "motion": 1}
```"""
    judged = pf.parse_judgement(text)
    assert [j["id"] for j in judged] == [1, 2]
    assert judged[0]["fit"] == 5


def test_parse_judgement_requires_every_id_exactly_once():
    text = """{"id": 1, "fit": 5, "specificity": 4, "motion": 3}
{"id": 1, "fit": 4, "specificity": 3, "motion": 2}"""
    with pytest.raises(pf.JudgementError):
        pf.parse_judgement(text, expected_ids=(1, 2))


def test_parse_judgement_rejects_out_of_range_scores():
    text = '{"id": 1, "fit": 9, "specificity": 4, "motion": 3}'
    with pytest.raises(pf.JudgementError):
        pf.parse_judgement(text)


def test_selection_prefers_fit_then_weighted_score():
    candidates = ["a", "b", "c", "d"]
    judged = [
        {"id": 1, "fit": 5, "specificity": 5, "motion": 5, "text": "a"},
        {"id": 2, "fit": 4, "specificity": 5, "motion": 5, "text": "b"},
        {"id": 3, "fit": 3, "specificity": 5, "motion": 5, "text": "c"},
        {"id": 4, "fit": 5, "specificity": 4, "motion": 3, "text": "d"},
    ]
    # a scores 20, b scores 18, d scores 17; c has the best specificity and
    # motion but is excluded for scoring fit 3.
    chosen = pf.select(judged, count=2)
    assert [c["text"] for c in chosen] == ["a", "b"]
    assert all(c["fit"] >= pf.MIN_FIT for c in chosen)


def test_selection_is_deterministic_on_ties():
    judged = [
        {"id": 2, "fit": 5, "specificity": 4, "motion": 3, "text": "second"},
        {"id": 1, "fit": 5, "specificity": 4, "motion": 3, "text": "first"},
    ]
    assert [c["text"] for c in pf.select(judged, count=1)] == ["first"]


def test_selection_refuses_to_pad_with_low_fit_prompts():
    judged = [{"id": i, "fit": 1, "specificity": 5, "motion": 5, "text": str(i)} for i in range(1, 4)]
    with pytest.raises(pf.JudgementError):
        pf.select(judged, count=3)


def test_judge_prompt_hides_the_scene_label():
    for label, description in pf.SCENES.items():
        prompt = pf.judge_prompt(description, ["one prompt", "two prompt"])
        assert label not in prompt, (label, "leaked into the judge prompt")
        assert "one prompt" in prompt and "two prompt" in prompt


def test_the_committed_prompt_set_matches_the_contract():
    """The corpus is generated from these files, so their shape is a contract."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "v0"
    for scene in pf.SCENES:
        path = root / f"{scene}.txt"
        assert path.exists(), f"missing prompt file for {scene}"
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 32, f"{scene}: {len(lines)} prompts, expected 32"
        assert len(set(lines)) == len(lines), f"{scene}: duplicate prompts"
        for line in lines:
            words = line.split()
            assert pf.MIN_WORDS <= len(words) <= pf.MAX_WORDS, f"{scene}: {len(words)} words: {line!r}"
        assert (root / f"{scene}.judged.jsonl").exists(), f"{scene}: no judgement log"


def test_every_committed_prompt_cleared_the_fit_floor():
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "v0"
    for scene in pf.SCENES:
        chosen = {
            line.strip()
            for line in (root / f"{scene}.txt").read_text().splitlines()
            if line.strip()
        }
        kept = [
            json.loads(line)
            for line in (root / f"{scene}.judged.jsonl").read_text().splitlines()
            if json.loads(line)["text"] in chosen
        ]
        assert len(kept) == len(chosen), f"{scene}: a kept prompt has no judgement"
        assert all(entry["fit"] >= pf.MIN_FIT for entry in kept), f"{scene}: kept a low-fit prompt"
