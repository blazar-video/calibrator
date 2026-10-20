"""Dataset work keys.

The point of the key is to make generation incremental: adding a prompt must not
re-render the other 191, and changing a setting that affects pixels must not
silently reuse an old clip. So the key covers exactly the inputs that determine
the pixels, and nothing else.
"""

import json

import pytest

from tools import dataset as ds


BASE = dict(
    model="Wan2.1-T2V-1.3B-Diffusers",
    model_revision="8cf6f5a1",
    prompt="A slow dolly across a salt flat at noon, white crust, 50mm lens",
    seed=0,
    width=832,
    height=480,
    fps=16,
    frames=81,
    inference={"steps": 50, "guidance": 5.0, "sampler": "unipc"},
)


def test_the_same_inputs_give_the_same_key():
    assert ds.clip_key(**BASE) == ds.clip_key(**BASE)


def test_key_does_not_depend_on_dict_ordering():
    reordered = dict(reversed(list(BASE["inference"].items())))
    assert ds.clip_key(**BASE) == ds.clip_key(**{**BASE, "inference": reordered})


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "LTX-Video-0.9.8-2B"),
        ("model_revision", "deadbeef"),
        ("prompt", "A different prompt at all, with a camera move and a lens"),
        ("seed", 1),
        ("width", 640),
        ("height", 384),
        ("fps", 24),
        ("frames", 121),
        ("inference", {"steps": 30, "guidance": 5.0, "sampler": "unipc"}),
    ],
)
def test_changing_anything_that_affects_pixels_changes_the_key(field, value):
    assert ds.clip_key(**{**BASE, field: value}) != ds.clip_key(**BASE)


def test_adding_a_prompt_leaves_the_other_keys_alone():
    prompts = ["first prompt with enough words in it", "second prompt with enough words"]
    one = [ds.clip_key(**{**BASE, "prompt": prompt}) for prompt in prompts]
    prompts.append("third prompt with enough words in it too")
    two = [ds.clip_key(**{**BASE, "prompt": prompt}) for prompt in prompts]
    assert one == two[:2]


def test_key_is_readable_and_full_length_is_kept():
    key = ds.clip_key(**BASE)
    assert len(key.short) == 16 and int(key.short, 16) >= 0
    assert len(key.full) == 64
    assert key.full.startswith(key.short)


def test_plan_skips_clips_that_already_exist():
    prompts = ["one prompt with enough words", "two prompt with enough words"]
    items = ds.plan(prompts, ["Wan2.1-T2V-1.3B-Diffusers"], BASE, seed=0)
    assert len(items) == 2
    existing = {items[0].key.full}
    again = ds.plan(prompts, ["Wan2.1-T2V-1.3B-Diffusers"], BASE, seed=0, existing=existing)
    assert [item.prompt for item in again] == ["two prompt with enough words"]


def test_plan_covers_every_model(tmp_path):
    items = ds.plan(["one prompt with enough words"], ["A", "B"], BASE, seed=0)
    assert {item.model for item in items} == {"A", "B"}
    assert len({item.key.full for item in items}) == 2


def test_index_round_trips(tmp_path):
    index = tmp_path / "index.jsonl"
    entry = ds.Clip(key=ds.clip_key(**BASE), path="clips/abc.mkv", **BASE)
    ds.append_index(index, [entry])
    ds.append_index(index, [entry])  # idempotent: same key must not duplicate
    loaded = ds.load_index(index)
    assert len(loaded) == 1
    assert loaded[0].key.full == entry.key.full
    assert loaded[0].prompt == BASE["prompt"]


def test_index_lines_are_one_json_object_each(tmp_path):
    index = tmp_path / "index.jsonl"
    entry = ds.Clip(key=ds.clip_key(**BASE), path="clips/abc.mkv", **BASE)
    ds.append_index(index, [entry])
    lines = [json.loads(line) for line in index.read_text().splitlines()]
    assert lines[0]["model"] == BASE["model"]
    assert lines[0]["inference"]["steps"] == 50
