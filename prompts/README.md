# Prompts

The frozen text-to-video prompt set that the dataset corpus is generated from.
One file per scene, one prompt per line, stable order, 32 lines each.

| file | prompts |
|---|---|
| `v0/static_talking_head.txt` | 32 |
| `v0/cinematic_slowpan.txt` | 32 |
| `v0/high_motion_action.txt` | 32 |
| `v0/fine_texture_nature.txt` | 32 |
| `v0/graphics_text.txt` | 32 |
| `v0/animation_flat.txt` | 32 |

## Why frozen

The corpus is only comparable across calibration runs if the prompts are. Changing a
prompt changes the distribution of content, which changes the RD curve, which changes the
calibration. Treat these files as versioned data: a new set is `v1/`, never an edit to `v0/`.

## How they were made

`tools/prompt_factory.py`:

1. **Generate** — `kiro-cli` writes 48 candidate prompts per scene from the scene
   description in `tools/prompt_factory.py::SCENES`.
2. **Judge** — a *fresh* session scores each candidate 1-5 on `fit`, `specificity` and
   `motion`. The judge gets the scene description but not the scene **label**, so it cannot
   reward prompts that merely name the category.
3. **Select** — rank by `2*fit + specificity + motion`, ties by id, hard floor `fit >= 4`.
   A scene that cannot fill 32 above the floor fails and is regenerated rather than padded.

`v0/<scene>.judged.jsonl` keeps every candidate and its scores, so a reviewer can see what
was rejected and why, not only what survived.

Regenerate with:

```sh
python3 -m tools.prompt_factory --tmux --want 48 --keep 32 --out prompts/v0
```

## What a prompt is expected to contain

A subject, an action, and a camera framing or movement, plus enough visual specifics
(materials, light, weather, colour, lens) that two clips from it are recognisably the same
scene. Camera language is not decoration: focal length changes how much fine detail reaches
the encoder, which is what CRF interacts with.
