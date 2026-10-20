"""Incremental dataset generation, keyed on what determines the pixels.

A clip's identity is a hash over the prompt, the model and revision, the seed, the
geometry, the frame count and the inference settings. Nothing else. That gives two
properties we need:

* **Adding prompts is cheap.** A new prompt gets a new key; every existing clip
  keeps its key, so nothing is re-rendered.
* **Changing a pixel-affecting setting is loud.** Bump the steps, swap the model
  revision, change the seed: the key changes and the clip is regenerated rather
  than silently reused.

Diffusion on a GPU is not bit-reproducible across driver or hardware changes, so
the key is a *work key*, not a fingerprint of the bytes. The index records the
machine and driver alongside it so a later reviewer can tell which clips were made
where.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "Clip",
    "ClipKey",
    "WorkItem",
    "append_index",
    "clip_key",
    "load_index",
    "plan",
]


@dataclass(frozen=True)
class ClipKey:
    short: str
    """16 hex characters: enough for filenames, and what the index is keyed on."""
    full: str
    """Full sha256, so two short keys can never be confused in an audit."""


@dataclass(frozen=True)
class WorkItem:
    model: str
    prompt: str
    key: ClipKey


@dataclass(frozen=True)
class Clip:
    key: ClipKey
    path: str
    model: str
    model_revision: str
    prompt: str
    seed: int
    width: int
    height: int
    fps: float
    frames: int
    inference: dict = field(default_factory=dict)
    machine: dict = field(default_factory=dict)
    """Provenance only: never part of the key, because hardware does not define
    the request, only how faithfully it was served."""


def clip_key(
    *,
    model: str,
    model_revision: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    fps: float,
    frames: int,
    inference: dict,
) -> ClipKey:
    """Hash exactly the inputs that determine the pixels."""
    payload = {
        "model": model,
        "model_revision": model_revision,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "fps": float(fps),
        "frames": frames,
        # Sorted so the key does not depend on dict ordering.
        "inference": {key: inference[key] for key in sorted(inference)},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    full = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ClipKey(short=full[:16], full=full)


def plan(
    prompts: list[str],
    models: list[str],
    base: dict,
    *,
    seed: int,
    existing: set[str] | None = None,
) -> list[WorkItem]:
    """Work still to do: every (model, prompt) whose key is not already on disk."""
    done = existing or set()
    items: list[WorkItem] = []
    for model in models:
        for prompt in prompts:
            key = clip_key(**{**base, "model": model, "prompt": prompt, "seed": seed})
            if key.full in done:
                continue
            items.append(WorkItem(model=model, prompt=prompt, key=key))
    return items


def _clip_to_json(clip: Clip) -> dict:
    payload = asdict(clip)
    payload["key"] = {"short": clip.key.short, "full": clip.key.full}
    return payload


def append_index(path: Path, clips: list[Clip]) -> None:
    """Append new clips, skipping any key already present. Idempotent on rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {clip.key.full for clip in load_index(path)} if path.exists() else set()
    with path.open("a", encoding="utf-8") as handle:
        for clip in clips:
            if clip.key.full in seen:
                continue
            handle.write(json.dumps(_clip_to_json(clip), ensure_ascii=False) + "\n")
            seen.add(clip.key.full)


def load_index(path: Path) -> list[Clip]:
    if not path.exists():
        return []
    clips: list[Clip] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        key = ClipKey(short=payload["key"]["short"], full=payload["key"]["full"])
        clips.append(
            Clip(
                key=key,
                path=payload["path"],
                model=payload["model"],
                model_revision=payload["model_revision"],
                prompt=payload["prompt"],
                seed=payload["seed"],
                width=payload["width"],
                height=payload["height"],
                fps=payload["fps"],
                frames=payload["frames"],
                inference=payload.get("inference", {}),
                machine=payload.get("machine", {}),
            )
        )
    return clips


def main(argv: list[str] | None = None) -> int:
    """Dry run: show what a dataset build would do without touching the GPU."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset build planner")
    parser.add_argument("--prompts", default="prompts/v0")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-revision", default="unpinned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--index", default="datasets/index.jsonl")
    parser.add_argument(
        "--inference",
        default="{}",
        help='JSON of inference settings that affect pixels, e.g. \'{"steps":50}\'',
    )
    args = parser.parse_args(argv)

    prompt_dir = Path(args.prompts)
    prompts = [
        line.strip()
        for path in sorted(prompt_dir.glob("*.txt"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    base = {
        "model_revision": args.model_revision,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frames": args.frames,
        "inference": json.loads(args.inference),
    }
    index = Path(args.index)
    existing = {clip.key.full for clip in load_index(index)}
    items = plan(prompts, args.models, base, seed=args.seed, existing=existing)
    print(
        f"{len(prompts)} prompts x {len(args.models)} models = "
        f"{len(prompts) * len(args.models)} clips; "
        f"{len(existing)} indexed; {len(items)} to generate"
    )
    for item in items[:5]:
        print(f"  {item.key.short}  {item.model}  {item.prompt[:60]}")
    if len(items) > 5:
        print(f"  ... and {len(items) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
