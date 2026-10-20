"""Generate and judge text-to-video prompts with kiro-cli.

Two passes, deliberately separate:

1. **Generate.** One kiro-cli session per scene asks for more prompts than we
   need, so the judge has something to choose between.
2. **Judge.** A *fresh* session scores the candidates blind: it is given the
   scene's description but not its label, so it cannot reward prompts that
   merely name the category.

Selection then keeps the top prompts by fit, with a floor, so a scene with a bad
batch fails loudly instead of being padded with off-type prompts.

The model's output is treated as untrusted text: every line passes through
parsing, duplicates are dropped, and a judgement that does not cover each id
exactly once is an error rather than a partial result.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SCENES",
    "MIN_FIT",
    "JudgementError",
    "generate",
    "judge",
    "judge_prompt",
    "parse_judgement",
    "parse_prompts",
    "select",
]

# Descriptions, not just labels: the generator and the judge both work from
# these, so "fit" is judged against intent rather than against a string.
SCENES: dict[str, str] = {
    "static_talking_head": (
        "A stationary or near-stationary shot of one person talking to camera, "
        "face and shoulders filling part of the frame, minimal camera movement."
    ),
    "cinematic_slowpan": (
        "A slow camera pan or dolly across a wide scene, smooth movement, "
        "skies or walls with gentle gradients, deliberate pacing."
    ),
    "high_motion_action": (
        "Fast movement filling the frame: sport, vehicles, falls, chase, or "
        "crowds. Short shutter, motion blur, rapid change between frames."
    ),
    "fine_texture_nature": (
        "Natural detail at close range: foliage, water, fur, sand, rain. The "
        "kind of texture a codec quantises away first."
    ),
    "graphics_text": (
        "On-screen text, titles, signage or UI: hard edges, flat colour, high "
        "contrast, fine strokes."
    ),
    "animation_flat": (
        "Illustrated or animated content with large flat colour regions, "
        "clean lines and limited palettes."
    ),
}

MIN_FIT = 4
"""A prompt scoring below this on scene fit is never selected."""

MIN_WORDS = 8
MAX_WORDS = 70


class JudgementError(ValueError):
    """The judge's output could not be used."""


@dataclass(frozen=True)
class Prompt:
    scene: str
    text: str


def generate_prompt(description: str, count: int) -> str:
    return (
        "You write prompts for a text-to-video model. The target content is:\n"
        f"{description}\n\n"
        f"Write {count} prompts for that target. Rules:\n"
        "- one prompt per line, plain text, no numbering, no bullets, no quotes\n"
        "- each names a subject, an action, and a camera framing or movement\n"
        "- each is visually specific: materials, light, weather, colour, lens\n"
        "- vary subject, setting and time of day; do not repeat a scene\n"
        "- no instructions to the model, no mentions of codecs, quality or video models\n"
        "- output the prompts only, nothing before or after\n"
    )


def judge_prompt(description: str, candidates: list[str]) -> str:
    """Score candidates blind: the scene label is deliberately absent."""
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(candidates, 1))
    return (
        "You are grading prompts for a text-to-video model. The intended content is:\n"
        f"{description}\n\n"
        "Score each prompt from 1 to 5 on:\n"
        "- fit: how well it matches the intended content\n"
        "- specificity: how much concrete visual detail it gives\n"
        "- motion: how clearly it implies movement (camera or subject)\n\n"
        f"Prompts:\n{numbered}\n\n"
        "Answer with one JSON object per line and nothing else:\n"
        '{"id": <number>, "fit": <1-5>, "specificity": <1-5>, "motion": <1-5>}\n'
    )


def parse_prompts(text: str) -> list[str]:
    """Extract prompts from model output, dropping prose, numbering and repeats."""
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("```", "#", "//")):
            continue
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        line = line.strip('"').strip()
        words = line.split()
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue
        if line.endswith(":"):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(line)
    return prompts


def parse_judgement(text: str, expected_ids: tuple[int, ...] | None = None) -> list[dict]:
    """Parse the judge's JSON lines, rejecting anything ambiguous."""
    judged: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            entry = {
                "id": int(payload["id"]),
                "fit": int(payload["fit"]),
                "specificity": int(payload["specificity"]),
                "motion": int(payload["motion"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise JudgementError(f"malformed judgement line {line!r}: {error}") from error
        for field in ("fit", "specificity", "motion"):
            if not 1 <= entry[field] <= 5:
                raise JudgementError(f"{field}={entry[field]} out of range in {line!r}")
        judged.append(entry)

    ids = [entry["id"] for entry in judged]
    if len(set(ids)) != len(ids):
        raise JudgementError("the judge scored the same id more than once")
    if expected_ids is not None and set(ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(ids))
        raise JudgementError(f"judge skipped ids: {missing}")
    return judged


def select(judged: list[dict], count: int) -> list[dict]:
    """Top prompts by fit-weighted score, deterministically, with a floor."""
    eligible = [entry for entry in judged if entry["fit"] >= MIN_FIT]
    if len(eligible) < count:
        raise JudgementError(
            f"only {len(eligible)} prompts scored fit >= {MIN_FIT}, need {count}: "
            "regenerate this scene rather than padding it"
        )
    ranked = sorted(
        eligible,
        key=lambda entry: (
            -(entry["fit"] * 2 + entry["specificity"] + entry["motion"]),
            entry["id"],
        ),
    )
    return ranked[:count]


def _ask(prompt: str, model: str | None = None) -> str:
    """One non-interactive kiro-cli turn. Stderr carries its logging."""
    command = ["kiro-cli", "chat", "--no-interactive", "--trust-tools="]
    if model:
        command += ["--model", model]
    command.append(prompt)
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"kiro-cli failed ({result.returncode}): {result.stderr[-400:]}")
    return result.stdout


def generate(scene: str, count: int, model: str | None = None) -> list[str]:
    prompts = parse_prompts(_ask(generate_prompt(SCENES[scene], count), model))
    if len(prompts) < count:
        raise JudgementError(f"{scene}: asked for {count}, parsed {len(prompts)}")
    return prompts


def judge(scene: str, candidates: list[str], model: str | None = None) -> list[dict]:
    raw = _ask(judge_prompt(SCENES[scene], candidates), model)
    judged = parse_judgement(raw, expected_ids=tuple(range(1, len(candidates) + 1)))
    for entry in judged:
        entry["text"] = candidates[entry["id"] - 1]
    return judged


def write_prompts(scene: str, prompts: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scene}.txt"
    path.write_text("".join(f"{entry['text']}\n" for entry in prompts))
    return path


def _run_scene(scene: str, want: int, keep: int, out_dir: Path, model: str | None) -> Path:
    """Generate, judge, select and write one scene. Returns the prompt file."""
    candidates = generate(scene, want, model)
    judged = judge(scene, candidates, model)
    chosen = select(judged, keep)
    # Provenance: the full judged set, so a later reviewer can see what was cut
    # and why, rather than only the survivors.
    log = out_dir / f"{scene}.judged.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in judged))
    return write_prompts(scene, chosen, out_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "prompts" / "v0"))
    parser.add_argument("--want", type=int, default=48, help="candidates to generate per scene")
    parser.add_argument("--keep", type=int, default=32, help="prompts to keep per scene")
    parser.add_argument("--model", default=None, help="kiro-cli model override")
    parser.add_argument("--scene", action="append", help="restrict to these scenes")
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="run each scene in its own tmux session instead of sequentially",
    )
    args = parser.parse_args(argv)

    scenes = args.scene or list(SCENES)
    unknown = [scene for scene in scenes if scene not in SCENES]
    if unknown:
        parser.error(f"unknown scenes: {unknown}; known: {list(SCENES)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tmux:
        # One session per scene: kiro-cli turns take minutes, so running them in
        # parallel is the difference between one coffee and six.
        for scene in scenes:
            # tmux starts a login shell, which may have neither `python` nor
            # this working directory: be explicit about both.
            root = Path(__file__).resolve().parent.parent
            command = (
                f"cd {root} && PYTHONPATH={root} python3 -m tools.prompt_factory "
                f"--out {out} --want {args.want} "
                f"--keep {args.keep} --scene {scene}"
                + (f" --model {args.model}" if args.model else "")
                + f" 2>&1 | tee {out}/{scene}.log"
            )
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", f"prompts-{scene}", command], check=True
            )
            print(f"started tmux session prompts-{scene}")
        print(f"attach with: tmux attach -t prompts-{scenes[0]}")
        return 0

    for scene in scenes:
        path = _run_scene(scene, args.want, args.keep, out, args.model)
        print(f"{scene}: {args.keep} prompts -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
