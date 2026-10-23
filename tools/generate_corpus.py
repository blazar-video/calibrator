"""Generate the calibration corpus with LTX 2.5 and ship it to S3.

One clip per (prompt, seed). Everything that determines the pixels goes into the
clip key, the key is hashed into the object path, and the resulting artefacts are
recorded with their own sha256 so a later reader can prove what it downloaded.

Layout in the bucket:

    dataset/<model>/scene=<scene>/<clip_id>/master.mkv     FFV1 rgb24, lossless
    dataset/<model>/scene=<scene>/<clip_id>/preview.mp4    H.264 CRF 18, viewing only
    dataset/<model>/scene=<scene>/<clip_id>/clip.json      key, prompt, sizes, hashes

`clip_id` is sha256 over the pixel-determining inputs: model, revision,
weight_dtype, sampler_steps, distill_lora, prompt, seed, width, height, frames, fps.
Resolution is in the hash because two clips at different sizes are not the same
content; the calibration index chooses not to key on it, which is a separate
decision.

Usage (on the GPU box, inside the venv that has diffusers + torchao):

    python tools/generate_corpus.py --prompts prompts.jsonl --dry-run
    python tools/generate_corpus.py --prompts prompts.jsonl --bucket blazar-video-encoded-854934822967
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

WIDTH, HEIGHT, FRAMES, FPS = 1920, 1088, 121, 24.0
MODEL = "ltx-2.5"
MODEL_REPO = "unsloth/LTX-2.5-FP8"
MODEL_REVISION = "main"
WEIGHT_DTYPE = "fp8-e4m3"
SAMPLER_STEPS = 8
DISTILL_LORA = "none"
SIGMAS = None  # the model's own schedule; set a list to override


@dataclass(frozen=True)
class ClipKey:
    model: str
    model_repo: str
    model_revision: str
    weight_dtype: str
    sampler_steps: int
    distill_lora: str
    scene: str
    prompt_id: str
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    fps: float

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def clip_id(self) -> str:
        return self.sha256()[:32]


@dataclass
class Clip:
    key: dict
    clip_id: str
    prompt: str
    scene: str
    seed: int
    width: int
    height: int
    frames: int
    fps: float
    model_repo: str
    model_revision: str
    weight_dtype: str
    sampler_steps: int
    generated_seconds: float = 0.0
    artefacts: dict = field(default_factory=dict)
    machine: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_master(frames, path: Path, ffmpeg: str) -> None:
    """FFV1 rgb24 in MKV: lossless of the frames, verified bit-exact elsewhere."""
    n, h, w, _ = frames.shape
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24", str(path)],
        input=frames.tobytes(), check=True,
    )


def encode_preview(frames, path: Path, ffmpeg: str, crf: int = 18) -> None:
    """H.264 for humans. Not a master: 4:2:0 and lossy."""
    n, h, w, _ = frames.shape
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        input=frames.tobytes(), check=True,
    )


def load_pipeline(root: str):
    """Load the FP8 DiT onto the GPU without ever materialising a dense copy."""
    import json as _json

    import torch
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

    with torch.device("meta"):
        transformer = LTX2VideoTransformer3DModel.from_config(
            _json.load(open(f"{root}/transformer/config.json"))
        )
    with safe_open(f"{root}/LTX-2.5-FP8.safetensors", framework="pt") as f:
        meta = f.metadata() or {}
        raw = {k: f.get_tensor(k) for k in f.keys()}
    state = unflatten_tensor_state_dict(raw, meta)
    state = state[0] if isinstance(state, tuple) else state
    # These three live under an unsloth-specific prefix and the unflattener drops
    # them rather than renaming; without them `strict=True` refuses.
    for name in ("keyframes_abs_pos_embedding", "scale_shift_table", "audio_scale_shift_table"):
        if name not in state and f"unsloth_root::{name}" in raw:
            state[name] = raw[f"unsloth_root::{name}"]
    transformer.load_state_dict(state, strict=True, assign=True)
    del state, raw
    return LTX2Pipeline.from_pretrained(root, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")


def generate(pipe, prompt: str, seed: int):
    import numpy as np
    import torch

    kwargs = dict(
        prompt=prompt, width=WIDTH, height=HEIGHT, num_frames=FRAMES,
        num_inference_steps=SAMPLER_STEPS, guidance_scale=1.0,
        # Mandatory for a corpus: otherwise a second LLM rewrites the prompt and
        # the key no longer describes what generated the clip.
        enable_prompt_enhancement=False,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    )
    if SIGMAS is not None:
        kwargs.pop("num_inference_steps")
        kwargs["sigmas"] = SIGMAS
    out = pipe(**kwargs)
    frames = np.asarray(out.frames)
    frames = frames[0] if frames.ndim == 5 else frames
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return frames


def upload(local: Path, bucket: str, key: str, dry_run: bool) -> str:
    import boto3

    if dry_run:
        return f"dry-run://{bucket}/{key}"
    boto3.client("s3").upload_file(str(local), bucket, key)
    return f"s3://{bucket}/{key}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prompts", required=True, type=Path,
                   help="JSONL: one {scene, prompt_id, prompt} per line")
    p.add_argument("--root", default="/home/ubuntu/ltx25", help="diffusers pipeline dir")
    p.add_argument("--out", default=Path("/home/ubuntu/corpus"), type=Path)
    p.add_argument("--bucket", default="blazar-video-encoded-854934822967")
    p.add_argument("--seeds", type=int, default=1, help="seeds per prompt")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="generate and encode, skip S3")
    p.add_argument("--force", action="store_true", help="regenerate clips already on disk")
    args = p.parse_args(argv)

    import numpy as np  # noqa: F401  (imported for the frame path below)

    items = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    jobs = []
    for item in items:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            key = ClipKey(
                model=MODEL, model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
                weight_dtype=WEIGHT_DTYPE, sampler_steps=SAMPLER_STEPS, distill_lora=DISTILL_LORA,
                scene=item["scene"], prompt_id=item["prompt_id"], prompt=item["prompt"],
                seed=seed, width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS,
            )
            jobs.append((key, item))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} clips from {len(items)} prompts", flush=True)

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    pipe = None
    done = 0
    for key, item in jobs:
        scene_dir = args.out / MODEL / f"scene={key.scene}" / key.clip_id
        clip_json = scene_dir / "clip.json"
        if clip_json.exists() and not args.force:
            print(f"skip {key.clip_id} (on disk)", flush=True)
            continue
        if pipe is None:
            started = time.perf_counter()
            pipe = load_pipeline(args.root)
            print(f"[load] {time.perf_counter()-started:.1f} s", flush=True)

        scene_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        frames = generate(pipe, key.prompt, key.seed)
        elapsed = time.perf_counter() - started

        master = scene_dir / "master.mkv"
        preview = scene_dir / "preview.mp4"
        encode_master(frames, master, ffmpeg)
        encode_preview(frames, preview, ffmpeg)

        clip = Clip(
            key=asdict(key), clip_id=key.clip_id, prompt=key.prompt, scene=key.scene,
            seed=key.seed, width=key.width, height=key.height, frames=int(frames.shape[0]),
            fps=key.fps, model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
            weight_dtype=WEIGHT_DTYPE, sampler_steps=SAMPLER_STEPS,
            generated_seconds=round(elapsed, 2),
            machine={"host": "g7e.2xlarge", "gpu": "RTX PRO 6000 Blackwell 96GB"},
            artefacts={
                name: {"bytes": (scene_dir / name).stat().st_size,
                       "sha256": sha256_file(scene_dir / name)}
                for name in ("master.mkv", "preview.mp4")
            },
        )
        clip_json.write_text(json.dumps(asdict(clip), indent=2, sort_keys=True))

        prefix = f"dataset/{MODEL}/scene={key.scene}/{key.clip_id}"
        for name in ("master.mkv", "preview.mp4", "clip.json"):
            upload(scene_dir / name, args.bucket, f"{prefix}/{name}", args.dry_run)

        done += 1
        print(f"[{done}/{len(jobs)}] {key.clip_id} scene={key.scene} seed={key.seed} "
              f"{elapsed:.1f} s master={clip.artefacts['master.mkv']['bytes']/1e6:.1f} MB", flush=True)
    print(f"generated {done} clips", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
