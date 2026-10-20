"""Generate one clip with vLLM-Omni and store it losslessly.

Written for the dataset builder: it renders one (model, prompt, seed) clip to
FFV1 RGB in MKV, which is lossless of the frames the model produced, and writes a
sidecar plus an index line keyed by tools.dataset.clip_key.

Run it from the vLLM-Omni environment (on mosk: ~/.vllm-stack). The engine is a
generation appliance: it is shut down inside this process before the file is
finalised, because 16 GiB does not hold the engine and anything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.dataset import Clip, append_index, clip_key


def frames_to_uint8(frames) -> np.ndarray:
    """(F, H, W, 3) uint8 from the pipeline's host output, which is in [0,1]."""
    import torch

    if isinstance(frames, (list, tuple)):
        item = frames[0] if len(frames) == 1 else None
        if item is not None:
            frames = item
    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]

    if hasattr(frames, "detach"):
        tensor = frames.detach().float().cpu()
        if tensor.dim() == 5:
            tensor = tensor[0]
        if tensor.dim() == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        array = tensor.numpy()
    else:
        array = np.asarray(frames)
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4 and array.shape[0] in (3, 4):
            array = array.transpose(1, 2, 3, 0)
        array = array.astype(np.float32)

    if array.min() < -0.01:
        raise SystemExit(
            f"frames have min {array.min():.3f}, outside [0,1]: the engine is handing "
            "back unnormalised values, fix the conversion rather than clipping blind"
        )
    return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)


def write_lossless(path: Path, frames: np.ndarray, fps: float) -> None:
    """FFV1 RGB in MKV: lossless of the frames, no chroma resampling."""
    count, height, width, _ = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24", str(path),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"encoder produced no data for {count} frames")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=Path("/var/tmp/blazar/datasets/v0/clips"))
    parser.add_argument("--index", type=Path, default=Path("/var/tmp/blazar/datasets/v0/index.jsonl"))
    args = parser.parse_args(argv)

    key = clip_key(
        model=args.model,
        model_revision=args.model_revision,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        inference={"steps": args.steps, "guidance": args.guidance},
    )
    clip_path = args.out_dir / f"{key.short}.mkv"
    if clip_path.exists():
        print(f"skip {key.short}: already on disk")
        return 0

    import torch
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = dict(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    try:
        params = OmniDiffusionSamplingParams(**sampling, fps=args.fps, frame_rate=args.fps)
    except TypeError:
        params = OmniDiffusionSamplingParams(**sampling)

    started = time.perf_counter()
    omni = Omni(model=args.model)
    loaded = time.perf_counter()
    output = omni.generate({"prompt": args.prompt, "modalities": ["video"]}, params)
    generated = time.perf_counter()
    if isinstance(output, list):
        output = output[0] if output else None
    if output is None:
        raise SystemExit("vLLM-Omni returned no output")
    frames = frames_to_uint8(output.images)
    del omni, output
    torch.cuda.empty_cache()

    write_lossless(clip_path, frames, args.fps)
    finished = time.perf_counter()

    clip = Clip(
        key=key,
        path=str(clip_path),
        model=args.model,
        model_revision=args.model_revision,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=int(frames.shape[0]),
        inference={"steps": args.steps, "guidance": args.guidance},
        machine={"host": "mosk", "gpu": "RTX 4080 SUPER"},
    )
    clip_path.with_suffix(".json").write_text(json.dumps({**clip.__dict__, "key": {"short": key.short, "full": key.full}}, indent=2, default=str))
    append_index(args.index, [clip])

    print(f"clip      : {clip_path} ({clip_path.stat().st_size / 1e6:.1f} MB, {frames.shape[0]} frames)")
    print(f"key       : {key.short}")
    print(f"load      : {loaded - started:.1f} s")
    print(f"generate  : {generated - loaded:.1f} s ({frames.shape[0] / max(generated - loaded, 1e-9):.2f} fps)")
    print(f"write     : {finished - generated:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
