"""Compare Blazar's encode paths on real frames: device vs host, hardware vs software.

Three arms over the same clip:

  device  frames already on the GPU -> submit_device_batch  (the SGLang seam)
  host    frames copied to numpy    -> submit               (what a stock pipeline does)
  ffmpeg  frames written to disk    -> libx264 via ffmpeg   (the usual fallback)

Run on a machine with the Blazar binding and a GPU:

    python tools/bench_encode_paths.py --clip clip.mkv --backend nvenc_h264 --repeats 3
    python tools/bench_encode_paths.py --clip clip.mkv --backend x264 --repeats 3

The clip is decoded once, before any timing, so the numbers are encode costs and
not decoder costs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def decode_rgb(path: Path, width: int, height: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, height, width, 3)


def make_encoder(calibrations, backend: str, codec: str, output: Path, width: int, height: int):
    import blazar

    return blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend=backend,
        codec=codec,
        output=str(output),
    )


def run_arm(name, frames, width, height, fps, backend, codec, out_dir, repeats):
    import blazar
    import torch

    timings = []
    size = 0
    for attempt in range(repeats):
        output = out_dir / f"{name}_{attempt}.mp4"
        encoder = make_encoder(
            calibrations, backend, codec, output, width, height
        )
        # Frames reach the GPU before the clock starts: this measures encoding,
        # not the upload.
        device = torch.from_numpy(frames).cuda().float() / 255.0
        torch.cuda.synchronize()
        started = time.perf_counter()
        if name.startswith("device"):
            # (1, 3, T, H, W): the model's BCTHW layout, one submit for the clip.
            bcthw = device.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            encoder.submit_device_batch(bcthw, fps=fps, range="unit01")
        else:
            host = frames.astype(np.float32) / 255.0
            for frame in host:
                encoder.submit(
                    frame, width=width, height=height, fps=fps, layout="thwc", range="unit01"
                )
        encoder.close()
        timings.append(time.perf_counter() - started)
        size = output.stat().st_size
    best = min(timings)
    return best, best * 1000.0 / len(frames), size


def main(argv=None) -> int:
    global calibrations
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", type=Path, help="video file to decode")
    parser.add_argument("--from-npy", type=Path, help="uint8 THWC frames, skips decoding")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--backend", default="nvenc_h264")
    parser.add_argument("--codec", default="h264")
    parser.add_argument("--calibrations", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0, help="limit to the first N frames")
    parser.add_argument("--out-dir", type=Path, default=Path("/var/tmp/blazar/bench_paths"))
    args = parser.parse_args(argv)

    import blazar

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_npy:
        frames = np.load(args.from_npy)
        if frames.dtype != np.uint8:
            raise SystemExit(f"--from-npy expects uint8 THWC, got {frames.dtype}")
        frames = frames[: args.frames] if args.frames else frames
    else:
        if args.clip is None:
            raise SystemExit("pass --clip or --from-npy")
        frames = decode_rgb(args.clip, args.width, args.height)[: args.frames or None]
    print(f"clip: {len(frames)} frames {args.width}x{args.height} -> {args.backend}")

    results = []
    for arm in ("device", "host"):
        best, ms, size = run_arm(
            arm, frames, args.width, args.height, args.fps,
            args.backend, args.codec, args.out_dir, args.repeats,
        )
        results.append((arm, best, ms, size))
    print(f"{'arm':8s} {'wall s':>8s} {'ms/frame':>9s} {'bytes':>10s}")
    for arm, best, ms, size in results:
        print(f"{arm:8s} {best:8.3f} {ms:9.2f} {size:10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
