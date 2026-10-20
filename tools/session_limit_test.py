"""How many concurrent Blazar encoders can this GPU actually open?

Real deployments serve many streams per box, and every Blazar Stream opens its
own NVENC session. Consumer cards cap concurrent sessions; datacenter cards do
not. This measures where the cap is on whatever GPU is present, and what
aggregate throughput looks like as concurrency rises.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import blazar

CAL = "/home/james/dev/code/blazar-video/calibrations/default"
W, H, FPS = 832, 480, 24
FRAMES = 16


def make(path):
    return blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced", backend="nvenc_h264", codec="h264", output=str(path),
    )


def one_stream(index, frames, out_dir):
    try:
        encoder = make(out_dir / f"conc_{index}.mp4")
        started = time.perf_counter()
        encoder.submit_device_batch(frames, fps=FPS, range="unit01")
        encoder.close()
        elapsed = time.perf_counter() - started
        return ("ok", elapsed, (out_dir / f"conc_{index}.mp4").stat().st_size)
    except Exception as error:  # noqa: BLE001 - the failure mode IS the result
        return ("fail", str(error)[:120], 0)


calibrations = blazar.Calibrations.load(CAL)
out_dir = __import__("pathlib").Path("/var/tmp/blazar/session_test")
out_dir.mkdir(parents=True, exist_ok=True)
frames = torch.rand((1, 3, FRAMES, H, W), device="cuda", dtype=torch.float32)
torch.cuda.synchronize()

print(f"frames per stream: {FRAMES} @ {W}x{H}")
print(f"{'streams':>8s} {'ok':>3s} {'worst ms':>9s} {'total ms':>9s} {'ms/frame':>9s}  failure")
for count in (1, 2, 4, 6, 8, 12, 16):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda i: one_stream(i, frames, out_dir), range(count)))
    total_ms = (time.perf_counter() - started) * 1000
    ok = [r for r in results if r[0] == "ok"]
    worst = max((r[1] for r in ok), default=0) * 1000
    failure = next((r[1] for r in results if r[0] != "ok"), "")
    per_frame = total_ms / (FRAMES * count) if ok else float("nan")
    print(f"{count:8d} {len(ok):3d} {worst:9.1f} {total_ms:9.1f} {per_frame:9.2f}  {failure}")
