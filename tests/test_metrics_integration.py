"""Metric layer against real ffmpeg.

These tests are the reason the metric definitions can be trusted: they check our
numbers against ffmpeg's on the same real encoded pair, not against themselves.
Skipped when the machine has no ffmpeg.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

from calibrator import metrics

W, H, FRAMES = 64, 64, 3


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg not installed")
    return path


def raw_rgb(frames: np.ndarray) -> bytes:
    return np.ascontiguousarray(frames, dtype=np.uint8).tobytes()


def make_pair(tmp_path) -> tuple[np.ndarray, np.ndarray]:
    """A reference clip and a lossy encode of it, as uint8 RGB arrays."""
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (FRAMES, H, W, 3), dtype=np.uint8)
    ref_path = tmp_path / "ref.mkv"
    dist_path = tmp_path / "dist.mp4"

    for out, extra in (
        (ref_path, ["-c:v", "ffv1"]),
        (dist_path, ["-c:v", "libx264", "-crf", "40", "-pix_fmt", "yuv420p"]),
    ):
        subprocess.run(
            [
                ffmpeg(), "-hide_banner", "-nostdin", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                "-i", "-", *extra, str(out),
            ],
            input=raw_rgb(reference),
            check=True,
            capture_output=True,
        )

    decoded = []
    for path in (ref_path, dist_path):
        result = subprocess.run(
            [ffmpeg(), "-hide_banner", "-nostdin", "-v", "error",
             "-i", str(path), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            check=True,
            capture_output=True,
        )
        decoded.append(np.frombuffer(result.stdout, np.uint8).reshape(FRAMES, H, W, 3))
    return decoded[0], decoded[1]


def test_our_psnr_matches_ffmpeg_on_a_real_encode(tmp_path):
    reference, distorted = make_pair(tmp_path)
    theirs = subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-nostdin", "-v", "info",
            "-i", str(tmp_path / "dist.mp4"), "-i", str(tmp_path / "ref.mkv"),
            "-lavfi", "[0:v]format=rgb24[a];[1:v]format=rgb24[b];[a][b]psnr",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    reported = dict(re.findall(r"(?:PSNR )?([rgb]):([\d.]+)", theirs.stderr))
    assert set(reported) == {"r", "g", "b"}, theirs.stderr[-400:]

    ours = metrics.psnr_planes(reference, distorted)
    for channel, name in zip(ours, "rgb"):
        assert channel == pytest.approx(float(reported[name]), abs=0.05), name


def test_our_ssim_is_in_the_same_ballpark_as_ffmpeg(tmp_path):
    reference, distorted = make_pair(tmp_path)
    theirs = subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-nostdin", "-v", "info",
            "-i", str(tmp_path / "dist.mp4"), "-i", str(tmp_path / "ref.mkv"),
            "-lavfi", "[0:v][1:v]ssim",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"SSIM Y:([\d.]+)", theirs.stderr)
    assert match, theirs.stderr[-400:]
    ours = metrics.ssim(reference, distorted)
    # Different window definitions (Gaussian vs ffmpeg's blocks), so this is a
    # sanity band rather than equality.
    assert abs(ours - float(match.group(1))) < 0.05, (ours, match.group(1))


def test_vmaf_scores_a_real_encode(tmp_path):
    caps = metrics.capabilities()
    if not caps.has_libvmaf:
        pytest.skip("this ffmpeg has no libvmaf")

    reference, _ = make_pair(tmp_path)
    # make_pair already wrote ref.mkv and dist.mp4 into tmp_path.
    result = metrics.vmaf(
        tmp_path / "ref.mkv",
        tmp_path / "dist.mp4",
        width=W,
        height=H,
        workdir=tmp_path,
    )
    assert 0.0 < result.score <= 100.0
    assert result.implementation in ("cpu", "cuda")
    assert result.model.startswith("vmaf_v")
