"""Quality metrics for calibration runs.

Two rules shape this module, both of them learned from numbers that looked fine
and were not:

* **Nothing changes the pixels unless a caller asked for it.** A metric that
  silently resizes (or accepts floats in the wrong range) makes between-run
  comparisons meaningless. Dimension mismatches and non-uint8 inputs are errors.
* **VMAF scaling is explicit.** VMAF is sensitive to how frames are scaled: a
  bilinear default instead of bicubic moves scores by whole points. The command
  builder therefore emits no scale filter unless one is requested, and then it
  says ``flags=bicubic``.

PSNR is computed on the RGB frames as decoded, averaged over channels, which
matches the numbers already recorded for this project. SSIM is computed on luma
because that is the channel the eye integrates.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Capabilities",
    "MetricError",
    "VmafError",
    "VmafResult",
    "build_vmaf_command",
    "capabilities",
    "psnr",
    "ssim",
    "vmaf",
]


class MetricError(ValueError):
    """Inputs the metric layer refuses to score."""


class VmafError(RuntimeError):
    """VMAF was requested but could not produce a score."""


@dataclass(frozen=True)
class Capabilities:
    """What the local ffmpeg can do, probed rather than assumed."""

    has_libvmaf: bool
    has_libvmaf_cuda: bool
    ffmpeg: str | None


@dataclass(frozen=True)
class VmafResult:
    score: float
    """Harmonic-mean VMAF over the clip."""
    implementation: str
    """``cuda`` or ``cpu``: which filter produced this number."""
    model: str
    """VMAF model version, e.g. ``vmaf_v0.6.1``."""


def _check_pair(reference: np.ndarray, distorted: np.ndarray) -> None:
    for name, frame in (("reference", reference), ("distorted", distorted)):
        if frame.dtype != np.uint8:
            raise MetricError(
                f"{name} frames must be uint8, got {frame.dtype}; "
                "convert deliberately so the value range is not in question"
            )
        if frame.ndim not in (2, 3, 4):
            raise MetricError(
                f"{name} frames must be HW, HWC, or THWC, got shape {frame.shape}"
            )
    if reference.shape != distorted.shape:
        raise MetricError(
            f"reference is {_size(reference)}, distorted is {_size(distorted)}; "
            "metrics are only comparable at identical dimensions "
            "(scaling is a separate, explicit step)"
        )


def _size(frame: np.ndarray) -> str:
    """Size as WxH, the convention used everywhere else in this project."""
    return f"{frame.shape[1]}x{frame.shape[0]}"


def psnr_planes(
    reference: np.ndarray, distorted: np.ndarray, peak: float = 255.0
) -> tuple[float, ...]:
    """Per-channel PSNR in dB.

    This is the definition ffmpeg reports, one number per plane, so a run can be
    compared against numbers recorded elsewhere. Computed in float64: uint8
    differences squared overflow a 32-bit accumulator at 1080p.
    """
    _check_pair(reference, distorted)
    difference = reference.astype(np.float64) - distorted.astype(np.float64)
    if difference.ndim == 2:
        planes = (difference,)
    else:
        # Channels are the last axis for both HWC and THWC.
        planes = tuple(difference[..., index] for index in range(difference.shape[-1]))
    result = []
    for plane in planes:
        mse = float(np.mean(plane * plane))
        result.append(float("inf") if mse == 0.0 else float(10.0 * np.log10((peak * peak) / mse)))
    return tuple(result)


def psnr(reference: np.ndarray, distorted: np.ndarray, peak: float = 255.0) -> float:
    """Mean PSNR in dB across channels.

    Bit-exact channels are excluded from the mean (they would otherwise report
    ``inf`` and swamp the average); all-exact frames give ``inf``. Encoded video
    never has an exact plane in practice, so the exclusion only affects
    synthetic comparisons.
    """
    values = [value for value in psnr_planes(reference, distorted, peak) if value != float("inf")]
    if not values:
        return float("inf")
    return float(np.mean(values))


def _luma(frame: np.ndarray) -> np.ndarray:
    """BT.709 luma from RGB, or the frame itself if it is already single-plane."""
    if frame.ndim == 2:
        return frame.astype(np.float64)
    rgb = frame[..., :3].astype(np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _frames(frame: np.ndarray):
    """Yield the frames of a clip, so a single frame and a clip share one path."""
    if frame.ndim == 4:
        return list(frame)
    return [frame]


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    offsets = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable Gaussian blur with reflect padding.

    Implemented here rather than pulled from scikit-image: the window is fixed by
    the SSIM definition, and a dependency that can change its default padding
    between releases would move every recorded score.
    """
    pad = len(kernel) // 2
    padded = np.pad(image, pad, mode="reflect")
    rows = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), 1, padded)
    return np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), 0, rows)


def ssim(reference: np.ndarray, distorted: np.ndarray, peak: float = 255.0) -> float:
    """Mean structural similarity on luma (11x11 Gaussian, sigma 1.5).

    Identical frames give 1.0. Values are in ``[-1, 1]``; a mean below zero means
    the frames are anti-correlated, which for decoded video indicates a real bug.
    """
    _check_pair(reference, distorted)
    values = [
        _ssim_frame(_luma(a), _luma(b), peak)
        for a, b in zip(_frames(reference), _frames(distorted))
    ]
    return float(np.mean(values))


def _ssim_frame(x: np.ndarray, y: np.ndarray, peak: float) -> float:
    kernel = _gaussian_kernel()

    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2

    mu_x = _blur(x, kernel)
    mu_y = _blur(y, kernel)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = _blur(x * x, kernel) - mu_xx
    sigma_yy = _blur(y * y, kernel) - mu_yy
    sigma_xy = _blur(x * y, kernel) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return float(np.mean(numerator / denominator))


def capabilities(ffmpeg: str = "ffmpeg") -> Capabilities:
    """Probe the local ffmpeg for VMAF filters.

    A machine without VMAF is not an error: the run records that VMAF was
    unavailable instead of inventing a score.
    """
    path = shutil.which(ffmpeg)
    if path is None:
        return Capabilities(has_libvmaf=False, has_libvmaf_cuda=False, ffmpeg=None)
    try:
        listing = subprocess.run(
            [path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return Capabilities(has_libvmaf=False, has_libvmaf_cuda=False, ffmpeg=path)

    # Match the filter *name* column, not a substring anywhere in the line.
    names = {
        line.split()[1]
        for line in listing.splitlines()
        if len(line.split()) > 1 and not line.startswith("Filters:")
    }
    return Capabilities(
        has_libvmaf="libvmaf" in names,
        has_libvmaf_cuda="libvmaf_cuda" in names,
        ffmpeg=path,
    )


def build_vmaf_command(
    reference: str | Path,
    distorted: str | Path,
    *,
    width: int,
    height: int,
    vmaf_log: str | Path,
    scale_to: tuple[int, int] | None = None,
    cuda: bool = False,
    model: str = "vmaf_v0.6.1",
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg invocation that scores a distorted clip against a reference.

    No scale filter is emitted unless ``scale_to`` is given, and then it is
    bicubic: VMAF is sensitive to the scaler, and a bilinear default would bias
    every comparison.
    """
    filter_name = "libvmaf_cuda" if cuda else "libvmaf"
    options = f"model=version={model}:log_fmt=json:log_path={vmaf_log}"
    # Both filters take two inputs, distorted first. When scaling is requested it
    # applies to the distorted stream only, which is the reference-vs-playback
    # case; the reference must stay untouched or the comparison is meaningless.
    if scale_to is not None:
        target_w, target_h = scale_to
        graph = (
            f"[0:v]scale={target_w}:{target_h}:flags=bicubic[scaled];"
            f"[scaled][1:v]{filter_name}={options}"
        )
    else:
        graph = f"[0:v][1:v]{filter_name}={options}"

    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(distorted),
        "-i",
        str(reference),
        "-lavfi",
        graph,
        "-f",
        "null",
        "-",
    ]


def vmaf(
    reference: str | Path,
    distorted: str | Path,
    *,
    width: int,
    height: int,
    workdir: str | Path,
    prefer_cuda: bool = True,
    scale_to: tuple[int, int] | None = None,
    ffmpeg: str = "ffmpeg",
) -> VmafResult:
    """Score ``distorted`` against ``reference`` with VMAF.

    Raises ``VmafError`` when the local build has no VMAF; a caller that can live
    without the score should check ``capabilities()`` first and record the gap.
    """
    caps = capabilities(ffmpeg)
    if not caps.has_libvmaf:
        raise VmafError(f"{ffmpeg} has no libvmaf filter; VMAF unavailable on this machine")

    log = Path(workdir) / "vmaf.json"
    use_cuda = prefer_cuda and caps.has_libvmaf_cuda
    command = build_vmaf_command(
        reference,
        distorted,
        width=width,
        height=height,
        vmaf_log=log,
        scale_to=scale_to,
        cuda=use_cuda,
        ffmpeg=caps.ffmpeg or ffmpeg,
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not log.exists():
        raise VmafError(
            f"vmaf failed ({result.returncode}): {result.stderr.strip()[:400] or 'no stderr'}"
        )
    import json

    payload = json.loads(log.read_text())
    pooled = payload.get("pooled_metrics", {}).get("vmaf", {})
    score = pooled.get("harmonic_mean")
    if score is None:
        raise VmafError("vmaf log has no pooled harmonic mean")
    return VmafResult(
        score=float(score),
        implementation="cuda" if use_cuda else "cpu",
        model="vmaf_v0.6.1",
    )
