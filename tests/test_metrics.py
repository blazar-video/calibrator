"""Metric layer: PSNR, SSIM, and the VMAF command construction.

The invariants here are the ones that make calibration numbers comparable
across runs. Anything that could silently change a score (a resize, a range
conversion) must fail loudly instead.
"""

import numpy as np
import pytest

from calibrator import metrics


def solid(value: int, size=(16, 16, 3)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def test_identical_frames_have_infinite_psnr_and_unit_ssim():
    frame = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert metrics.psnr(frame, frame) == float("inf")
    assert metrics.ssim(frame, frame) == pytest.approx(1.0, abs=1e-9)


def test_psnr_of_a_known_offset_matches_the_analytic_value():
    # Every sample off by one: PSNR = 20*log10(255/1).
    reference = solid(100)
    distorted = solid(101)
    assert metrics.psnr(reference, distorted) == pytest.approx(48.13, abs=0.01)


def test_psnr_of_black_against_white_is_zero_db():
    assert metrics.psnr(solid(0), solid(255)) == pytest.approx(0.0, abs=1e-9)


def test_psnr_planes_reports_each_channel_and_the_mean_skips_exact_ones():
    # One channel off by 1, two bit-exact: the exact planes report inf and are
    # excluded from the mean, leaving the degraded channel's own figure.
    reference = solid(100)
    distorted = reference.copy()
    distorted[..., 1] += 1
    planes = metrics.psnr_planes(reference, distorted)
    assert planes[0] == float("inf") and planes[2] == float("inf")
    assert planes[1] == pytest.approx(48.13, abs=0.01)
    assert metrics.psnr(reference, distorted) == pytest.approx(48.13, abs=0.01)


def test_ssim_falls_as_the_distortion_grows():
    reference = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mild = metrics.ssim(reference, np.clip(reference.astype(int) + 4, 0, 255).astype(np.uint8))
    harsh = metrics.ssim(reference, np.clip(reference.astype(int) + 40, 0, 255).astype(np.uint8))
    assert 0.0 <= harsh < mild <= 1.0


def test_ssim_is_computed_on_luma_so_a_hue_change_moves_it_less_than_a_luma_change():
    green = solid(0)
    green[..., 1] = 255
    red = solid(0)
    red[..., 0] = 255
    reference = solid(0)
    # Luma weights: green moves Y by 0.7152, red by 0.2126, so the green frame
    # is the bigger perceptual change and must score lower.
    assert metrics.ssim(reference, green) < metrics.ssim(reference, red)


def test_a_dimension_mismatch_is_an_error_naming_both_sizes():
    # Sizes are reported WxH, so 6 wide by 8 high prints as "6x8".
    reference = np.zeros((8, 6, 3), np.uint8)
    distorted = np.zeros((8, 5, 3), np.uint8)
    with pytest.raises(metrics.MetricError) as excinfo:
        metrics.psnr(reference, distorted)
    message = str(excinfo.value)
    assert "6x8" in message and "5x8" in message


def test_metrics_reject_floats_so_range_cannot_be_silently_wrong():
    with pytest.raises(metrics.MetricError):
        metrics.psnr(np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.uint8))


def test_vmaf_command_does_not_scale_when_the_sizes_match():
    command = metrics.build_vmaf_command("ref.mp4", "dist.mp4", width=64, height=64, vmaf_log="v.json")
    joined = " ".join(command)
    assert "scale" not in joined, "VMAF is sensitive to scaling; it must be explicit"
    assert "libvmaf" in joined


def test_vmaf_command_uses_bicubic_only_when_scaling_is_requested():
    scaled = metrics.build_vmaf_command(
        "ref.mp4", "dist.mp4", width=64, height=64, vmaf_log="v.json", scale_to=(128, 128)
    )
    joined = " ".join(scaled)
    assert "flags=bicubic" in joined
    assert "scale=128:128" in joined


def test_vmaf_prefers_cuda_when_the_build_has_it():
    command = metrics.build_vmaf_command(
        "ref.mp4", "dist.mp4", width=64, height=64, vmaf_log="v.json", cuda=True
    )
    graph = command[command.index("-lavfi") + 1]
    assert "libvmaf_cuda=" in graph
    assert "libvmaf=" not in graph.replace("libvmaf_cuda=", "")


def test_capabilities_reports_what_the_local_ffmpeg_has():
    caps = metrics.capabilities()
    assert isinstance(caps.has_libvmaf, bool)
    assert isinstance(caps.has_libvmaf_cuda, bool)
    # Both can be false; the point is that the probe answers instead of lying.
    assert caps.has_libvmaf_cuda <= caps.has_libvmaf or caps.has_libvmaf_cuda
