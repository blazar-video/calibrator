# blazar-video calibrator

The offline measurement pipeline (Python). Drives generative video models, dumps
frames, sweeps encoder parameters, computes quality metrics (VMAF/SSIM and the
VMAF-trust investigation), emits RD curves, and writes tuned calibration files
that the Rust `blazar` runtime consumes.

Status: scaffold. The runtime (`blazar`) is being built engineering-first; the
calibrator is Phase 3. A **default calibration** ships with `blazar` so the
runtime is useful from day one without any calibrator output, which is why this
can come later.

See the `blazar` and `calibrations` repos for the runtime and the data contract.

## License

Apache-2.0. See `LICENSE`.
