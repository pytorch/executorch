# Swin2SR VGF Export

This example provides two scripts:

- `prepare_demo_assets.py` — Creates a deterministic local demo dataset from a
  repo-local screenshot so the export and runtime steps can be reproduced
  without an external SR dataset.
- `export_super_resolution.py` — Loads a checkpoint, applies optional
  post-training quantization, evaluates PSNR/SSIM on paired LR/HR samples, and
  exports a VGF-ready ExecuTorch program for host execution.

## Requirements

- Python 3.10+ with `executorch` and the dependencies in
  `examples/arm/super_resolution_example_vgf/requirements.txt`.
- ML SDK dependencies installed through `examples/arm/setup.sh
  --disable-ethos-u-deps --enable-mlsdk-deps`.

## Quick demo assets

To generate the text-heavy demo crop used by the runtime walkthrough, along
with the small LR/HR directories used for INT8 calibration and evaluation, run:

```bash
python examples/arm/super_resolution_example_vgf/model_export/prepare_demo_assets.py \
  --output-dir ./demo_assets
```

This writes:

```text
demo_assets/
  calibration/hr/
  calibration/lr/
  eval/hr/
  eval/lr/
  runtime/demo_hr_128.png
  runtime/demo_lr_64.png
  metadata.json
```

The export flow expects paired RGB image directories with matching relative
paths on the LR and HR sides when evaluation metrics are requested. For each
pair, the HR image must be exactly `upscale x` larger than the LR image. The
exporter crops LR inputs to `--input-height` x `--input-width` and crops the HR
target to the matching scaled patch.

## Export and evaluate for VGF

The concrete quick-demo commands below use the pinned revision currently cached
for `caidas/swin2SR-classical-sr-x2-64`:
`cee1c923c6a37361c6e5650b65dcf4be821e5d52`.

### FP export

```bash
python examples/arm/super_resolution_example_vgf/model_export/export_super_resolution.py \
  --model-name swin2sr \
  --checkpoint caidas/swin2SR-classical-sr-x2-64 \
  --checkpoint-revision cee1c923c6a37361c6e5650b65dcf4be821e5d52 \
  --input-height 64 \
  --input-width 64 \
  --quantization-mode none \
  --eval-lr-dir ./demo_assets/eval/lr \
  --eval-hr-dir ./demo_assets/eval/hr \
  --num-eval-samples 2 \
  --output-path ./demo_assets/swin2sr_x2_vgf_fp.pte
```

### INT8 export

```bash
python examples/arm/super_resolution_example_vgf/model_export/export_super_resolution.py \
  --model-name swin2sr \
  --checkpoint caidas/swin2SR-classical-sr-x2-64 \
  --checkpoint-revision cee1c923c6a37361c6e5650b65dcf4be821e5d52 \
  --input-height 64 \
  --input-width 64 \
  --quantization-mode int8 \
  --calibration-lr-dir ./demo_assets/calibration/lr \
  --eval-lr-dir ./demo_assets/eval/lr \
  --eval-hr-dir ./demo_assets/eval/hr \
  --num-calibration-samples 4 \
  --num-eval-samples 2 \
  --output-path ./demo_assets/swin2sr_x2_vgf_int8.pte
```

For FP export, set `--quantization-mode none`. INT8 export requires
`--calibration-lr-dir`; the exporter no longer falls back to random calibration
inputs. The exporter first tries installed ExecuTorch quantized kernels and
then local build outputs such as `cmake-out/kernels/quantized` or
`arm_test/*/kernels/quantized` to register the quantized out-variant ops needed
by `to_executorch()`.

When `--eval-lr-dir` and `--eval-hr-dir` are provided, the exporter compares
the exported program module against the paired HR images and writes PSNR/SSIM
metrics. The quick-demo dataset is intentionally tiny, so these metrics are a
smoke signal for gross quality regressions rather than a benchmark target; use
a larger paired validation set when setting release-quality thresholds.

In the OOTB smoke flow, FP export over the generated 2-sample eval set produced
approximately PSNR 34.85 / SSIM 0.994, while INT8 PTQ produced approximately
PSNR 22.71 / SSIM 0.870. The INT8 drop is expected; these numbers are included
only as a smoke-test reference for the generated demo assets.

## Output artifacts

For an export path such as `./swin2sr_x2_vgf_int8.pte`, the exporter writes:

- `swin2sr_x2_vgf_int8.pte` — The ExecuTorch program.
- `swin2sr_x2_vgf_int8.json` — Static input/output metadata consumed by the
  runtime helper.
- `swin2sr_x2_vgf_int8_delegation.txt` — A summary of delegated and
  non-delegated operators.
- `swin2sr_x2_vgf_int8_metrics.json` — Optional PSNR/SSIM evaluation metrics
  when `--eval-lr-dir` and `--eval-hr-dir` are provided.
