# Practical-RIFE Accuracy Flow

`qat_loop.py` measures Practical-RIFE eager, PTQ, and QAT accuracy on real
frame triples. It is intended as the accuracy companion to VGF performance
measurements, so each reported artifact can be tied back to a reproducible
model, ExecuTorch revision, and export command.

## Inputs

Use a standard Practical-RIFE checkout with this layout:

```text
<model-root>/train_log/IFNet_HDv3.py
<model-root>/train_log/flownet.pkl
```

The triples file must identify frame interpolation samples in this order:

```text
name,input0,input1,target
sample_00000,/path/frame_0.png,/path/frame_2.png,/path/frame_1.png
```

The script treats `input0` and `input1` as the two model inputs and `target`
as the expected middle frame.

## Real Accuracy Run

Use real frame triples for accuracy reporting. This example compares eager,
PTQ, and QAT at the default 768x384 shape while preserving uint8 image IO:

```bash
python examples/arm/QAT_example/qat_loop.py \
  --triples-list <path-to-real-triples.csv> \
  --model-root <path-to-Practical-RIFE> \
  --height 768 \
  --width 384 \
  --max-triples 10 \
  --calibration-samples 8 \
  --mode all \
  --io-quantization uint8 \
  --qat-samples 8 \
  --qat-steps 50 \
  --qat-lr 1e-5 \
  --output-dir out/rife_accuracy_10_real_triples_qat50
```

The output directory contains:

- `report.md`: human-readable aggregate metrics.
- `metrics.json`: aggregate and per-sample metrics.
- `metrics.csv`: tabular per-sample metrics.
- `quantization_coverage/`: PTQ and QAT graph coverage details.

## Smoke Runs

Small random or short QAT runs are useful only for checking that the flow runs.
Do not use them as model-quality evidence. For example, two samples and three
QAT steps can produce very low PSNR while still proving that eager, PTQ, and
QAT execution is wired correctly.

## Reference Samples

Keep reference images outside this repository and pass them through
`--triples-list`. For internal comparisons, use a fixed triples list and keep
the image bundle unchanged across graph and quantization experiments.

The triples list is the reproducibility contract. It should use repository- or
bundle-relative paths where possible, so another user can unpack the same image
bundle and run the same command without editing absolute paths.

## Reporting Checklist

When sharing accuracy or performance data, include:

- Practical-RIFE commit hash.
- ExecuTorch commit hash.
- `qat_loop.py` command line and flags.
- Dataset or triples-list identity.
- Input shape and preprocessing mode.
- IO quantization mode.
- PTQ calibration sample count.
- QAT sample count, step count, and learning rate.
- Reference image bundle identity.
- Whether a VGF or PTE was exported from the same run.

This keeps accuracy numbers comparable with board performance results and
avoids mixing smoke-test numbers with real-data validation.
