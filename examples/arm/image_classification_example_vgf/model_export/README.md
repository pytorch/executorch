# DEiT Fine-Tuning & VGF Export

This example provides two scripts:

- `train_deit.py` — Fine-tunes the DEiT-Tiny model on the Oxford-IIIT Pet
  dataset.
- `export_deit.py` — Loads the trained checkpoint, applies post-training
  quantization (PT2E), evaluates, and exports a VGF-ready ExecuTorch program for
  host execution.

## Requirements

- Python 3.10+ with `executorch` and the dependencies in
  `examples/arm/image_classification_example_vgf/requirements.txt`.
- Internet access to download pretrained weights and the Oxford-IIIT Pet
  dataset.

## Fine-tuning DEiT Tiny

```bash
python examples/arm/image_classification_example_vgf/model_export/train_deit.py \
  --output-dir ./deit-tiny-oxford-pet \
  --num-epochs 3
```

## Export and quantize for VGF

```bash
python examples/arm/image_classification_example_vgf/model_export/export_deit.py \
  --model-path ./deit-tiny-oxford-pet/final_model \
  --output-path ./deit_quantized_vgf.pte \
  --num-calibration-samples 300 \
  --num-test-samples 100
```

During export, the script:
- Exports the FP32 model using `torch.export.export()`.
- Applies symmetric quantization to each operator.
- Delegates the network to the VGF backend.
- Writes the ExecuTorch program to the requested path.

Use the generated `.pte` with the host `executor_runner` built with
`EXECUTORCH_BUILD_VGF=ON`.
