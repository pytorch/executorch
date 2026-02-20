# DEiT Fine-Tuning & Export

This example provides two scripts:

- `train_deit.py` — Fine-tunes the DEiT-Tiny model, initially trained on ImageNet 1k, on the Oxford-IIIT Pet dataset to repurpose the network
to classify cat and dog breeds. This is intended to demonstrate the process of preparing a model for a new usecase, before lowering it via ExecuTorch.
- `export_deit.py` — Loads the trained checkpoint from `train_deit.py`, applies post-training quantization (PT2E), evaluates, and exports an Ethos-U–ready ExecuTorch program.

The Oxford-IIIT Pet dataset is used by both scripts as it's a relatively small dataset, allowing this example to be run on a high-end laptop or desktop.

See the sections below for requirements and exact commands.

## Requirements

- Python 3.10+ with `executorch` and the dependencies in `examples/arm/image_classification_example/requirements.txt`.
- Internet access to download pretrained weights and the Oxford-IIIT Pet dataset.

## Fine-tuning DEiT Tiny

The `train_deit.py` script can be run as follows:

```bash
python examples/arm/image_classification_example/model_export/train_deit.py \
  --output-dir ./deit-tiny-oxford-pet \
  --num-epochs 3
```

The script splits the training set for validation, fine-tunes the model, reports test accuracy, and by default outputs the model to `deit-tiny-oxford-pet/final_model`.
Running this script achieves a test set accuracy of 86.10% in FP32.

## Export and quantize

The `export_deit.py` script can be run as follows:

```bash
python examples/arm/image_classification_example/model_export/export_deit.py \
  --model-path ./deit-tiny-oxford-pet/final_model \
  --output-path ./deit_quantized_exported.pte \
  --num-calibration-samples 300 \
  --num-test-samples 100
```

During export, the script:
- Exports the FP32 model using `torch.export.export()`.
- Applies symmetric quantization to each operator.
- Targets `Ethos-U85-256` with shared SRAM and lowers the network to Ethos-U.
- Writes the ExecuTorch program to the requested path.

Running this script following the `train_deit.py` script achieves a test set accuracy of 85.00% for the quantized model on 100 samples.

### Interpreting Vela Output

After the model has been compiled for Ethos-U, the Vela compiler will output a network summary. You will see output similar to:

```
Network summary for out
Accelerator configuration               Ethos_U85_256
System configuration             Ethos_U85_SYS_DRAM_Mid
Memory mode                               Shared_Sram
Accelerator clock                                1000 MHz
Design peak SRAM bandwidth                      29.80 GB/s
Design peak DRAM bandwidth                      11.18 GB/s

Total SRAM used                               1291.80 KiB
Total DRAM used                               5289.91 KiB

CPU operators = 0 (0.0%)
NPU operators = 898 (100.0%)

... (Truncated)
```

Some of this information is key to understanding the example application, which will run this model on device:

- The `Accelerator configuration` is `Ethos_U85_256`, so it will only work on an Ethos-U85 system. The FVP for this is Corstone-320.
- The `Memory mode` is `Shared_Sram`, so the tensor arena is allocated in SRAM while the model data is read from flash and DRAM.
- The `Total SRAM used` is `1291.80 KiB`, so at least this much memory will need to be allocated for the tensor arena.
