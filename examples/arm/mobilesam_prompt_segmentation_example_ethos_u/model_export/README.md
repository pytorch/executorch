# MobileSAM Export

The exporter keeps one tested model configuration so the ExecuTorch steps are
visible without a layer of command-line configuration.

From the repository root, run:

```bash
python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/prepare_mobilesam.py
python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/export_mobilesam.py
```

The first script prepares the pinned official MobileSAM source and verified
checkpoint in `~/.cache/executorch/mobilesam`. It applies the included patch
there to support the smaller input size. Neither source nor checkpoint is
copied into the ExecuTorch repository.

The second script performs the model flow directly:

1. Wrap MobileSAM with the fixed point prompt.
2. Export with `torch.export`.
3. Calibrate and convert with PT2E, using A8W8 generally and A16W8 attention
   activations to preserve mask quality.
4. Check the FP32 and quantized masks have at least `0.9` IoU.
5. Lower with `EthosUPartitioner` and require one delegated subgraph.
6. Save `arm_test/mobilesam/export/mobilesam.pte`.

The export directory also contains the input tensor, host masks, validation
metrics, delegation summary, and TOSA/Vela intermediate artifacts.
