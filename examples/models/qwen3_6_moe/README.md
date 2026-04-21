# Qwen 3.6 MoE

Qwen 3.6 MoE uses the same architecture and runner as Qwen 3.5 MoE.
See [examples/models/qwen3_5_moe](../qwen3_5_moe/) for export, build,
and inference instructions.

Prequantized weights are available at
[SocialLocalMobile/Qwen3.6-35B-A3B-HQQ-INT4](https://huggingface.co/SocialLocalMobile/Qwen3.6-35B-A3B-HQQ-INT4).

Qwen 3.6 does not have quantization-aware training, so it requires
`--sensitive` for quantization. `--hqq` is recommended for better
expert weight accuracy. See the model card for details.

**Note:** This model has not been tested or evaluated. It is provided
mainly for development purposes.
