# LLM Export

High-level API for exporting LLMs to .pte format.

## Supported Models
Llama 2/3/3.1/3.2, Qwen 2.5/3, Phi 3.5/4-mini, SmolLM2

Full list: `extension/llm/export/config/llm_config.py`

For other models (Gemma, Mistral, BERT, Whisper): use optimum-executorch (see `/setup` skill).

## Basic Usage

```bash
python -m executorch.extension.llm.export.export_llm \
  --config path/to/config.yaml
```

## Config Structure

```yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'

model:
  use_kv_cache: True              # recommended
  use_sdpa_with_kv_cache: True    # recommended
  use_attention_sink: False       # extend generation
  quantize_kv_cache: False        # int8 KV cache

quantization:
  qmode: 8da4w                    # int8 activation + int4 weight
  group_size: 32
  embedding_quantize: 4,32

backend:
  xnnpack:
    enabled: True
    extended_ops: True

debug:
  verbose: True                   # show delegation table
  generate_etrecord: True         # for devtools profiling
```

## Quantization Modes

**TorchAO (XNNPACK)**:
- `8da4w`: int8 dynamic activation + int4 weight
- `int8`: int8 weight-only
- `torchao:8da4w`: low-bit kernels for Arm

**pt2e (QNN, CoreML, Vulkan)**: Use for non-CPU backends.

## Config Classes
All options in `extension/llm/export/config/llm_config.py`:
- `LlmConfig` - top level
- `ExportConfig` - max_seq_length, max_context_length
- `ModelConfig` - model optimizations
- `QuantizationConfig` - quantization options
- `BackendConfig` - backend settings
- `DebugConfig` - verbose, etrecord, profiling
