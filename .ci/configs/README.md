# CI Configuration Files for LLM Export

This directory contains YAML configuration files used by CI tests for exporting LLM models with the new `extension.llm.export.export_llm` command.

## Usage

These config files can be used with the export command like this:

```bash
python -m extension.llm.export.export_llm --config path/to/config.yaml
```

Or you can override specific parameters:

```bash
python -m extension.llm.export.export_llm --config ci_stories110m_xnnpack_quantized.yaml base.checkpoint=my_checkpoint.pt
```

## Configuration Files

### CI Test Configurations
- `ci_stories110m_xnnpack_quantized.yaml` - Stories110M with XNNPACK quantization (used in test_llama.sh)
- `ci_stories110m_mps.yaml` - Stories110M with MPS backend
- `ci_stories110m_coreml.yaml` - Stories110M with CoreML backend  
- `ci_stories110m_qnn.yaml` - Stories110M with QNN backend

### Performance Test Configurations
- `llama3_spinquant.yaml` - Llama3 with SpinQuant (used in apple-perf.yml, android-perf.yml)
- `llama3_qlora.yaml` - Llama3 with QLoRA (QAT + LoRA)
- `llama3_coreml_ane.yaml` - Llama3 with CoreML ANE
- `xnnpack_8da4w_basic.yaml` - Basic XNNPACK 8da4w quantization
- `qwen3_xnnpack_8da4w.yaml` - Qwen3 with XNNPACK 8da4w quantization

### Specialized Configurations
- `stories110m_torchao_lowbit.yaml` - Stories110M with TorchAO lowbit quantization
- `xnnpack_custom_quantized.yaml` - XNNPACK with custom ops and quantization

## Background

These configuration files were created as part of migrating CI tests from the old `examples.models.llama.export_llama` command to the new `extension.llm.export.export_llm` command with hydra configuration support.

The config files help reduce duplication in CI scripts and make it easier to maintain consistent export settings across different test scenarios.