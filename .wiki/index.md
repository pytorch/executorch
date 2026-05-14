# ExecuTorch Knowledge Base

> Auto-synthesized from 2,200+ GitHub issues and 99 discussions.
> Published corpus: GitHub issue threads and discussions. PR-specific review comments are not part of wiki v3 yet.
> Last updated: 2026-04-15 (wiki v3)

This knowledge base captures tribal knowledge — debugging steps, backend quirks, quantization recipes, and workarounds — that would otherwise live only in scattered issue threads and maintainers' heads.

**For AI agents**: Read this index first, then navigate to the relevant article. Knowledge articles have YAML frontmatter with `backends`, `category`, `source_issues`, `last_validated`, and `socs` when the topic is SoC-specific.

**For humans**: Browse by section or search for error messages, op names, or backend names.

---

## Backends

### QNN (Qualcomm AI Engine Direct)
- [Overview](backends/qnn/overview.md) — Architecture, delegation flow, supported hardware
- [SoC Compatibility Matrix](backends/qnn/soc-compatibility.md) — V68/V69/V73/V75/V79/V81 feature support, device-to-SoC mappings, arch-specific errors
- [Quantization Guide](backends/qnn/quantization.md) — Scheme selection, per-model recipes, mixed precision, common errors
- [Debugging Guide](backends/qnn/debugging.md) — Debug logging, diagnostic methodology, error message reference, profiling, memory analysis
- [Known Issues](backends/qnn/known-issues.md) — Active issues with workarounds, resolved instructive issues, version notes

### XNNPACK
- [Overview](backends/xnnpack/overview.md) — CPU backend capabilities, platform support, delegation patterns
- [Known Issues](backends/xnnpack/known-issues.md) — Operator gaps, dynamic shapes, threading, platform-specific bugs

### Vulkan
- [Overview](backends/vulkan/overview.md) — GPU backend, shader compilation, supported GPUs
- [Known Issues](backends/vulkan/known-issues.md) — GPU-specific bugs (PowerVR, Mali, Adreno), progressive model slicing, shader issues

### CoreML
- [Overview](backends/coreml/overview.md) — Apple hardware targets, MPS integration, iOS/macOS deployment, dim_order issues

### Arm (Ethos-U)
- [Overview](backends/arm/overview.md) — Ethos-U55/U85, TOSA, Vela compiler, Cortex-M deployment, FVP setup
- [Known Issues](backends/arm/known-issues.md) — Dynamic shapes, submodule issues, NHWC conversion, build problems

### Cadence (Xtensa)
- [Overview](backends/cadence/overview.md) — No-delegation architecture, preserved_ops pattern

## Export & Lowering
- [Common Pitfalls](export/common-pitfalls.md) — torch.export errors, dim_order gotchas, delegation failures, PTE size issues
- [Model-Specific Patterns](export/model-specific.md) — LLM export (dual codepaths), vision models, custom ops, dynamic shapes

## Quantization
- [Recipe Selection Guide](quantization/recipes.md) — When to use 8a8w vs 16a4w vs 16a8w, PT2E flow, calibration best practices
- [Accuracy Debugging](quantization/debugging.md) — Gibberish output diagnosis, accuracy drops, PTE size sanity checks

## Troubleshooting
- [Build Failures](troubleshooting/build-failures.md) — Submodule issues, platform-specific builds, CMake, dependency conflicts
- [Runtime Errors](troubleshooting/runtime-errors.md) — Missing ops, delegation fallthrough, model loading, memory issues
- [Performance](troubleshooting/performance.md) — ETDump profiling, FlameGraph, bottleneck analysis, benchmarking methodology

---

## How This Knowledge Base Works

- **Source**: Synthesized from pytorch/executorch GitHub issue threads and discussions. PR-specific review comments are documented in the synthesis guide but are not part of the published v3 corpus yet
- **Freshness**: Articles include `last_validated` dates and `source_issues` for traceability
- **Confidence**: Claims cite specific issue numbers — check the source if in doubt
- **Staleness**: Articles referencing functions/files that no longer exist need updating
- **Contributing**: Update articles when you resolve a new issue with reusable knowledge

## Quick Links

| If you're seeing... | Go to... |
|---|---|
| Build errors | [Build Failures](troubleshooting/build-failures.md) |
| Export/lowering errors | [Export Pitfalls](export/common-pitfalls.md) |
| Runtime crashes | [Runtime Errors](troubleshooting/runtime-errors.md) |
| Bad model accuracy | [Quantization Debugging](quantization/debugging.md) |
| Slow inference | [Performance](troubleshooting/performance.md) |
| QNN-specific errors | [QNN Debugging](backends/qnn/debugging.md) |
| "Missing operator" | [Runtime Errors](troubleshooting/runtime-errors.md) |
