(api-section)=
# API

ExecuTorch has two runtime layers:

- The **core runtime** loads and executes methods in a compatible `.pte`
  program. Use it when your application owns preprocessing, postprocessing,
  and task orchestration.
- The **LLM runner** builds on the core runtime with tokenization, prefill and
  decode orchestration, sampling, and streaming generation for text and
  multimodal models.

The table below is the entry point for choosing a language binding. Maturity
follows the {doc}`api-life-cycle`: an API is stable unless it is explicitly
marked experimental or deprecated. An annotation on an individual API takes
precedence over this summary.

```{list-table} Runtime and LLM APIs by language
:header-rows: 1
:widths: 18 33 33 16

* - **Language / platform**
  - **Core runtime**
  - **LLM runner**
  - **Maturity**
* - C++
  - {doc}`extension-module` for the high-level `Module` API, or
    {doc}`executorch-runtime-api-reference` for `Program` and `Method`
  - {doc}`llm/run-with-c-plus-plus` for `TextLLMRunner` and
    `MultimodalRunner`
  - Core: **stable**; LLM: **experimental**
* - Python
  - {doc}`runtime-python-api-reference` for host-side loading, execution, and
    validation
  - [Python LLM runner bindings](https://github.com/pytorch/executorch/blob/main/extension/llm/runner/README.md#python-api),
    whose availability depends on the installed package or source build
  - Core: **stable**; LLM: **experimental**
* - Android: Java / Kotlin
  - {doc}`using-executorch-android` and the
    [Javadoc](https://pytorch.org/executorch/main/javadoc/) for `Module`,
    `Tensor`, and `EValue`
  - {doc}`llm/run-on-android` for `LlmModule`
  - Core: **experimental**; LLM: **experimental**
* - Apple: Swift / Objective-C
  - {doc}`using-executorch-ios` for `Module`, `Tensor`, and `Value`
  - {doc}`llm/run-on-ios` for `TextRunner` and `MultimodalRunner`
  - Core: **experimental**; LLM: **experimental**
* - Browser: JavaScript / WebAssembly
  - [WebAssembly `Module` and `Tensor`](https://github.com/pytorch/executorch/blob/main/extension/wasm/README.md),
    currently built from source
  - No high-level LLM runner API
  - Core: **experimental**
```

Use the core runtime for any compatible exported model, including vision,
audio, and custom workloads. Use an LLM runner only when the exported program
and tokenizer satisfy that runner's model metadata and packaging requirements.

## Export and reference documentation

- {doc}`export-to-executorch-api-reference` — Export to ExecuTorch API Reference
- {doc}`executorch-runtime-api-reference` — ExecuTorch Runtime API Reference
- {doc}`runtime-python-api-reference` — Runtime Python API Reference
- {doc}`api-life-cycle` — API Life Cycle
- [Android API reference](https://pytorch.org/executorch/main/javadoc/): Java/Kotlin API documentation
- {doc}`extension-module` — Extension Module
- {doc}`extension-tensor` — Extension Tensor
- {doc}`running-a-model-cpp-tutorial` — Detailed C++ Runtime APIs Tutorial

```{toctree}
:hidden:
:maxdepth: 1
:caption: API Reference

export-to-executorch-api-reference
executorch-runtime-api-reference
runtime-python-api-reference
api-life-cycle
extension-module
extension-tensor
running-a-model-cpp-tutorial
