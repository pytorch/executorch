# Exporting LLMs

Instead of needing to manually write code to call torch.export(), use ExecuTorch's assortment of lowering APIs, or even interact with TorchAO quantize_ APIs for quantization, we have provided an out of box experience which performantly exports a selection of supported models to ExecuTorch.

## Prerequisites

The LLM export functionality requires the `pytorch_tokenizers` package. If you encounter a `ModuleNotFoundError: No module named 'pytorch_tokenizers'` error, install it from the ExecuTorch source code:

```bash
pip install -e ./extension/llm/tokenizers/
```

## Supported Models

As of this doc, the list of supported LLMs include the following:
- Llama 2/3/3.1/3.2
- Qwen 2.5/3
- Phi 3.5/4-mini
- SmolLM2

The up-to-date list of supported LLMs can be found in the code [here](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L32).

**Note:** For a Hugging Face architecture not covered here, use the
[experimental Transformers ExecuTorch exporter](https://huggingface.co/docs/transformers/en/exporters)
for broad programmatic graph export, or [Optimum ExecuTorch](export-llm-optimum.md)
for its tested task-level recipes, quantization, and higher-level APIs. Architectures
that need model-specific changes can start with [Exporting custom LLMs](export-custom-llm.md).

## The export_llm API
`export_llm` is ExecuTorch's high-level export API for LLMs. In this tutorial, we will focus on exporting Llama 3.2 1B using this API. `export_llm`'s arguments are specified either through CLI args or through a yaml configuration whose fields are defined in [`LlmConfig`](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py). To call `export_llm`:

```
python -m executorch.extension.llm.export.export_llm \
  --config <path-to-config-yaml> \
  +base.<additional-CLI-overrides>
```

## Basic export

To perform a basic export of Llama3.2, we will first need to download the checkpoint file (`consolidated.00.pth`) and params file (`params.json`). You can find these from the [Llama website](https://www.llama.com/llama-downloads/) or [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main/original).

Then, we specify the `model_class`, `checkpoint` (path to checkpoint file), and `params` (path to params file) as arguments. Additionally, later when we run the exported .pte with our runner APIs, the runner will need to know about the bos and eos ids for this model to know when to terminate. These are exposed through bos and eos getter methods in the .pte, which we can add by specifying bos and eos ids in a `metadata` argument. The values for these tokens can usually be found in the model's `tokenizer_config.json` on Hugging Face.

```
# path/to/config.yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```

We only require manually specifying a checkpoint path for the Llama model family, since it is our most optimized model and we have more advanced optimizations such as [SpinQuant](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#spinquant) that require custom checkpoints.

For the other supported LLMs, the checkpoint will be downloaded from Hugging Face automatically, and the param files can be found in their respective directories under `executorch/examples/models`, for instance `executorch/examples/models/qwen3/config/0_6b_config.json`.

## Export settings
[ExportConfig](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py) contains settings for the exported `.pte`, such as `max_seq_length` (max length of the prompt) and `max_context_length` (max length of the model's memory/cache).

## Adding optimizations
`export_llm` performs a variety of optimizations to the model before export, during export, and during lowering. Quantization and delegation to accelerator backends are the main ones and will be covered in the next two sections. All other optimizations can be found under [`ModelConfig`](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L120). We will go ahead and add a few optimizations.

```
# path/to/config.yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```

`use_kv_cache` and `use_sdpa_with_kv_cache` are recommended to export any LLM, while other options are useful situationally. For example:
- `use_shared_embedding` can help for models with tied input/output embedding layers, given that you quantize using TorchAO low bit ops (`quantization.qmode: torchao:8da(\\d+)w` or `quantization.qmode: torchao:fpa(\d+)w`), see more [here](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L307).
- `use_attention_sink` to extend generation by removing from the beginning of the KV cache when the max context length is reached.
- `quantize_kv_cache` quantizes the KV cache in int8.
- `local_global_attention` implements [Local-Global Attention](https://arxiv.org/abs/2411.09604), making specific attention layers use a much smaller localized sliding window KV cache.

## Quantization
Quantization options are defined by [`QuantizationConfig`](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L283). ExecuTorch does quantization in two ways:
1. TorchAO [`quantize_`](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html) API
2. [pt2e quantization](https://docs.pytorch.org/ao/main/pt2e_quantization/pt2e_quant_ptq.html)

### TorchAO (XNNPACK)
TorchAO quantizes at the source code level, swapping out Linear modules for QuantizedLinear modules.
**To quantize on XNNPACK backend, this is the quantization path to follow.**
The quantization modes are defined [here](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L306).

Common ones to use are:
- `8da4w`: short for int8 dynamic activation + int4 weight quantization.
- `int8`: int8 weight-only quantization.

Group size is specified with:
- `group_size`: 8, 32, 64, etc.

For Arm CPUs, there are also [low-bit kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for int8 dynamic activation + int[1-8] weight quantization. Note that this should not be used alongside XNNPACK, and experimentally we have found that the performance could sometimes even be better for the equivalent `8da4w`. To use these, specify `qmode` to either:
- `torchao:8da(\d+)w`: int8 dynamic activation + int[1-8] weights, for example `torchao:8da5w`
- `torchao:fpa(\d+)w`: int[1-8] weight only, for example `torchao:fpa4w`

To quantize embeddings, specify either `embedding_quantize: <bitwidth>,<groupsize>` (`bitwidth` here must be 2, 4, or 8), or for low-bit kernels use `embedding_quantize: torchao:<bitwidth>,<groupsize>` (`bitwidth` can be from 1-8).

```
# path/to/config.yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
quantization:
  embedding_quantize: 4,32
  qmode: 8da4w

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```

### pt2e (QNN, CoreML, and Vulkan)
pt2e quantizes at the post-export graph level, swapping nodes and injecting quant/dequant nodes.
**To quantize on non-CPU backends (QNN, CoreML, Vulkan), this is the quantization path to follow.**
Read more about pt2e [here](https://docs.pytorch.org/ao/main/pt2e_quantization/pt2e_quant_ptq.html), and how ExecuTorch uses pt2e [here](https://github.com/pytorch/executorch/blob/main/docs/source/quantization-overview.md).

*CoreML and Vulkan support for export_llm is currently experimental and limited. To read more about QNN export, please read [Running on Android (Qualcomm)](build-run-llama3-qualcomm-ai-engine-direct-backend.md).*


## Backend support
Backend options are defined by [`BackendConfig`](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L434). Each backend has their own backend configuration options. Here is an example of lowering the LLM to XNNPACK for CPU acceleration:

```
# path/to/config.yaml
base:
  model_class: llama3_2
  checkpoint: path/to/consolidated.00.pth
  params: path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
quantization:
  embedding_quantize: 4,32
  qmode: 8da4w
backend:
  xnnpack:
    enabled: True
    extended_ops: True  # Expand the selection of ops delegated to XNNPACK.

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```


## Profiling and Debugging
To see which ops got delegated to the backend and which didn't, specify `verbose: True`:

```
# path/to/config.yaml
...
debug:
  verbose: True
...

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```

In the logs, there will be a table of all ops in the graph, and which ones were and were not delegated.

Here is an example:
<details>
<summary>Click to see delegation details</summary>

Total delegated subgraphs: 368 <br/>
Number of delegated nodes: 2588 <br/>
Number of non-delegated nodes: 2513 <br/>


|    |  op_type                                 |  # in_delegated_graphs  |  # in_non_delegated_graphs  |
|----|---------------------------------|------- |-----|
|  0  |  _assert_scalar  |  0  |  167  |
|  1  |  _local_scalar_dense  |  0  |  123  |
|  2  |  add  |  0  |  31  |
|  3  |  aten__to_copy_default  |  0  |  44  |
|  4  |  aten_add_tensor  |  418  |  44  |
|  5  |  aten_alias_copy_default  |  0  |  52  |
|      |  ...  |    |    |
|  15  |  aten_linear_default  |  183  |  0  |
|  18  |  aten_mul_tensor  |  445  |  0  |
|  20  |  aten_pow_tensor_scalar  |  157  |  0  |
|  22  |  aten_rsqrt_default  |  157  |  0  |
|  27  |  aten_view_copy_default  |  0  |  126  |
|  31  |  getitem  |  366  |  628  |
|      |  ...  |    |    |
|  41  |  torchao_quantize_affine_default  |  183  |  0  |
|  42  |  Total  |  2588  |  2513  |

</details>
<br/>

For further performance analysis, use [ExecuTorch's Developer Tools](../devtools-overview.md) to trace individual operator performance back to source code, inspect memory planning, and debug intermediate activations. To generate an ETRecord that links the `.pte` program back to source code, use:

```
# path/to/config.yaml
...
debug:
  generate_etrecord: True
...

# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml
```

Other debug and profiling options can be found in [DebugConfig](https://github.com/pytorch/executorch/blob/main/extension/llm/export/config/llm_config.py#L228).

For example:

- `profile_memory`: Generates an activation memory profile in Chrome trace format. Use it to visualize tensor lifetimes, overlap, provenance, and their effect on runtime memory use. See [Inspecting Memory Planning](../memory-planning-inspection.md) for details.
- `profile_path`: Generates an HTML time profile of `export_llm` components such as `torch.export`, quantization, `to_edge`, and `to_backend` delegation. The flame graph or icicle view helps ExecuTorch developers identify expensive export stages. For background, see [Icicle graph anatomy](https://www.parca.dev/docs/icicle-graph-anatomy/).
