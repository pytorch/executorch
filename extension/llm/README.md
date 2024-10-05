This subtree contains libraries and utils of running generative AI, including Large Language Models (LLM) using ExecuTorch.
Below is a list of sub folders.
## export
Model preparation codes are in _export_ folder. The main entry point is the _LLMEdgeManager_ class. It hosts a _torch.nn.Module_, with a list of methods that can be used to prepare the LLM model for ExecuTorch runtime.
Note that ExecuTorch supports two [quantization APIs](https://pytorch.org/docs/stable/quantization.html#quantization-api-summary): eager mode quantization (aka source transform based quantization) and PyTorch 2 Export based quantization (aka pt2e quantization).

Commonly used methods in this class include:
- _set_output_dir_: where users want to save the exported .pte file.
- _to_dtype_: override the data type of the module.
- _source_transform_: execute a series of source transform passes. Some transform passes include
  - weight only quantization, which can be done at source (eager mode) level.
  - replace some torch operators to a custom operator. For example, _replace_sdpa_with_custom_op_.
- _capture_pre_autograd_graph_: get a graph that is ready for pt2 graph-based quantization.
- _pt2e_quantize_ with passed in quantizers.
  - util functions in _quantizer_lib.py_ can help to get different quantizers based on the needs.
- _export_to_edge_: export to edge dialect
- _to_backend_: lower the graph to an acceleration backend.
- _to_executorch_: get the executorch graph with optional optimization passes.
- _save_to_pte_: finally, the lowered and optimized graph can be saved into a .pte file for the runtime.

Some usage of LLMEdgeManager can be found in executorch/examples/models/llama2, and executorch/examples/models/llava.

When the .pte file is exported and saved, we can load and run it in a runner (see below).

## tokenizer
Currently, we support two types of tokenizers: sentencepiece and Tiktoken.
- In Python:
  - _utils.py_: get the tokenizer from a model file path, based on the file format.
  - _tokenizer.py_: rewrite a sentencepiece tokenizer model to a serialization format that the runtime can load.
- In C++:
  - _tokenizer.h_: a simple tokenizer interface. Actual tokenizer classes can be implemented based on this. In this folder, we provide two tokenizer implementations:
    - _bpe_tokenizer_. Note: we need the rewritten version of tokenizer artifact (refer to _tokenizer.py_ above), for bpe tokenizer to work.
    - _tiktoken_. For llama3 and llama3.1.

## sampler
A sampler class in C++ to sample the logistics given some hyperparameters.

## custom_ops
Contains custom op, such as:
- custom sdpa: implements CPU flash attention and avoids copies by taking the kv cache as one of its arguments.
  - _sdpa_with_kv_cache.py_, _op_sdpa_aot.cpp_: custom op definition in PyTorch with C++ registration.
  - _op_sdpa.cpp_: the optimized operator implementation and registration of _sdpa_with_kv_cache.out_.

## runner
It hosts the libary components used in a C++ llm runner. Currently, it hosts _stats.h_ on runtime status like token numbers and latency.

With the components above, an actual runner can be built for a model or a series of models. An example is in //executorch/examples/models/llama2/runner, where a C++ runner code is built to run Llama 2, 3, 3.1 and other models using the same architecture.

Usages can also be found in the [torchchat repo](https://github.com/pytorch/torchchat/tree/main/runner).
