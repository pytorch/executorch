# Summary
This example demonstrates how to Export a Llama 2 model in ExecuTorch.
For Llama2, please refer to [the llama's github page](https://github.com/facebookresearch/llama) for details.
Pretrained parameters are not included in this repo. Users are suggested to download them through [the llama's download page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

# Notes
1. This example is to show the feasibility of exporting a Llama2 model in ExecuTorch. There is no guarantee for performance.
2. It's targeted to a reasonable size for edge devices. Depending on the model size, the memory usage of exporting the model can be high. TODO: improve memory usage in EXIR emitter.
3. The provided check point, demo_rand_params.pth is a dummy checkpoint with random parameters. It does not provide meaningful results. It's only for the purpose of demonstration and fast iterations.

# Limitations
This example tries to reuse the Python code, with modifications to make it compatible with current ExecuTorch:
1. Since ExecuTorch does not support complex Tensor data type, use the customized functions to have rotary embedding with real numbers. TODO: support complex Tensor data type in ExecuTorch.
2. No KV cache. The current cache implementation in the original Llama2 repo is not supported by ExecuTorch, because ExecuTorch runtime assumes model data attributes being static. TODO: add support of mutable buffers in ExecuTorch.
3. No CUDA. ExecuTorch is focused on Edge use cases where CUDA is not available on most of the edge devices.
4. No dependencies on fairscale. The ColumnParallelLinear, ParallelEmbedding and training are not needed and supported in ExecuTorch.


# Instructions:
1. Follow the [tutorial](https://github.com/pytorch/executorch/blob/main/docs/website/docs/tutorials/00_setting_up_executorch.md) to set up ExecuTorch
2. `cd examples/third-party/llama`
3. `pip install -e .`
4. Go back to `executorch` root, run `python3 -m examples.export.export_example --model_name="llama2"`. The exported program, llama2.pte would be saved in current directory
5. Use the `executor_runner` (build instruction in step 1) to load and run llama2.pte, `executor_runner --model_path llama2.pte`
