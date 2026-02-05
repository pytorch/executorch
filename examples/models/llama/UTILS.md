# Utility tools for Llama enablement

## Stories110M model

If you want to deploy and run a smaller model for educational purposes, you can try stories110M model. It has the same architecture as Llama, but just smaller. It can be also used for fast iteration and verification during development.

### Export:

From `executorch` root:

1. Download `stories110M.pt` and `tokenizer.model` from Github.
    ```
    wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
    wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
    ```
2. Create params file.
    ```
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
    ```
3. Export model and generate `.pte` file.
    ```
    python -m extension.llm.export.export_llm base.checkpoint=stories110M.pt base.params=params.json backend.xnnpack.enabled=True model.use_kv_cache=True
    ```

## Smaller model delegated to other backends

Currently we supported lowering the stories model to other backends, including, CoreML, MPS and QNN. Please refer to the instruction
for each backend ([CoreML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html), [MPS](https://docs.pytorch.org/executorch/main/backends/mps/mps-overview.html), [QNN](https://docs.pytorch.org/executorch/main/backends-qualcomm.html)) before trying to lower them. After the backend library is installed, the script to export a lowered model is

- Lower to CoreML: `python -m extension.llm.export.export_llm model.use_kv_cache=True model.enable_dynamic_shape=False backend.coreml.enabled=True base.checkpoint=stories110M.pt base.params=params.json`
- MPS: `python -m extension.llm.export.export_llm model.use_kv_cache=True model.enable_dynamic_shape=False backend.mps.enabled=True base.checkpoint=stories110M.pt base.params=params.json`
- QNN: `python -m extension.llm.export.export_llm model.use_kv_cache=True model.enable_dynamic_shape=False backend.qnn.enabled=True base.checkpoint=stories110M.pt base.params=params.json`

The iOS LLAMA app supports the CoreML and MPS model and the Android LLAMA app supports the QNN model. On Android, it also allow to cross compiler the llama runner binary, push to the device and run.

For CoreML, there are 2 additional optional arguments:
* `backend.coreml.ios`: Specify the minimum iOS version to deploy (and turn on available optimizations). E.g. `backend.coreml.ios=18` will turn on [in-place KV cache](https://developer.apple.com/documentation/coreml/mlstate?language=objc) and [fused scaled dot product attention kernel](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS18.transformers.scaled_dot_product_attention) (the resulting model will then need at least iOS 18 to run, though)
* `backend.coreml.quantize`: Use [quantization tailored for CoreML](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html). E.g. `backend.coreml.quantize="b4w"` will perform per-block 4-bit weight-only quantization in a way tailored for CoreML

To deploy the large 8B model on the above backends, [please visit this section](non_cpu_backends.md).

## Download models from Hugging Face and convert from safetensor format to state dict

You can also download above models from [Hugging Face](https://huggingface.co/). Since ExecuTorch starts from a PyTorch model, a script like below can be used to convert the Hugging Face safetensors format to PyTorch's state dict. It leverages the utils provided by [TorchTune](https://github.com/pytorch/torchtune).


```Python
from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.models import convert_weights
import torch

# Convert from safetensors to TorchTune. Suppose the model has been downloaded from Hugging Face
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir='/home/.cache/huggingface/hub/models/snapshots/hash-number',
    checkpoint_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'],
    output_dir='/the/destination/dir' ,
    model_type='LLAMA3' # or other types that TorchTune supports
)

print("loading checkpoint")
sd = checkpointer.load_checkpoint()

# Convert from TorchTune to Meta (PyTorch native)
sd = convert_weights.tune_to_meta(sd['model'])

print("saving checkpoint")
torch.save(sd, "/the/destination/dir/checkpoint.pth")
```

## Finetuning

If you want to finetune your model based on a specific dataset, PyTorch provides [TorchTune](https://github.com/pytorch/torchtune) - a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

Once you have [TorchTune installed](https://github.com/pytorch/torchtune?tab=readme-ov-file#get-started) you can finetune Llama2 7B model using LoRA on a single GPU, using the following command. This will produce a checkpoint where the LoRA weights are merged with the base model and so the output checkpoint will be in the same format as the original Llama2 model.

```
tune run lora_finetune_single_device \
--config llama2/7B_lora_single_device \
checkpointer.checkpoint_dir=<path_to_checkpoint_folder>  \
tokenizer.path=<path_to_checkpoint_folder>/tokenizer.model
```

To run full finetuning with Llama2 7B on a single device, you can use the following command.

```
tune run full_finetune_single_device \
--config llama2/7B_full_single_device \
checkpointer.checkpoint_dir=<path_to_checkpoint_folder> \
tokenizer.path=<path_to_checkpoint_folder>/tokenizer.model
```
