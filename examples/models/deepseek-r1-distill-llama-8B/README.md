# Summary
This example demonstrates how to run [Deepseek R1 Distill Llama 8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) 3.8B model via ExecuTorch. The architecture of this distilled model is exactly the same as Llama and thus all the instructions mentioned in the [Llama README](../llama/README.md) apply as is.

# Instructions
## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_executorch.sh`

2. Run the installation step for Llama specific requirements
```
./examples/models/llama/install_requirements.sh
```

## Step 2: Prepare and run the model
1. Download the model
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir /target_dir/DeepSeek-R1-Distill-Llama-8B --local-dir-use-symlinks False
```

2. Download the [tokenizer.model](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main/original) from the Llama3.1 repo which will be needed later on when running the model using the runtime.

3. Convert the model to pth file.
```
pip install torchtune
```

Run this python code:
```
from torchtune.models import convert_weights
from torchtune.training import FullModelHFCheckpointer
import torch

# Convert from safetensors to TorchTune. Suppose the model has been downloaded from Hugging Face
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir='/target_dir/DeepSeek-R1-Distill-Llama-8B ',
    checkpoint_files=['model-00001-of-000002.safetensors', 'model-00002-of-000002.safetensors'],
    output_dir='/tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/' ,
    model_type='LLAMA3' # or other types that TorchTune supports
)

print("loading checkpoint")
sd = checkpointer.load_checkpoint()

# Convert from TorchTune to Meta (PyTorch native)
sd = convert_weights.tune_to_meta(sd['model'])

print("saving checkpoint")
torch.save(sd, "/tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/checkpoint.pth")
```

4. Download and save the [params.json](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main/original) file.

5. Generate a PTE file for use with the Llama runner.
```
python -m extension.llm.export.export_llm \
    --config examples/models/deepseek-r1-distill-llama-8B/config/deepseek-r1-distill-llama-8B
    +base.checkpoint=/tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/checkpoint.pth \
	+base.params=params.json \
	+export.output_name="DeepSeek-R1-Distill-Llama-8B.pte"
```

6. Run the model on your desktop for validation or integrate with iOS/Android apps. Instructions for these are available in the Llama [README](../llama/README.md) starting at Step 3.
