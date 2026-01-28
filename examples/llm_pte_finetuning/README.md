# ExecuTorch Finetuning example

In this tutorial, we show how to fine-tune an LLM using executorch.

## Pre-requisites

You will need to have a model's checkpoint, in the Hugging Face format. For example:

```console
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
```

You will need to install [torchtune](https://github.com/pytorch/torchtune) following [its installation instructions](https://github.com/pytorch/torchtune?tab=readme-ov-file#installation).

You might run into an issue with the `triton` package when installing `torchtune`. You can build `triton` locally following the [instructions in their repo](https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source).

## Config Files

The directory structure of the `llm_pte_finetuning` is:

```console
examples/llm_pte_finetuning
├── README.md
├── TARGETS
├── __init__.py
│   ├── model_loading_lib.cpython-312.pyc
│   └── training_lib.cpython-312.pyc
├── model_exporter.py
├── model_loading_lib.py
├── phi3_alpaca_code_config.yaml
├── phi3_config.yaml
├── qwen_05b_config.yaml
├── runner.py
└── training_lib.py
```

We already provide configs out of the box. The following sections explain how you can setup the config for your own model or dataset.

As mentioned in the previous section, we internally use `torchtune` APIs, and thus, we use config files that follow `torchtune`'s structure. Typically, in the following sections we go through a working example which can be found in the `phi3_config.yaml` config file.

### Tokenizer

We need to define the tokenizer. Let's suppose we would like to use [PHI3 Mini Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model from Microsoft. We need to define the tokenizer component:

```yaml
tokenizer:
  _component_: torchtune.models.phi3.phi3_mini_tokenizer
  path: /tmp/Phi-3-mini-4k-instruct/tokenizer.model
  max_seq_len: 1024
```

This will load the tokenizer, and set the max sequence length to 1024. The class that will be instantiated will be [`Phi3MiniTokenizer`](https://github.com/pytorch/torchtune/blob/ee343e61804f9942b2bd48243552bf17b5d0d553/torchtune/models/phi3/_tokenizer.py#L30).

### Dataset

In this example we use the [Alpaca-Cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned). We need to define the following parameters:

```yaml
dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
seed: null
shuffle: True
batch_size: 1
```

Torchtune supports datasets using huggingface dataloaders, so custom datasets could also be defined. For examples on defining your own datasets, review the [torchtune docs](https://meta-pytorch.org/torchtune/main/basics/text_completion_datasets.html#loading-text-completion-datasets-from-hugging-face).

### Loss

For the loss function, we use PyTorch losses. In this example we use the `CrossEntropyLoss`:

```yaml
loss:
  _component_: torch.nn.CrossEntropyLoss
```

### Model

Model parameters can be set, in this example we replicate the configuration for phi3 mini instruct benchmarks:

```yaml
model:
  _component_: torchtune.models.phi3.lora_phi3_mini
  lora_attn_modules: ['q_proj', 'v_proj']
  apply_lora_to_mlp: False
  apply_lora_to_output: False
  lora_rank: 8
  lora_alpha: 16
```

### Checkpointer

Depending on how your model is defined, you will need to instantiate different components. In these examples we use checkpoints from HF (hugging face format), and thus we will need to instantiate a `FullModelHFCheckpointer` object. We need to pass the checkpoint directory, the files with the tensors, the output directory for training and the model type:

```yaml
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: /tmp/Phi-3-mini-4k-instruct
  checkpoint_files: [
    model-00001-of-00002.safetensors,
    model-00002-of-00002.safetensors
  ]
  recipe_checkpoint: null
  output_dir: /tmp/Phi-3-mini-4k-instruct/
  model_type: PHI3_MINI
```

### Device

Torchtune supports `cuda` and `bf16` tensors. However, for ExecuTorch training we only support `cpu` and `fp32`:

```yaml
device: cpu
dtype: fp32
```

## Running the example

### Step 1: Generate the ExecuTorch PTE (checkpoint)

The `model_exporter.py` exports the LLM checkpoint into an ExecuTorch checkpoint (.pte). It has two parameters:

* `cfg`: Configuration file
* `output_file`: The `.pte` output path

```console
python model_exporter.py \
    --cfg=qwen_05b_config.yaml \
    --output_file=qwen2_0_5B.pte
```

### Step 2: Run the fine-tuning job

To run the fine-tuning job:

```console
python runner.py \
    --cfg=qwen_05b_config.yaml \
    --model_file=qwen2_0_5B.pte \
    --num_training_steps=10 \
    --num_eval_steps=5
```

You need to use **the same** config file from the previous step. The `model_file` arg is the `.pte` model from the previous step.

Example output:

```console
Evaluating the model before training
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:47<00:00,  9.45s/it]
Eval loss:  tensor(0.9441)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:30<00:00,  9.09s/it]
Losses:  [0.5646533966064453, 1.3464953899383545, 1.297974705696106, 1.2249481678009033, 0.6750457286834717, 0.7721152901649475, 1.0774847269058228, 0.7962403893470764, 0.8448256850242615, 0.8731598854064941]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:45<00:00,  9.18s/it]
Eval loss:  tensor(0.7679)
```
