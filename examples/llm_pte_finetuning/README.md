# ExecuTorch Finetuning example

In this tutorial, we show how to fine-tune an LLM using executorch.

## Pre-requisites

You will need to have a model's checkpoint, in the Hugging Face format. For example:

```
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
```

You will need to install [torchtune](https://github.com/pytorch/torchtune) following [its installation instructions](https://github.com/pytorch/torchtune?tab=readme-ov-file#installation).

## Config Files

As mentioned in the previous section, we internally use `torchtune` APIs, and thus, we use config files that follow `torchtune`'s structure. Typically, in the following sections we go through a working example which can be found in the `phi3_config.yaml` config file.

### Tokenizer

We need to define the tokenizer. Let's suppose we would like to use [PHI3 Mini Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model from Microsoft. We need to define the tokenizer component:

```
tokenizer:
  _component_: torchtune.models.phi3.phi3_mini_tokenizer
  path: /tmp/Phi-3-mini-4k-instruct/tokenizer.model
  max_seq_len: 1024
```

This will load the tokenizer, and set the max sequence length to 1024. The class that will be instantiated will be [`Phi3MiniTokenizer`](https://github.com/pytorch/torchtune/blob/ee343e61804f9942b2bd48243552bf17b5d0d553/torchtune/models/phi3/_tokenizer.py#L30).

### Dataset

In this example we use the [Alpaca-Cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned). We need to define the following parameters:

```
dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
seed: null
shuffle: True
batch_size: 1
```

Torchtune supports datasets using huggingface dataloaders, so custom datasets could also be defined. For examples on defining your own datasets, review the [torchtune docs](https://pytorch.org/torchtune/stable/tutorials/datasets.html#hugging-face-datasets).

### Loss

For the loss function, we use PyTorch losses. In this example we use the `CrossEntropyLoss`:

```
loss:
  _component_: torch.nn.CrossEntropyLoss
```

### Model

Model parameters can be set, in this example we replicate the configuration for phi3 mini instruct benchmarks:

```
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

```
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

```
device: cpu
dtype: fp32
```

## Running the example

### Step 1: Generate the ExecuTorch PTE (checkpoint)

The `model_exporter.py` exports the LLM checkpoint into an ExecuTorch checkpoint (.pte). It has two parameters:

* `cfg`: Configuration file
* `output_file`: The `.pte` output path

```
python model_exporter.py --cfg=phi3_config.yaml --output_file=phi3_mini_lora.pte
```

### Step 2: Run the fine-tuning job

To run the fine-tuning job:

```
python runner.py --cfg=phi3_config.yaml --model_file=phi3_mini_lora.pte
```

You need to use **the same** config file from the previous step. The `model_file` arg is the `.pte` model from the previous step.

Example output:

```
Evaluating the model before training...
100%|██████████████████████████████████████████████████████████████████████████████████████| 3/3 [31:23<00:00, 627.98s/it]
Eval loss:  tensor(2.3778)
100%|██████████████████████████████████████████████████████████████████████████████████████| 5/5 [52:29<00:00, 629.84s/it]
Losses:  [2.7152762413024902, 0.7890686988830566, 2.249271869659424, 1.4777560234069824, 0.8378427624702454]
100%|██████████████████████████████████████████████████████████████████████████████████████| 3/3 [30:35<00:00, 611.90s/it]
Eval loss:  tensor(0.8464)
```
