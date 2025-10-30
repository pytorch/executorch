# CIFAR 10 End-to-End Fine-Tuning Tutorial

## Objective:

This tutorial guides the users through the training process of a simple PyTorch CNN model on the server and subsequently fine-tune the model on their edge devices.

### Key Objectives

1. **Server-Side Training**: Users can leverage the computational resource of the server to perform initial model training using PyTorch.
2. **Edge Device Fine-Tuning**: Pre-trained models are lowered and deployed on mobile devices through ExecuTorch where they undergo fine-tuning.
3. **Performance Benchmarking**: To track comprehensive performance metrics for on-device fine-tuning operations, measuring factors such as training speed, memory usage, and model accuracy to evaluate ExecuTorch's effectiveness in the edge environment.

## ExecuTorch Environment Setup

For easier management of Python environments and packages, we recommended using a Python environment management tool such as `conda`, `venv`, or `uv`. In this demonstration, we will use `uv` to set up the Python environment.

To install ExecuTorch in a [`uv`](https://docs.astral.sh/uv/getting-started/installation/) Python environment use the following commands:

```bash
$ git clone https://github.com/pytorch/executorch.git --recurse-submodules
$ cd executorch
$ uv venv --seed --prompt et --python 3.10
$ source .venv/bin/activate
$ git fetch origin
$ git submodule sync --recursive
$ git submodule update --init --recursive
$ ./install_executorch.sh
```

## Data Preparation

We can download the CIFAR-10 dataset from the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it to the desired location. Alternatively, we can also use the following command to download, extract, and create a balanced dataset:

```bash
python data_utils.py --train-data-batch-path ./data/cifar-10/cifar-10-batches-py/data_batch_1 --train-output-path ./data/cifar-10/extracted_data/train_data.bin --test-data-batch-path ./data/cifar-10/cifar-10-batches-py/test_batch --test-output-path ./data/cifar-10/extracted_data/test_data.bin --train-images-per-class 100
```

## Model Export

Alternatively, if the users have a pre-trained pytorch model, they can export the standalone `pte`file using the following command:

```bash
python export.py --train-model-path cifar10_model.pth --pte-only-model-path cifar10_model.pte
```

For getting the `pte` and `ptd` files, they can use the following command:

```bash
python export.py --train-model-path cifar10_model.pth --with-ptd --pte-model-path cifar10_model.pte --ptd-model-path .
```

## Model Training and Fine-Tuning

To run the end-to-end example, the users can use the following command:

```bash
python main.py --data-dir ./data --model-path cifar10_model.pth --pte-model-path cifar10_model.pte --split-pte-model-path cifar10_model_pte_only.pte --save-pt-json cifar10_pt.json --save-et-json cifar10_et.json --ptd-model-dir . --epochs 1 --fine-tune-epochs 1
```
