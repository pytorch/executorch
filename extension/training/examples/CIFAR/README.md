## Objective:

This project enables the users to train PyTorch models on server infrastructure and get the required files to subsequently fine-tune these models on their edge devices.

### Key Objectives

1. **Server-Side Training**: Users can leverage server computational resources to perform initial model training using PyTorch, leveraging a more powerful hardware setup for the computationally intensive training phase.
2. **Edge Device Fine-Tuning**: Pre-trained models are lowered and deployed on mobile devices through ExecuTorch where they undergo fine-tuning. This allows us to create a more personalized model while maintaining data privacy and allowing the users to be in control of their data.
3. **Performance Benchmarking**: We will track comprehensive performance metrics for fine-tuning operations across various environments to see if the performance is consistent across various runtimes.

### ExecuTorch Installation

To install ExecuTorch in a python environment we can use the following commands in a new terminal:

```bash
$ git clone https://github.com/pytorch/executorch.git
$ cd executorch
$ uv venv --seed --prompt et --python 3.10
$ source .venv/bin/activate
$ which python
$ git fetch origin
$ git submodule sync --recursive
$ git submodule update --init --recursive
$ ./install_requirements.sh
$ ./install_executorch.sh
```

### Prerequisites

We need the following packages for this example:
1. torch
2. torchvision
3. executorch
4. tqdm

Make sure these are installed in the `et` venv created in the previous steps. Torchvision and Toech are installed by the installation script of ExecuTorch. Tqdm might have to be installed manually.

### Dataset

For simplicity and replicatability we will be using the [CIFAR 10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 test images.

### PyTorch Model Architecture

Here is a simple CNN Model that we have used for the classification of the CIFAR 10 dataset:

```python
class CIFAR10Model(torch.nn.Module):

	def __init__(self, num_classes=10) -> None:
		super(CIFAR10Model, self).__init__()
		self.features = torch.nn.Sequential(
			torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(128 * 4 * 4, 512),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(512, num_classes),
		)

	def forward(self, x) -> torch.Tensor:
		"""
		The forward function takes the input image and applies the convolutional
		layers and the fully connected layers to extract the features and
		classify the image respectively.
		"""
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
```

While this implementation demonstrates a relatively simple convolutional neural network, it incorporates fundamental building blocks essential for developing sophisticated computer vision models: convolutional layers for feature extraction and max-pooling layers for spatial dimension reduction and computational efficiency.

#### Core Components

1. **Convolutional Layers**: Extract hierarchical features from input images, learning patterns ranging from edges and textures in the images for a more complex and comprehensive object representations.
2. **Max-Pooling Layers**: Reduce spatial dimensions while preserving the most important features, improving computational efficiency and providing translation invariance.

### Exporting the PyTorch model to ExecuTorch runtime

To enable efficient on-device execution and fine-tuning, the trained PyTorch model must be converted to the ExecuTorch format. This conversion process involves several key steps that optimize the model for mobile deployment while preserving its ability to be fine-tuned on edge devices.

#### Wrapping the model with the loss function before export

```python
class ModelWithLoss(torch.nn.Module):

	"""
	NOTE: A wrapper class that combines a model and the loss function into a
	single module. Used for capturing the entire computational graph, i.e.
	forward pass and the loss calculation, to be captured during export. Our
	objective is to enable on-device training, so the loss calculation should
	also be included in the exported graph.
	"""

	def __init__(self, model, criterion):
		super().__init__()
		self.model = model
		self.criterion = criterion

	def forward(self, x, target):
		# Forward pass through the model
		output = self.model(x)
		# Calculate loss
		loss = self.criterion(output, target)
		# Return loss and predicted class
		return loss, output.detach().argmax(dim=1)
```

#### Conversion of PyTorch model to ExecuTorch

1. **Graph Capture**: The PyTorch model's computational graph is captured and serialized, creating a portable representation that can be executed across different hardware platforms without requiring the full PyTorch runtime.
	1. The exported format can run consistently across different mobile operating systems and hardware configurations.
2. **Runtime Optimization**: The model is optimized for the ExecuTorch runtime environment, which is specifically designed for resource-constrained edge devices. This includes memory layout optimizations and operator fusion where applicable.
	1. ExecuTorch models have significantly lower memory requirements compared to full PyTorch models.
3. **Fine-Tuning Compatibility**: The exported model retains the necessary metadata and structure to support gradient computation and parameter updates, enabling on-device fine-tuning capabilities.
	1. Optimized execution paths provide improved inference performance on mobile hardware.
	2. Traditionally the models are exported as `.pte` files which are immutable. Therefore, we need the `.ptd` files decoupled from `.pte` to perform fine-tuning and save the updated weights and biases for future use.
	3. Unlike traditional inference-only exports, we will set the flags during the model export to preserve the ability to perform gradient-based updates for fine-tuning.
##### Tracing the model:

The `strict=True` flag in the `export()`method controls the tracing method used during model export. If we set `strict=True`:
*   Export method uses TorchDynamo for tracing
*   It ensures complete soundness of the resulting graph by validating all implicit assumptions
*   It provides stronger guarantees about the correctness of the exported model
*   **Caveats:** TorchDynamo has limited Python feature coverage, so you may encounter more errors during export
##### Capturing the forward and backward graphs:

`_export_forward_backward()` transforms a forward-only exported PyTorch model into a **joint forward-backward graph** that includes both the forward pass and the automatically generated backward pass (gradients) needed for training.
We get an `ExportedProgram` containing only the forward computation graph as the output of the `export()`method.
Steps carried out by this method:

1. Apply core ATen decompositions to break down complex operations into simpler, more fundamental operations that are easier to handle during training.
2. Automatically generates the backward pass (gradient computation) for the forward graph, creating a joint graph that can compute both:
	*   Forward pass: Input → Output
	*   Backward pass: Loss gradients → Parameter gradients
3. **Graph Optimization**:
	*   Removes unnecessary `detach` operations that would break gradient flow. (During model export, sometimes unnecessary detach operations get inserted that would prevent gradients from flowing backward through the model. Removing these ensures the training graph remains connected.)
	*   Eliminates dead code to optimize the graph. (Dead code refers to computational nodes in the graph that are never used by any output and don't contribute to the final result.)
	*   Preserves the graph structure needed for gradient computation.

##### Transform model from **ATen dialect** to **Edge dialect**
`to_edge()`converts exported PyTorch programs from ATen (A Tensor Library) dialect to Edge dialect, which is optimized for edge device deployment.

`EdgeCompileConfig(_check_ir_validity=False)`skips intermediate representation (IR) validity checks during transformation and permits operations that might not pass strict validation.

### Fine-tuning the ExecuTorch model

The fine-tuning process involves updating the model's weights and biases based on new training data, typically collected from the edge devices. The support for `PTE` files is baked into the ExecuTorch runtime, which enables the model to be fine-tuned on the edge devices. However, at the time of writing, the support for training with `PTD` files is not yet available in the ExecuTorch Python runtime. Therefore, we export these files to be used in our `C++` and `Java` runtimes.

### Command Line Arguments

The training script supports various command line arguments to customize the training process. Here is a comprehensive list of all available flags:

#### Data Configuration
- `--data-dir` (str, default: `./data`)
  - Directory to download and store CIFAR-10 dataset
  - Example: `--data-dir /path/to/data`

- `--batch-size` (int, default: `4`)
  - Batch size for data loaders during training and validation
  - Example: `--batch-size 32`

- `--use-balanced-dataset` (flag, default: `True`)
  - Use balanced dataset instead of full CIFAR-10
  - When enabled, creates a subset with equal representation from each class
  - Example: `--use-balanced-dataset` (to enable) or omit flag to use full dataset

- `--images-per-class` (int, default: `100`)
  - Number of images per class for balanced dataset
  - Only applies when `--use-balanced-dataset` is enabled
  - Example: `--images-per-class 200`

#### Model Paths
- `--model-path` (str, default: `cifar10_model.pth`)
  - Path to save/load the PyTorch model
  - Example: `--model-path models/my_cifar_model.pth`

- `--pte-model-path` (str, default: `cifar10_model.pte`)
  - Path to save the PTE (PyTorch ExecuTorch) model file
  - Example: `--pte-model-path models/cifar_model.pte`

- `--split-pte-model-path` (str, default: `split_cifar10_model.pte`)
  - Path to save the split PTE model (model architecture without weights)
  - Used in conjunction with PTD files for external weight storage
  - Example: `--split-pte-model-path models/split_model.pte`

- `--ptd-model-dir` (str, default: `.`)
  - Directory path to save PTD (PyTorch Tensor Data) files
  - Contains external weights and constants separate from the PTE file
  - Example: `--ptd-model-dir ./model_data`

#### Training History and Logging
- `--save-pt-json` (str, default: `cifar10_pt_model_finetuned_history.json`)
  - Path to save PyTorch model training history as JSON
  - Contains metrics like loss, accuracy, and timing information
  - Example: `--save-pt-json results/pytorch_history.json`

- `--save-et-json` (str, default: `cifar10_et_pte_only_model_finetuned_history.json`)
  - Path to save ExecuTorch model fine-tuning history as JSON
  - Contains metrics from the ExecuTorch fine-tuning process
  - Example: `--save-et-json results/executorch_history.json`

#### Training Hyperparameters
- `--epochs` (int, default: `1`)
  - Number of epochs for initial PyTorch model training
  - Example: `--epochs 5`

- `--fine-tune-epochs` (int, default: `10`)
  - Number of epochs for fine-tuning the ExecuTorch model
  - Example: `--fine-tune-epochs 20`

- `--learning-rate` (float, default: `0.001`)
  - Learning rate for both PyTorch training and ExecuTorch fine-tuning
  - Example: `--learning-rate 0.01`
