# Vulkan Delegate Export Examples

This directory contains scripts for exporting models with the Vulkan delegate in ExecuTorch. Vulkan delegation allows you to run your models on devices with Vulkan-capable GPUs, potentially providing significant performance improvements over CPU execution.

## Scripts

- `export.py`: Basic export script for models to use with Vulkan delegate
- `aot_compiler.py`: Advanced export script with quantization support

## Usage

### Basic Export

```bash
python -m executorch.examples.vulkan.export -m <model_name> -o <output_dir>
```

### Export with Quantization (Experimental)

```bash
python -m executorch.examples.vulkan.aot_compiler -m <model_name> -q -o <output_dir>
```

### Dynamic Shape Support

```bash
python -m executorch.examples.vulkan.export -m <model_name> -d -o <output_dir>
```

### Additional Options

- `-s/--strict`: Export with strict mode (default: True)
- `-a/--segment_alignment`: Specify segment alignment in hex (default: 0x1000)
- `-e/--external_constants`: Save constants in external .ptd file (default: False)
- `-r/--etrecord`: Generate and save an ETRecord to the given file location

## Examples

```bash
# Export MobileNetV2 with Vulkan delegate
python -m executorch.examples.vulkan.export -m mobilenet_v2 -o ./exported_models

# Export MobileNetV3 with quantization
python -m executorch.examples.vulkan.aot_compiler -m mobilenet_v3 -q -o ./exported_models

# Export with dynamic shapes
python -m executorch.examples.vulkan.export -m mobilenet_v2 -d -o ./exported_models

# Export with ETRecord for debugging
python -m executorch.examples.vulkan.export -m mobilenet_v2 -r ./records/mobilenet_record.etrecord -o ./exported_models
```

## Supported Operations

The Vulkan delegate supports various operations including:

- Basic arithmetic (add, subtract, multiply, divide)
- Activations (ReLU, Sigmoid, Tanh, etc.)
- Convolutions (Conv1d, Conv2d, ConvTranspose2d)
- Pooling operations (MaxPool2d, AvgPool2d)
- Linear/Fully connected layers
- BatchNorm, GroupNorm
- Various tensor operations (cat, reshape, permute, etc.)

For a complete list of supported operations, refer to the Vulkan delegate implementation in the ExecuTorch codebase.

## Debugging and Optimization

If you encounter issues with Vulkan delegation:

1. Use `-r/--etrecord` to generate an ETRecord for debugging
2. Check if your operations are supported by the Vulkan delegate
3. Ensure your Vulkan drivers are up to date
4. Try using the export script with `--strict False` if strict mode causes issues

## Requirements

- Vulkan runtime libraries (libvulkan.so.1)
- A Vulkan-capable GPU with appropriate drivers
- PyTorch with Vulkan support
