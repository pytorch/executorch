#!/usr/bin/env python3
"""
Extract convolution layer parameters from PyTorch ExecuTorch .pte file

This script:
1. Loads .pte model using ExecuTorch Inspector
2. Extracts all convolution operations with their parameters
3. Outputs layer information for generate_idma_buffers.py

Usage:
    python extract_layers_from_pte.py model.pte --output layers_config.json
"""

import sys
import json
import argparse
from pathlib import Path

def extract_conv_layers_executorch(pte_file):
    """
    Extract convolution layers using ExecuTorch Inspector
    
    Returns: List of layer configurations
    [
        {
            "layer_id": 0,
            "layer_name": "conv1",
            "input_shape": [1, 3, 224, 224],    # [N, C, H, W]
            "output_shape": [1, 64, 112, 112],
            "kernel_shape": [64, 3, 7, 7],      # [out_c, in_c, kH, kW]
            "stride": [2, 2],
            "padding": [3, 3],
            "dilation": [1, 1]
        },
        ...
    ]
    """
    try:
        from executorch.sdk import Inspector
        
        inspector = Inspector(pte_file_path=str(pte_file))
        
        layers = []
        layer_id = 0
        
        # Get execution plan
        plan = inspector.get_operator_list()
        
        for op_idx, op in enumerate(plan):
            # Look for convolution operations
            if 'conv' in op['name'].lower():
                # Extract tensor shapes from metadata
                try:
                    # Get input/output tensor info
                    input_spec  = inspector.get_tensor_spec(op['inputs'][0])
                    output_spec = inspector.get_tensor_spec(op['outputs'][0])
                    weight_spec = inspector.get_tensor_spec(op['inputs'][1])
                    
                    # Extract parameters from operator attributes
                    stride = op.get('stride', [1, 1])
                    padding = op.get('padding', [0, 0])
                    dilation = op.get('dilation', [1, 1])
                    
                    layer_info = {
                        "layer_id": layer_id,
                        "layer_name": f"{op['name']}_{op_idx}",
                        "input_shape": input_spec['shape'],
                        "output_shape": output_spec['shape'],
                        "kernel_shape": weight_spec['shape'],
                        "stride": stride if isinstance(stride, list) else [stride, stride],
                        "padding": padding if isinstance(padding, list) else [padding, padding],
                        "dilation": dilation if isinstance(dilation, list) else [dilation, dilation]
                    }
                    
                    layers.append(layer_info)
                    layer_id += 1
                    
                except Exception as e:
                    print(f"Warning: Could not extract details for op {op_idx}: {e}")
                    continue
        
        return layers
        
    except ImportError:
        print("ERROR: executorch not installed. Install with: pip install executorch")
        return None
    except Exception as e:
        print(f"ERROR extracting from .pte: {e}")
        return None


def extract_conv_layers_onnx(onnx_file):
    """
    Extract convolution layers from ONNX model (alternative method)
    
    Returns: List of layer configurations (same format as executorch)
    """
    try:
        import onnx
        from onnx import numpy_helper
        
        model = onnx.load(str(onnx_file))
        graph = model.graph
        
        layers = []
        layer_id = 0
        
        # Build tensor shape map
        tensor_shapes = {}
        for init in graph.initializer:
            tensor_shapes[init.name] = list(init.dims)
        
        for value_info in graph.value_info:
            if value_info.type.HasField('tensor_type'):
                shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                tensor_shapes[value_info.name] = shape
        
        for input_info in graph.input:
            if input_info.type.HasField('tensor_type'):
                shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                tensor_shapes[input_info.name] = shape
        
        for output_info in graph.output:
            if output_info.type.HasField('tensor_type'):
                shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
                tensor_shapes[output_info.name] = shape
        
        # Extract Conv nodes
        for node in graph.node:
            if node.op_type == 'Conv':
                # Get attributes
                stride = [1, 1]
                padding = [0, 0]
                dilation = [1, 1]
                
                for attr in node.attribute:
                    if attr.name == 'strides':
                        stride = list(attr.ints)
                    elif attr.name == 'pads':
                        pads = list(attr.ints)
                        padding = [pads[0], pads[1]]  # Assume symmetric
                    elif attr.name == 'dilations':
                        dilation = list(attr.ints)
                
                # Get shapes
                input_name = node.input[0]
                weight_name = node.input[1]
                output_name = node.output[0]
                
                input_shape = tensor_shapes.get(input_name, [1, 0, 0, 0])
                kernel_shape = tensor_shapes.get(weight_name, [0, 0, 0, 0])
                output_shape = tensor_shapes.get(output_name, [1, 0, 0, 0])
                
                layer_info = {
                    "layer_id": layer_id,
                    "layer_name": node.name or f"conv_{layer_id}",
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "kernel_shape": kernel_shape,
                    "stride": stride,
                    "padding": padding,
                    "dilation": dilation
                }
                
                layers.append(layer_info)
                layer_id += 1
        
        return layers
        
    except ImportError:
        print("ERROR: onnx not installed. Install with: pip install onnx")
        return None
    except Exception as e:
        print(f"ERROR extracting from ONNX: {e}")
        return None


def convert_to_generate_config(layers):
    """
    Convert extracted layers to format used by generate_idma_buffers.py
    
    Input format (from .pte/ONNX):
        input_shape: [N, C, H, W]
        kernel_shape: [out_c, in_c, kH, kW]
    
    Output format (for generate_idma_buffers.py):
        input: (W, H, C)
        output: (W, H, C)
        kernel: (kW, kH, in_c, out_c)
    """
    configs = []
    
    for layer in layers:
        # Extract dimensions
        _, in_c, in_h, in_w = layer['input_shape']
        _, out_c, out_h, out_w = layer['output_shape']
        out_channels, in_channels, k_h, k_w = layer['kernel_shape']
        
        stride_h, stride_w = layer['stride']
        pad_h, pad_w = layer['padding']
        dil_h, dil_w = layer['dilation']
        
        # Create config in Python script format
        config = {
            'layer_id': layer['layer_id'],
            'name': layer['layer_name'],
            'input': (in_w, in_h, in_c),
            'output': (out_w, out_h, out_c),
            'kernel': (k_w, k_h, in_channels, out_channels),
            'stride': (stride_w, stride_h),
            'padding': (pad_w, pad_h),
            'dilation': (dil_w, dil_h),
            # These will be calculated by generate_idma_buffers.py
            'conv_params': None  # (strideX, strideY, accumShift, reluMax, outputShift, outputScale, dilation, kernelH, kernelW)
        }
        
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Extract convolution layers from .pte or ONNX model')
    parser.add_argument('model_file', help='Path to .pte or .onnx model file')
    parser.add_argument('--output', '-o', default='layers_config.json', 
                       help='Output JSON file (default: layers_config.json)')
    parser.add_argument('--format', choices=['json', 'python'], default='json',
                       help='Output format (json or python dict)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_file)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return 1
    
    # Determine file type and extract
    print(f"Extracting layers from {model_path}...")
    
    if model_path.suffix == '.pte':
        layers = extract_conv_layers_executorch(model_path)
    elif model_path.suffix == '.onnx':
        layers = extract_conv_layers_onnx(model_path)
    else:
        print(f"ERROR: Unsupported file type: {model_path.suffix}")
        print("Supported: .pte, .onnx")
        return 1
    
    if layers is None or len(layers) == 0:
        print("ERROR: No layers extracted")
        return 1
    
    print(f"Extracted {len(layers)} convolution layers")
    
    # Convert to generate_idma_buffers.py format
    configs = convert_to_generate_config(layers)
    
    # Output
    output_path = Path(args.output)
    
    if args.format == 'json':
        with open(output_path, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Saved to {output_path}")
    else:  # python format
        output_path = output_path.with_suffix('.py')
        with open(output_path, 'w') as f:
            f.write("# Auto-generated layer configurations\n")
            f.write("# Generated from: {}\n\n".format(model_path))
            f.write("LAYER_CONFIGS = [\n")
            for config in configs:
                f.write("    {\n")
                for key, value in config.items():
                    f.write(f"        '{key}': {repr(value)},\n")
                f.write("    },\n")
            f.write("]\n")
        print(f"Saved to {output_path}")
    
    # Print summary
    print("\nLayer Summary:")
    print("-" * 80)
    for layer in configs:
        print(f"Layer {layer['layer_id']:2d}: {layer['name']:20s} "
              f"{layer['input']} → {layer['output']} "
              f"kernel={layer['kernel'][:2]} stride={layer['stride']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
