"""
Extract convolution parameters from PyTorch Edge (.pte) files

This script loads a .pte file and extracts convolution layer parameters
including kernel size, stride, padding, input/output channels, and dimensions.
"""

import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
import json
from pathlib import Path


def extract_conv_params_from_pte(pte_path):
    """
    Extract convolution parameters from a .pte file
    
    Args:
        pte_path: Path to the .pte file
        
    Returns:
        List of dictionaries containing conv layer parameters
    """
    print(f"Loading .pte file: {pte_path}")
    
    # Load the ExecuTorch program
    try:
        # Method 1: Using executorch library
        et_module = _load_for_executorch(str(pte_path))
        print(f"Successfully loaded {pte_path}")
    except Exception as e:
        print(f"Error loading with executorch: {e}")
        print("Trying alternative method...")
        
        # Method 2: Direct torch.load (may work for some .pte files)
        try:
            program = torch.load(pte_path)
            print(f"Loaded with torch.load: {type(program)}")
        except Exception as e2:
            print(f"Error with torch.load: {e2}")
            return None
    
    # Extract layer information
    conv_layers = []
    
    # Method to traverse and find conv operations
    # This depends on the specific structure of your .pte file
    
    return conv_layers


def extract_from_onnx(onnx_path):
    """
    Alternative: Extract from ONNX model (easier to parse)
    
    Args:
        onnx_path: Path to ONNX model file
        
    Returns:
        List of conv layer parameters
    """
    import onnx
    
    model = onnx.load(onnx_path)
    conv_layers = []
    
    for node in model.graph.node:
        if node.op_type == 'Conv':
            params = {
                'name': node.name,
                'op_type': 'Conv',
            }
            
            # Extract attributes
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    params['kernel_h'] = attr.ints[0]
                    params['kernel_w'] = attr.ints[1] if len(attr.ints) > 1 else attr.ints[0]
                elif attr.name == 'strides':
                    params['stride_y'] = attr.ints[0]
                    params['stride_x'] = attr.ints[1] if len(attr.ints) > 1 else attr.ints[0]
                elif attr.name == 'pads':
                    params['pad_top'] = attr.ints[0]
                    params['pad_left'] = attr.ints[1]
                    params['pad_bottom'] = attr.ints[2]
                    params['pad_right'] = attr.ints[3]
                elif attr.name == 'dilations':
                    params['dilation'] = attr.ints[0]
                elif attr.name == 'group':
                    params['groups'] = attr.i
            
            # Get input/output shapes from value_info
            for value_info in model.graph.value_info:
                if value_info.name == node.input[0]:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) == 4:  # NCHW format
                        params['input_channels'] = shape[1]
                        params['input_h'] = shape[2]
                        params['input_w'] = shape[3]
                        
                if value_info.name == node.output[0]:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) == 4:
                        params['output_channels'] = shape[1]
                        params['output_h'] = shape[2]
                        params['output_w'] = shape[3]
            
            conv_layers.append(params)
    
    return conv_layers


def extract_from_pytorch_model(model_path):
    """
    Extract from regular PyTorch model (.pt or .pth)
    
    Args:
        model_path: Path to PyTorch model
        
    Returns:
        List of conv layer parameters
    """
    model = torch.load(model_path)
    
    if isinstance(model, dict) and 'state_dict' in model:
        state_dict = model['state_dict']
    elif isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model
    
    conv_layers = []
    
    for name, param in state_dict.items():
        if 'conv' in name.lower() and 'weight' in name:
            # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
            if len(param.shape) == 4:
                params = {
                    'layer_name': name,
                    'output_channels': param.shape[0],
                    'input_channels': param.shape[1],
                    'kernel_h': param.shape[2],
                    'kernel_w': param.shape[3],
                }
                conv_layers.append(params)
    
    return conv_layers


def print_conv_params(conv_layers):
    """Print convolution parameters in a readable format"""
    print("\n" + "="*80)
    print("CONVOLUTION LAYERS FOUND:")
    print("="*80)
    
    for i, layer in enumerate(conv_layers, 1):
        print(f"\nLayer {i}:")
        print("-" * 40)
        for key, value in layer.items():
            print(f"  {key:20s}: {value}")


def generate_config_from_params(conv_layers, output_file="conv_configs.py"):
    """
    Generate Python config dictionary from extracted parameters
    
    Args:
        conv_layers: List of conv layer parameters
        output_file: Output Python file path
    """
    with open(output_file, 'w') as f:
        f.write("# Auto-generated convolution configurations\n\n")
        
        for i, layer in enumerate(conv_layers):
            config_name = f"config_{layer.get('name', f'conv{i}')}"
            
            f.write(f"{config_name} = {{\n")
            f.write(f"    'input_w': {layer.get('input_w', 'UNKNOWN')},\n")
            f.write(f"    'input_h': {layer.get('input_h', 'UNKNOWN')},\n")
            f.write(f"    'input_channels': {layer.get('input_channels', 'UNKNOWN')},\n")
            f.write(f"    'output_w': {layer.get('output_w', 'UNKNOWN')},\n")
            f.write(f"    'output_h': {layer.get('output_h', 'UNKNOWN')},\n")
            f.write(f"    'output_channels': {layer.get('output_channels', 'UNKNOWN')},\n")
            f.write(f"    'kernel_w': {layer.get('kernel_w', 'UNKNOWN')},\n")
            f.write(f"    'kernel_h': {layer.get('kernel_h', 'UNKNOWN')},\n")
            f.write(f"    'stride_x': {layer.get('stride_x', 1)},\n")
            f.write(f"    'stride_y': {layer.get('stride_y', 1)},\n")
            f.write(f"    'padding': {layer.get('padding', 0)},\n")
            f.write(f"    'dilation': {layer.get('dilation', 1)},\n")
            f.write("}\n\n")
    
    print(f"\nGenerated config file: {output_file}")


def main():
    """Main function - example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract convolution parameters from model files')
    parser.add_argument('model_path', help='Path to .pte, .onnx, .pt, or .pth file')
    parser.add_argument('--output', '-o', default='conv_configs.py', 
                       help='Output config file (default: conv_configs.py)')
    parser.add_argument('--format', '-f', choices=['pte', 'onnx', 'pytorch', 'auto'],
                       default='auto', help='Model format (default: auto-detect)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: File not found: {model_path}")
        return
    
    # Auto-detect format
    if args.format == 'auto':
        suffix = model_path.suffix.lower()
        if suffix == '.pte':
            args.format = 'pte'
        elif suffix == '.onnx':
            args.format = 'onnx'
        elif suffix in ['.pt', '.pth']:
            args.format = 'pytorch'
        else:
            print(f"Unknown format for {suffix}, please specify --format")
            return
    
    # Extract parameters
    conv_layers = None
    
    if args.format == 'pte':
        conv_layers = extract_conv_params_from_pte(model_path)
    elif args.format == 'onnx':
        conv_layers = extract_from_onnx(model_path)
    elif args.format == 'pytorch':
        conv_layers = extract_from_pytorch_model(model_path)
    
    if conv_layers:
        print_conv_params(conv_layers)
        generate_config_from_params(conv_layers, args.output)
    else:
        print("No convolution layers found or error during extraction")


if __name__ == '__main__':
    # Example usage without command line args
    # Uncomment and modify for your use case:
    
    # For .pte file:
    # conv_layers = extract_conv_params_from_pte('model.pte')
    
    # For ONNX (recommended - easier to parse):
    # conv_layers = extract_from_onnx('model.onnx')
    
    # For PyTorch model:
    # conv_layers = extract_from_pytorch_model('model.pth')
    
    # Or run with command line:
    main()
