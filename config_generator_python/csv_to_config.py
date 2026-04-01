"""
Convert ResNet CSV convolution list to config format for generate_idma_buffers.py
"""

import csv
import re


def parse_shape(shape_str):
    """Parse shape string like '1,64,112,112' to list"""
    return [int(x.strip()) for x in shape_str.split(',')]


def parse_tuple(tuple_str):
    """Parse tuple string like '2, 2' to integer"""
    values = [int(x.strip()) for x in tuple_str.split(',')]
    return values[0]  # Assuming symmetric stride/padding


def csv_to_configs(csv_file):
    """
    Convert ResNet CSV to config dictionaries
    
    CSV format:
    layer_name, input, kernel, stride, padding, dilation, transposed, output_padding, groups, output
    """
    configs = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Parse input shape: [batch, channels, height, width]
            input_shape = parse_shape(row['input'])
            input_c = input_shape[1]
            input_h = input_shape[2]
            input_w = input_shape[3]
            
            # Parse kernel shape: [out_channels, in_channels, kernel_h, kernel_w]
            kernel_shape = parse_shape(row['kernel'])
            output_c = kernel_shape[0]
            kernel_h = kernel_shape[2]
            kernel_w = kernel_shape[3]
            
            # Parse output shape
            output_shape = parse_shape(row['output'])
            output_h = output_shape[2]
            output_w = output_shape[3]
            
            # Parse stride and padding
            stride = parse_tuple(row['stride'])
            padding = parse_tuple(row['padding'])
            dilation = parse_tuple(row['dilation'])
            
            # Generate config name
            layer_name = row[''].strip() if row[''] else f'conv_{len(configs)}'
            
            # Determine config name based on kernel and stride
            config_key = f"{kernel_h}x{kernel_w}j{stride}d{dilation}"
            config_name = f"config_{config_key}_{layer_name}"
            
            config = {
                'name': config_name,
                'layer_name': layer_name,
                'input_w': input_w,
                'input_h': input_h,
                'input_channels': input_c,
                'output_w': output_w,
                'output_h': output_h,
                'output_channels': output_c,
                'kernel_w': kernel_w,
                'kernel_h': kernel_h,
                'stride_x': stride,
                'stride_y': stride,
                'padding': padding,
                'dilation': dilation,
                # For generate_idma_buffers.py format:
                'conv_params': (stride, stride, 8, 4000, 11, 0, dilation, 1, 1)
            }
            
            configs.append(config)
    
    return configs


def print_configs(configs):
    """Print configs in Python dictionary format"""
    print("# Convolution configurations from ResNet CSV\n")
    
    for cfg in configs:
        print(f"# {cfg['layer_name']}: {cfg['input_h']}x{cfg['input_w']}x{cfg['input_channels']} "
              f"-> {cfg['output_h']}x{cfg['output_w']}x{cfg['output_channels']}")
        print(f"{cfg['name']} = {{")
        print(f"    'input_w': {cfg['input_w']},")
        print(f"    'input_h': {cfg['input_h']},")
        print(f"    'input_channels': {cfg['input_channels']},")
        print(f"    'output_w': {cfg['output_w']},")
        print(f"    'output_h': {cfg['output_h']},")
        print(f"    'output_channels': {cfg['output_channels']},")
        print(f"    'kernel_w': {cfg['kernel_w']},")
        print(f"    'kernel_h': {cfg['kernel_h']},")
        print(f"    'stride_xy': ({cfg['stride_x']}, {cfg['stride_y']}),")
        print(f"    'padding': {cfg['padding']},")
        print(f"    'dilation': {cfg['dilation']},")
        print(f"    'conv_params': {cfg['conv_params']}")
        print("}\n")


def generate_unique_configs(configs):
    """Generate unique configurations (by kernel size and stride)"""
    unique = {}
    
    for cfg in configs:
        key = (cfg['kernel_h'], cfg['kernel_w'], cfg['stride_x'], cfg['dilation'])
        
        if key not in unique:
            unique[key] = cfg
        else:
            # Keep the one with most representative size (e.g., largest feature map)
            if cfg['input_w'] * cfg['input_h'] > unique[key]['input_w'] * unique[key]['input_h']:
                unique[key] = cfg
    
    return list(unique.values())


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ResNet CSV to config format')
    parser.add_argument('csv_file', help='Path to CSV file (e.g., resnet18_conv_list.csv)')
    parser.add_argument('--unique', action='store_true', 
                       help='Only output unique kernel/stride combinations')
    parser.add_argument('--output', '-o', help='Output Python file')
    
    args = parser.parse_args()
    
    configs = csv_to_configs(args.csv_file)
    
    if args.unique:
        configs = generate_unique_configs(configs)
        print(f"# Found {len(configs)} unique configurations\n")
    
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            print_configs(configs)
        sys.stdout = original_stdout
        print(f"Wrote configs to {args.output}")
    else:
        print_configs(configs)


if __name__ == '__main__':
    # Example usage:
    # python csv_to_config.py resnet18_conv_list.csv --unique
    # python csv_to_config.py resnet18_conv_list.csv --output resnet18_configs.py
    
    main()
