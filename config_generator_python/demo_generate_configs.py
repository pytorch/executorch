#!/usr/bin/env python3
"""
Demo: Generate lookup table from existing ResNet CSV

This demonstrates the complete workflow using your existing resnet18_conv_list.csv
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 70)
    print("Model-Driven Architecture Demo")
    print("=" * 70)
    print()
    
    # Check if CSV exists
    csv_file = Path("resnet18_conv_list.csv")
    if not csv_file.exists():
        print(f"ERROR: {csv_file} not found")
        print("Please ensure resnet18_conv_list.csv is in the current directory")
        return 1
    
    print("Step 1: Found resnet18_conv_list.csv")
    print()
    
    # Generate lookup table
    print("Step 2: Generating conv_layer_configs.h...")
    print()
    
    cmd = [
        sys.executable,
        "generate_layer_configs.py",
        str(csv_file),
        "--dram0", "32768",
        "--dram1", "32768",
        "--output", "conv_layer_configs.h"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print()
        print("Generated file: conv_layer_configs.h")
        print()
        print("Next steps:")
        print("  1. Copy conv_layer_configs.h to test_cnn_depthwise_convolve_MOD2/test/")
        print("  2. Include in your code:")
        print("       #include \"conv_layer_configs.h\"")
        print("  3. Use the API:")
        print("       conv_execute_layer(0, input, output, weights, bias, outscale);")
        print()
        print("Example usage:")
        print("  // Print configuration")
        print("  print_layer_config(0);")
        print()
        print("  // Execute layer")
        print("  const conv_layer_config_t* config = get_layer_config(0);")
        print("  printf(\"Executing %s\\n\", config->layer_name);")
        print()
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"ERROR: Command failed with exit code {e.returncode}")
        return 1
    except FileNotFoundError:
        print()
        print("ERROR: generate_layer_configs.py not found")
        print("Please ensure the script is in the current directory")
        return 1

if __name__ == '__main__':
    sys.exit(main())
