#!/usr/bin/env python3
"""
Example script showing how to load a saved LLM config and use it.
"""

import os
import sys

# Add the executorch path to import modules
sys.path.append("/home/gasoonjia/executorch")

from executorch.examples.models.llama.export_llama_lib import export_llama
from executorch.extension.llm.export.config.llm_config import LlmConfig
from executorch.extension.llm.export.export_llm import (
    load_config_from_file,
    save_config_to_file,
)


def load_and_use_saved_config():
    """Load a previously saved config and use it for export."""

    # Method 1: Load from a saved YAML file
    try:
        config_obj = load_config_from_file("used_config_llama3.yaml")
        print("✓ Successfully loaded config from used_config_llama3.yaml")

        # Optional: Modify the loaded config
        print("Original quantization mode:", config_obj.quantization.qmode)
        config_obj.quantization.qmode = "8da4w"  # Change quantization
        config_obj.debug.verbose = True  # Enable verbose logging
        print("Modified quantization mode:", config_obj.quantization.qmode)

        # Use the config for export
        print("Starting export with loaded config...")
        output_file = export_llama(config_obj)
        print(f"✓ Export completed! Output: {output_file}")

    except FileNotFoundError:
        print("❌ Config file 'used_config_llama3.yaml' not found.")
        print("First save a config by running the main export script.")
        return False

    return True


def create_and_save_custom_config():
    """Create a custom config and save it."""

    # Create a new config from scratch
    custom_config = LlmConfig()

    # Configure the model
    custom_config.base.model_class = "llama3"
    custom_config.base.checkpoint = (
        "/path/to/your/checkpoint.pth"  # Set your checkpoint path
    )

    # Configure model settings
    custom_config.model.use_kv_cache = True
    custom_config.model.use_sdpa_with_kv_cache = True
    custom_config.model.dtype_override = "fp32"

    # Configure export settings
    custom_config.export.max_seq_length = 2048
    custom_config.export.output_dir = "./outputs"

    # Configure backend
    custom_config.backend.xnnpack.enabled = True
    custom_config.backend.xnnpack.extended_ops = True

    # Configure quantization
    custom_config.quantization.qmode = "8da4w"

    # Configure debug
    custom_config.debug.verbose = True

    # Save the custom config
    config_filename = "my_custom_llama_config.yaml"
    save_config_to_file(custom_config, config_filename)
    print(f"✓ Custom config saved to {config_filename}")

    # Load it back to verify
    loaded_config = load_config_from_file(config_filename)
    print("✓ Verified: Config loaded successfully")

    return loaded_config


def main():
    print("=== LLM Config Load/Save Examples ===\n")

    # Example 1: Try to load a previously saved config
    print("1. Attempting to load saved config...")
    success = load_and_use_saved_config()

    if not success:
        print("\n2. Creating and saving a custom config...")
        custom_config = create_and_save_custom_config()

        print("\n3. Using the custom config for export...")
        try:
            output_file = export_llama(custom_config)
            print(f"✓ Export completed with custom config! Output: {output_file}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print("Make sure to set a valid checkpoint path in the config.")


if __name__ == "__main__":
    main()
