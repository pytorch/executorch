# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import logging

# force=True to ensure logging while in debugger. Set up logger before any
# other imports.
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, force=True)

import sys

import torch

from .export_llama_lib import build_args_parser, export_llama

from pathlib import Path

sys.setrecursionlimit(4096)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    parser = build_args_parser()
    args = parser.parse_args()
    # Create config from all sources
    from .config.loader import create_config, get_model_args_from_config
    from .config.validation import ConfigValidationError

    try:
        config = create_config(
            yaml_path=args.config,
            cli_args=args,
            params_json_path=args.params  # Include params.json
        )
    except ConfigValidationError as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        logging.error("Please check your configuration and try again.")
        return 1
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        return 1

    # Convert config back to args format for backward compatibility
    args_dict = {
        'model': config['model']['name'],
        'output_dir': config['export']['output_dir'],
        'checkpoint': config['export']['checkpoint'],
        'checkpoint_dir': config['export']['checkpoint_dir'],
        'params': config['export']['params'],
        'tokenizer_path': config['export']['tokenizer_path'],
        'pt2e_quantize': config['quantization']['pt2e_quantize'],
        'embedding_quantize': config['quantization']['embedding_quantize'],
        'quantization_mode': config['quantization']['quantization_mode'],
        'group_size': config['quantization']['group_size'],
        'use_qnn_sha': config['quantization']['use_qnn_sha'],
        'use_kv_cache': config['kv_cache']['enabled'],
        'quantize_kv_cache': config['kv_cache']['quantize'],
        'use_sdpa_with_kv_cache': config['kv_cache']['use_sdpa'],
        'enable_dynamic_shape': config['export']['enable_dynamic_shape'],
        'generate_full_logits': config['export']['generate_full_logits'],
        'dtype_override': config['export']['dtype_override'],
        'output_name': config['export']['output_name'],
        'profile_memory': config['export']['profile_memory'],
        'profile_path': config['export']['profile_path'],
        'num_sharding': config['export']['num_sharding'],
        'xnnpack': config['backends']['xnnpack']['enabled'],
        'xnnpack_extended_ops': config['backends']['xnnpack']['extended_ops'],
        'vulkan': config['backends']['vulkan']['enabled'],
        'mps': config['backends']['mps']['enabled'],
        'coreml': config['backends']['coreml']['enabled'],
        'coreml-enable-state': config['backends']['coreml']['enable_state'],
        'coreml-preserve-sdpa': config['backends']['coreml']['preserve_sdpa'],
        'verbose': config['misc']['verbose'],
        'fairseq2': config['model']['type'] == 'FAIRSEQ2',
        'optimized_rotation_path': config['misc']['optimized_rotation_path'],
        'metadata': config['misc']['metadata'],
        'so_library': config['misc']['so_library'],
        'max_context_length': config['limits']['max_context_len'],
        'max_seq_length': config['limits']['max_seq_len'],
    }

    # Update args with config values
    for k, v in args_dict.items():
        setattr(args, k, v)

    # # Create ModelArgs from config
    # model_args_dict = get_model_args_from_config(config)

    export_llama(args)

    # import json
    # with open("tmp/args_test.json", "w") as f:
    #     json.dump(vars(args), f, indent=4)

    if args.output_dir:
        final_config_path = Path(args.output_dir) / "used_config.yaml"
        from .config.loader import save_yaml_config
        save_yaml_config(config, final_config_path)
        logging.info(f"Saved final configuration to: {final_config_path}")


if __name__ == "__main__":
    main()  # pragma: no cover
