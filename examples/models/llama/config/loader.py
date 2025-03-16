import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from argparse import Namespace
import sys

from .validation import validate_config, ConfigValidationError

def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary using dot notation."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Convert a flattened dictionary back to nested format."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result

def _convert_cli_to_config_format(args: Namespace) -> Dict[str, Any]:
    """Convert CLI arguments to config format."""
    config = {}
    
    # Get only arguments that were explicitly set on command line
    specified_args = {}
    # _get_kwargs() returns list of (key, value) tuples for all arguments
    for key, value in args._get_kwargs():
        # Check if this argument was explicitly set on command line
        if value is not None:
            # Skip if it's False, assuming all bool args are with
            # action="store_true"
            if isinstance(value, bool) and value == False:
                continue
            specified_args[key] = value
    
    # Only set sections if they have specified values
    model_config = {}
    if 'model' in specified_args:
        model_config['name'] = specified_args['model']
    if 'fairseq2' in specified_args:
        model_config['type'] = 'FAIRSEQ2' if specified_args['fairseq2'] else 'LLAMA'
    if model_config:
        config['model'] = model_config
    
    # Export settings
    export_config = {}
    export_mappings = {
        'output_dir': 'output_dir',
        'checkpoint': 'checkpoint',
        'checkpoint_dir': 'checkpoint_dir',
        'tokenizer_path': 'tokenizer_path',
        'output_name': 'output_name',
        'enable_dynamic_shape': 'enable_dynamic_shape',
        'generate_full_logits': 'generate_full_logits',
        'dtype_override': 'dtype_override',
        'profile_memory': 'profile_memory',
        'profile_path': 'profile_path',
        'num_sharding': 'num_sharding',
        'params': 'params'  # Add params to export settings
    }
    for config_key, arg_key in export_mappings.items():
        if arg_key in specified_args:
            export_config[config_key] = specified_args[arg_key]
    if export_config:
        config['export'] = export_config
    
    # KV Cache settings
    kv_cache_config = {}
    kv_cache_mappings = {
        'enabled': 'use_kv_cache',
        'quantize': 'quantize_kv_cache',
        'use_sdpa': 'use_sdpa_with_kv_cache'
    }
    for config_key, arg_key in kv_cache_mappings.items():
        if arg_key in specified_args:
            kv_cache_config[config_key] = specified_args[arg_key]
    if kv_cache_config:
        config['kv_cache'] = kv_cache_config
    
    # Quantization settings
    quant_config = {}
    quant_mappings = {
        'pt2e_quantize': 'pt2e_quantize',
        'embedding_quantize': 'embedding_quantize',
        'quantization_mode': 'quantization_mode',
        'group_size': 'group_size',
        'use_qnn_sha': 'use_qnn_sha'
    }
    for config_key, arg_key in quant_mappings.items():
        if arg_key in specified_args:
            quant_config[config_key] = specified_args[arg_key]
    if quant_config:
        config['quantization'] = quant_config
    
    # Backend settings
    backend_config = {}
    if 'xnnpack' in specified_args or 'xnnpack_extended_ops' in specified_args:
        backend_config['xnnpack'] = {}
        if 'xnnpack' in specified_args:
            backend_config['xnnpack']['enabled'] = specified_args['xnnpack']
        if 'xnnpack_extended_ops' in specified_args:
            backend_config['xnnpack']['extended_ops'] = specified_args['xnnpack_extended_ops']
    
    if 'vulkan' in specified_args:
        backend_config['vulkan'] = {'enabled': specified_args['vulkan']}
    
    if 'mps' in specified_args:
        backend_config['mps'] = {'enabled': specified_args['mps']}
    
    if any(k in specified_args for k in ['coreml', 'coreml_enable_state', 'coreml_preserve_sdpa']):
        backend_config['coreml'] = {}
        if 'coreml' in specified_args:
            backend_config['coreml']['enabled'] = specified_args['coreml']
        if 'coreml_enable_state' in specified_args:
            backend_config['coreml']['enable_state'] = specified_args['coreml_enable_state']
        if 'coreml_preserve_sdpa' in specified_args:
            backend_config['coreml']['preserve_sdpa'] = specified_args['coreml_preserve_sdpa']
    
    if backend_config:
        config['backends'] = backend_config
    
    # Misc settings
    misc_config = {}
    misc_mappings = {
        'verbose': 'verbose',
        'fairseq2': 'fairseq2',
        'optimized_rotation_path': 'optimized_rotation_path',
        'metadata': 'metadata',
        'so_library': 'so_library'
    }
    for config_key, arg_key in misc_mappings.items():
        if arg_key in specified_args:
            misc_config[config_key] = specified_args[arg_key]
    if misc_config:
        config['misc'] = misc_config
    
    return config

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json_params(params_path: Union[str, Path]) -> Dict[str, Any]:
    """Load model parameters from JSON file."""
    with open(params_path, 'r') as f:
        return json.load(f)

def _convert_params_to_config_format(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert params.json format to our config structure."""
    # Mapping from params.json keys to config keys
    params_mapping = {
        'dim': 'architecture.dim',
        'n_layers': 'architecture.n_layers',
        'n_heads': 'architecture.n_heads',
        'n_kv_heads': 'architecture.n_kv_heads',
        'vocab_size': 'architecture.vocab_size',
        'hidden_dim': 'architecture.hidden_dim',
        'head_dim': 'architecture.head_dim',
        'multiple_of': 'architecture.multiple_of',
        'ffn_dim_multiplier': 'architecture.ffn_dim_multiplier',
        'norm_eps': 'architecture.norm_eps',
        'max_batch_size': 'limits.max_batch_size',
        'max_seq_len': 'limits.max_seq_len',
        'max_context_len': 'limits.max_context_len',
        'moe': 'moe.enabled',
        'num_experts': 'moe.num_experts',
        'num_activated_experts': 'moe.num_activated_experts',
        'use_kv_cache': 'kv_cache.enabled',
        'use_sdpa_with_kv_cache': 'kv_cache.use_sdpa',
        'generate_full_logits': 'export.generate_full_logits',
        'enable_dynamic_shape': 'export.enable_dynamic_shape',
        'use_hf_rope': 'rope.use_hf_rope',
        'rope_theta': 'rope.theta',
        'rope_freq_base': 'rope.freq_base',
        'use_scaled_rope': 'rope.use_scaled_rope',
        'rope_scale_factor': 'rope.scale_factor',
        'bos_idx': 'special_tokens.bos_idx',
        'eos_idx': 'special_tokens.eos_idx',
        'bos_count': 'special_tokens.bos_count',
        'eos_count': 'special_tokens.eos_count',
    }
    
    # Convert flattened params to nested config
    mapped_dict = {}
    for param_key, value in params.items():
        config_key = params_mapping.get(param_key)
        if config_key:
            mapped_dict[config_key] = value
        else:
            # For any params not in our mapping, put them in a special section
            mapped_dict[f'additional_params.{param_key}'] = value
    
    return _unflatten_dict(mapped_dict)

def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries."""
    merged = base.copy()
    
    for key, value in override.items():
        if (
            key in merged and 
            isinstance(merged[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        elif value is not None:  # Only override if value is not None
            merged[key] = value
            
    return merged

def create_config(
    yaml_path: Optional[Union[str, Path]] = None,
    cli_args: Optional[Namespace] = None,
    model_args: Optional[Any] = None,
    params_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Create configuration from multiple sources.
    Priority (highest to lowest):
    1. CLI arguments
    2. params.json file (for model architecture)
    3. YAML config file
    4. Default values
    """
    # Start with default config
    config = load_yaml_config(Path(__file__).parent / "default.yaml")
    
    # Load from YAML if provided
    if yaml_path:
        yaml_config = load_yaml_config(yaml_path)
        config = merge_configs(config, yaml_config)
    
    # Load and apply params.json if provided
    if params_json_path:
        params = load_json_params(params_json_path)
        params_config = _convert_params_to_config_format(params)
        # Params override YAML config for architecture settings
        config = merge_configs(config, params_config)
    
    # Override with ModelArgs if provided
    if model_args:
        model_config = {
            "architecture": {
                "dim": model_args.dim,
                "n_layers": model_args.n_layers,
                "n_heads": model_args.n_heads,
                "n_kv_heads": model_args.n_kv_heads,
                "vocab_size": model_args.vocab_size,
                "hidden_dim": model_args.hidden_dim,
                "head_dim": model_args.head_dim,
                "multiple_of": model_args.multiple_of,
                "ffn_dim_multiplier": model_args.ffn_dim_multiplier,
                "norm_eps": model_args.norm_eps,
            },
            "limits": {
                "max_batch_size": model_args.max_batch_size,
                "max_seq_len": model_args.max_seq_len,
                "max_context_len": model_args.max_context_len,
            },
            "moe": {
                "enabled": model_args.moe,
                "num_experts": model_args.num_experts,
                "num_activated_experts": model_args.num_activated_experts,
            },
            "rope": {
                "use_hf_rope": model_args.use_hf_rope,
                "theta": model_args.rope_theta,
                "freq_base": model_args.rope_freq_base,
                "use_scaled_rope": model_args.use_scaled_rope,
                "scale_factor": model_args.rope_scale_factor,
            },
            "special_tokens": {
                "bos_idx": model_args.bos_idx,
                "eos_idx": model_args.eos_idx,
                "bos_count": model_args.bos_count,
                "eos_count": model_args.eos_count,
            },
        }
        config = merge_configs(config, model_config)
    
    # Override with CLI args if provided
    if cli_args:
        cli_config = _convert_cli_to_config_format(cli_args)
        config = merge_configs(config, cli_config)
    
    # Validate the final configuration
    try:
        validate_config(config)
    except ConfigValidationError as e:
        raise ConfigValidationError(f"Configuration validation failed: {str(e)}")
    
    return config

def get_model_args_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config back to ModelArgs format."""
    return {
        'dim': config['architecture']['dim'],
        'n_layers': config['architecture']['n_layers'],
        'n_heads': config['architecture']['n_heads'],
        'n_kv_heads': config['architecture']['n_kv_heads'],
        'vocab_size': config['architecture']['vocab_size'],
        'hidden_dim': config['architecture']['hidden_dim'],
        'head_dim': config['architecture']['head_dim'],
        'multiple_of': config['architecture']['multiple_of'],
        'ffn_dim_multiplier': config['architecture']['ffn_dim_multiplier'],
        'norm_eps': config['architecture']['norm_eps'],
        'max_batch_size': config['limits']['max_batch_size'],
        'max_seq_len': config['limits']['max_seq_len'],
        'max_context_len': config['limits']['max_context_len'],
        'moe': config['moe']['enabled'],
        'num_experts': config['moe']['num_experts'],
        'num_activated_experts': config['moe']['num_activated_experts'],
        'use_kv_cache': config['kv_cache']['enabled'],
        'use_sdpa_with_kv_cache': config['kv_cache']['use_sdpa'],
        'generate_full_logits': config['export']['generate_full_logits'],
        'enable_dynamic_shape': config['export']['enable_dynamic_shape'],
        'use_hf_rope': config['rope']['use_hf_rope'],
        'rope_theta': config['rope']['theta'],
        'rope_freq_base': config['rope']['freq_base'],
        'use_scaled_rope': config['rope']['use_scaled_rope'],
        'rope_scale_factor': config['rope']['scale_factor'],
        'bos_idx': config['special_tokens']['bos_idx'],
        'eos_idx': config['special_tokens']['eos_idx'],
        'bos_count': config['special_tokens']['bos_count'],
        'eos_count': config['special_tokens']['eos_count'],
    } 