from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model section of config."""
    if "model" not in config:
        raise ConfigValidationError("Missing required 'model' section")
    
    model = config["model"]
    if "name" not in model:
        raise ConfigValidationError("Missing required 'model.name' field")
    if "type" not in model:
        raise ConfigValidationError("Missing required 'model.type' field")
        
    valid_models = [
        "stories110m", "llama2", "llama3", "llama3_1", "llama3_2", 
        "static_llama", "qwen2_5", "phi-4-mini", "llama3_2_vision"
    ]
    if model["name"] not in valid_models:
        raise ConfigValidationError(f"Invalid model name. Must be one of: {valid_models}")
        
    valid_types = ["LLAMA", "FAIRSEQ2"]
    if model["type"] not in valid_types:
        raise ConfigValidationError(f"Invalid model type. Must be one of: {valid_types}")

def validate_architecture_config(config: Dict[str, Any]) -> None:
    """Validate architecture section of config."""
    if "architecture" not in config:
        raise ConfigValidationError("Missing required 'architecture' section")
    
    arch = config["architecture"]
    required_fields = ["dim", "n_layers", "n_heads"]
    for field in required_fields:
        if field not in arch:
            raise ConfigValidationError(f"Missing required 'architecture.{field}' field")
    
    # Validate numeric values
    if not isinstance(arch["dim"], int) or arch["dim"] <= 0:
        raise ConfigValidationError("architecture.dim must be a positive integer")
    if not isinstance(arch["n_layers"], int) or arch["n_layers"] <= 0:
        raise ConfigValidationError("architecture.n_layers must be a positive integer")
    if not isinstance(arch["n_heads"], int) or arch["n_heads"] <= 0:
        raise ConfigValidationError("architecture.n_heads must be a positive integer")
    
    # Validate optional fields
    if "multiple_of" in arch and (not isinstance(arch["multiple_of"], int) or arch["multiple_of"] <= 0):
        raise ConfigValidationError("architecture.multiple_of must be a positive integer")
    if "norm_eps" in arch and (not isinstance(arch["norm_eps"], float) or arch["norm_eps"] <= 0):
        raise ConfigValidationError("architecture.norm_eps must be a positive float")

def validate_limits_config(config: Dict[str, Any]) -> None:
    """Validate limits section of config."""
    if "limits" not in config:
        raise ConfigValidationError("Missing required 'limits' section")
    
    limits = config["limits"]
    required_fields = ["max_batch_size", "max_seq_len", "max_context_len"]
    for field in required_fields:
        if field not in limits:
            raise ConfigValidationError(f"Missing required 'limits.{field}' field")
    
    # Validate numeric values
    if not isinstance(limits["max_batch_size"], int) or limits["max_batch_size"] <= 0:
        raise ConfigValidationError("limits.max_batch_size must be a positive integer")
    if not isinstance(limits["max_seq_len"], int) or limits["max_seq_len"] <= 0:
        raise ConfigValidationError("limits.max_seq_len must be a positive integer")
    if not isinstance(limits["max_context_len"], int) or limits["max_context_len"] <= 0:
        raise ConfigValidationError("limits.max_context_len must be a positive integer")
    
    # Validate relationships
    if limits["max_context_len"] < limits["max_seq_len"]:
        raise ConfigValidationError("limits.max_context_len must be >= limits.max_seq_len")

def validate_rope_config(config: Dict[str, Any]) -> None:
    """Validate RoPE section of config."""
    if "rope" not in config:
        return  # Optional section
    
    rope = config["rope"]
    if "freq_base" in rope and (not isinstance(rope["freq_base"], float) or rope["freq_base"] <= 0):
        raise ConfigValidationError("rope.freq_base must be a positive float")
    if "scale_factor" in rope and (not isinstance(rope["scale_factor"], (int, float)) or rope["scale_factor"] <= 0):
        raise ConfigValidationError("rope.scale_factor must be a positive number")
    
    # Validate boolean fields
    bool_fields = ["use_hf_rope", "use_scaled_rope"]
    for field in bool_fields:
        if field in rope and not isinstance(rope[field], bool):
            raise ConfigValidationError(f"rope.{field} must be a boolean")

def validate_kv_cache_config(config: Dict[str, Any]) -> None:
    """Validate KV cache section of config."""
    if "kv_cache" not in config:
        return  # Optional section
    
    kv_cache = config["kv_cache"]
    bool_fields = ["enabled", "quantize", "use_sdpa"]
    for field in bool_fields:
        if field in kv_cache and not isinstance(kv_cache[field], bool):
            raise ConfigValidationError(f"kv_cache.{field} must be a boolean")

def validate_export_config(config: Dict[str, Any]) -> None:
    """Validate export section of config."""
    if "export" not in config:
        raise ConfigValidationError("Missing required 'export' section")
    
    export = config["export"]
    required_fields = ["output_dir"]
    for field in required_fields:
        if field not in export:
            raise ConfigValidationError(f"Missing required 'export.{field}' field")
    
    # Validate paths
    if "checkpoint" in export and export["checkpoint"]:
        if not isinstance(export["checkpoint"], str):
            raise ConfigValidationError("export.checkpoint must be a string path")
    
    if "checkpoint_dir" in export and export["checkpoint_dir"]:
        if not isinstance(export["checkpoint_dir"], str):
            raise ConfigValidationError("export.checkpoint_dir must be a string path")
    
    # Validate dtype_override
    valid_dtypes = ["fp32", "fp16", "bf16"]
    if "dtype_override" in export and export["dtype_override"] not in valid_dtypes:
        raise ConfigValidationError(f"export.dtype_override must be one of: {valid_dtypes}")

def validate_quantization_config(config: Dict[str, Any]) -> None:
    """Validate quantization section of config."""
    if "quantization" not in config:
        return  # Optional section
    
    quant = config["quantization"]
    
    # Validate pt2e_quantize
    valid_pt2e = [
        "xnnpack_dynamic", "xnnpack_dynamic_qc4", "qnn_8a8w", "qnn_16a16w",
        "qnn_16a4w", "coreml_c4w", "coreml_8a_c8w", "coreml_8a_c4w",
        "coreml_baseline_8a_c8w", "coreml_baseline_8a_c4w", "vulkan_8w", None,
    ]
    if "pt2e_quantize" in quant and quant["pt2e_quantize"] not in valid_pt2e:
        raise ConfigValidationError(f"quantization.pt2e_quantize must be one of: {valid_pt2e}")
    
    # Validate embedding_quantize format
    if "embedding_quantize" in quant and quant["embedding_quantize"]:
        pattern = r'^\d+,\d+$'
        if not re.match(pattern, quant["embedding_quantize"]):
            raise ConfigValidationError("quantization.embedding_quantize must be in format 'bitwidth,groupsize' (e.g., '8,1024')")

def validate_backends_config(config: Dict[str, Any]) -> None:
    """Validate backends section of config."""
    if "backends" not in config:
        return  # Optional section
    
    backends = config["backends"]
    
    # Validate backend-specific settings
    for backend in ["xnnpack", "vulkan", "mps", "coreml"]:
        if backend in backends:
            if not isinstance(backends[backend], dict):
                raise ConfigValidationError(f"backends.{backend} must be a dictionary")
            if "enabled" in backends[backend]:
                if not isinstance(backends[backend]["enabled"], bool):
                    raise ConfigValidationError(f"backends.{backend}.enabled must be a boolean")

    # Validate CoreML specific settings
    if "coreml" in backends and backends["coreml"].get("enabled"):
        coreml = backends["coreml"]
        if "ios_version" in coreml and coreml["ios_version"] not in [15, 16, 17, 18]:
            raise ConfigValidationError("backends.coreml.ios_version must be one of: 15, 16, 17, 18")
        if "compute_units" in coreml and coreml["compute_units"] not in ["cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"]:
            raise ConfigValidationError("backends.coreml.compute_units must be one of: cpu_only, cpu_and_gpu, cpu_and_ne, all")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate entire configuration."""
    validators = [
        validate_model_config,
        validate_architecture_config,
        validate_limits_config,
        validate_rope_config,
        validate_kv_cache_config,
        validate_export_config,
        validate_quantization_config,
        validate_backends_config
    ]
    
    for validator in validators:
        validator(config)
    
    # Cross-section validations
    if config.get("kv_cache", {}).get("enabled"):
        if config.get("export", {}).get("enable_dynamic_shape"):
            if any(config.get("backends", {}).get(b, {}).get("enabled") for b in ["coreml", "mps"]):
                raise ConfigValidationError("Dynamic shape is not supported with CoreML or MPS backends when KV cache is enabled") 