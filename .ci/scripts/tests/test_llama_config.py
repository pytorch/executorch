import os
import pytest
from pathlib import Path
from argparse import Namespace
from examples.models.llama.config.loader import create_config, merge_configs
from examples.models.llama.config.validation import ConfigValidationError

def test_basic_config_validation():
    """Test basic configuration validation."""
    # Valid config
    valid_config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "architecture": {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "vocab_size": 32000,
            "multiple_of": 256,
            "norm_eps": 1.0e-5
        },
        "limits": {
            "max_batch_size": 32,
            "max_seq_len": 2048,
            "max_context_len": 2048
        },
        "export": {
            "output_dir": "output"
        }
    }
    
    try:
        create_config(cli_args=None, model_args=None, params_json_path=None)
    except ConfigValidationError as e:
        assert "Missing required 'model' section" in str(e)
    
    # Test invalid model name
    invalid_config = valid_config.copy()
    invalid_config["model"]["name"] = "invalid_model"
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=invalid_config)
    assert "Invalid model name" in str(exc.value)
    
    # Test invalid architecture values
    invalid_config = valid_config.copy()
    invalid_config["architecture"]["dim"] = -1
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=invalid_config)
    assert "architecture.dim must be a positive integer" in str(exc.value)

def test_config_merging():
    """Test configuration merging logic."""
    base_config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "architecture": {
            "dim": 4096,
            "n_layers": 32
        }
    }
    
    override_config = {
        "architecture": {
            "dim": 2048,
            "n_heads": 16
        }
    }
    
    merged = merge_configs(base_config, override_config)
    assert merged["architecture"]["dim"] == 2048  # Override wins
    assert merged["architecture"]["n_layers"] == 32  # Base preserved
    assert merged["architecture"]["n_heads"] == 16  # New value added

def test_cli_override():
    """Test CLI argument overrides."""
    base_config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "quantization": {
            "pt2e_quantize": "xnnpack_dynamic"
        }
    }
    
    cli_args = Namespace(
        pt2e_quantize="xnnpack_dynamic_qc4",
        config=None
    )
    
    config = create_config(config_dict=base_config, cli_args=cli_args)
    assert config["quantization"]["pt2e_quantize"] == "xnnpack_dynamic_qc4"

def test_backend_compatibility():
    """Test backend compatibility validations."""
    config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "architecture": {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32
        },
        "limits": {
            "max_batch_size": 32,
            "max_seq_len": 2048,
            "max_context_len": 2048
        },
        "export": {
            "output_dir": "output",
            "enable_dynamic_shape": True
        },
        "kv_cache": {
            "enabled": True
        },
        "backends": {
            "coreml": {
                "enabled": True
            }
        }
    }
    
    # Should raise error due to dynamic shape + CoreML + KV cache incompatibility
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=config)
    assert "Dynamic shape is not supported with CoreML" in str(exc.value)

def test_quantization_validation():
    """Test quantization configuration validation."""
    config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "architecture": {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32
        },
        "limits": {
            "max_batch_size": 32,
            "max_seq_len": 2048,
            "max_context_len": 2048
        },
        "export": {
            "output_dir": "output"
        },
        "quantization": {
            "pt2e_quantize": "invalid_quantizer",
            "embedding_quantize": "invalid_format"
        }
    }
    
    # Test invalid quantizer
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=config)
    assert "quantization.pt2e_quantize must be one of" in str(exc.value)
    
    # Test invalid embedding quantization format
    config["quantization"]["pt2e_quantize"] = "xnnpack_dynamic"
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=config)
    assert "quantization.embedding_quantize must be in format" in str(exc.value)

def test_path_validation():
    """Test path validation in configuration."""
    config = {
        "model": {
            "name": "llama3",
            "type": "LLAMA"
        },
        "architecture": {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32
        },
        "limits": {
            "max_batch_size": 32,
            "max_seq_len": 2048,
            "max_context_len": 2048
        },
        "export": {
            "output_dir": "output",
            "checkpoint": 123,  # Should be string
            "checkpoint_dir": True  # Should be string
        }
    }
    
    with pytest.raises(ConfigValidationError) as exc:
        create_config(config_dict=config)
    assert "checkpoint must be a string path" in str(exc.value) 