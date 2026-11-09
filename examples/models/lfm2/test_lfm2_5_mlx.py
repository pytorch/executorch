import json
from pathlib import Path

from executorch.examples.models.llama.export_llama_lib import EXECUTORCH_DEFINED_MODELS
from executorch.extension.llm.export.config.model_identity import MODEL_SPECS, ModelType

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "models" / "lfm2" / "config"


def _load_json_config(name: str) -> dict:
    with open(CONFIG_DIR / name, "r") as f:
        return json.load(f)


def test_lfm2_5_models_are_registered() -> None:
    assert "lfm2_5_350m" in EXECUTORCH_DEFINED_MODELS
    assert "lfm2_5_1_2b" in EXECUTORCH_DEFINED_MODELS
    assert ModelType.lfm2_5_350m.value == "lfm2_5_350m"
    assert ModelType.lfm2_5_1_2b.value == "lfm2_5_1_2b"
    assert MODEL_SPECS["lfm2_5_350m"].hf_repo_id == "LiquidAI/LFM2.5-350M"
    assert MODEL_SPECS["lfm2_5_1_2b"].hf_repo_id == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_lfm2_5_architecture_configs_match_expected_shapes() -> None:
    expected = {
        "lfm2_5_350m_config.json": {
            "dim": 1024,
            "hidden_dim": 4608,
            "n_heads": 16,
            "n_kv_heads": 8,
        },
        "lfm2_5_1_2b_config.json": {
            "dim": 2048,
            "hidden_dim": 8192,
            "n_heads": 32,
            "n_kv_heads": 8,
        },
    }

    for filename, expected_fields in expected.items():
        cfg = _load_json_config(filename)
        for key, value in expected_fields.items():
            assert cfg[key] == value
        assert cfg["n_layers"] == 16
        assert len(cfg["layer_types"]) == cfg["n_layers"]
        assert cfg["layer_types"].count("full_attention") == 6
        assert cfg["layer_types"].count("conv") == 10
        assert cfg["vocab_size"] == 65536
        assert cfg["rope_theta"] == 1000000.0
        assert cfg["use_hf_rope"] is True
        assert cfg["use_qk_norm"] is True
        assert cfg["qk_norm_before_rope"] is True


def test_lfm2_mlx_config_enables_mlx_backend() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "lfm2_mlx_4w.yaml")
    assert cfg.base.metadata == '{"get_bos_id": 1, "get_eos_ids":[7]}'
    assert cfg.model.use_kv_cache is True
    assert cfg.model.use_sdpa_with_kv_cache is True
    assert cfg.model.dtype_override == "bf16"
    assert cfg.quantization.qmode == "4w"
    assert cfg.quantization.group_size == 64
    assert cfg.backend.mlx.enabled is True
