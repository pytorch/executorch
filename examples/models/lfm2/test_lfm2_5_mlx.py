import ast
import json
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "models" / "lfm2" / "config"
EXPORT_LLAMA_LIB = REPO_ROOT / "examples" / "models" / "llama" / "export_llama_lib.py"
LLM_CONFIG = REPO_ROOT / "extension" / "llm" / "export" / "config" / "llm_config.py"


def _load_json_config(name: str) -> dict:
    with open(CONFIG_DIR / name, "r") as f:
        return json.load(f)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _literal_assignment(module: ast.Module, name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _class_string_assignments(module: ast.Module, class_name: str) -> dict[str, str]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values = {}
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    values[stmt.targets[0].id] = ast.literal_eval(stmt.value)
            return values
    raise AssertionError(f"{class_name} not found")


def test_lfm2_5_models_are_registered() -> None:
    export_module = _module_ast(EXPORT_LLAMA_LIB)
    model_types = _class_string_assignments(_module_ast(LLM_CONFIG), "ModelType")
    executor_defined_models = _literal_assignment(
        export_module, "EXECUTORCH_DEFINED_MODELS"
    )
    hf_repo_ids = _literal_assignment(export_module, "HUGGING_FACE_REPO_IDS")

    assert "lfm2_5_350m" in executor_defined_models
    assert "lfm2_5_1_2b" in executor_defined_models
    assert model_types["lfm2_5_350m"] == "lfm2_5_350m"
    assert model_types["lfm2_5_1_2b"] == "lfm2_5_1_2b"
    assert hf_repo_ids["lfm2_5_350m"] == "LiquidAI/LFM2.5-350M"
    assert hf_repo_ids["lfm2_5_1_2b"] == "LiquidAI/LFM2.5-1.2B-Instruct"


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
