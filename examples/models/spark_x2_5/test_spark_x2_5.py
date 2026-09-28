import ast
import json
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "models" / "spark_x2_5" / "config"
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


def test_spark_x2_5_models_are_registered() -> None:
    export_module = _module_ast(EXPORT_LLAMA_LIB)
    model_types = _class_string_assignments(_module_ast(LLM_CONFIG), "ModelType")
    executor_defined_models = _literal_assignment(
        export_module, "EXECUTORCH_DEFINED_MODELS"
    )
    hf_repo_ids = _literal_assignment(export_module, "HUGGING_FACE_REPO_IDS")

    assert "spark_x2_5_1_7b" in executor_defined_models
    assert "spark_x2_5_4b" in executor_defined_models
    assert model_types["spark_x2_5_1_7b"] == "spark_x2_5_1_7b"
    assert model_types["spark_x2_5_4b"] == "spark_x2_5_4b"
    assert hf_repo_ids["spark_x2_5_1_7b"] == "XHToken/Spark-X2.5-1.7B"
    assert hf_repo_ids["spark_x2_5_4b"] == "XHToken/Spark-X2.5-4B"


def test_spark_x2_5_architecture_configs_match_expected_shapes() -> None:
    expected = {
        "spark_x2_5_1_7b_config.json": {
            "dim": 2048,
            "hidden_dim": 6656,
            "n_heads": 8,
            "n_kv_heads": 2,
            "head_dim": 256,
        },
        "spark_x2_5_4b_config.json": {
            "dim": 2560,
            "hidden_dim": 10240,
            "n_heads": 16,
            "n_kv_heads": 4,
            "head_dim": 256,
        },
    }

    for filename, expected_fields in expected.items():
        cfg = _load_json_config(filename)
        for key, value in expected_fields.items():
            assert (
                cfg[key] == value
            ), f"{filename}: {key} expected {value}, got {cfg[key]}"

    # 1.7B: 28 layers = 7 groups of 4
    cfg_1_7b = _load_json_config("spark_x2_5_1_7b_config.json")
    assert cfg_1_7b["n_layers"] == 28
    assert len(cfg_1_7b["layer_types"]) == 28
    assert cfg_1_7b["layer_types"].count("full_attention") == 7
    assert cfg_1_7b["layer_types"].count("sliding_attention") == 21

    # 4B: 36 layers = 9 groups of 4
    cfg_4b = _load_json_config("spark_x2_5_4b_config.json")
    assert cfg_4b["n_layers"] == 36
    assert len(cfg_4b["layer_types"]) == 36
    assert cfg_4b["layer_types"].count("full_attention") == 9
    assert cfg_4b["layer_types"].count("sliding_attention") == 27

    # Shared architecture features.
    for filename in expected:
        cfg = _load_json_config(filename)
        assert cfg["vocab_size"] == 131072
        assert cfg["sliding_window"] == 512
        assert cfg["use_hf_rope"] is True
        assert cfg["headwise_attn_output_gate"] is True
        assert cfg["act_fn"] == "gelu"
        assert cfg["rope_parameters"]["full_attention"]["rope_theta"] == 5000000
        assert cfg["rope_parameters"]["full_attention"]["partial_rotary_factor"] == 0.25
        assert cfg["rope_parameters"]["sliding_attention"]["rope_theta"] == 10000
        assert (
            cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] == 1.0
        )


def test_spark_x2_5_xnnpack_q8da4w_config_enables_xnnpack_backend() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "spark_x2_5_xnnpack_q8da4w.yaml")
    assert cfg.base.metadata == '{"get_bos_id": 0, "get_eos_ids":[1]}'
    assert cfg.model.use_kv_cache is True
    assert cfg.model.use_sdpa_with_kv_cache is True
    assert cfg.model.dtype_override == "fp32"
    assert cfg.quantization.qmode == "8da4w"
    assert cfg.backend.xnnpack.enabled is True
