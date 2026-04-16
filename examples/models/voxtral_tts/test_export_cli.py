from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_export_module():
    module_path = Path(__file__).resolve().with_name("export_voxtral_tts.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("voxtral_tts_export", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_xnnpack_disables_embedding_quantization() -> None:
    module = _load_export_module()

    plan = module.resolve_effective_quantization(
        backend="xnnpack",
        qlinear="4w",
        qembedding="4w",
    )

    assert plan["qlinear"] == "4w"
    assert plan["qembedding"] is None
    assert "embedding" in plan["warning"]
    assert "xnnpack" in plan["warning"].lower()


def test_portable_preserves_embedding_quantization() -> None:
    module = _load_export_module()

    plan = module.resolve_effective_quantization(
        backend="portable",
        qlinear="4w",
        qembedding="8w",
    )

    assert plan == {
        "qlinear": "4w",
        "qembedding": "8w",
        "warning": None,
    }


def test_apply_model_quantization_can_scope_decoder_to_feed_forward(monkeypatch) -> None:
    module = _load_export_module()
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        module,
        "quantize_model_",
        lambda target, **kwargs: calls.append(
            (getattr(target, "_label", target.__class__.__name__), kwargs)
        ),
    )

    layer0 = SimpleNamespace(
        attention=SimpleNamespace(_label="attn0"),
        feed_forward=SimpleNamespace(_label="ffn0"),
    )
    layer1 = SimpleNamespace(
        attention=SimpleNamespace(_label="attn1"),
        feed_forward=SimpleNamespace(_label="ffn1"),
    )
    fake_model = SimpleNamespace(
        decoder=SimpleNamespace(layers=[layer0, layer1]),
        flow_head=SimpleNamespace(_label="flow_head"),
        audio_token_embedding=SimpleNamespace(_label="audio_embed"),
    )

    module.apply_model_quantization(
        fake_model,
        qlinear="8da8w",
        qlinear_group_size=64,
        qlinear_packing_format=None,
        qembedding=None,
        qembedding_group_size=None,
        decoder_qlinear_scope="feed_forward",
    )

    assert calls == [
        (
            "ffn0",
            {
                "qlinear_config": "8da8w",
                "qlinear_group_size": 64,
                "qlinear_packing_format": None,
            },
        ),
        (
            "ffn1",
            {
                "qlinear_config": "8da8w",
                "qlinear_group_size": 64,
                "qlinear_packing_format": None,
            },
        ),
        (
            "flow_head",
            {
                "qlinear_config": "8da8w",
                "qlinear_group_size": 64,
                "qlinear_packing_format": None,
                "skip_incompatible_shapes": True,
            },
        ),
    ]
