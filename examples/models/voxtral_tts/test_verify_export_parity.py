from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def _load_parity_module():
    module_path = Path(__file__).resolve().with_name("verify_export_parity.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("voxtral_verify_export_parity", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_requested_methods_all_includes_token_embedding() -> None:
    module = _load_parity_module()

    methods = module.resolve_requested_methods("all")

    assert methods == {
        "token_embedding",
        "text_decoder",
        "semantic_head",
        "predict_velocity",
        "audio_token_embedding",
    }


def test_apply_quantization_matches_export_policy(monkeypatch) -> None:
    module = _load_parity_module()
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        module,
        "quantize_model_",
        lambda target, **kwargs: calls.append((target.__class__.__name__, kwargs)),
    )

    fake_decoder = SimpleNamespace(tok_embeddings=object())
    fake_model = SimpleNamespace(
        decoder=fake_decoder,
        flow_head=SimpleNamespace(),
        audio_token_embedding=object(),
    )

    module.apply_quantization(
        fake_model,
        qlinear="4w",
        qlinear_group_size=128,
        qlinear_packing_format="opaque",
        qembedding="8w",
        qembedding_group_size=64,
    )

    assert calls == [
        (
            "SimpleNamespace",
            {
                "qlinear_config": "4w",
                "qlinear_group_size": 128,
                "qlinear_packing_format": "opaque",
            },
        ),
        (
            "SimpleNamespace",
            {
                "qlinear_config": "4w",
                "qlinear_group_size": 128,
                "qlinear_packing_format": "opaque",
                "skip_incompatible_shapes": True,
            },
        ),
        (
            "TokenEmbeddingExport",
            {
                "qembedding_config": "8w",
                "qembedding_group_size": 64,
            },
        ),
        (
            "AudioTokenEmbeddingExport",
            {
                "qembedding_config": "8w",
                "qembedding_group_size": 64,
            },
        ),
    ]


def test_apply_quantization_can_scope_decoder_to_attention(monkeypatch) -> None:
    module = _load_parity_module()
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
    fake_decoder = SimpleNamespace(tok_embeddings=object(), layers=[layer0, layer1])
    fake_model = SimpleNamespace(
        decoder=fake_decoder,
        flow_head=SimpleNamespace(_label="flow_head"),
        audio_token_embedding=object(),
    )

    module.apply_quantization(
        fake_model,
        qlinear="8da4w",
        qlinear_group_size=32,
        qlinear_packing_format=None,
        qembedding=None,
        qembedding_group_size=None,
        decoder_qlinear_scope="attention",
    )

    assert calls == [
        (
            "attn0",
            {
                "qlinear_config": "8da4w",
                "qlinear_group_size": 32,
                "qlinear_packing_format": None,
            },
        ),
        (
            "attn1",
            {
                "qlinear_config": "8da4w",
                "qlinear_group_size": 32,
                "qlinear_packing_format": None,
            },
        ),
        (
            "flow_head",
            {
                "qlinear_config": "8da4w",
                "qlinear_group_size": 32,
                "qlinear_packing_format": None,
                "skip_incompatible_shapes": True,
            },
        ),
    ]


def test_build_export_and_runtime_modules_uses_requested_backend(monkeypatch, tmp_path: Path) -> None:
    module = _load_parity_module()
    lower_backends: list[str] = []

    class FakeExportedProgram:
        def module(self):
            return "exported-module"

    class FakeExecutorchProgram:
        def write_to_file(self, file_obj) -> None:
            file_obj.write(b"pte")

    monkeypatch.setattr(module, "export", lambda *args, **kwargs: FakeExportedProgram())
    monkeypatch.setattr(
        module,
        "lower_to_executorch",
        lambda programs, metadata, backend: lower_backends.append(backend) or FakeExecutorchProgram(),
    )
    monkeypatch.setattr(module, "_load_for_executorch", lambda path: {"path": path})
    monkeypatch.setattr(module.gc, "collect", lambda: None)

    config = SimpleNamespace(dim=4, n_codebooks=37, acoustic_dim=36)
    fake_model = SimpleNamespace(
        config=config,
        decoder=SimpleNamespace(tok_embeddings=torch.nn.Identity()),
    )

    export_modules, runtime_modules = module.build_export_and_runtime_modules(
        fake_model,
        {"token_embedding"},
        max_seq_len=16,
        backend="xnnpack",
        temp_dir=tmp_path,
        temp_prefix="quantized",
    )

    assert lower_backends == ["xnnpack"]
    assert export_modules == {"token_embedding": "exported-module"}
    assert runtime_modules["token_embedding"]["path"].endswith("quantized_token_embedding.pte")


def test_semantic_triplet_report_returns_stage_metrics_and_topk() -> None:
    module = _load_parity_module()

    eager = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
    export = torch.tensor([[0.1, 0.7, 0.3]], dtype=torch.float32)
    runtime = torch.tensor([[0.05, 0.8, 0.2]], dtype=torch.float32)

    report, topk = module.semantic_triplet_report(
        eager,
        export,
        runtime,
        atol=0.15,
    )

    assert report["eager_vs_export"]["ok"] is False
    assert report["eager_vs_runtime"]["ok"] is True
    assert topk["eager"][0] == {"id": 1, "logit": 0.8999999761581421}
    assert topk["export"][0] == {"id": 1, "logit": 0.699999988079071}
    assert topk["runtime"][0] == {"id": 1, "logit": 0.800000011920929}
