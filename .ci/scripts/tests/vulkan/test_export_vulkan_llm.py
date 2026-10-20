# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import importlib
import importlib.util
import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]


def load_existing_module(relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location("existing_export_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_download_cache_is_writable_with_read_only_runner_cache(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulkan.yml").read_text())
    script = workflow["jobs"]["test-model-vulkan-e2e"]["with"]["script"]
    setup, _ = script.split("source .ci/scripts/setup-vulkan-linux-deps.sh", 1)
    shared_cache = tmp_path / "shared-cache"
    shared_hub = shared_cache / "hub"
    shared_hub.mkdir(parents=True)
    shared_hub.chmod(0o555)
    shared_cache.chmod(0o555)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probe = """
import json
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils._fixes import WeakFileLock

cache = Path(HF_HUB_CACHE)
lock = cache / ".locks" / "test.lock"
lock.parent.mkdir(parents=True, exist_ok=True)
with WeakFileLock(str(lock)):
    (cache / "download-probe").write_text("download")
print(json.dumps({"cache": str(cache), "wrote": True}))
"""
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                setup + f"\n{shlex.quote(sys.executable)} - <<'PY'\n{probe}\nPY",
            ],
            env={
                **os.environ,
                "RUNNER_TEMP": str(scratch),
                "HF_HOME": str(shared_cache),
                "HF_HUB_CACHE": str(shared_hub),
            },
            capture_output=True,
            text=True,
        )
    finally:
        shared_cache.chmod(0o755)
        shared_hub.chmod(0o755)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "cache": str(scratch / "hf_cache" / "hub"),
        "wrote": True,
    }
    assert not (shared_hub / "download-probe").exists()


def test_existing_hf_download_accepts_pinned_revision_and_propagates_failure(
    monkeypatch,
):
    module = load_existing_module("examples/models/llama/hf_download.py")

    def failed_download(**kwargs):
        assert kwargs["repo_id"] == "test/model"
        assert kwargs["revision"] == "r" * 40
        assert kwargs["token"] is False
        raise RuntimeError("download failed")

    monkeypatch.setattr(module, "snapshot_download", failed_download)
    with pytest.raises(RuntimeError, match="download failed"):
        module.download_hf_snapshot("test/model", "r" * 40)


def test_existing_hf_download_rejects_corrupt_weights_and_records_hash(
    tmp_path, monkeypatch
):
    module = load_existing_module("examples/models/llama/hf_download.py")
    corrupt = b"corrupt weights"
    (tmp_path / "model.safetensors").write_bytes(corrupt)
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"model": "test/model"}')
    monkeypatch.setattr(module, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(ValueError, match="SHA256"):
        module.download_hf_snapshot(
            "test/model",
            "r" * 40,
            expected_sha256=hashlib.sha256(b"original weights").hexdigest(),
            metadata_path=metadata,
        )
    assert json.loads(metadata.read_text())["observed_checkpoint_sha256"] == (
        hashlib.sha256(corrupt).hexdigest()
    )


def test_existing_hf_download_accepts_matching_hash(tmp_path, monkeypatch):
    module = load_existing_module("examples/models/llama/hf_download.py")
    weights = b"original weights"
    (tmp_path / "model.safetensors").write_bytes(weights)
    monkeypatch.setattr(module, "snapshot_download", lambda **kwargs: str(tmp_path))
    assert module.download_hf_snapshot(
        "test/model", "r" * 40, expected_sha256=hashlib.sha256(weights).hexdigest()
    ) == str(tmp_path)


def test_snapshot_cli_exposes_pinned_download_and_checksum(
    tmp_path, monkeypatch, capsys
):
    hub = importlib.import_module("huggingface_hub")
    weights = b"checkpoint"
    (tmp_path / "model.safetensors").write_bytes(weights)
    metadata = tmp_path / "metadata.json"
    checksum = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_download",
            "snapshot",
            "test/model",
            "--revision",
            "r" * 40,
            "--expected-sha256",
            checksum,
            "--metadata-path",
            str(metadata),
        ],
    )
    runpy.run_module(
        "executorch.examples.models.llama.hf_download", run_name="__main__"
    )
    assert capsys.readouterr().out.strip() == str(tmp_path)
    assert json.loads(metadata.read_text())["observed_checkpoint_sha256"] == checksum


def test_download_and_convert_cli_exposes_converter_callback(
    tmp_path, monkeypatch, capsys
):
    import torch
    from safetensors.torch import save_file

    hub = importlib.import_module("huggingface_hub")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "tie_word_embeddings": True,
            }
        )
    )
    embeddings = torch.arange(80, dtype=torch.float32).view(10, 8)
    save_file({"model.embed_tokens.weight": embeddings}, snapshot / "model.safetensors")
    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: str(snapshot))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_download",
            "download-and-convert",
            "test/model",
            "--converter",
            "executorch.examples.models.llama.convert_weights:convert_hf_to_meta",
        ],
    )
    runpy.run_module(
        "executorch.examples.models.llama.hf_download", run_name="__main__"
    )
    output = cache / ".cache/meta_checkpoints/test_model.pth"
    assert capsys.readouterr().out.strip() == str(output)
    torch.testing.assert_close(
        torch.load(output, weights_only=True)["tok_embeddings.weight"], embeddings
    )

    def unexpected_download(**kwargs):
        raise AssertionError("cached conversion must not redownload")

    monkeypatch.setattr(hub, "snapshot_download", unexpected_download)
    runpy.run_module(
        "executorch.examples.models.llama.hf_download", run_name="__main__"
    )
    assert capsys.readouterr().out.strip() == str(output)


def test_vulkan_artifact_inspection_does_not_import_runtime_bindings(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "executorch.extension.pybindings.portable_lib", None
    )
    module = load_existing_module("backends/vulkan/test/utils.py")
    assert callable(module.verify_vulkan_llm_pte)


@pytest.fixture
def artifact_inspector(tmp_path, monkeypatch):
    module = load_existing_module("backends/vulkan/test/utils.py")
    schema = importlib.import_module("executorch.exir.schema")
    deserializer = importlib.import_module("executorch.exir._serialize._program")
    serializer = importlib.import_module(
        "executorch.backends.vulkan.serialization.vulkan_graph_serialize"
    )
    forward = SimpleNamespace(
        name="forward",
        delegates=[
            SimpleNamespace(id="VulkanBackend", processed=SimpleNamespace(index=0))
        ],
        chains=[
            SimpleNamespace(
                instructions=[SimpleNamespace(instr_args=schema.DelegateCall(0, []))]
            )
        ],
    )
    program = SimpleNamespace(
        execution_plan=[forward],
        backend_delegate_data=[SimpleNamespace(data=b"vulkan")],
    )
    operators = [
        SimpleNamespace(name="et_vk.linear_dq8ca_q4gsw.default"),
        SimpleNamespace(name="et_vk.embedding_q4gsw.default"),
        SimpleNamespace(name="sdpa_with_kv_cache.default"),
    ]
    monkeypatch.setattr(
        deserializer,
        "deserialize_pte_binary",
        lambda _: SimpleNamespace(program=program),
    )
    monkeypatch.setattr(serializer, "extract_vk_flatbuffer", lambda data: data)
    monkeypatch.setattr(
        serializer, "flatbuffer_to_vk_graph", lambda _: SimpleNamespace(chain=operators)
    )
    model = tmp_path / "model.pte"
    model.write_bytes(b"model")
    return module, model, program, operators


def test_required_called_vulkan_operators_pass_and_record_counts(artifact_inspector):
    module, model, _, _ = artifact_inspector
    module.verify_vulkan_llm_pte(model, quantized_embedding=True)
    assert json.loads((model.parent / "delegation.json").read_text()) == {
        "forward_delegates": ["VulkanBackend"]
    }
    assert (
        json.loads((model.parent / "vulkan_operators.json").read_text())[
            "sdpa_with_kv_cache.default"
        ]
        == 1
    )


@pytest.mark.parametrize("failure", ["cpu_only", "unused", "metadata_only"])
def test_nonexecuted_vulkan_delegate_cannot_pass(artifact_inspector, failure):
    module, model, program, _ = artifact_inspector
    forward = program.execution_plan[0]
    if failure == "unused":
        forward.chains.clear()
    else:
        forward.delegates[0].id = "XnnpackBackend"
    if failure == "metadata_only":
        program.execution_plan.insert(
            0,
            SimpleNamespace(
                name="get_vocab_size", delegates=[SimpleNamespace(id="VulkanBackend")]
            ),
        )
    with pytest.raises(ValueError, match="called VulkanBackend"):
        module.verify_vulkan_llm_pte(model)


@pytest.mark.parametrize(
    "operator",
    [
        "et_vk.linear_dq8ca_q4gsw.default",
        "et_vk.embedding_q4gsw.default",
        "sdpa_with_kv_cache.default",
    ],
)
def test_partial_lowering_missing_required_operator_fails(artifact_inspector, operator):
    module, model, _, operators = artifact_inspector
    operators[:] = [op for op in operators if op.name != operator]
    with pytest.raises(ValueError, match="Missing Vulkan operator family"):
        module.verify_vulkan_llm_pte(model, quantized_embedding=True)


def test_weight_only_case_does_not_require_dynamic_quantization_or_embedding(
    artifact_inspector,
):
    module, model, _, operators = artifact_inspector
    operators[:] = [
        SimpleNamespace(name="et_vk.linear_q4gsw.default"),
        SimpleNamespace(name="sdpa_with_kv_cache.default"),
    ]
    module.verify_vulkan_llm_pte(model, qmode="4w")


def test_model_size_limit_is_enforced_before_deserialization(artifact_inspector):
    module, model, _, _ = artifact_inspector
    with model.open("wb") as output:
        output.truncate(1024 * 1024 + 1)
    with pytest.raises(ValueError, match="1 MiB"):
        module.verify_vulkan_llm_pte(model, max_size_mib=1)
