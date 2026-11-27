# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check pruning layout and mask persistence through PT2E preparation."""

import csv
import json
import sys

import pytest
import torch

from examples.arm.QAT_example import qat_loop
from examples.arm.QAT_example.pruning import TwoOfFourPruner
from examples.arm.QAT_example.qat_loop import (
    build_ptq_model,
    build_qat_model,
    prepare_cached_model,
)
from PIL import Image


def test_magnitude_pruning_groups_input_channels_at_each_kernel_position():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(8, 2, 2, bias=False),
        torch.nn.Conv2d(2, 2, 1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 65).reshape(2, 8, 2, 2))
        model[0].weight[:, 3::4].neg_()
    original = model[0].weight.detach().clone()
    dense = model[1].weight.detach().clone()
    TwoOfFourPruner(model, ["0"])
    expected = original.clone()
    expected[:, 0::4] = 0
    expected[:, 1::4] = 0
    torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(model[1].weight, dense, rtol=0, atol=0)
    assert not model[0]._forward_pre_hooks
    graph = torch.export.export(model, (torch.rand(1, 8, 4, 4),)).graph
    assert all(node.target != torch.ops.aten.masked_fill.Scalar for node in graph.nodes)


def test_masks_are_stable_and_restored_after_optimizer_updates():
    model = torch.nn.Sequential(torch.nn.Conv2d(8, 4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1)
    pruner = TwoOfFourPruner(model, ["0"])
    expected = model[0].weight.detach().clone()
    assert torch.all(expected[:, [0, 1, 4, 5]] == 1)
    assert torch.all(expected[:, [2, 3, 6, 7]] == 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(1, 8, 2, 2)).sum().backward()
        optimizer.step()
        pruner.apply_(model)
        assert torch.all(model[0].weight[expected == 0] == 0)
    assert not torch.equal(model[0].weight[expected != 0], expected[expected != 0])


@pytest.mark.parametrize("pattern", ["missing", "1", "2", "*"])
def test_invalid_selection_does_not_partially_prune(pattern):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(3, 4, 1),
    )
    before = model[0].weight.detach().clone()
    with pytest.raises(ValueError):
        TwoOfFourPruner(model, ["0", pattern])
    torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)


class _Model(torch.nn.Module):
    def __init__(self, input_channels=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, 8, 3, padding=1)
        self.head = torch.nn.Conv2d(8, 3, 1)

    def forward(self, previous, current):
        return self.head(torch.relu(self.conv(torch.cat((previous, current), dim=1))))


class _CachedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flownet = torch.nn.Module()
        self.flownet.encode = torch.nn.Conv2d(4, 4, 1)
        self.flownet.blend = torch.nn.Conv2d(4, 3, 1)

    def forward(self, previous, current, features):
        return self.flownet.blend(features), self.flownet.encode(current)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("qat", [False, True])
def test_pt2e_preserves_sparse_weights_and_dense_source_model(cache, qat):
    torch.manual_seed(2026)
    model = (_CachedModel() if cache else _Model()).eval()
    dense = {name: p.detach().clone() for name, p in model.named_parameters()}
    frames = (torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8))
    target = torch.rand(1, 3, 8, 8)
    if cache:
        converted = prepare_cached_model(
            model,
            [(frames, target)],
            is_qat=qat,
            io_quantization="int8",
            steps=2,
            lr=1e-2,
            prune_layers=["flownet.blend"],
        )
    elif qat:
        converted = build_qat_model(
            model,
            [(frames, target)],
            steps=2,
            lr=1e-2,
            io_quantization="int8",
            prune_layers=["conv"],
        ).model
    else:
        converted = build_ptq_model(
            model,
            [frames],
            io_quantization="int8",
            prune_layers=["conv"],
        ).model
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dense[name], rtol=0, atol=0)
    node = next(
        n for n in converted.graph.nodes if n.target == torch.ops.aten.conv2d.default
    )
    dequantize = node.args[1]
    codes = converted.get_buffer(dequantize.args[0].target)
    assert torch.all(converted.get_buffer(dequantize.args[2].target) == 0)
    groups = codes.permute(0, 2, 3, 1).reshape(-1, 4)
    assert torch.all(torch.count_nonzero(groups, dim=-1) <= 2)


@pytest.mark.parametrize(
    "mode,prune", [("eager", False), ("eager", True), ("ptq", True), ("all", True)]
)
def test_accuracy_flow_reports_pruned_fp32_and_preserves_dense_reference(
    mode, prune, tmp_path, monkeypatch, capsys
):
    torch.manual_seed(2026)
    model = _Model(input_channels=6).eval()
    with torch.no_grad():
        model.head.weight.fill_(0.01)
        model.head.bias.fill_(0.1)
    dense = {name: p.detach().clone() for name, p in model.named_parameters()}
    monkeypatch.setattr(qat_loop, "load_rife_model", lambda *_: model)
    paths = []
    for index in range(3):
        path = tmp_path / f"frame{index}.png"
        Image.fromarray((torch.rand(64, 64, 3) * 255).to(torch.uint8).numpy()).save(
            path
        )
        paths.append(path)
    triples = tmp_path / "triples.csv"
    with triples.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "input0", "input1", "target"])
        writer.writerow(["pair", paths[0], paths[2], paths[1]])
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qat_loop.py",
            "--model-root",
            str(tmp_path),
            "--triples-list",
            str(triples),
            "--height",
            "64",
            "--width",
            "64",
            "--mode",
            mode,
            "--qat-steps",
            "1",
            "--output-dir",
            str(output),
        ]
        + (["--prune-2-4", "head"] if prune else []),
    )
    assert qat_loop.main() == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dense[name], rtol=0, atol=0)
    payload = json.loads((output / "metrics.json").read_text())
    row = payload["rows"][0]
    aggregate = payload["aggregate"]
    with (output / "metrics.csv").open(newline="") as file:
        csv_row = next(csv.DictReader(file))
    report = (output / "report.md").read_text()
    console = capsys.readouterr().out
    assert "eager_psnr" in row
    assert ("eager_pruned_psnr" in row) == prune
    assert ("Pruned eager FP32 PSNR" in report) == prune
    assert ("eager_pruned:" in console) == prune
    if prune:
        assert row["eager_pruned_vs_eager_mae"] > 0
        assert aggregate["eager_pruned_psnr_mean"] == row["eager_pruned_psnr"]
        assert float(csv_row["eager_pruned_psnr"]) == row["eager_pruned_psnr"]
    if mode == "eager":
        assert payload["quantization_coverage"] == {}
        assert "ptq_psnr" not in row
        assert "qat_psnr" not in row
    else:
        for prefix in ("ptq", "qat") if mode == "all" else ("ptq",):
            assert f"{prefix}_vs_eager_mae" in row
            assert f"{prefix}_vs_eager_pruned_mae_mean" in aggregate
            assert f"{prefix.upper()} vs pruned eager MAE" in report
