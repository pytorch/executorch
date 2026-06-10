# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.nn as nn
import torchao
from executorch.backends.cadence.aot.remove_ops import RemoveBNTrackingMutationsPass
from executorch.exir import to_edge
from torch.export.graph_signature import InputKind, OutputKind
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


class MultiBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)
        return self.fc(x)


class ReadBNTrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_batches_tracked = self.bn.num_batches_tracked
        assert num_batches_tracked is not None
        return self.bn(x) + num_batches_tracked.to(dtype=x.dtype)


def _qat_export_to_edge(
    model: nn.Module,
    example_input: tuple[torch.Tensor, ...],
) -> torch.export.ExportedProgram:
    """Simulate the QAT export path that produces BN tracking mutations.

    QAT models are traced in training mode (model.train()), then converted
    back to eval via move_exported_model_to_eval(). run_decompositions()
    in to_edge() then re-introduces num_batches_tracked mutations.
    """
    from executorch.backends.cadence.aot.quantizer.quantizer import (
        CadenceFusedConvReluQuantizer,
    )

    model.train()
    captured = torch.export.export(model, example_input, strict=False).module()
    prepared = prepare_pt2e(captured, CadenceFusedConvReluQuantizer(is_qat=True))

    for _ in range(3):
        prepared(*example_input)

    torchao.quantization.pt2e.move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)

    exported = torch.export.export(converted, example_input)
    edge = to_edge(exported)
    return edge.exported_program()


class RemoveBNTrackingMutationsTest(unittest.TestCase):
    def _get_nbt_mutations(self, ep: torch.export.ExportedProgram) -> dict[str, str]:
        return {
            k: v
            for k, v in ep.graph_signature.buffers_to_mutate.items()
            if "num_batches_tracked" in v
        }

    def _get_nbt_placeholders(self, ep: torch.export.ExportedProgram) -> list[str]:
        placeholders: list[str] = []
        for n in ep.graph_module.graph.nodes:
            if (
                n.op == "placeholder"
                and isinstance(n.target, str)
                and "num_batches_tracked" in n.target
            ):
                placeholders.append(n.target)
        return placeholders

    def _get_nbt_input_specs(self, ep: torch.export.ExportedProgram) -> list[str]:
        input_specs: list[str] = []
        for s in ep.graph_signature.input_specs:
            if (
                s.kind == InputKind.BUFFER
                and s.target is not None
                and "num_batches_tracked" in s.target
            ):
                input_specs.append(s.target)
        return input_specs

    def _run_remove_pass_on_qat_model(
        self,
        model: nn.Module,
        example_input: tuple[torch.Tensor, ...],
    ) -> torch.export.ExportedProgram:
        edge_ep = _qat_export_to_edge(model, example_input)
        nbt = self._get_nbt_mutations(edge_ep)
        self.assertGreater(
            len(nbt), 0, "expected pre-pass num_batches_tracked mutations"
        )

        result = RemoveBNTrackingMutationsPass()(edge_ep)
        self.assertTrue(result.modified)
        return result.exported_program

    def test_single_bn_no_tracking_mutations(self) -> None:
        model = SimpleBNModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))
        nbt = self._get_nbt_mutations(edge_ep)
        self.assertEqual(len(nbt), 0, f"num_batches_tracked mutations present: {nbt}")

    def test_multi_bn_no_tracking_mutations(self) -> None:
        model = MultiBNModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))
        nbt = self._get_nbt_mutations(edge_ep)
        self.assertEqual(len(nbt), 0, f"num_batches_tracked mutations present: {nbt}")

    def test_no_nbt_output_specs(self) -> None:
        model = MultiBNModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))
        nbt_specs = [
            s
            for s in edge_ep.graph_signature.output_specs
            if s.kind == OutputKind.BUFFER_MUTATION
            and s.target is not None
            and "num_batches_tracked" in s.target
        ]
        self.assertEqual(
            len(nbt_specs), 0, f"num_batches_tracked output specs present: {nbt_specs}"
        )

    def test_no_nbt_input_placeholders(self) -> None:
        """All num_batches_tracked input placeholders should be removed."""
        model = MultiBNModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))
        nbt_placeholders = self._get_nbt_placeholders(edge_ep)
        self.assertEqual(
            len(nbt_placeholders),
            0,
            f"num_batches_tracked placeholders still present: {nbt_placeholders}",
        )

    def test_no_nbt_input_specs(self) -> None:
        """No input_specs for num_batches_tracked buffers should remain."""
        model = MultiBNModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))
        nbt_input_specs = self._get_nbt_input_specs(edge_ep)
        self.assertEqual(
            len(nbt_input_specs),
            0,
            f"num_batches_tracked input specs still present: {nbt_input_specs}",
        )

    def test_live_nbt_input_spec_preserved(self) -> None:
        model = ReadBNTrackingModel()
        edge_ep = self._run_remove_pass_on_qat_model(model, (torch.randn(1, 3, 32),))

        nbt_placeholders = self._get_nbt_placeholders(edge_ep)
        nbt_input_specs = self._get_nbt_input_specs(edge_ep)
        self.assertGreater(
            len(nbt_placeholders),
            0,
            "expected live num_batches_tracked placeholder to remain",
        )
        self.assertEqual(len(nbt_placeholders), len(nbt_input_specs))

    def test_no_bn_model_unaffected(self) -> None:
        class NoBNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = NoBNModel()
        model.eval()
        ep = torch.export.export(model, (torch.randn(1, 8),))
        edge_ep = to_edge(ep).exported_program()
        result = RemoveBNTrackingMutationsPass()(edge_ep)
        self.assertFalse(result.modified)
        self.assertEqual(
            len(result.exported_program.graph_signature.buffers_to_mutate), 0
        )
