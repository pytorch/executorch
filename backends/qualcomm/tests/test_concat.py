# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.backends.qualcomm.quantizer.observers.concat_observer import (
    ConcatObserver,
)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from torchao.quantization.pt2e import FakeQuantizeBase
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

_CAT = torch.ops.aten.cat.default


class DeepCat(torch.nn.Module):
    """A cat at the end of a node chain long enough to exhaust the default
    recursion limit if the chain is walked recursively."""

    def __init__(self, depth: int = 300):
        super().__init__()
        self.depth = depth

    def forward(self, x, y):
        for _ in range(self.depth):
            x = x + 0.5
        return torch.cat([x, y], dim=1)


class TwoRangeCat(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x + 1.0, y * 4.0], dim=1)


def _prepare(module, example_inputs, is_qat=False):
    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(QuantDtype.use_8a8w, is_qat=is_qat)
    exported = torch.export.export(module, example_inputs, strict=True).module()
    prepare = prepare_qat_pt2e if is_qat else prepare_pt2e
    prepared = prepare(exported, quantizer)
    prepared(*example_inputs)
    return prepared


def _named(module, cls):
    return [(n, m) for n, m in module.named_modules() if isinstance(m, cls)]


def _only_cat(gm):
    [cat] = [n for n in gm.graph.nodes if n.target == _CAT]
    return cat


class ConcatObserverDeepcopyTest(unittest.TestCase):
    """`copy.deepcopy` of a prepared model, as done before `convert_pt2e` by
    training frameworks, must not walk the graph through `ConcatObserver`'s
    `fx.Node` references."""

    def setUp(self):
        self.example_inputs = (torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        self.prepared = _prepare(DeepCat().eval(), self.example_inputs)

    def test_deepcopy_and_convert(self):
        copied = copy.deepcopy(self.prepared)
        convert_pt2e(copied)

    def test_copy_input_observers_point_into_the_copy(self):
        [(name, observer)] = _named(self.prepared, ConcatObserver)
        self.assertEqual(len(observer.input_observers), 2)
        names = {id(m): n for n, m in self.prepared.named_modules() if n}
        input_names = [names[id(obs)] for obs in observer.input_observers]

        copied = copy.deepcopy(self.prepared)
        copied_observer = copied.get_submodule(name)
        for input_name, copied_input in zip(
            input_names, copied_observer.input_observers
        ):
            self.assertIs(copied_input, copied.get_submodule(input_name))
        self.assertIs(copied_observer.concat_node, observer.concat_node)
        torch.testing.assert_close(copied_observer.min_val, observer.min_val)
        torch.testing.assert_close(copied_observer.max_val, observer.max_val)


class ConcatQatTest(unittest.TestCase):
    """Under QAT a cat must be one FakeQuantize shared by its inputs and output.

    `ConcatObserver` is a plain observer, so it never fake-quantizes the cat
    output, and it aligns input ranges by writing `min_val` onto the input
    observers, which a FakeQuantize ignores.
    """

    def setUp(self):
        self.example_inputs = (torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        self.prepared = _prepare(TwoRangeCat(), self.example_inputs, is_qat=True)

    def _edge_modules(self):
        cat = _only_cat(self.prepared)
        edges = list(cat.args[0]) + list(cat.users)
        return [self.prepared.get_submodule(n.target) for n in edges]

    def test_inputs_and_output_share_one_fake_quant(self):
        self.assertEqual(_named(self.prepared, ConcatObserver), [])
        modules = self._edge_modules()
        self.assertEqual(len(modules), 3)
        self.assertEqual(len({id(m) for m in modules}), 1)
        self.assertIsInstance(modules[0], FakeQuantizeBase)

    def test_convert_gives_inputs_and_output_one_qparam(self):
        converted = convert_pt2e(self.prepared)
        cat = _only_cat(converted)
        [quantize] = list(cat.users)
        qparams = {tuple(n.args[1:3]) for n in list(cat.args[0]) + [quantize]}
        self.assertEqual(len(qparams), 1, qparams)
