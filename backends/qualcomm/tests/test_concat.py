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
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


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


def _prepare(module, example_inputs):
    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(QuantDtype.use_8a8w, is_qat=False)
    exported = torch.export.export(module, example_inputs, strict=True).module()
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*example_inputs)
    return prepared


def _named(module, cls):
    return [(n, m) for n, m in module.named_modules() if isinstance(m, cls)]


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
