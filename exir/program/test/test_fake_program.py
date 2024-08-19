# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest

import torch

from executorch.exir.program._fake_program import (
    get_fake_program,
    update_to_real_program,
)
from torch.export import export, ExportedProgram


def get_exported_program() -> ExportedProgram:
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.register_buffer("buf", torch.randn(10, 10), persistent=False)

        def forward(self, arg) -> torch.Tensor:
            return self.linear(arg) + self.buf

    linear = Linear()
    exported_program = export(
        linear,
        args=(torch.randn(10, 10),),
    ).run_decompositions()
    return exported_program


class TestFakeProgram(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_fake_program(self) -> None:
        exported_program = get_exported_program()
        fake_program = get_fake_program(exported_program)
        print(f"Exported program size: {sys.getsizeof(exported_program.state_dict)}")
        print(f"Fake program size: {sys.getsizeof(fake_program.state_dict)}")

        # Fake program deep copies attributes besides verifier, state_dict and constants.
        self.assertEqual(exported_program.graph_signature, fake_program.graph_signature)
        self.assertNotEqual(
            id(exported_program.graph_signature), id(fake_program.graph_signature)
        )
        self.assertEqual(
            exported_program.module_call_graph, fake_program.module_call_graph
        )
        self.assertNotEqual(
            id(exported_program.module_call_graph), id(fake_program.module_call_graph)
        )

        # Verifier is static.
        self.assertEqual(exported_program.verifier, fake_program.verifier)
        self.assertEqual(id(exported_program.verifier), id(fake_program.verifier))

        # Fake program uses fake tensors for the state dict. Size should be smaller.
        self.assertLess(
            sys.getsizeof(fake_program.state_dict),
            sys.getsizeof(exported_program.state_dict),
        )

        # Do not copy constants.
        self.assertEqual(exported_program.constants, fake_program.constants)
        self.assertEqual(id(exported_program.constants), id(fake_program.constants))

        update_to_real_program(fake_program, exported_program)
        self.assertEqual(exported_program.state_dict, fake_program.state_dict)
        self.assertEqual(
            exported_program.state_dict.keys(), fake_program.state_dict.keys()
        )
