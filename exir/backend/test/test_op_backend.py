# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.op_backend import _lower_and_verify, OpBackend


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # atan is non-terminal, so retargeting it leaves the graph signature's
        # user outputs intact.
        return torch.relu(torch.atan(self.linear(x)))


def _edge_program() -> torch.export.ExportedProgram:
    exported = torch.export.export(_Model(), (torch.randn(1, 4),))
    return to_edge(exported).exported_program()


class TestToOpBackend(unittest.TestCase):
    def test_it_returns_what_the_backend_lowered(self) -> None:
        rewritten = _edge_program()

        class Backend(OpBackend):
            saw = None

            def lower(self, exported_program, method_name):
                self.saw = (method_name, exported_program)
                return rewritten

        backend = Backend()
        program = _edge_program()
        self.assertIs(_lower_and_verify(program, backend, "forward"), rewritten)
        # The backend is handed a copy, not the caller's program, so that
        # rewriting in place -- which most of them do -- cannot reach back.
        self.assertEqual(backend.saw[0], "forward")
        self.assertIsNot(backend.saw[1], program)

    def test_a_backend_that_returns_nothing_is_named(self) -> None:
        # The common authoring slip is a missing return; without this the
        # None travels on and fails somewhere unrelated.
        class Forgetful(OpBackend):
            def lower(self, exported_program, method_name):
                pass

        with self.assertRaisesRegex(
            TypeError, r"Forgetful\.lower\(\) .* got <class 'NoneType'>"
        ):
            _lower_and_verify(_edge_program(), Forgetful(), "forward")

    def test_an_unregistered_placeholder_is_named(self) -> None:
        class AddsRogueInput(OpBackend):
            def lower(self, exported_program, method_name):
                program = exported_program
                graph = program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("rogue")
                return program

        with self.assertRaisesRegex(
            ValueError,
            r"AddsRogueInput\.lower\(\) on 'forward' returned an inconsistent "
            r"program: Number of graph inputs \(4\)",
        ):
            _lower_and_verify(_edge_program(), AddsRogueInput(), "forward")

    def test_the_caller_s_program_is_untouched(self) -> None:
        # Backends are built from passes that rewrite in place, so the copy is
        # what makes the contract real rather than a request.
        class Vandal(OpBackend):
            def lower(self, exported_program, method_name):
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("rogue")
                exported_program.state_dict["rogue"] = torch.ones(1)
                raise RuntimeError("gave up half way")

        program = _edge_program()
        nodes, weights = len(list(program.graph.nodes)), sorted(program.state_dict)
        with self.assertRaises(RuntimeError):
            _lower_and_verify(program, Vandal(), "forward")
        self.assertEqual(len(list(program.graph.nodes)), nodes)
        self.assertEqual(sorted(program.state_dict), weights)

    def test_adding_a_constant_does_not_reach_the_caller(self) -> None:
        # create_constant_placeholder, which lower()'s contract points at,
        # inserts into graph_signature.input_specs. Sharing the signature would
        # land that insert in the caller's program, which then fails to emit
        # with a message-less assertion.
        from executorch.backends.transforms.utils import create_constant_placeholder
        from torch.export.graph_signature import InputKind

        class AddsConstant(OpBackend):
            def lower(self, exported_program, method_name):
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    create_constant_placeholder(
                        exp_program=exported_program,
                        graph=graph,
                        kind=InputKind.BUFFER,
                        name="added_by_the_backend",
                        data=torch.ones(1),
                        persistent_buffer=True,
                    )
                return exported_program

        manager = _manager()
        program = manager.exported_program()
        before = len(program.graph_signature.input_specs)

        manager.to_op_backend(AddsConstant())

        self.assertEqual(len(program.graph_signature.input_specs), before)
        self.assertNotIn("added_by_the_backend", program.state_dict)
        self.assertGreater(len(manager.to_executorch().buffer), 0)

    def test_editing_an_input_spec_does_not_reach_the_caller(self) -> None:
        # InputSpec is a mutable dataclass, so giving the signature its own
        # lists is not enough; the entries have to be its own too.
        class RetargetsEveryParameter(OpBackend):
            def lower(self, exported_program, method_name):
                for spec in exported_program.graph_signature.input_specs:
                    if spec.target is not None:
                        spec.target = "renamed_by_the_backend"
                return exported_program

        manager = _manager()
        program = manager.exported_program()
        before = [spec.target for spec in program.graph_signature.input_specs]
        self.assertIn("linear.weight", before)

        # Retargeting without moving the weight leaves the program inconsistent,
        # so this is also the case that matters most: the edit must not survive
        # a lowering that failed.
        with self.assertRaises(ValueError):
            manager.to_op_backend(RetargetsEveryParameter())

        self.assertEqual(
            [spec.target for spec in program.graph_signature.input_specs], before
        )

    def test_a_constant_the_backend_adds_is_accepted(self) -> None:
        # The thing a real operator backend does: install a kernel and the
        # constant it needs. Nothing else here exercises a legal rewrite, so
        # without this the checks are never run against a changed program.
        from executorch.backends.transforms.utils import create_constant_placeholder
        from torch.export.graph_signature import InputKind

        class AddsAConstant(OpBackend):
            def lower(self, exported_program, method_name):
                exported_program = _edge_program()
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    create_constant_placeholder(
                        exp_program=exported_program,
                        graph=graph,
                        kind=InputKind.BUFFER,
                        name="op_backend_scale",
                        data=torch.ones(1),
                        persistent_buffer=True,
                    )
                return exported_program

        lowered = _lower_and_verify(_edge_program(), AddsAConstant(), "forward")
        self.assertIn("op_backend_scale", lowered.state_dict)


_kernels = torch.library.Library("op_backend_test", "FRAGMENT")
_kernels.define("atan(Tensor x) -> Tensor")
_kernels.impl("atan", lambda x: x.atan(), "CompositeExplicitAutograd")


class _Retargets(OpBackend):
    """Installs an operator the edge verifier does not know, which is what an
    operator backend is for."""

    def lower(self, exported_program, method_name):
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        graph_module = exported_program.graph_module
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and "atan" in str(node.target):
                with graph_module.graph.inserting_after(node):
                    replacement = graph_module.graph.call_function(
                        torch.ops.op_backend_test.atan.default, node.args
                    )
                replacement.meta.update(node.meta)
                node.replace_all_uses_with(replacement)
                graph_module.graph.erase_node(node)
                break
        graph_module.graph.lint()
        graph_module.recompile()
        return _transform(exported_program, ExportPass())


class _Counting(OpBackend):
    def __init__(self) -> None:
        self.seen: list = []

    def lower(self, exported_program, method_name):
        self.seen.append(method_name)
        return exported_program


def _manager(names=("forward",), **kwargs):
    programs = {
        name: torch.export.export(_Model(), (torch.randn(1, 4),)) for name in names
    }
    return to_edge(
        programs,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        **kwargs,
    )


class TestEdgeProgramManagerToOpBackend(unittest.TestCase):
    """The manager-level peer of ``EdgeProgramManager.to_backend``."""

    def test_every_method_is_lowered(self) -> None:
        backend = _Counting()
        _manager(("zeta", "alpha")).to_op_backend(backend)
        self.assertCountEqual(backend.seen, ["alpha", "zeta"])

    def test_constant_methods_and_etrecord_are_carried(self) -> None:
        manager = _manager(constant_methods={"get_max_seq_len": 128})
        manager._etrecord = "sentinel"
        lowered = manager.to_op_backend(_Counting())
        self.assertEqual(lowered._config_methods, {"get_max_seq_len": 128})
        self.assertEqual(lowered._etrecord, "sentinel")
        self.assertIsNot(lowered._config_methods, manager._config_methods)

    def test_a_backend_s_own_operators_survive(self) -> None:
        lowered = _manager().to_op_backend(_Retargets())
        targets = {
            str(n.target)
            for n in lowered.exported_program("forward").graph.nodes
            if n.op == "call_function"
        }
        self.assertIn("op_backend_test.atan.default", targets)

    def test_ir_validity_must_be_off_before_to_edge(self) -> None:
        # Programs keep the verifier they were built with, so this cannot be
        # rescued here: the backend's own _transform rejects its kernels before
        # this method sees them. Pinned because it is the caveat in OpBackend's
        # docstring, and the first thing a backend author trips over.
        strict = to_edge(
            {"forward": torch.export.export(_Model(), (torch.randn(1, 4),))}
        )
        self.assertTrue(strict.compile_config._check_ir_validity)
        with self.assertRaisesRegex(Exception, "is not an Edge operator"):
            strict.to_op_backend(_Retargets())

    def test_it_verifies_what_each_backend_returns(self) -> None:
        # Every other verification test calls the helper directly, so nothing
        # pinned that the manager method routes through it.
        class AddsRogueInput(OpBackend):
            def lower(self, exported_program, method_name):
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("rogue")
                return exported_program

        with self.assertRaisesRegex(ValueError, "inconsistent program"):
            _manager().to_op_backend(AddsRogueInput())

    def test_it_clears_ir_validity_on_the_rebuilt_manager(self) -> None:
        # A strict config would make EdgeProgramManager.__init__ re-verify the
        # lowered programs against a dialect they have deliberately left.
        manager = to_edge(
            {"forward": torch.export.export(_Model(), (torch.randn(1, 4),))}
        )
        self.assertTrue(manager.compile_config._check_ir_validity)
        self.assertFalse(
            manager.to_op_backend(_Counting()).compile_config._check_ir_validity
        )

    def test_the_rest_of_the_compile_config_is_carried(self) -> None:
        # to_backend replaces the config wholesale, dropping preserve_ops with
        # it; carrying it is the one place this deliberately differs.
        manager = to_edge(
            {"forward": torch.export.export(_Model(), (torch.randn(1, 4),))},
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, preserve_ops=[torch.ops.aten.linear.default]
            ),
        )
        config = manager.to_op_backend(_Counting()).compile_config
        self.assertIn(torch.ops.aten.linear.default, config.preserve_ops)
        self.assertFalse(config._check_ir_validity)

    def test_a_rewritten_program_still_reaches_a_pte(self) -> None:
        # The reason this method exists: to_executorch is only reachable from a
        # manager, so returning a bare program leaves a hand-staged export with
        # nowhere to put the result.
        lowered = _manager().to_op_backend(_Counting())
        self.assertGreater(len(lowered.to_executorch().buffer), 0)


if __name__ == "__main__":
    unittest.main()
