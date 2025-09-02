# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
import torch
from executorch.backends.cadence.aot.program_builder import IrMode, ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from later.unittest import TestCase  # type: ignore[import-not-found]
from torch._export.verifier import SpecViolationError
from torch.export.graph_signature import InputKind, OutputKind


class TestProgramBuilder(TestCase):
    def test_user_input_with_parameter(self) -> None:
        inp = torch.randn([3, 5])
        w = torch.nn.Parameter(torch.randn([5]))
        # Create a exported program with one user input and one parameter.
        # Returns inp + w, w + 2 tuple.
        builder = ProgramBuilder()
        inp_proxy = builder.placeholder("inp", inp)
        w_proxy = builder.placeholder("w", w, input_kind=InputKind.PARAMETER)
        add = builder.call_operator(torch.ops.aten.add.Tensor, (inp_proxy, w_proxy))
        add_w = builder.call_operator(torch.ops.aten.add.Scalar, (w_proxy, 2))
        builder.output([add, add_w])
        program = builder.get_program()

        self.assertEqual(len(program.graph_signature.input_specs), 2)
        self.assertEqual(
            program.graph_signature.input_specs[0].kind, InputKind.USER_INPUT
        )
        self.assertEqual(
            program.graph_signature.input_specs[1].kind, InputKind.PARAMETER
        )
        self.assertEqual(len(program.graph_signature.output_specs), 2)
        self.assertEqual(
            program.graph_signature.output_specs[0].kind, OutputKind.USER_OUTPUT
        )
        self.assertEqual(
            program.graph_signature.output_specs[1].kind, OutputKind.USER_OUTPUT
        )

    def test_user_input_with_constant(self) -> None:
        inp = torch.randn([3, 5])
        const = torch.randn([5])
        # Create a exported program with one user input and one constant tensor.
        # Returns inp + const
        builder = ProgramBuilder()
        inp_proxy = builder.placeholder("inp", inp)
        const_proxy = builder.placeholder(
            "const", const, input_kind=InputKind.CONSTANT_TENSOR
        )
        add = builder.call_operator(torch.ops.aten.add.Tensor, (inp_proxy, const_proxy))
        builder.output([add])
        program = builder.get_program()

        # Verify the program has the correct inputs and outputs
        self.assertEqual(len(program.graph_signature.input_specs), 2)
        self.assertEqual(
            program.graph_signature.input_specs[0].kind, InputKind.USER_INPUT
        )
        self.assertEqual(
            program.graph_signature.input_specs[1].kind, InputKind.CONSTANT_TENSOR
        )
        self.assertEqual(len(program.graph_signature.output_specs), 1)
        self.assertEqual(
            program.graph_signature.output_specs[0].kind, OutputKind.USER_OUTPUT
        )

    def test_mutable_buffer(self) -> None:
        inp = torch.randn([3, 5])
        buffer = torch.randn([5])
        # Create a exported program with one user input and one buffer that gets mutated.
        # Returns inp + buffer, updated_buffer
        builder = ProgramBuilder()
        inp_proxy = builder.placeholder("inp", inp)
        buffer_proxy = builder.placeholder(
            "buffer", buffer, input_kind=InputKind.BUFFER
        )
        add = builder.call_operator(
            torch.ops.aten.add.Tensor, (inp_proxy, buffer_proxy)
        )
        # Mutate the buffer by adding 1
        updated_buffer = builder.call_operator(
            torch.ops.aten.add.Scalar, (buffer_proxy, 1)
        )
        builder.output(
            [add, updated_buffer], [OutputKind.USER_OUTPUT, OutputKind.BUFFER_MUTATION]
        )
        program = builder.get_program()

        # Verify the program has the correct inputs and outputs
        self.assertEqual(len(program.graph_signature.input_specs), 2)
        self.assertEqual(
            program.graph_signature.input_specs[0].kind, InputKind.USER_INPUT
        )
        self.assertEqual(program.graph_signature.input_specs[1].kind, InputKind.BUFFER)
        self.assertEqual(len(program.graph_signature.output_specs), 2)
        self.assertEqual(
            program.graph_signature.output_specs[0].kind, OutputKind.USER_OUTPUT
        )
        self.assertEqual(
            program.graph_signature.output_specs[1].kind, OutputKind.BUFFER_MUTATION
        )

    def test_user_input_mutation(self) -> None:
        inp = torch.randn([3, 5])
        # Create a exported program with one user input that gets mutated.
        # Returns updated_inp
        builder = ProgramBuilder()
        inp_proxy = builder.placeholder("inp", inp)
        # Mutate the input by adding 1
        updated_inp = builder.call_operator(torch.ops.aten.add.Scalar, (inp_proxy, 1))
        builder.output([updated_inp], [OutputKind.USER_INPUT_MUTATION])
        program = builder.get_program()

        # Verify the program has the correct inputs and outputs
        self.assertEqual(len(program.graph_signature.input_specs), 1)
        self.assertEqual(
            program.graph_signature.input_specs[0].kind, InputKind.USER_INPUT
        )
        self.assertEqual(len(program.graph_signature.output_specs), 1)
        self.assertEqual(
            program.graph_signature.output_specs[0].kind, OutputKind.USER_INPUT_MUTATION
        )

    def test_get_verifier_exir_mode(self) -> None:
        """Test that get_verifier returns EXIREdgeDialectVerifier for EXIR mode."""
        builder = ProgramBuilder(mode=IrMode.EXIR)
        verifiers = builder.get_verifiers()
        self.assertIsNotNone(verifiers)
        self.assertEqual(len(verifiers), 1)  # type: ignore[arg-type]

    def test_get_verifier_aten_mode(self) -> None:
        """Test that get_verifier returns None for ATEN mode."""
        builder = ProgramBuilder(mode=IrMode.ATEN)
        verifiers = builder.get_verifiers()
        self.assertIsNone(verifiers)

    def test_get_verifier_default_mode(self) -> None:
        """Test that get_verifier returns EXIREdgeDialectVerifier for default mode."""
        builder = ProgramBuilder()  # Should default to EXIR
        self.assertEqual(builder.mode, IrMode.EXIR)
        verifiers = builder.get_verifiers()
        self.assertIsNotNone(verifiers)
        self.assertEqual(len(verifiers), 1)  # type: ignore[arg-type]

    def test_aten_add_tensor_exir_mode(self) -> None:
        """Test using torch.ops.aten.add.Tensor with EXIR mode."""
        inp = torch.randn([3, 5])
        buffer = torch.randn([5])

        builder = ProgramBuilder(mode=IrMode.EXIR)
        inp_proxy = builder.placeholder("inp", inp)
        buffer_proxy = builder.placeholder(
            "buffer", buffer, input_kind=InputKind.BUFFER
        )
        add = builder.call_operator(
            torch.ops.aten.add.Tensor, (inp_proxy, buffer_proxy)
        )
        builder.output([add])
        builder.get_program()

    def test_aten_add_tensor_aten_mode(self) -> None:
        """Test using torch.ops.aten.add.Tensor with ATEN mode."""
        inp = torch.randn([3, 5])
        buffer = torch.randn([5])

        builder = ProgramBuilder(mode=IrMode.ATEN)
        inp_proxy = builder.placeholder("inp", inp)
        buffer_proxy = builder.placeholder(
            "buffer", buffer, input_kind=InputKind.BUFFER
        )
        add = builder.call_operator(
            torch.ops.aten.add.Tensor, (inp_proxy, buffer_proxy)
        )
        builder.output([add])
        program = builder.get_program()

        # Verify the program was created successfully
        self.assertEqual(len(program.graph_signature.input_specs), 2)
        self.assertEqual(len(program.graph_signature.output_specs), 1)
        self.assertEqual(builder.mode, IrMode.ATEN)

    def test_exir_edge_aten_add_tensor_exir_mode(self) -> None:
        """Test using exir_ops.edge.aten.add.Tensor with EXIR mode."""
        inp = torch.randn([3, 5])
        buffer = torch.randn([5])

        builder_exir = ProgramBuilder(mode=IrMode.EXIR)
        inp_proxy_exir = builder_exir.placeholder("inp", inp)
        buffer_proxy_exir = builder_exir.placeholder(
            "buffer", buffer, input_kind=InputKind.BUFFER
        )
        add_exir = builder_exir.call_operator(
            exir_ops.edge.aten.add.Tensor, (inp_proxy_exir, buffer_proxy_exir)
        )
        builder_exir.output([add_exir])
        program_exir = builder_exir.get_program()

        # Verify the program was created successfully
        self.assertEqual(len(program_exir.graph_signature.input_specs), 2)
        self.assertEqual(len(program_exir.graph_signature.output_specs), 1)
        self.assertEqual(builder_exir.mode, IrMode.EXIR)

    def test_exir_edge_aten_add_tensor_aten_mode(self) -> None:
        """Test using exir_ops.edge.aten.add.Tensor with ATEN mode."""
        inp = torch.randn([3, 5])
        buffer = torch.randn([5])

        builder_aten = ProgramBuilder(mode=IrMode.ATEN)
        inp_proxy_aten = builder_aten.placeholder("inp", inp)
        buffer_proxy_aten = builder_aten.placeholder(
            "buffer", buffer, input_kind=InputKind.BUFFER
        )
        add_aten = builder_aten.call_operator(
            exir_ops.edge.aten.add.Tensor, (inp_proxy_aten, buffer_proxy_aten)
        )
        builder_aten.output([add_aten])

        with self.assertRaises(
            SpecViolationError, msg="Operator '<EdgeOpOverload: aten.add.Tensor>"
        ):
            builder_aten.get_program()
