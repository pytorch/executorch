# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from executorch.backends.cuda.scripts._atomic_output import write_atomically
from executorch.backends.cuda.scripts.remap_pte_devices import (
    collect_cuda_device_indices,
    dense_device_mapping,
    main as remap_main,
    parse_device_mapping,
    remap_program_devices,
    remap_pte_bytes,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize._program import (
    deserialize_pte_binary,
    PTEFile,
    serialize_pte_binary,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    ContainerMetadata,
    DataLocation,
    DeviceType,
    EValue,
    ExecutionPlan,
    ExtraTensorInfo,
    NonConstBufferDevice,
    Program,
    SubsegmentOffsets,
    Tensor,
    TensorShapeDynamism,
)


def _program(target_device: bytes = b"cuda:9") -> Program:
    delegate = BackendDelegate(
        id="CudaBackend",
        processed=BackendDelegateDataReference(DataLocation.INLINE, 0),
        compile_specs=[
            CompileSpec("method_name", b"forward"),
            CompileSpec("target_device", target_device),
        ],
    )
    tensor = Tensor(
        scalar_type=ScalarType.FLOAT,
        storage_offset=0,
        sizes=[4],
        dim_order=[0],
        requires_grad=False,
        layout=0,
        data_buffer_idx=0,
        allocation_info=None,
        shape_dynamism=TensorShapeDynamism.STATIC,
        extra_tensor_info=ExtraTensorInfo(
            fully_qualified_name="weight",
            device_type=DeviceType.CUDA,
            device_index=7,
        ),
    )
    plan = ExecutionPlan(
        name="forward",
        container_meta_type=ContainerMetadata("", ""),
        values=[EValue(tensor)],
        inputs=[],
        outputs=[],
        chains=[],
        operators=[],
        delegates=[delegate],
        non_const_buffer_sizes=[0, 16, 32, 64],
        non_const_buffer_device=[
            NonConstBufferDevice(
                buffer_idx=1, device_type=DeviceType.CUDA, device_index=3
            ),
            NonConstBufferDevice(
                buffer_idx=2, device_type=DeviceType.CUDA, device_index=5
            ),
            NonConstBufferDevice(
                buffer_idx=3, device_type=DeviceType.CPU, device_index=0
            ),
        ],
    )
    return Program(
        version=0,
        execution_plan=[plan],
        constant_buffer=[],
        backend_delegate_data=[
            BackendDelegateInlineData(b"fake-processed-delegate-bytes")
        ],
        segments=[],
        constant_segment=SubsegmentOffsets(0, []),
    )


def _spec_value(program: Program) -> bytes:
    return program.execution_plan[0].delegates[0].compile_specs[1].value


class TestRemapPteDevices(unittest.TestCase):
    def test_collects_indices_from_all_three_places(self) -> None:
        # Planned buffers on 3 and 5, tensor info on 7, delegate spec on 9:
        # each place contributes an index no other place has.
        self.assertEqual(collect_cuda_device_indices(_program()), {3, 5, 7, 9})

    def test_dense_mapping_keeps_order(self) -> None:
        self.assertEqual(dense_device_mapping({7, 3, 5}), {3: 0, 5: 1, 7: 2})
        self.assertEqual(dense_device_mapping({0, 1}), {0: 0, 1: 1})

    def test_dense_remap_rewrites_all_three_places(self) -> None:
        program = _program()
        report = remap_program_devices(program, dense_device_mapping({3, 5, 7, 9}))
        plan = program.execution_plan[0]
        devices = {
            entry.buffer_idx: entry.device_index
            for entry in plan.non_const_buffer_device or []
        }
        self.assertEqual(devices, {1: 0, 2: 1, 3: 0})
        tensor = plan.values[0].val
        assert isinstance(tensor, Tensor)
        assert tensor.extra_tensor_info is not None
        self.assertEqual(tensor.extra_tensor_info.device_index, 2)
        self.assertEqual(_spec_value(program), b"cuda:3")
        self.assertEqual(report.planned_buffers_changed, 2)
        self.assertEqual(report.tensor_infos_changed, 1)
        self.assertEqual(report.compile_specs_changed, 1)

    def test_explicit_mapping(self) -> None:
        self.assertEqual(parse_device_mapping("3=0, 5=1"), {3: 0, 5: 1})
        with self.assertRaisesRegex(ValueError, "not of the form OLD=NEW"):
            parse_device_mapping("cuda:3=0")
        program = _program()
        remap_program_devices(program, {3: 0, 5: 1, 7: 3, 9: 2})
        self.assertEqual(_spec_value(program), b"cuda:2")

    def test_mapping_must_cover_every_recorded_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover"):
            remap_program_devices(_program(), {3: 0, 5: 1, 7: 2})
        with self.assertRaisesRegex(ValueError, "does not record"):
            remap_program_devices(_program(), {3: 0, 5: 1, 7: 2, 9: 3, 42: 0})

    def test_mapping_rejects_repeated_sources_and_negative_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "repeats source"):
            parse_device_mapping("3=0,3=1")
        with self.assertRaisesRegex(ValueError, "0..=127"):
            remap_program_devices(_program(), {3: -1, 5: 0, 7: 1, 9: 2})
        with self.assertRaisesRegex(ValueError, "0..=127"):
            remap_program_devices(_program(), {3: 128, 5: 0, 7: 1, 9: 2})

    def test_mapping_may_collapse_to_one_device(self) -> None:
        program = _program()
        remap_program_devices(program, {3: 0, 5: 0, 7: 0, 9: 0})
        self.assertEqual(collect_cuda_device_indices(program), {0})

    def test_bare_and_uppercase_target_device_specs(self) -> None:
        for spec in (b"cuda", b"CUDA"):
            with self.subTest(spec=spec):
                program = _program(target_device=spec)
                self.assertIn(0, collect_cuda_device_indices(program))
                remap_program_devices(program, {0: 4, 3: 0, 5: 1, 7: 2})
                self.assertEqual(_spec_value(program), b"cuda:4")
        program = _program(target_device=b"CUDA:3")
        self.assertEqual(collect_cuda_device_indices(program), {3, 5, 7})
        remap_program_devices(program, {3: 0, 5: 1, 7: 2})
        self.assertEqual(_spec_value(program), b"cuda:0")

    def test_non_cuda_target_device_spec_is_left_alone(self) -> None:
        program = _program(target_device=b"cpu")
        self.assertEqual(collect_cuda_device_indices(program), {3, 5, 7})
        report = remap_program_devices(program, {3: 0, 5: 1, 7: 2})
        self.assertEqual(_spec_value(program), b"cpu")
        self.assertEqual(report.compile_specs_changed, 0)

    def test_other_backends_target_device_spec_is_ignored(self) -> None:
        program = _program()
        delegate = program.execution_plan[0].delegates[0]
        other = BackendDelegate(
            id="OtherBackend",
            processed=delegate.processed,
            compile_specs=[CompileSpec("target_device", b"npu:0")],
        )
        program.execution_plan[0].delegates.append(other)
        undecodable = BackendDelegate(
            id="OtherBackend",
            processed=delegate.processed,
            compile_specs=[CompileSpec("target_device", b"\xff:0")],
        )
        program.execution_plan[0].delegates.append(undecodable)
        self.assertEqual(collect_cuda_device_indices(program), {3, 5, 7, 9})
        remap_program_devices(program, {3: 0, 5: 0, 7: 0, 9: 0})
        self.assertEqual(other.compile_specs[0].value, b"npu:0")

    def test_malformed_target_device_spec_raises(self) -> None:
        program = _program(target_device=b"cuda:three")
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            remap_program_devices(program, {3: 0, 5: 1, 7: 2, 9: 3})

    def test_program_without_cuda_ids_is_refused(self) -> None:
        plan = ExecutionPlan(
            name="forward",
            container_meta_type=ContainerMetadata("", ""),
            values=[],
            inputs=[],
            outputs=[],
            chains=[],
            operators=[],
            delegates=[],
            non_const_buffer_sizes=[0, 16],
            non_const_buffer_device=[
                NonConstBufferDevice(
                    buffer_idx=1, device_type=DeviceType.CPU, device_index=0
                )
            ],
        )
        program = Program(
            version=0,
            execution_plan=[plan],
            constant_buffer=[],
            backend_delegate_data=[],
            segments=[],
            constant_segment=SubsegmentOffsets(0, []),
        )
        pte_data = bytes(serialize_pte_binary(PTEFile(program=program)))
        with self.assertRaisesRegex(ValueError, "no CUDA device ids"):
            remap_pte_bytes(pte_data)

    def _segmented_pte_bytes(self) -> bytes:
        store = NamedDataStore()
        store.add_named_data("so_key", b"fake-so-bytes")
        return bytes(
            serialize_pte_binary(
                PTEFile(
                    program=_program(),
                    named_data=store.get_named_data_store_output(),
                ),
                extract_delegate_segments=True,
            )
        )

    def test_round_trip_through_pte_bytes(self) -> None:
        remapped, _ = remap_pte_bytes(self._segmented_pte_bytes())
        pte_file = deserialize_pte_binary(remapped)
        program = pte_file.program
        self.assertEqual(collect_cuda_device_indices(program), {0, 1, 2, 3})
        # Non-device content survives the rewrite.
        assert pte_file.named_data is not None
        self.assertEqual(pte_file.named_data.buffers, [b"fake-so-bytes"])
        self.assertEqual(list(pte_file.named_data.pte_data), ["so_key"])
        plan = program.execution_plan[0]
        self.assertEqual(plan.non_const_buffer_sizes, [0, 16, 32, 64])
        tensor = plan.values[0].val
        assert isinstance(tensor, Tensor)
        self.assertEqual(tensor.sizes, [4])

    def test_identity_remap_reproduces_the_input_bytes(self) -> None:
        pte_data = self._segmented_pte_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            remapped, _ = remap_pte_bytes(pte_data, {3: 3, 5: 5, 7: 7, 9: 9})
        self.assertEqual(remapped, pte_data)

    def test_multi_device_result_warns(self) -> None:
        pte_data = self._segmented_pte_bytes()
        with self.assertWarnsRegex(UserWarning, "current device") as warning:
            remap_pte_bytes(pte_data)
        # The suggested collapse can be pasted after --mapping as-is.
        self.assertIn("3=0,5=0,7=0,9=0", str(warning.warning))
        # An explicit mapping onto several devices warns just the same.
        with self.assertWarnsRegex(UserWarning, "current device"):
            remap_pte_bytes(pte_data, {3: 0, 5: 1, 7: 0, 9: 0})
        # So does a result on a single non-zero device.
        with self.assertWarnsRegex(UserWarning, "current device"):
            remap_pte_bytes(pte_data, {3: 1, 5: 1, 7: 1, 9: 1})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            remap_pte_bytes(pte_data, {3: 0, 5: 0, 7: 0, 9: 0})
        self.assertEqual(caught, [])

    def test_shared_compile_spec_is_rewritten_once(self) -> None:
        program = _program()
        delegate = program.execution_plan[0].delegates[0]
        twin = BackendDelegate(
            id=delegate.id,
            processed=delegate.processed,
            compile_specs=delegate.compile_specs,
        )
        program.execution_plan[0].delegates.append(twin)
        report = remap_program_devices(program, {3: 0, 5: 0, 7: 0, 9: 0})
        self.assertEqual(_spec_value(program), b"cuda:0")
        self.assertEqual(report.compile_specs_changed, 1)

    def test_second_execution_plan_is_rewritten_too(self) -> None:
        program = _program()
        second = _program(target_device=b"cuda:11").execution_plan[0]
        for entry in second.non_const_buffer_device or []:
            if entry.device_type == DeviceType.CUDA:
                entry.device_index = 11
        program.execution_plan.append(second)
        self.assertIn(11, collect_cuda_device_indices(program))
        remap_program_devices(program, {3: 0, 5: 0, 7: 0, 9: 0, 11: 0})
        self.assertEqual(collect_cuda_device_indices(program), {0})

    def test_exported_program_round_trip(self) -> None:
        import torch
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
        from executorch.exir.capture._config import ExecutorchBackendConfig

        class Add(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        def export_for(device: str) -> bytes:
            torch.manual_seed(0)
            exported = torch.export.export(
                Add().eval(), (torch.randn(2, 2), torch.randn(2, 2))
            )
            manager = to_edge_transform_and_lower(
                exported, partitioner=[DeviceAwarePartitioner(device)]
            ).to_executorch(
                ExecutorchBackendConfig(enable_non_cpu_memory_planning=True)
            )
            return bytes(manager.buffer)

        pte_data = export_for("cuda:3")
        before = collect_cuda_device_indices(deserialize_pte_binary(pte_data).program)
        self.assertEqual(before, {3})
        remapped, _ = remap_pte_bytes(pte_data)
        remapped_program = deserialize_pte_binary(remapped).program
        after = collect_cuda_device_indices(remapped_program)
        self.assertEqual(after, {0})
        # A device id recorded anywhere the tool does not know about
        # would show up as a difference from a native cuda:0 export.
        native_program = deserialize_pte_binary(export_for("cuda:0")).program
        self.assertEqual(remapped_program, native_program)

    def test_cli_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_pte = Path(directory) / "model.pte"
            input_pte.write_bytes(
                bytes(serialize_pte_binary(PTEFile(program=_program())))
            )
            stdout = io.StringIO()
            with mock.patch(
                "sys.argv",
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--dry-run",
                ],
            ), redirect_stdout(stdout):
                remap_main()
            self.assertIn(
                "CUDA device mapping: {3: 0, 5: 1, 7: 2, 9: 3}", stdout.getvalue()
            )
            self.assertEqual(list(Path(directory).iterdir()), [input_pte])

    def test_failed_atomic_write_keeps_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pte"
            source.write_bytes(b"input")
            output = root / "output.pte"
            output.write_bytes(b"previous")

            def write_partial(stream) -> None:
                stream.write(b"partial")
                raise OSError("write failed")

            with self.assertRaisesRegex(OSError, "write failed"):
                write_atomically(output, write_partial, mode_from=source)
            self.assertEqual(output.read_bytes(), b"previous")
            self.assertEqual(
                {entry.name for entry in root.iterdir()},
                {"input.pte", "output.pte"},
            )

    def test_cli_input_errors_exit_like_argparse_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_pte = Path(directory) / "model.pte"
            input_pte.write_bytes(
                bytes(serialize_pte_binary(PTEFile(program=_program())))
            )
            cases = [
                ["remap_pte_devices", "--input-pte", str(input_pte)],
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--dry-run",
                    "--mapping",
                    "",
                ],
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--dry-run",
                    "--mapping",
                    "3=0",
                ],
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(Path(directory) / "missing.pte"),
                    "--dry-run",
                ],
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--dry-run",
                    "--output-pte",
                    str(Path(directory) / "out.pte"),
                ],
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--output-pte",
                    directory,
                    "--mapping",
                    "3=0,5=0,7=0,9=0",
                ],
            ]
            for argv in cases:
                with self.subTest(argv=argv), mock.patch("sys.argv", argv):
                    with self.assertRaises(SystemExit) as raised:
                        remap_main()
                    self.assertEqual(raised.exception.code, 2)

    def test_cli_writes_remapped_pte_with_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_pte = Path(directory) / "model.pte"
            input_pte.write_bytes(
                bytes(serialize_pte_binary(PTEFile(program=_program())))
            )
            output_pte = Path(directory) / "nested" / "model_cuda0.pte"
            with mock.patch(
                "sys.argv",
                [
                    "remap_pte_devices",
                    "--input-pte",
                    str(input_pte),
                    "--output-pte",
                    str(output_pte),
                    "--mapping",
                    "3=0,5=0,7=0,9=0",
                ],
            ):
                remap_main()
            program = deserialize_pte_binary(output_pte.read_bytes()).program
            self.assertEqual(collect_cuda_device_indices(program), {0})
            # The output keeps the input's readable permissions.
            self.assertEqual(
                output_pte.stat().st_mode & 0o777,
                input_pte.stat().st_mode & 0o777,
            )


if __name__ == "__main__":
    unittest.main()
