# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NeutronTester -- backend-specific Tester subclass for the Neutron backend.

Usage in a test:

    NeutronTester(model, example_inputs) \
        .quantize() \
        .export() \
        .to_edge_transform_and_lower() \
        .check_count({"executorch_exir_dialects_edge__ops_aten_convolution_default": 0}) \
        .to_executorch() \
        .serialize() \
        .run_method_and_compare_outputs()

The tester integrates into the shared test/suite operator test suite when a
NeutronTestFlow is registered in backends/test/suite/flows/nxp.py.
"""

import logging
import os
import tempfile
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import (
    core_aten_ops_exception_list,
    generate_neutron_compile_spec,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.nxp.tests.nsys_testing import execute_cmd
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import (
    Serialize,
    Stage,
    StageType,
    ToEdgeTransformAndLower,
    ToExecutorch,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from torch.export import ExportedProgram
from torch.utils._pytree import tree_flatten, tree_unflatten

# ---------------------------------------------------------------------------
# Module logger used by NeutronSerialize for diagnostic warnings.
# ---------------------------------------------------------------------------
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default number of random calibration samples used when no custom calibration
# function is provided.
# ---------------------------------------------------------------------------
_DEFAULT_NUM_CALIBRATION_SAMPLES = 4


def _calibration_tensor_like(t: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Return a tensor suitable for PTQ calibration with the same shape and dtype as t.

    For floating-point tensors, returns values uniformly drawn from [eps, 1) to
    avoid NaN in ops like log/sqrt. For integer and bool tensors, clones the
    original example tensor so that calibration inputs have realistic values
    (e.g. valid embedding indices, bool masks).
    """
    if t.is_floating_point():
        eps = torch.finfo(torch.float32).eps
        return torch.rand_like(t) * (1.0 - eps) + eps
    # Integer and bool dtypes: reuse the original example tensor value so that
    # HistogramObserver never receives a freshly-generated integer tensor.
    return original.clone()


def _make_random_calibration_inputs(
    example_inputs,
    num_samples: int = _DEFAULT_NUM_CALIBRATION_SAMPLES,
) -> List[Tuple[torch.Tensor, ...]]:
    """Generate random calibration samples compatible with example_inputs.

    example_inputs may be a tuple of tensors or a flat sequence that has already
    been tree-flattened. Non-tensor items are passed through unchanged.
    """
    flat, spec = tree_flatten(example_inputs)
    samples = []
    for _ in range(num_samples):
        flat_sample = [
            (
                _calibration_tensor_like(item, item)
                if isinstance(item, torch.Tensor)
                else item
            )
            for item in flat
        ]
        samples.append(tree_unflatten(flat_sample, spec))
    return samples


# ---------------------------------------------------------------------------
# Stage 1: Quantize
# ---------------------------------------------------------------------------


class NeutronQuantize(Stage):
    """Quantization stage for the Neutron backend.

    Applies NeutronQuantizer followed by calibration and convert_pt2e.
    Accepts either a pre-built list of calibration samples or a callable that
    produces them from the example inputs.
    """

    def __init__(
        self,
        target: str = "imxrt700",
        calibration_samples: Optional[Iterable[Tuple[torch.Tensor, ...]]] = None,
        get_calibration_inputs_fn: Optional[
            Callable[
                [Tuple[torch.Tensor, ...]],
                Iterable[Tuple[torch.Tensor, ...]],
            ]
        ] = None,
        num_calibration_samples: int = _DEFAULT_NUM_CALIBRATION_SAMPLES,
        is_qat: bool = False,
        train_fn: Optional[Callable[[torch.fx.GraphModule], None]] = None,
    ):
        """
        :param target: Neutron target string (e.g. 'imxrt700').
        :param calibration_samples: Fixed list of (inputs...) tuples to use for
            calibration.  If provided, get_calibration_inputs_fn and
            num_calibration_samples are ignored.
        :param get_calibration_inputs_fn: Callable(example_inputs) -> iterable of
            input tuples.  Called lazily during run() if calibration_samples is
            None.
        :param num_calibration_samples: Number of random samples to generate when
            neither calibration_samples nor get_calibration_inputs_fn is given.
        :param is_qat: Whether to use QAT-style prepare/convert.
        :param train_fn: Optional training function for QAT.
        """
        self._target = target
        self._calibration_samples = calibration_samples
        self._get_calibration_inputs_fn = get_calibration_inputs_fn
        self._num_calibration_samples = num_calibration_samples
        self._is_qat = is_qat
        self._train_fn = train_fn
        self._quantized_module: Optional[torch.fx.GraphModule] = None

    # Stage protocol ---

    def stage_type(self) -> StageType:
        return StageType.QUANTIZE

    def run(
        self,
        artifact: torch.nn.Module,
        inputs: Optional[Tuple[torch.Tensor, ...]],
    ) -> None:
        target_spec = NeutronTargetSpec(self._target)
        quantizer = NeutronQuantizer(target_spec, is_qat=self._is_qat)

        # Resolve calibration data.
        if self._calibration_samples is not None:
            calibration_inputs = self._calibration_samples
        elif self._get_calibration_inputs_fn is not None:
            calibration_inputs = self._get_calibration_inputs_fn(inputs)
        else:
            calibration_inputs = _make_random_calibration_inputs(
                inputs, self._num_calibration_samples
            )

        # Export the module before quantization.
        from torch.export import export

        exported = export(artifact, inputs, strict=True)

        # Both QAT and PTQ paths go through calibrate_and_quantize, which handles
        # BN fusion, observer selection (including HistogramObserver -> MinMaxObserver
        # for non-float inputs), and convert_pt2e internally.
        self._quantized_module = calibrate_and_quantize(
            model=exported,
            calibration_inputs=calibration_inputs,
            quantizer=quantizer,
            is_qat=self._is_qat,
            train_fn=self._train_fn,
        )

    @property
    def artifact(self) -> torch.fx.GraphModule:
        return self._quantized_module

    @property
    def graph_module(self) -> torch.fx.GraphModule:
        return self._quantized_module

    def run_artifact(self, inputs):
        return self._quantized_module(*inputs)


# ---------------------------------------------------------------------------
# Stage 2: ToEdgeTransformAndLower (Neutron-specific)
# ---------------------------------------------------------------------------


class NeutronToEdgeTransformAndLower(ToEdgeTransformAndLower):
    """Runs to_edge_transform_and_lower with the Neutron partitioner.

    Inherits from ToEdgeTransformAndLower (the shared harness base) and
    overrides run() to inject the NeutronPartitioner, Neutron edge passes,
    and the optional post-quant state dict.
    """

    def __init__(
        self,
        target: str = "imxrt700",
        operators_not_to_delegate: Optional[List[str]] = None,
        custom_delegation_options: Optional[CustomDelegationOptions] = None,
        use_neutron_for_format_conversion: bool = True,
        use_quant_state_dict: bool = True,
    ):
        # Initialize the base with no partitioners -- we build the Neutron
        # partitioner inside run() because it requires the compiled spec and
        # the post-quant state dict from the exported artifact.
        super().__init__()
        self._target = target
        self._operators_not_to_delegate = operators_not_to_delegate or []
        self._custom_delegation_options = (
            custom_delegation_options or CustomDelegationOptions()
        )
        self._use_neutron_for_format_conversion = use_neutron_for_format_conversion
        self._use_quant_state_dict = use_quant_state_dict

    def run(
        self,
        artifact: ExportedProgram,
        inputs=None,
        generate_etrecord: bool = False,
    ) -> None:
        from torch.export import export

        # Re-export the quantized graph module to get a clean ExportedProgram.
        if isinstance(artifact, torch.fx.GraphModule):
            # artifact is the output of the Quantize stage (a GraphModule).
            # Re-export it as a proper ExportedProgram before lowering.
            if inputs is None:
                raise RuntimeError(
                    "NeutronToEdgeTransformAndLower requires inputs for re-export."
                )
            artifact = export(artifact, inputs, strict=True)

        compile_spec = generate_neutron_compile_spec(
            self._target,
            operators_not_to_delegate=self._operators_not_to_delegate,
            use_neutron_for_format_conversion=self._use_neutron_for_format_conversion,
        )

        # Build the post-quant state dict for the partitioner if requested.
        post_quant_state_dict = None
        if self._use_quant_state_dict:
            post_quant_state_dict = artifact.state_dict

        preserve_ops = [
            torch.ops.aten.pad.default,
            torch.ops.aten.prelu.default,
            torch.ops.aten.hardswish.default,
        ]

        partitioner = NeutronPartitioner(
            compile_spec,
            NeutronTargetSpec(self._target),
            self._custom_delegation_options,
            post_quant_state_dict,
            preserve_ops=preserve_ops,
        )

        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=core_aten_ops_exception_list,
        )

        edge_program_manager = to_edge_transform_and_lower(
            artifact,
            transform_passes=NeutronEdgePassManager(),
            partitioner=[partitioner],
            generate_etrecord=generate_etrecord,
            compile_config=edge_compile_config,
        )

        # Remove redundant QDQ clusters added by the partitioner.
        edge_program_manager = edge_program_manager.transform(
            NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
        )

        # Store using the attribute name expected by the base class so that the
        # inherited artifact property returns the correct EdgeProgramManager.
        self.edge_dialect_program = edge_program_manager


# ---------------------------------------------------------------------------
# Stage 3: ToExecutorch (Neutron-specific config)
# ---------------------------------------------------------------------------


class NeutronToExecutorch(ToExecutorch):
    """ToExecutorch with Neutron-compatible ExecutorchBackendConfig.

    Uses extract_delegate_segments=False to embed the NPU payload inline
    (matching the existing executorch_pipeline.py behaviour).
    """

    def __init__(self):
        super().__init__(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )


# ---------------------------------------------------------------------------
# ExecuTorch ScalarType int -> (torch.dtype, numpy dtype) mapping.
# Values match at::ScalarType in PyTorch / ExecuTorch.
# ---------------------------------------------------------------------------
_SCALAR_TYPE_TO_TORCH = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
}

_TORCH_TO_NUMPY = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
}


def _output_tensor_specs(buffer: bytes):
    """Return a list of (sizes, torch.dtype) tuples for each output of the
    'forward' method, extracted from the serialised .pte buffer via
    ExecuTorchModule.method_meta().
    """
    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
            Verification,
        )
    except ImportError:
        return None

    module = _load_for_executorch_from_buffer(
        buffer, program_verification=Verification.Minimal
    )
    meta = module.method_meta("forward")
    specs = []
    for i in range(meta.num_outputs()):
        tinfo = meta.output_tensor_meta(i)
        dtype = _SCALAR_TYPE_TO_TORCH.get(tinfo.dtype(), torch.float32)
        specs.append((list(tinfo.sizes()), dtype))
    return specs


# ---------------------------------------------------------------------------
# Stage 4: Serialize -- run inference via the NSYS simulator
# ---------------------------------------------------------------------------


def _resolve_nsys_paths():
    """Return (nsys_path, config_path, firmware_path) using the config_importer
    shim, which prefers the integration-repo config.py (with the correct firmware
    path) and falls back to the pure ExecuTorch config."""
    from executorch.backends.nxp.tests.config_importer import test_config

    return (
        str(test_config.NSYS_PATH),
        str(test_config.NSYS_CONFIG_PATH),
        str(test_config.NSYS_FIRMWARE_PATH),
    )


def _resolve_runner_path() -> Optional[str]:
    """Return the path to the nxp_executor_runner binary, or None if not found.

    Delegates to config_importer (which uses the same resolution logic as
    config.py: NXP_RUNNER_PATH env var, then PROJECT_DIR-based auto-detect).
    Using config_importer avoids duplicating the path arithmetic and ensures
    consistent behaviour regardless of how the test is launched.
    """
    from executorch.backends.nxp.tests.config_importer import test_config

    runner = str(test_config.NEUTRON_TEST_PATH)
    if os.path.isfile(runner):
        return runner
    return None


class NeutronSerialize(Serialize):
    """Serialize stage that runs inference via the NSYS Neutron simulator.

    Writes the .pte buffer to a temporary file, invokes nxp_executor_runner,
    and reads the binary output tensors back as torch.Tensor objects so that
    run_method_and_compare_outputs() can compare them to the eager reference.

    If the simulator infrastructure (nsys, nxp_executor_runner) is not
    available, run_artifact() raises RuntimeError with a clear message.
    """

    def __init__(self, target: str = "imxrt700"):
        super().__init__()
        self._target = target

    def run_artifact(self, inputs: Tuple[torch.Tensor, ...]):
        """Run the serialised .pte via the NSYS Neutron simulator.

        :param inputs: Tuple of float32 torch.Tensors (one per model input).
        :returns: Tuple of torch.Tensors produced by the NPU simulator.
        :raises RuntimeError: If the simulator or runner binary is not available,
            or if the runner exits with a non-zero return code.
        """
        nsys_path, nsys_config_path, firmware_path = _resolve_nsys_paths()

        runner_path = _resolve_runner_path()
        if runner_path is None:
            raise RuntimeError(
                "nxp_executor_runner not found. "
                "Either set NXP_RUNNER_PATH to the compiled binary path, "
                "or build it at examples/nxp/executor_runner/build/nxp_executor_runner."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # --- Write .pte ---
            pte_path = os.path.join(tmp_dir, "model.pte")
            with open(pte_path, "wb") as f:
                f.write(self.buffer)

            # --- Write input tensors as raw binary files ---
            # Input tensors are written as zero-padded numbered files:
            #   0000.bin, 0001.bin, ... (one file per tensor in the flat input list)
            # The runner CLI flags then differ by arity:
            #   --dataset <dir>    for single-input models (reads *.bin files in dir)
            #   --inputs p0,p1,... for multi-input models (one absolute path per tensor)
            dataset_dir = os.path.join(tmp_dir, "dataset")
            os.makedirs(dataset_dir)
            flat_inputs, _ = tree_flatten(inputs)
            input_paths = []
            for idx, tensor in enumerate(flat_inputs):
                inp_path = os.path.join(dataset_dir, f"{idx:04d}.bin")
                arr = tensor.detach().cpu().numpy()
                arr.tofile(inp_path)
                input_paths.append(inp_path)

            # --- Run simulator ---
            output_dir = os.path.join(tmp_dir, "outputs")
            os.makedirs(output_dir)

            if len(input_paths) == 1:
                # Single input: use --dataset (runner iterates over .bin files in dir)
                input_arg = f"--dataset {dataset_dir}"
            else:
                # Multi-input: use --inputs with comma-separated paths (one per tensor)
                input_arg = f"--inputs {','.join(input_paths)}"

            cmd = (
                f"{runner_path} "
                f"--model {pte_path} "
                f"{input_arg} "
                f"--output {output_dir} "
                f"--firmware {firmware_path} "
                f"--nsys {nsys_path} "
                f"--nsys_config {nsys_config_path}"
            )
            try:
                execute_cmd(cmd)
            except Exception as exc:
                raise RuntimeError(
                    f"nxp_executor_runner failed.\ncommand: {cmd}"
                ) from exc

            # --- Read output binary files ---
            # The runner writes outputs as:
            #   <output_dir>/<sample_name>/<output_index>.bin
            # where <sample_name> matches the input file name (e.g. "0000.bin").
            # Collect all .bin files recursively, sorted so output order is stable.
            output_files = sorted(
                os.path.join(root, fname)
                for root, _dirs, files in os.walk(output_dir)
                for fname in files
                if fname.endswith(".bin")
            )
            if not output_files:
                raise RuntimeError(
                    f"No output .bin files found in {output_dir} after simulator run."
                )

            # --- Determine output shapes and dtypes from the .pte metadata ---
            output_specs = _output_tensor_specs(self.buffer)

            results = []
            for i, fpath in enumerate(output_files):
                if output_specs is not None and i < len(output_specs):
                    sizes, dtype = output_specs[i]
                    np_dtype = _TORCH_TO_NUMPY.get(dtype, np.float32)
                else:
                    # Fallback: output tensor metadata could not be read from the
                    # .pte buffer (pybindings unavailable or output index out of
                    # range). Interpret raw bytes as a flat float32 tensor. Shape
                    # and dtype may be incorrect -- compare with caution.
                    _log.warning(
                        "Output tensor metadata unavailable for output %d (%s); "
                        "interpreting raw bytes as a flat float32 tensor. "
                        "Shape and dtype may be wrong.",
                        i,
                        os.path.basename(fpath),
                    )
                    sizes = None
                    np_dtype = np.float32

                arr = np.fromfile(fpath, dtype=np_dtype)
                t = torch.from_numpy(arr)
                if sizes is not None:
                    t = t.reshape(sizes)
                results.append(t)

            return tuple(results)


# ---------------------------------------------------------------------------
# NeutronTester
# ---------------------------------------------------------------------------


class NeutronTester(TesterBase):
    """Backend tester for the Neutron NPU delegate.

    Provides Neutron-specific implementations for the Quantize,
    ToEdgeTransformAndLower, ToExecutorch, and Serialize stages.

    Example:

        NeutronTester(MyModel(), example_inputs) \
            .quantize() \
            .export() \
            .to_edge_transform_and_lower() \
            .to_executorch() \
            .serialize() \
            .run_method_and_compare_outputs()
    """

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        target: str = "imxrt700",
        operators_not_to_delegate: Optional[List[str]] = None,
        custom_delegation_options: Optional[CustomDelegationOptions] = None,
        use_neutron_for_format_conversion: bool = True,
        use_quant_state_dict: bool = True,
        calibration_samples: Optional[Iterable[Tuple[torch.Tensor, ...]]] = None,
        get_calibration_inputs_fn: Optional[Callable] = None,
        num_calibration_samples: int = _DEFAULT_NUM_CALIBRATION_SAMPLES,
        dynamic_shapes=None,
    ):
        self._neutron_target = target
        self._operators_not_to_delegate = operators_not_to_delegate
        self._custom_delegation_options = custom_delegation_options
        self._use_neutron_for_format_conversion = use_neutron_for_format_conversion
        self._use_quant_state_dict = use_quant_state_dict
        self._calibration_samples = calibration_samples
        self._get_calibration_inputs_fn = get_calibration_inputs_fn
        self._num_calibration_samples = num_calibration_samples

        # Start from the base defaults so stages like EXPORT, PARTITION, RUN_PASSES,
        # and TO_EDGE remain functional, then override with Neutron-specific stages.
        stage_classes = TesterBase.default_stage_classes()
        stage_classes.update(
            {
                StageType.QUANTIZE: self._make_quantize_stage,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: self._make_lower_stage,
                StageType.TO_EXECUTORCH: NeutronToExecutorch,
                StageType.SERIALIZE: lambda: NeutronSerialize(
                    target=self._neutron_target
                ),
            }
        )

        super().__init__(
            module,
            example_inputs,
            stage_classes=stage_classes,
            dynamic_shapes=dynamic_shapes,
        )

    # ------------------------------------------------------------------
    # Internal factory helpers registered as stage_classes callables
    # ------------------------------------------------------------------

    def _make_quantize_stage(self) -> NeutronQuantize:
        return NeutronQuantize(
            target=self._neutron_target,
            calibration_samples=self._calibration_samples,
            get_calibration_inputs_fn=self._get_calibration_inputs_fn,
            num_calibration_samples=self._num_calibration_samples,
        )

    def _make_lower_stage(self) -> NeutronToEdgeTransformAndLower:
        return NeutronToEdgeTransformAndLower(
            target=self._neutron_target,
            operators_not_to_delegate=self._operators_not_to_delegate,
            custom_delegation_options=self._custom_delegation_options,
            use_neutron_for_format_conversion=self._use_neutron_for_format_conversion,
            use_quant_state_dict=self._use_quant_state_dict,
        )
