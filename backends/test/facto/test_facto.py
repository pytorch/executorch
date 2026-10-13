# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# This file contains logic to run generated operator tests using the FACTO
# library (https://github.com/meta-pytorch/FACTO). To run the tests, first
# clone and install FACTO by running pip install . from the FACTO source
# directory. Then, from the executorch root directory, run the following:
#
# python -m unittest backends.test.facto.test_facto.FactoTestsXNNPACK
#
# Useful environment variables:
# ARM_FACTO=1 enables Arm-generated FACTO test registration.
# FACTO_OPS="abs.default,acos.default" limits generated tests to selected ops.
# FACTO_MAX_CASES=10 limits the number of generated cases per op.
# FACTO_CASES_FILE=facto_cases.jsonl writes per-generated-case records.
# FACTO_FAILURES_FILE=facto_failures.jsonl writes structured failure records.
# FACTO_SUMMARY_FILE=facto_summary.jsonl writes per-op execution summaries.
#

import copy
import fnmatch
import functools
import json
import logging
import math
import os
import traceback
import unittest
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.test.harness.tester import Tester as BackendTester
from executorch.backends.xnnpack.test.tester.tester import Tester as XnnpackTester
from torch._ops import OpOverload


class _MissingFactoProxy:
    def __getattr__(self, _name: str) -> Any:
        raise unittest.SkipTest(_facto_unavailable_message())


FACTO_IMPORT_ERROR: ImportError | None = None
FACTO_AVAILABLE = False


def _facto_unavailable_message() -> str:
    message = (
        "FACTO is required to run generated operator tests. Install it with "
        "`pip install -e backends/cadence/utils/FACTO`."
    )
    if FACTO_IMPORT_ERROR is None:
        return message
    return f"{message} ({FACTO_IMPORT_ERROR})"


def _require_facto() -> None:
    if FACTO_AVAILABLE:
        return
    raise unittest.SkipTest(_facto_unavailable_message())


def _missing_facto_module(exc: ModuleNotFoundError) -> bool:
    return exc.name == "facto" or exc.name.startswith("facto.")


try:
    from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
    from facto.inputgen.specs.model import ConstraintProducer as cp, Spec
    from facto.inputgen.utils.random_manager import random_manager
    from facto.specdb.db import SpecDictDB
except ModuleNotFoundError as exc:
    if not _missing_facto_module(exc):
        raise
    FACTO_IMPORT_ERROR = exc
    ArgumentTupleGenerator = None
    cp = _MissingFactoProxy()
    Spec = Any
    SpecDictDB = {}
    ExtraSpecDB = {}
    CombinedSpecDB = {}
else:
    from .facto_specs import ExtraSpecDB

    CombinedSpecDB = SpecDictDB | ExtraSpecDB
    FACTO_AVAILABLE = True

if FACTO_AVAILABLE:
    COMMON_TENSOR_CONSTRAINTS = [
        cp.Rank.Ge(lambda deps: 1),  # Avoid zero and high rank tensors.
        cp.Rank.Le(lambda deps: 4),
        cp.Size.Ge(lambda deps, r, d: 1),  # Keep sizes reasonable.
        cp.Size.Le(lambda deps, r, d: 2**9),
    ]

    COMMON_SCALAR_CONSTRAINS = [
        cp.Value.Ge(lambda deps, dtype: -1000),
        cp.Value.Le(lambda deps, dtype: 1000),
    ]
else:
    COMMON_TENSOR_CONSTRAINTS = []
    COMMON_SCALAR_CONSTRAINS = []

# Operator args are treated as runtime graph inputs if the argument name is
# in this list.
RUNTIME_INPUT_NAMES = {
    "self",
    "tensor",
    "other",
}


TesterFactory = Callable[[torch.nn.Module, tuple[Any, ...]], BackendTester]
FACTO_MAX_CASES_ENV = "FACTO_MAX_CASES"
FACTO_OPS_ENV = "FACTO_OPS"
FACTO_CASES_FILE_ENV = "FACTO_CASES_FILE"
FACTO_FAILURES_FILE_ENV = "FACTO_FAILURES_FILE"
FACTO_SUMMARY_FILE_ENV = "FACTO_SUMMARY_FILE"
SUMMARY_REPR_LIMIT = 160
TENSOR_SAMPLE_LIMIT = 8
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FactoConfig:
    max_cases: int | None
    selected_op_names: list[str]
    selected_ops_error: str | None
    cases_file: str | None
    failures_file: str | None
    summary_file: str | None


def _matches_selected_patterns(op_name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(op_name, pattern) for pattern in patterns)


def _selected_op_names() -> list[str]:
    op_patterns = os.environ.get(FACTO_OPS_ENV)
    if not FACTO_AVAILABLE:
        return []

    if op_patterns is None or not op_patterns.strip():
        return list(CombinedSpecDB.keys())

    patterns = [pattern.strip() for pattern in op_patterns.split(",")]
    patterns = [pattern for pattern in patterns if pattern]
    return [
        op_name
        for op_name in CombinedSpecDB.keys()
        if _matches_selected_patterns(op_name, patterns)
    ]


def _facto_config() -> _FactoConfig:
    max_cases = os.environ.get(FACTO_MAX_CASES_ENV)
    selected_op_names = _selected_op_names()
    selected_ops_error = None
    op_patterns = os.environ.get(FACTO_OPS_ENV)
    if (
        FACTO_AVAILABLE
        and op_patterns is not None
        and op_patterns.strip()
        and not selected_op_names
    ):
        selected_ops_error = (
            f"{FACTO_OPS_ENV}={op_patterns!r} did not match any operators."
        )
    return _FactoConfig(
        max_cases=None if max_cases is None else int(max_cases),
        selected_op_names=selected_op_names,
        selected_ops_error=selected_ops_error,
        cases_file=os.environ.get(FACTO_CASES_FILE_ENV),
        failures_file=os.environ.get(FACTO_FAILURES_FILE_ENV),
        summary_file=os.environ.get(FACTO_SUMMARY_FILE_ENV),
    )


def _short_repr(value: object) -> str:
    value_repr = repr(value)
    if len(value_repr) <= SUMMARY_REPR_LIMIT:
        return value_repr
    return value_repr[: SUMMARY_REPR_LIMIT - 3] + "..."


def _json_scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        if value > 0:
            return "Infinity"
        return "-Infinity"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return _short_repr(value)


def _tensor_sample(value: torch.Tensor) -> list[object]:
    sample_values = value.detach().flatten()[:TENSOR_SAMPLE_LIMIT].cpu().tolist()
    if not isinstance(sample_values, list):
        sample_values = [sample_values]
    return [_json_scalar(item) for item in sample_values]


def _summarize_tensor(value: torch.Tensor) -> dict[str, object]:
    summary: dict[str, object] = {
        "type": "Tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "device": str(value.device),
        "requires_grad": value.requires_grad,
    }
    if value.numel() == 0:
        return summary

    try:
        summary["sample"] = _tensor_sample(value)

        if value.is_complex():
            return summary

        stats_tensor = value.detach().cpu()
        if stats_tensor.dtype == torch.bool:
            stats_tensor = stats_tensor.to(torch.int64)
        stats_tensor_float = stats_tensor.to(torch.float32)

        summary["min"] = _json_scalar(stats_tensor.min().item())
        summary["max"] = _json_scalar(stats_tensor.max().item())
        summary["mean"] = _json_scalar(stats_tensor_float.mean().item())
    except Exception as e:
        summary["summary_error"] = _short_repr(e)
    return summary


def _summarize_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return _summarize_tensor(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return _json_scalar(value)
    if isinstance(value, tuple):
        return [_summarize_value(item) for item in value]
    if isinstance(value, list):
        return [_summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _summarize_value(item) for key, item in value.items()}
    return {"type": type(value).__name__, "repr": _short_repr(value)}


def _append_jsonl(path: str, record: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _patch_spec(spec: Spec) -> Spec:
    _require_facto()
    spec = copy.deepcopy(spec)
    for inspec in spec.inspec:
        if inspec.type.is_tensor():
            inspec.constraints.extend(COMMON_TENSOR_CONSTRAINTS)
        elif inspec.type.is_scalar():
            inspec.constraints.extend(COMMON_SCALAR_CONSTRAINS)
    return spec


class OpModel(torch.nn.Module):
    """
    Wraps a single torch operator in an nn.Module.
    """

    def __init__(
        self,
        op: OpOverload,
        runtime_input_count: int,
        fixed_args: Sequence[Any],
        fixed_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.op = op
        self.runtime_input_count = runtime_input_count
        self.fixed_kwargs = fixed_kwargs

        # Register parameters for fixed tensors. Some things will choke on
        # constant tensor weights, for example.
        new_args = []
        for i, arg in enumerate(fixed_args):
            if isinstance(arg, torch.Tensor):
                param = torch.nn.Parameter(arg, requires_grad=False)
                param_name = f"arg_{i}_param"
                setattr(self, param_name, param)
                self.register_parameter(param_name, param)
                new_args.append(param)
            else:
                new_args.append(arg)
        self.fixed_args = tuple(new_args)

    def forward(self, *args, **kwargs):
        return self.op(*(args + self.fixed_args), **(kwargs | self.fixed_kwargs))


# The convolution model has some minor wrapper logic around the actual convolution
# operator. Most of the backends are expecting this form.
# TODO (gjcomer) Investigate these discrepencies.
class ConvModel(OpModel):
    def forward(self, *args, **kwargs):
        weight, bias, stride, padding, dilation, transposed, output_padding, groups = (
            self.fixed_args
        )

        if not transposed:
            if len(weight.shape) == 3:
                op = torch.nn.functional.conv1d
            elif len(weight.shape) == 4:
                op = torch.nn.functional.conv2d
            elif len(weight.shape) == 5:
                op = torch.nn.functional.conv3d

            return op(args[0], weight, bias, stride, padding, dilation, groups)
        else:
            if len(weight.shape) == 3:
                op = torch.nn.functional.conv_transpose1d
            elif len(weight.shape) == 4:
                op = torch.nn.functional.conv_transpose2d
            elif len(weight.shape) == 5:
                op = torch.nn.functional.conv_transpose3d

            return op(
                args[0], weight, bias, stride, padding, output_padding, groups, dilation
            )


def get_module_for_op(op: OpOverload):
    if op == torch.ops.aten.convolution.default:
        return ConvModel
    else:
        return OpModel


class FactoTestsBase(unittest.TestCase):
    __test__ = False

    def __init__(self, tester_factory: TesterFactory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tester_factory = tester_factory

    @classmethod
    def _generate_test(cls, op_name: str) -> None:
        # Find the torch op with the given name.
        sections = op_name.split(".")
        torch_op = functools.reduce(getattr, sections, torch.ops.aten)

        test_name = "test_" + op_name.replace(".", "_")

        def test_body(self):
            self._test_op(torch_op)

        setattr(cls, test_name, test_body)

    @classmethod
    def _generate_missing_facto_test(cls) -> None:
        def test_body(self):
            _require_facto()

        cls.test_facto_dependency_unavailable = test_body

    @classmethod
    def _generate_configuration_error_test(cls, test_name: str, message: str) -> None:
        def test_body(self):
            self.fail(message)

        setattr(cls, test_name, test_body)

    @staticmethod
    def get_runtime_input_count(spec: Spec):
        # Determine which inputs are fixed at tracing time (weights, for example),
        # vs inputs to the runtime graph. We currently assume that the runtime graph
        # inputs start at the beginning of the arg list and are contiguous.
        #
        # Args are consider to be runtime inputs if they are positional and are named
        # one of RUNTIME_INPUT_NAMES. If none match, we assume only the first arg is a
        # runtime input.
        runtime_input_count = 0
        for inspec in spec.inspec:
            is_runtime_input = (
                inspec.type.is_tensor() and inspec.name.lower() in RUNTIME_INPUT_NAMES
            )
            if is_runtime_input:
                runtime_input_count += 1
            else:
                break

        return max(1, runtime_input_count)

    def setUp(self):
        torch.set_printoptions(threshold=3)

    def _patch_spec_for_backend(self, spec: Spec, op_name: str) -> Spec:
        return spec

    def _should_skip_case(self, posargs: Sequence[Any]) -> bool:
        return False

    def _run_delegated_case(self, tester: BackendTester) -> None:
        tester.to_executorch().serialize().run_method_and_compare_outputs()

    def _should_fail_on_failures(self) -> bool:
        return False

    def _count_as_test_failures(
        self,
        *,
        eager_fail_count: int,
        export_fail_count: int,
        fail_count: int,
    ) -> int:
        return fail_count

    @staticmethod
    def _current_stage_name(
        tester: BackendTester | None, default: str = "setup"
    ) -> str:
        if tester is None or tester.cur is None:
            return default

        stage_names = {
            StageType.EXPORT: "export",
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: "lower",
            StageType.TO_EXECUTORCH: "run_delegated",
            StageType.SERIALIZE: "run_delegated",
        }
        return stage_names.get(tester.cur, tester.cur.name.lower())

    def _record_failure(
        self,
        config: _FactoConfig,
        op_name: str,
        case_index: int,
        posargs: Sequence[Any],
        kwargs: dict[str, Any],
        exception: BaseException,
        exception_text: str,
        stage: str,
        *,
        delegated: bool | None = None,
    ) -> None:
        record: dict[str, object] = {
            "backend": type(self).__name__,
            "case_index": case_index,
            "exception": exception_text,
            "exception_message": str(exception),
            "exception_type": type(exception).__name__,
            "op": op_name,
            "posargs": _summarize_value(list(posargs)),
            "kwargs": _summarize_value(kwargs),
            "stage": stage,
            "traceback": exception_text,
        }
        if delegated is not None:
            record["delegated"] = delegated
        record_line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(f"FACTO_FAILURE {record_line}")

        if config.failures_file:
            _append_jsonl(config.failures_file, record)

    def _record_failed_case(
        self,
        config: _FactoConfig,
        op_name: str,
        case_index: int,
        status: str,
        posargs: Sequence[Any],
        kwargs: dict[str, Any],
        exception: BaseException,
        exception_text: str,
        *,
        delegated: bool | None = None,
        stage: str,
    ) -> None:
        self._record_failure(
            config,
            op_name,
            case_index,
            posargs,
            kwargs,
            exception,
            exception_text,
            stage,
            delegated=delegated,
        )
        self._record_case(
            config,
            op_name,
            case_index,
            status,
            posargs,
            kwargs,
            delegated=delegated,
            stage=stage,
        )

    def _record_case(
        self,
        config: _FactoConfig,
        op_name: str,
        case_index: int,
        status: str,
        posargs: Sequence[Any],
        kwargs: dict[str, Any],
        *,
        delegated: bool | None = None,
        stage: str | None = None,
    ) -> None:
        if not config.cases_file:
            return

        record: dict[str, object] = {
            "backend": type(self).__name__,
            "case_index": case_index,
            "kwargs": _summarize_value(kwargs),
            "op": op_name,
            "posargs": _summarize_value(list(posargs)),
            "status": status,
        }
        if delegated is not None:
            record["delegated"] = delegated
        if stage is not None:
            record["stage"] = stage
        _append_jsonl(config.cases_file, record)

    def _record_summary(
        self,
        config: _FactoConfig,
        op_name: str,
        generated_count: int,
        skipped_count: int,
        eager_fail_count: int,
        export_fail_count: int,
        success_count_delegated: int,
        success_count_undelegated: int,
        fail_count: int,
    ) -> None:
        success_count = success_count_delegated + success_count_undelegated
        record: dict[str, object] = {
            "backend": type(self).__name__,
            "delegated": success_count_delegated,
            "eager_failures": eager_fail_count,
            "export_failures": export_fail_count,
            "failed": fail_count,
            "generated": generated_count,
            "op": op_name,
            "passed": success_count,
            "skipped": skipped_count,
            "undelegated": success_count_undelegated,
        }
        record_line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(f"FACTO_SUMMARY {record_line}")

        if config.summary_file:
            _append_jsonl(config.summary_file, record)

    def _run_generated_case(
        self,
        *,
        config: _FactoConfig,
        op: OpOverload,
        op_name: str,
        runtime_input_count: int,
        case_index: int,
        posargs: Sequence[Any],
        inkwargs: dict[str, Any],
    ) -> tuple[str, bool | None]:
        if self._should_skip_case(posargs):
            self._record_case(config, op_name, case_index, "skipped", posargs, inkwargs)
            return "skipped", None

        module_cls = get_module_for_op(op)
        model = module_cls(
            op, runtime_input_count, posargs[runtime_input_count:], inkwargs
        )

        try:
            model(*posargs[:runtime_input_count])
        except Exception as e:
            exception_text = traceback.format_exc()
            logger.error("Eager execution failed: %s", e)
            logger.error("%s", exception_text.rstrip())
            self._record_failed_case(
                config,
                op_name,
                case_index,
                "eager_failed",
                posargs,
                inkwargs,
                e,
                exception_text,
                stage="eager",
            )
            return "eager_failed", None

        tester = self._tester_factory(model, tuple(posargs[:runtime_input_count]))

        try:
            tester.export()
        except Exception as e:
            exception_text = traceback.format_exc()
            logger.error("Export failed.")
            logger.error("%s", exception_text.rstrip())
            self._record_failed_case(
                config,
                op_name,
                case_index,
                "export_failed",
                posargs,
                inkwargs,
                e,
                exception_text,
                stage=self._current_stage_name(tester),
            )
            return "export_failed", None

        tester.to_edge_transform_and_lower()
        is_delegated = any(
            n.target == torch._higher_order_ops.executorch_call_delegate
            for n in tester.stages[tester.cur].graph_module.graph.nodes
            if n.op == "call_function"
        )
        if is_delegated:
            self._run_delegated_case(tester)

        self._record_case(
            config,
            op_name,
            case_index,
            "passed",
            posargs,
            inkwargs,
            delegated=is_delegated,
        )
        return "passed", is_delegated

    def _test_op(self, op: OpOverload) -> None:  # noqa: C901
        """Run generated FACTO cases for one op and emit configured reports."""
        random_manager.seed(0)
        config = _facto_config()

        # Strip namespace
        op_name = op.name().split("::")[-1]

        # Default to .default overload
        if "." not in op_name:
            op_name += ".default"

        # Find and patch op spec
        if op_name not in CombinedSpecDB:
            raise ValueError(f"Operator {op_name} not found in SpecDictDB.")
        spec = _patch_spec(CombinedSpecDB[op_name])
        spec = self._patch_spec_for_backend(spec, op_name)

        runtime_input_count = FactoTestsBase.get_runtime_input_count(spec)

        logger.info("Op: %s, %s runtime inputs", op_name, runtime_input_count)

        # Run test cases
        generated_count = 0
        skipped_count = 0
        eager_fail_count = 0
        export_fail_count = 0
        success_count_delegated = 0
        success_count_undelegated = 0
        fail_count = 0

        for case_index, (posargs, inkwargs, _) in enumerate(
            ArgumentTupleGenerator(spec).gen()
        ):
            if config.max_cases is not None and case_index >= config.max_cases:
                break

            generated_count += 1
            is_delegated: bool | None = None

            try:
                status, is_delegated = self._run_generated_case(
                    config=config,
                    op=op,
                    op_name=op_name,
                    runtime_input_count=runtime_input_count,
                    case_index=case_index,
                    posargs=posargs,
                    inkwargs=inkwargs,
                )
                if status == "skipped":
                    skipped_count += 1
                    continue
                if status == "eager_failed":
                    eager_fail_count += 1
                    continue
                if status == "export_failed":
                    export_fail_count += 1
                    continue
                if is_delegated:
                    success_count_delegated += 1
                else:
                    success_count_undelegated += 1
            except Exception as e:
                fail_count += 1
                logger.error("Args:")
                for arg in posargs:
                    if isinstance(arg, torch.Tensor):
                        logger.error("  %s %s", arg.dtype, tuple(arg.shape))
                    else:
                        logger.error("  %s", arg)

                exception_text = traceback.format_exc()
                logger.error("%s", exception_text.rstrip())
                self._record_failed_case(
                    config,
                    op_name,
                    case_index,
                    "failed",
                    posargs,
                    inkwargs,
                    e,
                    exception_text,
                    delegated=is_delegated,
                    stage="lower_or_run",
                )

        logger.info(
            "%s PASS, %s FAIL",
            success_count_delegated + success_count_undelegated,
            fail_count,
        )
        logger.info(
            "  %s DELEGATED, %s UNDELEGATED",
            success_count_delegated,
            success_count_undelegated,
        )
        logger.info(
            "  %s GENERATED, %s SKIPPED, %s EAGER_FAILED, %s EXPORT_FAILED",
            generated_count,
            skipped_count,
            eager_fail_count,
            export_fail_count,
        )
        self._record_summary(
            config,
            op_name,
            generated_count,
            skipped_count,
            eager_fail_count,
            export_fail_count,
            success_count_delegated,
            success_count_undelegated,
            fail_count,
        )

        counted_failures = self._count_as_test_failures(
            eager_fail_count=eager_fail_count,
            export_fail_count=export_fail_count,
            fail_count=fail_count,
        )
        if counted_failures and self._should_fail_on_failures():
            self.fail(
                f"{op_name}: {counted_failures} FACTO-generated case(s) failed "
                f"(runtime={fail_count}, export={export_fail_count}, "
                f"eager={eager_fail_count})."
            )


# TODO Figure out where to put these
class FactoTestsXNNPACK(FactoTestsBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(XnnpackTester, *args, **kwargs)

    def _should_skip_case(self, posargs: Sequence[Any]) -> bool:
        if isinstance(posargs[0], torch.Tensor):
            # Temporary for getting around XNN crashes
            # (https://github.com/pytorch/executorch/issues/10960).
            # TODO Re-enable when resolved.
            if posargs[0].dtype in {torch.int8, torch.uint8}:
                print("Skipping (u)int8 case.")
                return True
        return False


_CONFIG = _facto_config()

for op_name in _CONFIG.selected_op_names:
    FactoTestsXNNPACK._generate_test(op_name)

if not FACTO_AVAILABLE:
    FactoTestsXNNPACK._generate_missing_facto_test()
elif _CONFIG.selected_ops_error is not None:
    FactoTestsXNNPACK._generate_configuration_error_test(
        "test_facto_ops_filter_matches_ops",
        _CONFIG.selected_ops_error,
    )


try:
    from executorch.backends.apple.coreml.test.tester import CoreMLTester

    class FactoTestsCoreML(FactoTestsBase):
        __test__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(CoreMLTester, *args, **kwargs)

    for op_name in _CONFIG.selected_op_names:
        FactoTestsCoreML._generate_test(op_name)
    if not FACTO_AVAILABLE:
        FactoTestsCoreML._generate_missing_facto_test()
    elif _CONFIG.selected_ops_error is not None:
        FactoTestsCoreML._generate_configuration_error_test(
            "test_facto_ops_filter_matches_ops",
            _CONFIG.selected_ops_error,
        )

except:
    print("Skipping Core ML facto tests as Core ML AOT is not available.")
