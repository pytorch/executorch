# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import itertools
import logging
import os
import random
import subprocess

import tempfile
import time
import traceback
import xml.etree.ElementTree as et
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from executorch import exir
from executorch.backends.qualcomm.export_utils import (
    convert_pt2e,
    generate_inputs,
    get_qnn_context_binary_alignment,
    prepare_pt2e,
    QnnConfig,
    QnnQuantizer,
    setup_common_args_and_variables,
    SimpleADB,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.utils import make_output_dir
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.program import ExecutorchProgramManager
from torchao.quantization.utils import compute_error


# test framework definitions
EXPECT_NOT_ANNOTATED = "expect to be annotated"
EXPECT_NOT_FULLY_DELEGATED = "expect to be fully delegated"
EXPECT_OUTPUT_CLOSE = "expect outputs to be closed\n"
EXPECT_OUTPUT_MATCH = "expect number of outputs to be matched"
TOTAL_TEST_COUNT = "total_test_count"

# et framework messages
EXCEPTION_EXIR_PROGRAM = "exir/program"
EXCEPTION_FROM_PASSES = "backends/qualcomm/_passes"
EXCEPTION_FROM_PREPROCESS = "backends/qualcomm/qnn_preprocess"


def check_exception(msg):
    def _check(msg, _: Exception):
        return msg in traceback.format_exc()

    return partial(_check, msg)


# extend this for backend agnostic tests
def default_property():
    @dataclass
    class Property:
        soc_model: str = "SM8750"

    return Property()


class Metrics(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __enter__(self):
        torch.set_printoptions(threshold=20)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_printoptions(threshold=1000)
        return exc_type is None

    def _check_bool(
        self,
        device_output: torch.Tensor,
        ref_output: torch.Tensor,
    ):
        device = device_output.to(torch.bool)
        ref = ref_output.to(torch.bool)
        device, ref = torch.broadcast_tensors(device, ref)
        assert torch.equal(device, ref), (
            f"{EXPECT_OUTPUT_CLOSE}"
            f"\toutput tensor shape: {device_output.shape}, dtype: {device_output.dtype}\n"
            f"\treference tensor shape: {ref_output.shape}, dtype: {ref_output.dtype}\n"
            f"\tmismatch count: {torch.count_nonzero(device ^ ref).item()} / {device.numel()}\n",
        )

    @abstractmethod
    def assert_close(
        self,
        device_output: Tuple[torch.Tensor],
        ref_output: Tuple[torch.Tensor],
    ):
        pass


class Tolerance(Metrics):
    def __init__(self, rtol=1, atol=1e-1):
        self.rtol = rtol
        self.atol = atol

    def assert_close(
        self,
        device_output: Tuple[torch.Tensor],
        ref_output: Tuple[torch.Tensor],
    ):
        assert len(ref_output) == len(device_output), EXPECT_OUTPUT_MATCH
        for i in range(len(ref_output)):
            if (
                device_output[i].dtype == torch.bool
                or ref_output[i].dtype == torch.bool
            ):
                self._check_bool(device_output[i], ref_output[i])
            else:
                isclose = torch.isclose(
                    device_output[i],
                    ref_output[i],
                    atol=self.atol,
                    rtol=self.rtol,
                )
                if not isclose.all():
                    mismatch = ~isclose
                    idx = mismatch.nonzero(as_tuple=False)
                    ref_vals = ref_output[i][mismatch]
                    dev_vals = device_output[i][mismatch]
                    total = int(mismatch.sum())
                    limit = 20
                    lines = [
                        f"  index={tuple(idx[k].tolist())}  ref={ref_vals[k].item()}  device={dev_vals[k].item()}"
                        for k in range(min(total, limit))
                    ]
                    if total > limit:
                        lines.append(f"  ... ({total - limit} more)")
                    raise AssertionError(
                        f"{EXPECT_OUTPUT_CLOSE}"
                        f"mismatch count: {total} / {mismatch.numel()}\n"
                        + "\n".join(lines)
                    )


class Sqnr(Metrics):
    def __init__(self, threshold):
        self.threshold = threshold

    def assert_close(
        self,
        device_output: Tuple[torch.Tensor],
        ref_output: Tuple[torch.Tensor],
    ):
        assert len(ref_output) == len(device_output), EXPECT_OUTPUT_MATCH
        for i in range(len(ref_output)):
            if (
                device_output[i].dtype == torch.bool
                or ref_output[i].dtype == torch.bool
            ):
                self._check_bool(device_output[i], ref_output[i])
            else:
                assert (
                    compute_error(ref_output[i], device_output[i]) >= self.threshold
                ), (
                    f"{EXPECT_OUTPUT_CLOSE}"
                    f"ref_output:\n{ref_output[i]}\n\ndevice_output:\n{device_output[i]}"
                )


class CosineSimilarity(Metrics):
    def __init__(self, threshold):
        self.threshold = threshold

    def assert_close(
        self,
        device_output: Tuple[torch.Tensor],
        ref_output: Tuple[torch.Tensor],
    ):
        assert len(ref_output) == len(device_output), EXPECT_OUTPUT_MATCH
        for i in range(len(ref_output)):
            if (
                device_output[i].dtype == torch.bool
                or ref_output[i].dtype == torch.bool
            ):
                self._check_bool(device_output[i], ref_output[i])
            else:
                assert (
                    torch.nn.functional.cosine_similarity(
                        ref_output[i].flatten(), device_output[i].flatten(), dim=0
                    )
                    >= self.threshold
                ), (
                    f"{EXPECT_OUTPUT_CLOSE}"
                    f"ref_output:\n{ref_output[i]}\n\ndevice_output:\n{device_output[i]}"
                )


class SkipOutputCheck(Metrics):
    def __init__(self):
        pass

    def assert_close(self, device_output, ref_output):
        pass


@pytest.fixture(autouse=True, scope="session")
def global_setup(request):
    # make sure things are reproducible
    seed = 1126
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # suppress stream
    stream_fd = [1, 2]
    std_stream = [os.dup(fd) for fd in stream_fd]

    with open(os.devnull, "w") as devnull:
        if not any(
            request.config.getoption(option) for option in ["--verbose", "--pdb"]
        ):
            for fd in stream_fd:
                os.dup2(devnull.fileno(), fd)
        yield

    # restore stream
    for i, fd in enumerate(stream_fd):
        os.dup2(std_stream[i], fd)
        os.close(std_stream[i])


@contextmanager
def temp_attribute(obj, attr, new_value):
    current_attr = getattr(obj, attr, None)
    try:
        setattr(obj, attr, new_value)
        yield
    finally:
        (
            setattr(obj, attr, current_attr)
            if current_attr is not None
            else delattr(obj, attr)
        )


# No direct depedency between qnn_config and global_setup.
# Keep following depedency for better practice so global_setup is done first.
@pytest.fixture(autouse=True, scope="session")
def qnn_config(global_setup, request):
    # generate QnnConfig for on-device test
    config = None
    try:
        config = QnnConfig.load_config(request.config.option)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f'invalid configuration detected, fall back to emulator workload:\n"{e}"'
        )
        config = QnnConfig(
            soc_model="unknown", build_folder="build-x86", compile_only=True
        )

    return config


def _get_expected_output(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
) -> None:
    with torch.no_grad():
        ref_outputs, outputs = [], module(*inputs)
        if isinstance(outputs, collections.OrderedDict):
            ref_outputs.append(outputs["out"].detach())
        elif isinstance(outputs, (list, tuple)):
            for output in outputs:
                ref_outputs.append(output.detach())
        else:
            ref_outputs.append(outputs.detach())

        return ref_outputs


class calibrate(object):
    def __init__(
        self,
        module: torch.nn.Module,
        inputs: List[Tuple[torch.Tensor]],
        quantizer: QnnQuantizer,
    ):
        self.module = module
        self.inputs = inputs
        self.quantizer = quantizer

    def __enter__(self):
        if self.quantizer is None:
            return self.module
        else:
            module = torch.export.export(
                self.module, self.inputs[0], strict=True
            ).module()
            module = prepare_pt2e(model=module, quantizer=self.quantizer)
            for input in self.inputs:
                module(*input)
            return convert_pt2e(model=module)

    def __exit__(self, *_):
        pass


def init_remote_env(func):
    initialized = False

    def wrapper(**kwargs):
        nonlocal initialized
        if not initialized:
            qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
            assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
            qnn_config = kwargs["qnn_config"]
            device_workspace = (
                f"{getattr(qnn_config, 'device_workspace', '')}_{qnn_config.backend}"
            )
            SimpleADB(
                qnn_config=qnn_config,
                pte_path=[],
                workspace=f"/data/local/tmp/{device_workspace}",
            ).push(
                init_env=True,
            )
            initialized = True
        return func(**kwargs)

    return wrapper


@init_remote_env
def invoke_remote(
    qnn_config: QnnConfig,
    executorch_prog: ExecutorchProgramManager,
    callback: callable,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pte_fname = f"{tmp_dir}/qnn_executorch_test.pte"
        device_workspace = (
            f"{getattr(qnn_config, 'device_workspace', '')}_{qnn_config.backend}"
        )
        with open(pte_fname, "wb") as file:
            executorch_prog.write_to_file(file)

        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=[pte_fname],
            workspace=f"/data/local/tmp/{device_workspace}",
        )
        adb.push()
        callback(adb)


@init_remote_env
def verify_output_remote(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    executorch_prog: ExecutorchProgramManager,
    metrics: Metrics,
    qnn_config: QnnConfig,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_outputs = _get_expected_output(module=module, inputs=inputs)
        output_dir = f"{tmp_dir}/outputs"
        device_outputs = []
        pte_fname = f"{tmp_dir}/qnn_executorch_test.pte"
        device_workspace = (
            f"{getattr(qnn_config, 'device_workspace', '')}_{qnn_config.backend}"
        )

        generate_inputs(tmp_dir, "input_list.txt", [inputs])
        make_output_dir(output_dir)
        with open(pte_fname, "wb") as file:
            executorch_prog.write_to_file(file)

        def post_process():
            for i, f in enumerate(
                sorted(f for f in os.listdir(output_dir) if f.endswith(".raw"))
            ):
                dtype = ref_outputs[i].numpy().dtype
                filename = os.path.join(output_dir, f)
                output = np.fromfile(filename, dtype=dtype)
                device_outputs.append(
                    torch.from_numpy(output).reshape(ref_outputs[i].shape)
                )

        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=[pte_fname],
            workspace=f"/data/local/tmp/{device_workspace}",
        )
        adb.push(inputs=[inputs], init_env=False)
        adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
        adb.execute(method_index=getattr(qnn_config, "method_index", 0))
        adb.pull(host_output_path=tmp_dir, callback=post_process)
        metrics.assert_close(device_output=device_outputs, ref_output=ref_outputs)


def verify_output_emulator(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    executorch_prog: ExecutorchProgramManager,
    metrics: Metrics,
    **_: Any,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_outputs = _get_expected_output(module=module, inputs=inputs)
        output_dir = f"{tmp_dir}/outputs"
        pte_fname = f"{tmp_dir}/qnn_executorch_test.pte"
        target = "x86_64-linux-clang"
        qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
        assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        build_folder = os.path.join(os.getcwd(), "build-x86")

        generate_inputs(tmp_dir, "input_list.txt", [inputs])
        make_output_dir(output_dir)
        with open(pte_fname, "wb") as file:
            executorch_prog.write_to_file(file)

        cmd = [
            f"{build_folder}/examples/qualcomm/executor_runner/qnn_executor_runner",
            "--model_path",
            pte_fname,
            "--input_list_path",
            f"{tmp_dir}/input_list.txt",
            "--output_folder_path",
            output_dir,
        ]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{qnn_sdk}/lib/{target}/:{build_folder}/lib"
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=tmp_dir,
        )
        assert (
            proc.returncode == 0
        ), f"qnn_executorch_runner return {proc.returncode}, STDOUT=\n {proc.stdout}"

        # post_process
        device_outputs = []
        for i, f in enumerate(
            sorted(f for f in os.listdir(output_dir) if f.endswith(".raw"))
        ):
            dtype = ref_outputs[i].numpy().dtype
            filename = os.path.join(output_dir, f)
            output = np.fromfile(filename, dtype=dtype)
            output = torch.from_numpy(output).reshape(ref_outputs[i].shape)
            device_outputs.append(output)
        metrics.assert_close(device_output=device_outputs, ref_output=ref_outputs)


def export_and_verify(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    qnn_config: QnnConfig,
    quantizer: QnnQuantizer,
    compile_specs: List[Any],
    metrics: Metrics,
):
    with calibrate(module, [inputs], quantizer) as exported_module:
        if quantizer is not None:
            nodes = {node.target for node in exported_module.graph.nodes}
            q_and_dq = {
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.quantized_decomposed.quantize_per_channel.default,
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                torch.ops.torchao.quantize_affine.default,
                torch.ops.torchao.dequantize_affine.default,
            }
            assert nodes.intersection(q_and_dq), EXPECT_NOT_ANNOTATED

    delegated_prog = to_edge_transform_and_lower_to_qnn(
        module=exported_module,
        inputs=inputs,
        compiler_specs=compile_specs,
    )
    executorch_prog = delegated_prog.to_executorch(
        exir.ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=not qnn_config.shared_buffer,
                alloc_graph_output=not qnn_config.shared_buffer,
            ),
            segment_alignment=get_qnn_context_binary_alignment(),
        )
    )
    execution_plan = executorch_prog.executorch_program.execution_plan[0]
    assert all(
        [
            len(execution_plan.delegates) == 1,
            execution_plan.delegates[0].id == "QnnBackend",
            len(execution_plan.operators) == 0,
        ]
    ), EXPECT_NOT_FULLY_DELEGATED

    mode = "emulator" if qnn_config.build_folder == "build-x86" else "remote"
    globals()[f"verify_output_{mode}"](
        module=module,
        inputs=inputs,
        executorch_prog=executorch_prog,
        metrics=metrics,
        qnn_config=qnn_config,
    )


def pytest_addoption(parser):
    setup_common_args_and_variables(parser=parser)
    parser.addoption(
        "--test_report",
        type=str,
        help="Specify report path while testing",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    from pytest_subtests.plugin import SubTestReport

    def get_num_subtests(reports):
        category = {}
        for report in reports:
            category.setdefault(report.nodeid, [[], []])
            category[report.nodeid][isinstance(report, SubTestReport)].append(report)

        return (
            sum(max(len(mono), len(sub)) for mono, sub in category.values()),
            category,
        )

    failed_tests = [
        report
        for report in itertools.chain.from_iterable(
            terminalreporter.stats.get(key, []) for key in ("failed",)
        )
        if report.when == "call"
    ]
    num_failed_tests, failed_reports = get_num_subtests(failed_tests)
    num_tests = sum(
        max(count - 1, 1) for count in config.stash[TOTAL_TEST_COUNT].values()
    )
    elapsed_time = time.time() - terminalreporter._session_start.time
    statistics = (
        f"QUALCOMM EXECUTORCH TEST SUMMARY [PASSED]/[TOTAL]:"
        f"[{num_tests - num_failed_tests}]/[{num_tests}] in {elapsed_time}s"
    )
    terminalreporter.write_sep("*", statistics, bold=True, yellow=True)

    if test_report := config.getoption("--test_report", None):
        root = et.Element("report")
        for id, (mono, sub) in failed_reports.items():
            current_test = et.SubElement(root, id)
            reports = sub if len(sub) > len(mono) else mono
            for report in reports:
                index = report.longreprtext.rfind("\n")
                short_exception = report.longreprtext[index + 1 :]
                is_subtests = isinstance(report, SubTestReport)
                et.SubElement(
                    current_test,
                    "subtest" if is_subtests else "test",
                    {
                        "msg": str(report.context.msg) if is_subtests else "none",
                        "error": short_exception,
                    },
                )

        et.indent(root)
        tree = et.ElementTree(root)
        tree.write(test_report, encoding="utf-8", xml_declaration=True)


def pytest_configure(config):
    config.stash[TOTAL_TEST_COUNT] = defaultdict(int)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        item.session.config.stash[TOTAL_TEST_COUNT][report.nodeid] += 1
