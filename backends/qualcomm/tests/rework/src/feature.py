# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
import tempfile
from functools import partial, reduce
from operator import mul

import pytest

import torch

from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_sample import (
    QcomCosineSimilarityComparator,
)
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    QNNIntermediateDebugger,
)
from executorch.backends.qualcomm.export_utils import (
    make_quantizer,
    QcomChipset,
    QnnConfig,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPerformanceMode,
    SimpleADB,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchGpuPerformanceMode,
    QnnExecuTorchLpaiClientPerf,
    QnnExecuTorchProfileLevel,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    calibrate,
    export_and_verify,
    invoke_remote,
    temp_attribute,
    verify_output_remote,
)
from executorch.backends.qualcomm.utils.utils import update_spill_fill_size
from executorch.devtools import Inspector
from executorch.devtools.inspector._inspector_utils import TimeScale


def unpack_fixtures(func):
    def wrapper(request, kwargs):
        params = inspect.signature(func).parameters
        extra_fixtures = set(params.keys()) - set(kwargs.keys())
        new_kwargs = {key: request.getfixturevalue(key) for key in extra_fixtures}
        # hack qnn_config to get unique test folder
        with temp_attribute(
            new_kwargs["qnn_config"], "device_workspace", __name__.replace(".", "_")
        ):
            return func(**new_kwargs, **kwargs)

    return wrapper


def get_quantizer(qnn_config: QnnConfig):
    return (
        make_quantizer(
            backend=qnn_config.backend,
            soc_model=qnn_config.soc_model,
        )
        if qnn_config.backend != QnnExecuTorchBackendType.kGpuBackend
        else None
    )


class Logging:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

    @staticmethod
    def _get_log_pattern(backend):
        return {
            QnnExecuTorchBackendType.kHtpBackend: "QnnDsp <V>",
            QnnExecuTorchBackendType.kGpuBackend: "OpenCL",
            # looks like no special keyword appears
            QnnExecuTorchBackendType.kLpaiBackend: "",
        }[backend]

    @staticmethod
    def _test(qnn_config, compile_specs, expected, aot):
        def callback(adb: SimpleADB, pattern):
            def verify(log):
                msg = log.stdout
                assert pattern in msg, f"{pattern} in log"

            # QnnExecuTorchLogLevel.kLogLevelVerbose
            adb.extra_cmds += "" if aot else " --log_level 4"
            adb.execute(output_callback=verify)

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=compile_specs,
                ).to_executorch()
                # remote testing
                invoke_remote(
                    qnn_config=qnn_config,
                    executorch_prog=executorch_prog_mgr,
                    callback=partial(
                        callback,
                        pattern=Logging._get_log_pattern(qnn_config.backend),
                    ),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                {"soc_model": soc_model, "debug": True, "use_fp16": False},
                {"soc_model": soc_model, "debug": False, "use_fp16": False},
            ],
            QnnExecuTorchBackendType.kGpuBackend: [
                {"soc_model": soc_model, "debug": True, "online_prepare": True},
                {"soc_model": soc_model, "debug": False, "online_prepare": True},
            ],
            QnnExecuTorchBackendType.kLpaiBackend: [
                {"soc_model": soc_model, "debug": True},
                {"soc_model": soc_model, "debug": False},
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=compile_specs(
                        tuple(backend_compile_specs[qnn_config.backend][i].items())
                    ),
                    expected=expected,
                    aot=config == "compile_time_option",
                )


class MultiGraph:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = 8192
            self.up_proj = torch.nn.Linear(512, hidden_dim)
            self.down_proj = torch.nn.Linear(hidden_dim, 512)

        def example_inputs(self):
            return (torch.randn(1, 256, 512),)

        def forward(self, x):
            return self.down_proj(self.up_proj(x))

    @staticmethod
    def _test(qnn_config, compile_specs, expected, weight_sharing=False):
        def compile(models, compile_specs):
            graph_names = ["up_down_proj", "up_proj"]
            modules_dict = {}
            sample_inputs_dict = {}
            compile_specs_dict = {}
            inputs = models[0].example_inputs()
            for i, graph_name in enumerate(graph_names):
                with calibrate(
                    models[i],
                    [inputs],
                    get_quantizer(qnn_config),
                ) as model:
                    modules_dict[graph_name] = model
                    sample_inputs_dict[graph_name] = inputs
                    compile_specs_dict[graph_name] = compile_specs

            # start lowering
            return to_edge_transform_and_lower_to_qnn(
                module=modules_dict,
                inputs=inputs,
                compiler_specs=compile_specs_dict,
            ).to_executorch()

        model = __class__.Model()
        models = [model, model.up_proj]

        if weight_sharing:
            vanilla, weight_shared = (
                compile(models=models, compile_specs=cs) for cs in compile_specs
            )
            assert len(weight_shared.buffer) < len(
                vanilla.buffer
            ), "weight-shared PTE size is expected to be smaller or equal"
        else:
            with expected as metrics:
                for method_index in range(2):
                    with temp_attribute(qnn_config, "method_index", method_index):
                        verify_output_remote(
                            module=models[method_index],
                            inputs=model.example_inputs(),
                            executorch_prog=compile(
                                models=models, compile_specs=compile_specs
                            ),
                            metrics=metrics,
                            qnn_config=qnn_config,
                        )

    @staticmethod
    @unpack_fixtures
    def test_weight_sharing(qnn_config, compile_specs, expected):
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                compile_specs(tuple(d.items()))
                for d in [
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "use_fp16": False,
                        "use_weight_sharing": False,
                    },
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "use_fp16": False,
                        "use_weight_sharing": True,
                    },
                ]
            ],
        }

        __class__._test(
            qnn_config=qnn_config,
            compile_specs=backend_compile_specs[qnn_config.backend],
            expected=expected,
            weight_sharing=True,
        )

    @staticmethod
    @unpack_fixtures
    def test_inference(qnn_config, compile_specs, expected):
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: {
                "soc_model": soc_model,
                "use_fp16": False,
            },
            QnnExecuTorchBackendType.kGpuBackend: {
                "soc_model": soc_model,
                "online_prepare": True,
            },
            QnnExecuTorchBackendType.kLpaiBackend: {"soc_model": soc_model},
        }

        __class__._test(
            qnn_config=qnn_config,
            compile_specs=compile_specs(
                tuple(backend_compile_specs[qnn_config.backend].items())
            ),
            expected=expected,
        )


class OnlinePrepare:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: {
                "soc_model": soc_model,
                "online_prepare": True,
                "use_fp16": False,
            },
            QnnExecuTorchBackendType.kGpuBackend: {
                "soc_model": soc_model,
                "online_prepare": True,
            },
        }

        module = __class__.Model()
        export_and_verify(
            module=module,
            inputs=module.example_inputs(),
            qnn_config=qnn_config,
            quantizer=get_quantizer(qnn_config),
            compile_specs=compile_specs(
                tuple(backend_compile_specs[qnn_config.backend].items())
            ),
            metrics=expected,
        )


class Performance:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

    @staticmethod
    def _test(qnn_config, compile_specs, expected, aot):
        # extend this for other backends
        def callback_htp(adb: SimpleADB, voltage):
            def verify(log):
                msg = log.stdout
                # refer to HtpDevice.cpp for the following values
                min_voltage = f"coreVoltageCornerMin {voltage}"
                assert min_voltage in msg, f"expecting '{min_voltage}' in log"

            # high power saver mode
            adb.extra_cmds += "" if aot else " --htp_performance_mode 6"
            adb.execute(output_callback=verify)

        # TODO: extend performance check for following backends
        def callback_gpu(adb: SimpleADB):
            adb.execute()

        def callback_lpai(adb: SimpleADB):
            adb.execute()

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=compile_specs,
                ).to_executorch()
                dispatcher = {
                    QnnExecuTorchBackendType.kHtpBackend: callback_htp,
                    QnnExecuTorchBackendType.kGpuBackend: callback_gpu,
                    QnnExecuTorchBackendType.kLpaiBackend: callback_lpai,
                }
                # remote testing
                invoke_remote(
                    qnn_config=qnn_config,
                    executorch_prog=executorch_prog_mgr,
                    callback=(
                        partial(dispatcher[qnn_config.backend], voltage=80)
                        if qnn_config.backend == QnnExecuTorchBackendType.kHtpBackend
                        else dispatcher[qnn_config.backend]
                    ),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                # compile_time option
                {
                    "soc_model": soc_model,
                    "debug": True,
                    "use_fp16": False,
                    "htp_performance_mode": QnnExecuTorchHtpPerformanceMode.kHtpHighPowerSaver,
                },
                # runtime_option (performance mode defaults to kHtpBurst)
                {"soc_model": soc_model, "debug": True, "use_fp16": False},
            ],
            QnnExecuTorchBackendType.kGpuBackend: [
                # compile_time option: set low perf hint to GPU
                {
                    "soc_model": soc_model,
                    "online_prepare": True,
                    "performance_mode": QnnExecuTorchGpuPerformanceMode.kGpuPerfHintLow,
                },
                # runtime_option: scaffold — GPUruntime perf hint not yet wired in C++
                # TODO: extend GPU runtime to accept dynamic performance settings
                {
                    "soc_model": soc_model,
                    "debug": True,
                    "online_prepare": True,
                },
            ],
            QnnExecuTorchBackendType.kLpaiBackend: [
                {
                    "soc_model": soc_model,
                    "fps": 30,
                    "ftrt_ratio": 10,
                    "client_perf_type": QnnExecuTorchLpaiClientPerf.kRealTime,
                },
                # runtime_option: scaffold — LPAI runtime perf hint not yet wired in C++
                # TODO: extend LPAI runtime to accept dynamic performance settings
                {"soc_model": soc_model, "debug": True},
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=compile_specs(
                        tuple(backend_compile_specs[qnn_config.backend][i].items())
                    ),
                    expected=expected,
                    aot=config == "compile_time_option",
                )


class Profile:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
            )
            self.relu = torch.nn.ReLU()

        def example_inputs(self):
            return (torch.randn(1, 32, 36, 36),)

        def forward(self, x):
            return self.relu(self.conv2d(x))

    @staticmethod
    def _test(qnn_config, compile_specs, expected, aot):
        def callback(adb: SimpleADB, executorch_prog_mgr, expected_profile_events):
            with tempfile.TemporaryDirectory() as tmp_dir:
                etdump_path = f"{tmp_dir}/etdump.etdp"
                etrecord_path = f"{tmp_dir}/etrecord.bin"

                adb.extra_cmds += "" if aot else " --profile_level 2"
                adb.execute()
                adb.pull_etdump(output_path=etdump_path)

                executorch_prog_mgr.get_etrecord().save(etrecord_path)
                inspector = Inspector(
                    etdump_path=etdump_path,
                    etrecord=etrecord_path,
                    source_time_scale=TimeScale.CYCLES,
                    target_time_scale=TimeScale.CYCLES,
                )
                assert len(inspector.to_dataframe().index) >= expected_profile_events, (
                    f"unexpected number of profile events, expecting "
                    f"{expected_profile_events}, but has {len(inspector.to_dataframe().index)} events.",
                )

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=compile_specs,
                    generate_etrecord=True,
                ).to_executorch()
                # remote testing
                invoke_remote(
                    qnn_config=qnn_config,
                    executorch_prog=executorch_prog_mgr,
                    callback=partial(
                        callback,
                        executorch_prog_mgr=executorch_prog_mgr,
                        expected_profile_events=2,
                    ),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                # compile_time option
                {
                    "soc_model": soc_model,
                    "profile_level": QnnExecuTorchProfileLevel.kProfileDetailed,
                    "use_fp16": False,
                },
                # runtime_option
                {"soc_model": soc_model, "use_fp16": False},
            ],
            QnnExecuTorchBackendType.kGpuBackend: [
                # compile_time option
                {
                    "soc_model": soc_model,
                    "profile_level": QnnExecuTorchProfileLevel.kProfileDetailed,
                    "online_prepare": True,
                },
                # runtime_option
                {"soc_model": soc_model, "online_prepare": True},
            ],
            QnnExecuTorchBackendType.kLpaiBackend: [
                # compile_time option
                {
                    "soc_model": soc_model,
                    "profile_level": QnnExecuTorchProfileLevel.kProfileDetailed,
                },
                # runtime_option
                {"soc_model": soc_model},
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=compile_specs(
                        tuple(backend_compile_specs[qnn_config.backend][i].items())
                    ),
                    expected=expected,
                    aot=config == "compile_time_option",
                )


class Saver:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
            flatbuffer_to_option,
            option_to_flatbuffer,
        )

        # saver=True is a top-level QnnExecuTorchOptions field; works across backends
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: {
                "soc_model": soc_model,
                "saver": True,
                "use_fp16": False,
            },
            QnnExecuTorchBackendType.kGpuBackend: {
                "soc_model": soc_model,
                "saver": True,
                "online_prepare": True,
            },
            QnnExecuTorchBackendType.kLpaiBackend: {
                "soc_model": soc_model,
                "saver": True,
            },
        }

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # hack saver output folder
                    cs = compile_specs(
                        tuple(backend_compile_specs[qnn_config.backend].items())
                    )
                    option = flatbuffer_to_option(cs[0].value)
                    option.saver_output_dir = f"{tmp_dir}/saver_output"
                    cs[0].value = option_to_flatbuffer(option)
                    with pytest.raises(SystemExit):
                        to_edge_transform_and_lower_to_qnn(
                            module=model,
                            inputs=inputs,
                            compiler_specs=cs,
                        )
                    assert all(
                        os.path.isfile(f)
                        for f in [
                            f"{tmp_dir}/saver_output/params.bin",
                            f"{tmp_dir}/saver_output/saver_output.c",
                        ]
                    )


class SharedBuffer:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        # shared_buffer=True is a top-level QnnExecuTorchOptions field; works across backends
        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: {
                "soc_model": soc_model,
                "shared_buffer": True,
                "use_fp16": False,
            },
            QnnExecuTorchBackendType.kGpuBackend: {
                "soc_model": soc_model,
                "shared_buffer": True,
                "online_prepare": True,
            },
            QnnExecuTorchBackendType.kLpaiBackend: {
                "soc_model": soc_model,
                "shared_buffer": True,
            },
        }

        module = __class__.Model()
        qnn_config.shared_buffer = True
        export_and_verify(
            module=module,
            inputs=module.example_inputs(),
            qnn_config=qnn_config,
            quantizer=get_quantizer(qnn_config),
            compile_specs=compile_specs(
                tuple(backend_compile_specs[qnn_config.backend].items())
            ),
            metrics=expected,
        )


class SpillFill:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = 8192
            self.up_proj = torch.nn.ModuleList(
                [torch.nn.Linear(512, hidden_dim) for _ in range(3)]
            )
            self.down_proj = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_dim, 512) for _ in range(3)]
            )

        def example_inputs(self):
            return (torch.randn(1, 256, 512),)

        def forward(self, x):
            up_proj = [linear(x) for linear in self.up_proj]
            return reduce(
                mul, [linear(up_proj[i]) for i, linear in enumerate(self.down_proj)]
            )

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "use_multi_contexts": True,
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        with expected:
            # perform ptq
            model = __class__.Model()
            inputs = model.example_inputs()
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=backend_compile_specs[qnn_config.backend],
                )
                max_spill_fill_size = update_spill_fill_size(
                    edge_prog_mgr.exported_program()
                )
                assert max_spill_fill_size > 0


class TensorDump:
    # Simple Conv2d+ReLU model that is supported by all backends
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()

        def example_inputs(self):
            return (torch.randn(1, 3, 8, 8),)

        def forward(self, x):
            return self.relu(self.conv(x))

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        def callback(adb: SimpleADB, debugger, expected_compared_events):
            with tempfile.TemporaryDirectory() as tmp_dir:
                etdump_path = f"{tmp_dir}/etdump.etdp"
                debug_output_path = f"{tmp_dir}/debug_output.bin"
                adb.execute()
                adb.pull_debug_output(
                    etdump_path=etdump_path, debug_buffer_path=debug_output_path
                )
                debugger.setup_inspector(
                    etdump_path=etdump_path,
                    debug_buffer_path=debug_output_path,
                )
                comparator = debugger.create_comparator(QcomCosineSimilarityComparator)
                numeric_results = debugger.inspector.calculate_numeric_gap(
                    distance=comparator,
                    reference_graph=debugger.reference_graph_name,
                )
                numeric_results = numeric_results.set_index("runtime_debug_handle")
                assert len(numeric_results) == expected_compared_events, (
                    f"unexpected number of compared events, expecting "
                    f"{expected_compared_events}, but has {len(numeric_results)} events."
                )
                for _, row in numeric_results.iterrows():
                    assert comparator.is_valid_score(row.gap[0]), (
                        f"Node {row.aot_ops} is failing "
                        f"{comparator.metric_name()} test, {row.gap[0]} is lower "
                        f"than {comparator.threshold}."
                    )

        soc_model = getattr(QcomChipset, qnn_config.soc_model)
        # dump_intermediate_outputs=True is a top-level QnnExecuTorchOptions field; works across backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: {
                "soc_model": soc_model,
                "dump_intermediate_outputs": True,
                "use_fp16": False,
            },
            QnnExecuTorchBackendType.kGpuBackend: {
                "soc_model": soc_model,
                "dump_intermediate_outputs": True,
                "online_prepare": True,
            },
            QnnExecuTorchBackendType.kLpaiBackend: {
                "soc_model": soc_model,
                "dump_intermediate_outputs": True,
            },
        }

        with expected:
            # perform ptq
            model = __class__.Model()
            inputs = model.example_inputs()
            with calibrate(model, [inputs], get_quantizer(qnn_config)) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=compile_specs(
                        tuple(backend_compile_specs[qnn_config.backend].items())
                    ),
                    generate_etrecord=True,
                ).to_executorch()

                with tempfile.TemporaryDirectory() as etrecord_dir:
                    etrecord_path = f"{etrecord_dir}/etrecord.bin"
                    etrecord = executorch_prog_mgr.get_etrecord()
                    debugger = QNNIntermediateDebugger(inputs)
                    debugger.set_etrecord_file_path(etrecord_path)
                    debugger.set_edge_ep(
                        edge_ep=etrecord.graph_map[debugger.reference_graph_name]
                    )
                    etrecord.update_representative_inputs(debugger.sample_input)
                    etrecord.save(etrecord_path)

                    # remote testing
                    qnn_config.dump_intermediate_outputs = True
                    invoke_remote(
                        qnn_config=qnn_config,
                        executorch_prog=executorch_prog_mgr,
                        inputs=inputs,
                        # conv + relu = 2 intermediate outputs
                        callback=partial(
                            callback,
                            debugger=debugger,
                            expected_compared_events=2,
                        ),
                    )
