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

from executorch.backends.qualcomm.export_utils import (
    make_quantizer,
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPerformanceMode,
    SimpleADB,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
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


class Logging:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def example_inputs(self):
            return (torch.randn(1, 2, 3, 4),)

        def forward(self, x):
            return torch.nn.ReLU()(x)

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
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
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
                    callback=partial(callback, pattern="QnnDsp <V>"),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                compile_specs(tuple(d.items()))
                for d in [
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "debug": True,
                        "use_fp16": False,
                    },
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "debug": False,
                        "use_fp16": False,
                    },
                ]
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=backend_compile_specs[qnn_config.backend][i],
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
                    make_quantizer(soc_model=qnn_config.soc_model),
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
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        __class__._test(
            qnn_config=qnn_config,
            compile_specs=backend_compile_specs[qnn_config.backend],
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
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "online_prepare": True,
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        module = __class__.Model()
        qnn_config.online_prepare = True
        export_and_verify(
            module=module,
            inputs=module.example_inputs(),
            qnn_config=qnn_config,
            quantizer=make_quantizer(soc_model=qnn_config.soc_model),
            compile_specs=backend_compile_specs[qnn_config.backend],
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

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=compile_specs,
                ).to_executorch()
                # verifier per backend
                dispatcher = {
                    QnnExecuTorchBackendType.kHtpBackend: callback_htp,
                }
                # remote testing
                invoke_remote(
                    qnn_config=qnn_config,
                    executorch_prog=executorch_prog_mgr,
                    callback=partial(dispatcher[qnn_config.backend], voltage=80),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                compile_specs(tuple(d.items()))
                for d in [
                    # compile_time option
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "debug": True,
                        "use_fp16": False,
                        "htp_performance_mode": QnnExecuTorchHtpPerformanceMode.kHtpHighPowerSaver,
                    },
                    # runtime_option (performance mode defaults to kHtpBurst)
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "debug": True,
                        "use_fp16": False,
                    },
                ]
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=backend_compile_specs[qnn_config.backend][i],
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
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
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
                        expected_profile_events=20,
                    ),
                )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, compile_specs, expected):
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: [
                compile_specs(tuple(d.items()))
                for d in [
                    # compile_time option
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "profile_level": QnnExecuTorchProfileLevel.kProfileDetailed,
                        "use_fp16": False,
                    },
                    # runtime_option
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "use_fp16": False,
                    },
                ]
            ],
        }

        for i, config in enumerate(["compile_time_option", "runtime_option"]):
            with subtests.test(msg=config):
                __class__._test(
                    qnn_config=qnn_config,
                    compile_specs=backend_compile_specs[qnn_config.backend][i],
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

        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "saver": True,
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        with expected:
            # model declaration
            model = __class__.Model()
            inputs = model.example_inputs()
            # perform ptq
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
                # start lowering
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # hack saver output folder
                    cs = backend_compile_specs[qnn_config.backend]
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
        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "shared_buffer": True,
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        module = __class__.Model()
        qnn_config.shared_buffer = True
        export_and_verify(
            module=module,
            inputs=module.example_inputs(),
            qnn_config=qnn_config,
            quantizer=make_quantizer(soc_model=qnn_config.soc_model),
            compile_specs=backend_compile_specs[qnn_config.backend],
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
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
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
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.idx_source = torch.rand(10, 3)

        def example_inputs(self):
            return (torch.randn(3, 10),)

        def forward(self, x):
            a, b = torch.topk(x, 3)
            return a + self.idx_source[b]

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, compile_specs, expected):
        def callback(adb: SimpleADB, expected_intermediate_events):
            with tempfile.TemporaryDirectory() as tmp_dir:
                etdump_path = f"{tmp_dir}/etdump.etdp"
                debug_output_path = f"{tmp_dir}/debug_output.bin"
                adb.execute()
                adb.pull_debug_output(
                    etdump_path=etdump_path, debug_buffer_path=debug_output_path
                )
                inspector = Inspector(
                    etdump_path=etdump_path, debug_buffer_path=debug_output_path
                )
                for event_block in inspector.event_blocks:
                    if event_block.name == "Execute":
                        assert (
                            len(event_block.events) == expected_intermediate_events
                        ), (
                            f"unexpected number of intermediate events, expecting "
                            f"{expected_intermediate_events}, but has {len(event_block.events)} events.",
                        )

        # extend this for other backends
        backend_compile_specs = {
            QnnExecuTorchBackendType.kHtpBackend: compile_specs(
                tuple(
                    {
                        "soc_model": getattr(QcomChipset, qnn_config.soc_model),
                        "dump_intermediate_outputs": True,
                        "use_fp16": False,
                    }.items()
                )
            ),
        }

        with expected:
            # perform ptq
            model = __class__.Model()
            inputs = model.example_inputs()
            with calibrate(
                model, [inputs], make_quantizer(soc_model=qnn_config.soc_model)
            ) as model:
                # start lowering
                executorch_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module=model,
                    inputs=inputs,
                    compiler_specs=backend_compile_specs[qnn_config.backend],
                    generate_etrecord=True,
                ).to_executorch()
                # remote testing
                qnn_config.dump_intermediate_outputs = True
                invoke_remote(
                    qnn_config=qnn_config,
                    executorch_prog=executorch_prog_mgr,
                    callback=partial(callback, expected_intermediate_events=9),
                )
