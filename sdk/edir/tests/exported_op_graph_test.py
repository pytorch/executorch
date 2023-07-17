import copy
import unittest
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from executorch import exir
from executorch.exir import (
    ExecutorchBackendConfig,
    ExecutorchProgram,
    ExirExportedProgram,
)
from executorch.exir.delegate import LoweredBackendModule, patch_lowered_functions
from executorch.sdk.edir.base_schema import OperatorNode
from executorch.sdk.edir.et_schema import (
    ExportedETOperatorGraph,
    FXOperatorGraph,
    InferenceRun,
)
from executorch.sdk.etdump.schema import ETDump, ProfileBlock, ProfileEvent, RunData

from parameterized import parameterized
from torch import Tensor


class TwoLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = torch.nn.Linear(4, 5)
        self.lin_2 = torch.nn.Linear(5, 6)

    def forward(self, x):
        x = self.lin_1(x)
        x = nn.functional.relu(x)
        return self.lin_2(x)

    def get_random_inputs(self):
        return (torch.randint(-10, 100, (4,), dtype=torch.float32),)

    # Helper to generate sample profile times
    # For testing, the end time of a sample is defined as the start time of the next sample
    def gen_sample_profile_times(self) -> List[List[int]]:
        sample_times = [
            10000,
            20000,
            50000,
            100000,
            1000000,
            1001000,
            1500000,
            15000000,
            15000500,
            15001000,
            15001100,
            16000000,
            16020000,
            20000000,
        ]
        profile_times_slow = [2 * x for x in sample_times]
        return [sample_times, profile_times_slow]

    def gen_inference_run(self) -> InferenceRun:
        samples = self.gen_sample_profile_times()
        node_metadata = {}

        ops_count = len(samples[0]) - 1
        for op_index in range(ops_count):
            profile_start_times = [
                samples[sample_index][op_index] for sample_index in range(len(samples))
            ]
            profile_end_times = [
                samples[sample_index][op_index + 1]
                for sample_index in range(len(samples))
            ]

            node_metadata[op_index] = {
                "profile_start_time": profile_start_times,
                "profile_end_time": profile_end_times,
            }

        return InferenceRun(
            node_metadata,
            {
                "profile_start_time": [sample[0] for sample in samples],
                "profile_end_time": [sample[-1] for sample in samples],
                "load_start_time": 5000,
                "load_end_time": 10000,
            },
        )

    def gen_et_dump(self) -> ETDump:
        samples = self.gen_sample_profile_times()
        profile_inference_blocks = []
        for profile_times in samples:
            profile_events = [
                ProfileEvent(
                    name="ExecPlan::execute",
                    debug_handle=0,
                    start_time=profile_times[0],
                    end_time=profile_times[-1],
                )
            ]
            for index, (start_time, end_time) in enumerate(
                zip(profile_times[:-1], profile_times[1:])
            ):
                profile_events.append(
                    ProfileEvent(
                        name="OPERATOR_CALL",
                        debug_handle=index,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
                profile_events.append(
                    ProfileEvent(
                        name=f"<operator_run_{index}>",
                        debug_handle=index,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            profile_inference_blocks.append(
                ProfileBlock(
                    name="inference block",
                    allocators=[],
                    profile_events=profile_events,
                    allocation_events=[],
                )
            )

        profile_blocks = [
            ProfileBlock(
                name="default",
                allocators=[],
                profile_events=[
                    ProfileEvent(
                        name="ExecPlan::init_execution_plan",
                        debug_handle=0,
                        start_time=5000,
                        end_time=10000,
                    )
                ],
                allocation_events=[],
            ),
        ] + profile_inference_blocks

        return ETDump(
            version=0,
            run_data=[
                RunData(
                    debug_blocks=[],
                    profile_blocks=profile_blocks,
                )
            ],
        )


# TODO: Use a gen_sample_profile_times function similar to TwoLinearModule
class MultiOutputNodeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    def forward(self, t1):
        x = self.p1.add(t1).asin().mul(t1)
        y = self.p1.mul(x)
        return torch.sigmoid(x + y), t1

    def get_random_inputs(self):
        return (torch.randint(-10, 100, (4,), dtype=torch.float32),)

    def gen_et_dump(self) -> ETDump:
        return ETDump(
            version=0,
            run_data=[
                RunData(
                    debug_blocks=[],
                    profile_blocks=[
                        ProfileBlock(
                            name="default",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::init_execution_plan",
                                    debug_handle=0,
                                    start_time=5000,
                                    end_time=10000,
                                )
                            ],
                            allocation_events=[],
                        ),
                        ProfileBlock(
                            name="inference loop",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::execute",
                                    debug_handle=0,
                                    start_time=10000,
                                    end_time=1500000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=0,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=0,
                                    start_time=10000,
                                    end_time=20000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=1,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=1,
                                    start_time=20000,
                                    end_time=50000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=2,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=2,
                                    start_time=50000,
                                    end_time=100000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=3,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=3,
                                    start_time=100000,
                                    end_time=1000000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=4,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=4,
                                    start_time=1000000,
                                    end_time=1001000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=5,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=5,
                                    start_time=1001000,
                                    end_time=1500000,
                                ),
                            ],
                            allocation_events=[],
                        ),
                    ],
                )
            ],
        )

    def gen_inference_run(self) -> InferenceRun:
        return InferenceRun(
            {
                0: {
                    "profile_start_time": [10000],
                    "profile_end_time": [20000],
                },
                1: {
                    "profile_start_time": [20000],
                    "profile_end_time": [50000],
                },
                2: {
                    "profile_start_time": [50000],
                    "profile_end_time": [100000],
                },
                3: {
                    "profile_start_time": [100000],
                    "profile_end_time": [1000000],
                },
                4: {
                    "profile_start_time": [1000000],
                    "profile_end_time": [1001000],
                },
                5: {
                    "profile_start_time": [1001000],
                    "profile_end_time": [1500000],
                },
            },
            {
                "profile_start_time": [10000],
                "profile_end_time": [1500000],
                "load_start_time": 5000,
                "load_end_time": 10000,
            },
        )


class CompositeDelegateModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        class DelegateAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return [x + y]

            def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
                return (torch.randn(1, 3), torch.randn(1, 3))

        delegated_m = DelegateAdd()
        edge_ir_m = exir.capture(
            delegated_m,
            delegated_m.get_random_inputs(),
            exir.CaptureConfig(pt2_mode=True),
        ).to_edge()
        lowered_module = LoweredBackendModule(
            edge_ir_m=edge_ir_m,
            backend_id="backend_demo",
            processed_bytes=bytes("basic_module_add", encoding="utf8"),
            compile_specs=[],
        )
        patch_lowered_functions(lowered_module)
        self.lowered_module: LoweredBackendModule = lowered_module

    def forward(self, a: exir.Value, b: exir.Value, s: Tensor) -> Tensor:
        res = self.lowered_module(a, b)
        res = res[0] * s
        return res

    def get_random_inputs(self) -> Tuple[Tensor, Tensor, Tensor]:
        return (torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3))

    def gen_et_dump(self) -> ETDump:
        return ETDump(
            version=0,
            run_data=[
                RunData(
                    debug_blocks=[],
                    profile_blocks=[
                        ProfileBlock(
                            name="default",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::init_execution_plan",
                                    debug_handle=0,
                                    start_time=5000,
                                    end_time=10000,
                                )
                            ],
                            allocation_events=[],
                        ),
                        ProfileBlock(
                            name="inference loop",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::execute",
                                    debug_handle=0,
                                    start_time=10000,
                                    end_time=1500000,
                                ),
                                ProfileEvent(
                                    name="DELEGATE_CALL",
                                    debug_handle=0,
                                    start_time=55555,
                                    end_time=55556,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=1,
                                    start_time=55557,
                                    end_time=55559,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=1,
                                    start_time=55557,
                                    end_time=55558,
                                ),
                            ],
                            allocation_events=[],
                        ),
                    ],
                )
            ],
        )

    def gen_inference_run(self) -> InferenceRun:
        return InferenceRun(
            {
                0: {
                    "profile_start_time": [55555],
                    "profile_end_time": [55556],
                },
                1: {
                    "profile_start_time": [55557],
                    "profile_end_time": [55558],
                },
            },
            {
                "profile_start_time": [10000],
                "profile_end_time": [1500000],
                "load_start_time": 5000,
                "load_end_time": 10000,
            },
        )


class SymIntTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        t = x.size()[0]
        return x[0:t]

    def get_random_inputs(self):
        return (torch.ones(4, 4),)

    def gen_et_dump(self) -> ETDump:
        return ETDump(
            version=0,
            run_data=[
                RunData(
                    debug_blocks=[],
                    profile_blocks=[
                        ProfileBlock(
                            name="default",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::init_execution_plan",
                                    debug_handle=0,
                                    start_time=5000,
                                    end_time=10000,
                                )
                            ],
                            allocation_events=[],
                        ),
                        ProfileBlock(
                            name="inference loop",
                            allocators=[],
                            profile_events=[
                                ProfileEvent(
                                    name="ExecPlan::execute",
                                    debug_handle=0,
                                    start_time=10000,
                                    end_time=1500000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=0,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=0,
                                    start_time=10000,
                                    end_time=20000,
                                ),
                                ProfileEvent(
                                    name="OPERATOR_CALL",
                                    debug_handle=1,
                                    start_time=55555,
                                    end_time=55555,
                                ),
                                ProfileEvent(
                                    name="<operator_run>",
                                    debug_handle=1,
                                    start_time=20000,
                                    end_time=50000,
                                ),
                            ],
                            allocation_events=[],
                        ),
                    ],
                )
            ],
        )

    def gen_inference_run(self) -> InferenceRun:
        return InferenceRun(
            {
                0: {
                    "profile_start_time": [10000],
                    "profile_end_time": [20000],
                },
                1: {
                    "profile_start_time": [20000],
                    "profile_end_time": [50000],
                },
            },
            {
                "profile_start_time": [10000],
                "profile_end_time": [1500000],
                "load_start_time": 5000,
                "load_end_time": 10000,
            },
        )


MODELS = [
    ("two_linear_module", TwoLinearModule()),
    ("mulitple_output_node_module", MultiOutputNodeModule()),
    ("composite_delegate_module", CompositeDelegateModule()),
    ("sym_int_test_module", SymIntTestModule()),
]

# Generate fixture files
def get_module_path(module_name: str) -> Path:
    curr_dir = Path(__file__).resolve().parents[0]
    fixture_path = curr_dir / "fixtures"
    module_path = fixture_path / f"{module_name}.txt"
    return module_path


def generate_op_graph(m: Any, inputs: Any) -> ExportedETOperatorGraph:
    """
    Given a module and its inputs, returns the Operator Graph
    """
    et_program = (
        exir.capture(m, inputs, exir.CaptureConfig(pt2_mode=True))
        .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
        .to_executorch(config=ExecutorchBackendConfig())
    )
    program = et_program.program
    op_graph = ExportedETOperatorGraph.gen_operator_graph(program)
    return op_graph


def generate_op_graph_file_contents(m: Any, inputs: Any) -> bytes:
    """
    Given a module and its inputs, returns the Operator Graph in the form of
    bytes for easy storage into files and comparison with the fixture tests.
    """
    op_graph = generate_op_graph(m, inputs)
    tag = "generated"
    heading = bytes(
        f"# @{tag} by //executorch/sdk/edir/tests:generate_fixtures\n\n",
        "utf-8",
    )
    return heading + bytes(str(op_graph), "utf-8")


def gen_fx_graph_file_contents(graph_module: torch.fx.GraphModule) -> bytes:
    """
    Given a module and its inputs, returns the Operator Graph in the form of
    bytes for easy storage into files and comparison with the fixture tests.
    """
    op_graph = FXOperatorGraph.gen_operator_graph(
        model=graph_module, skip_stack_trace=True
    )
    tag = "generated"
    heading = bytes(
        f"# @{tag} by //executorch/sdk/edir/tests:generate_fixtures\n\n",
        "utf-8",
    )
    return heading + bytes(str(op_graph), "utf-8")


def get_fx_op_graph_file_name(model_name: str, dialect_type: str) -> str:
    return f"{model_name}_fx_graph_{dialect_type}"


def write_fx_graph_file_contents(
    model_name: str, dialect_type: str, write_bytes: bytes
) -> None:
    with open(
        str(get_module_path(get_fx_op_graph_file_name(model_name, dialect_type))), "wb"
    ) as f:
        f.write(write_bytes)


def generate_json_fixtures() -> None:
    """
    Generates fixture tests for the models and saves them to a file under the fixtures/ folder.
    """
    for model_name, model in MODELS:
        output = generate_op_graph_file_contents(model, model.get_random_inputs())

        assert isinstance(model_name, str)
        with open(get_module_path(model_name), "wb") as f:
            f.write(output)


def gen_graphs_from_model(
    model: torch.nn.Module,
) -> Tuple[ExirExportedProgram, ExirExportedProgram, ExecutorchProgram]:
    et_aten = exir.capture(
        model,
        model.get_random_inputs(),
        exir.CaptureConfig(pt2_mode=True),
    )
    et_aten_copy = copy.deepcopy(et_aten)
    et_edge = et_aten.to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
    et_edge_copy = copy.deepcopy(et_edge)
    et_program = et_edge.to_executorch(
        config=ExecutorchBackendConfig(emit_stacktrace=False)
    )
    # Referencing the program attribute triggers the actual emission of the graph
    et_program.program
    return (et_aten_copy, et_edge_copy, et_program)


def generate_fx_json_fixture() -> None:
    """
    Generates fixture tests for the models and saves them to a file under the fixtures/ folder.
    """
    for model_name, model in MODELS:
        et_aten_copy, et_edge_copy, et_program = gen_graphs_from_model(model)
        write_fx_graph_file_contents(
            model_name,
            "aten_dialect",
            gen_fx_graph_file_contents(et_aten_copy.graph_module),
        )
        write_fx_graph_file_contents(
            model_name,
            "edge_dialect",
            gen_fx_graph_file_contents(et_edge_copy.graph_module),
        )
        write_fx_graph_file_contents(
            model_name,
            "et_dialect",
            gen_fx_graph_file_contents(et_program.dump_graph_module()),
        )


if __name__ == "__main__":
    generate_json_fixtures()
    generate_fx_json_fixture()


class ExportedFXGraphTest(unittest.TestCase):
    def check_graph_equal(self, op_graph, model_name: str, dialect_type: str) -> None:
        with open(
            get_module_path(get_fx_op_graph_file_name(model_name, dialect_type)), "rb"
        ) as f:
            expected_op_graph = f.read()

        self.assertEqual(
            expected_op_graph,
            op_graph,
            "Please run `//executorch/sdk/edir/tests:generate_fixtures` to regenerate the fixtures.",
        )

    # pyre-ignore
    @parameterized.expand(MODELS)
    def test_gen_from_fx_graph(self, model_name: str, model: torch.nn.Module) -> None:
        et_aten_copy, et_edge_copy, et_program = gen_graphs_from_model(model)
        op_graph = gen_fx_graph_file_contents(et_aten_copy.graph_module)
        self.check_graph_equal(op_graph, model_name, "aten_dialect")
        op_graph = gen_fx_graph_file_contents(et_edge_copy.graph_module)
        self.check_graph_equal(op_graph, model_name, "edge_dialect")
        op_graph = gen_fx_graph_file_contents(et_program.dump_graph_module())
        self.check_graph_equal(op_graph, model_name, "et_dialect")

    # pyre-ignore
    @parameterized.expand(MODELS)
    def test_metadata_attaching(self, model_name: str, model: torch.nn.Module) -> None:
        _, _, et_program = gen_graphs_from_model(model)
        op_graph = FXOperatorGraph.gen_operator_graph(et_program.dump_graph_module())
        inference_run = model.gen_inference_run()
        op_graph.attach_metadata(inference_run)

        def verify_metadata_containment(
            graph: FXOperatorGraph, inference_run: InferenceRun
        ) -> None:
            validation_map = inference_run.node_metadata

            for node in graph.elements:
                # Recursively check subgraph nodes
                if isinstance(node, FXOperatorGraph):
                    verify_metadata_containment(node, inference_run)
                # Check that each node contains the corresponding metadata fields
                if isinstance(node, OperatorNode) and node.metadata is not None:
                    metadata = node.metadata
                    debug_handle = metadata.get("debug_handle")
                    if debug_handle in validation_map:
                        self.assertDictContainsSubset(
                            validation_map[debug_handle], metadata
                        )

        # Check for run level metadata
        if op_graph.metadata is not None:
            self.assertDictContainsSubset(inference_run.run_metadata, op_graph.metadata)

        verify_metadata_containment(op_graph, inference_run)


class ExportedOpGraphTest(unittest.TestCase):
    # pyre-ignore
    @parameterized.expand(MODELS)
    def test_gen_from_emitted_program(
        self, model_name: str, model: torch.nn.Module
    ) -> None:
        op_graph = generate_op_graph_file_contents(model, model.get_random_inputs())

        with open(get_module_path(model_name), "rb") as f:
            expected_op_graph = f.read()

        self.assertEqual(
            expected_op_graph,
            op_graph,
            "Please run `//executorch/sdk/edir/tests:generate_fixtures` to regenerate the fixtures.",
        )

    # pyre-ignore
    @parameterized.expand(MODELS)
    def test_metadata_attaching(self, model_name: str, model: torch.nn.Module) -> None:
        op_graph = generate_op_graph(model, model.get_random_inputs())
        inference_run = model.gen_inference_run()
        op_graph.attach_metadata(inference_run)

        def verify_metadata_containment(
            graph: ExportedETOperatorGraph, inference_run: InferenceRun
        ) -> None:
            validation_map = inference_run.node_metadata

            for node in graph.elements:
                # Recursively check subgraph nodes
                if isinstance(node, ExportedETOperatorGraph):
                    verify_metadata_containment(node, inference_run)
                # Check that each node contains the corresponding metadata fields
                if isinstance(node, OperatorNode) and node.metadata is not None:
                    metadata = node.metadata
                    debug_handle = metadata.get("debug_handle")
                    if debug_handle in validation_map:
                        self.assertDictContainsSubset(
                            validation_map[debug_handle], metadata
                        )

        # Check for run level metadata
        if op_graph.metadata is not None:
            self.assertDictContainsSubset(inference_run.run_metadata, op_graph.metadata)

        verify_metadata_containment(op_graph, inference_run)


class InferenceRunTest(unittest.TestCase):
    # pyre-ignore
    @parameterized.expand(MODELS)
    def test_inference_run_construction(
        self, model_name: str, model: torch.nn.Module
    ) -> None:
        et_dump = model.gen_et_dump()
        inference_run = model.gen_inference_run()

        self.assertEqual(
            inference_run, InferenceRun.extract_runs_from_etdump(et_dump)[0]
        )
