# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time

import pytest
import torch
from executorch.backends.arm._passes.decompose_linear_pass import DecomposeLinearPass
from executorch.backends.xnnpack.test.tester import Tester

from executorch.devtools.visualization import (
    ModelExplorerServer,
    SingletonModelExplorerServer,
    visualization_utils,
    visualize,
    visualize_graph,
)
from executorch.exir import ExportedProgram, to_edge_transform_and_lower

try:
    from model_explorer.config import ModelExplorerConfig  # type: ignore
except ImportError:
    print(
        "Error: 'model_explorer' is not installed. Install using devtools/install_requirements.sh"
    )
    raise


@pytest.fixture
def server():
    """Mock relevant calls in visualization.visualize and check that parameters have their expected value."""
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context():
        _called_reuse_server = False

        def mock_set_reuse_server(self):
            nonlocal _called_reuse_server
            _called_reuse_server = True

        def mock_add_model_from_pytorch(self, name, exported_program, settings):
            assert isinstance(exported_program, ExportedProgram)

        def mock_visualize_from_config(cur_config, no_open_in_browser):
            pass

        monkeypatch.setattr(
            ModelExplorerConfig, "set_reuse_server", mock_set_reuse_server
        )
        monkeypatch.setattr(
            ModelExplorerConfig, "add_model_from_pytorch", mock_add_model_from_pytorch
        )
        monkeypatch.setattr(
            visualization_utils, "visualize_from_config", mock_visualize_from_config
        )
        yield monkeypatch.context
        assert _called_reuse_server, "Did not call reuse_server"


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.inputs = (torch.randn(5, 10, 25, in_features),)
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def get_inputs(self) -> tuple[torch.Tensor]:
        return self.inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_visualize_manual_export(server):
    with server():
        model = Linear(20, 30)
        exported_program = torch.export.export(model, model.get_inputs(), strict=True)
        visualize(exported_program)
        time.sleep(3.0)


def test_visualize_exported_program(server):
    with server():
        model = Linear(20, 30)
        (
            Tester(
                model,
                example_inputs=model.get_inputs(),
            )
            .export()
            .visualize()
        )


def test_visualize_to_edge(server):
    with server():
        model = Linear(20, 30)
        (
            Tester(
                model,
                example_inputs=model.get_inputs(),
            )
            .export()
            .to_edge()
            .visualize()
        )


def test_visualize_partition(server):
    with server():
        model = Linear(20, 30)
        (
            Tester(
                model,
                example_inputs=model.get_inputs(),
            )
            .export()
            .to_edge()
            .partition()
            .visualize()
        )


def test_visualize_to_executorch(server):
    with server():
        model = Linear(20, 30)
        (
            Tester(
                model,
                example_inputs=model.get_inputs(),
            )
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .visualize()
        )


def test_visualize_graph(server):
    with server():
        model = Linear(20, 30)
        exported_program = torch.export.export(model, model.get_inputs(), strict=True)
        exported_program = to_edge_transform_and_lower(
            exported_program
        ).exported_program()
        modified_gm = DecomposeLinearPass()(exported_program.graph_module).graph_module
        visualize_graph(modified_gm, exported_program)


if __name__ == "__main__":
    """A test to run locally to make sure that the web browser opens up
    automatically as intended.
    """

    test_visualize_manual_export(ModelExplorerServer)

    with SingletonModelExplorerServer():
        test_visualize_manual_export(SingletonModelExplorerServer)
        test_visualize_exported_program(SingletonModelExplorerServer)
        test_visualize_to_edge(SingletonModelExplorerServer)
        test_visualize_partition(SingletonModelExplorerServer)
        test_visualize_to_executorch(SingletonModelExplorerServer)
        test_visualize_graph(SingletonModelExplorerServer)

        # Sleep to give the server time to load the last graph before killing it.
        time.sleep(3.0)
