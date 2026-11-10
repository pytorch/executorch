# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess  # nosec B404

import pytest
import torch
from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.exir import to_edge


@pytest.mark.parametrize("server_mode", [False, True])
def test_etdump_records_all_executions(tmp_path, server_mode):
    runner = os.environ.get("EXECUTORCH_TEST_EXECUTOR_RUNNER")
    if not runner:
        pytest.skip("Set EXECUTORCH_TEST_EXECUTOR_RUNNER to a tracing-enabled runner")

    class Model(torch.nn.Module):
        def forward(self, value):
            return value + 1

    program = to_edge(torch.export.export(Model(), (torch.ones(1),))).to_executorch()
    model_path = tmp_path / "model.pte"
    model_path.write_bytes(program.buffer)
    trace_path = tmp_path / "model.etdump"
    command = [
        runner,
        f"--model_path={model_path}",
        f"--output_file={tmp_path / 'output'}",
        f"--etdump_path={trace_path}",
    ]
    if server_mode:
        command.append("--server_mode")
    result = subprocess.run(  # nosec B603
        command,
        input="\n\n" if server_mode else "",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    trace = deserialize_from_etdump_flatcc(trace_path.read_bytes())
    executions = [
        event.profile_event
        for block in trace.run_data
        for event in block.events or []
        if event.profile_event and event.profile_event.name == "Method::execute"
    ]
    assert len(executions) == (2 if server_mode else 1)
    assert all(event.end_time >= event.start_time for event in executions)
