# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import os
import selectors
import subprocess
import sys

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram

from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgram, ExecutorchProgramManager

# If quiet is true, suppress the printing of stdout and stderr output.
quiet = False


def _execute_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[str, str]:
    """
    `subprocess.run(cmd, capture_output=True)` captures stdout/stderr and only
    returns it at the end. This functions not only does that, but also prints out
    stdout/stderr non-blockingly when running the command.
    """
    logging.debug(f"cmd = \33[33m{cmd}\33[0m, cwd = {cwd}")
    stdout = ""
    stderr = ""

    PIPE = subprocess.PIPE
    with subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd) as p:
        sel = selectors.DefaultSelector()
        # pyre-fixme[6]: For 1st argument expected `Union[HasFileno, int]` but got
        #  `Optional[IO[bytes]]`.
        sel.register(p.stdout, selectors.EVENT_READ)
        # pyre-fixme[6]: For 1st argument expected `Union[HasFileno, int]` but got
        #  `Optional[IO[bytes]]`.
        sel.register(p.stderr, selectors.EVENT_READ)

        done = False
        while not done:
            for key, _ in sel.select():
                # pyre-fixme[16]: Item `HasFileno` of `Union[HasFileno, int]` has no
                #  attribute `read1`.
                data = key.fileobj.read1().decode()
                if not data:
                    done = True
                    break

                if key.fileobj is p.stdout:
                    if not quiet:
                        print(data, end="")
                    stdout += data
                else:
                    if not quiet:
                        print(data, end="", file=sys.stderr)
                    stderr += data

    # flush stdout and stderr in case there's no newline character at the end
    # from the subprocess
    sys.stdout.flush()
    sys.stderr.flush()

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, stdout, stderr)

    return stdout, stderr


def execute(args: List[str]) -> Tuple[str, str]:
    """
    Either a local execution (through subprocess.run) or a remote execution (in Hargow).
    Run the command described by args (the same way subprocess.run does). Ex: if you want to
    run "ls -al", you need to pass args = ["ls", "-al"]
    """
    # `import torch` will mess up PYTHONPATH. delete the messed up PYTHONPATH
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]

    try:
        return _execute_subprocess(args)
    except subprocess.CalledProcessError as e:
        fdb_cmd = f"fdb {' '.join(e.cmd)}"
        raise RuntimeError(
            f"Failed to execute. Use the following to debug:\n{fdb_cmd}"
        ) from e


class Executor:
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        working_dir: str = "",
    ):
        self.working_dir = working_dir
        self.executor_builder = "./backends/cadence/build_cadence_runner.sh"
        self.execute_runner = "./cmake-out/backends/cadence/cadence_runner"
        self.bundled_program_path: str = "CadenceDemoModel.bpte"

    def __call__(self) -> None:
        # build executor
        args = self.get_bash_command(self.executor_builder)
        logging.info(f"\33[33m{' '.join(args)}\33[0m")
        execute(args)

        # run executor
        cmd_args = {
            "bundled_program_path": os.path.join(
                self.working_dir, self.bundled_program_path
            ),
            "etdump_path": os.path.join(self.working_dir, "etdump.etdp"),
            "debug_output_path": os.path.join(self.working_dir, "debug_output.bin"),
        }
        args = self.get_bash_command(self.execute_runner, cmd_args)
        logging.info(f"\33[33m{' '.join(args)}\33[0m")
        execute(args)

    @staticmethod
    def get_bash_command(
        executable: str,
        cmd_args: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        # go through buck config and turn the dict into a list of "{key}=={value}"
        if cmd_args is None:
            cmd_args = {}

        cmd_args_strs = []
        for key, value in cmd_args.items():
            cmd_args_strs.extend([f"--{key}={value}"])

        return [executable] + cmd_args_strs


@dataclass
class BundledProgramTestData:
    method: str
    inputs: Sequence[Union[bool, float, int, torch.Tensor]]
    expected_outputs: Sequence[torch.Tensor]
    testset_idx: int = 0  # There is only one testset in the bundled program


class BundledProgramManager:
    """
    Stateful bundled program object
    Takes a BundledProgramTestData and generates a bundled program
    """

    def __init__(self, bundled_program_test_data: List[BundledProgramTestData]) -> None:
        self.bundled_program_test_data: List[BundledProgramTestData] = (
            bundled_program_test_data
        )

    @staticmethod
    # pyre-fixme[2]: Parameter `**args` has no type specified.
    def bundled_program_test_data_gen(**args) -> BundledProgramTestData:
        return BundledProgramTestData(**args)

    def get_method_test_suites(self) -> List[MethodTestSuite]:
        return [
            self._gen_method_test_suite(bptd) for bptd in self.bundled_program_test_data
        ]

    def _gen_method_test_suite(self, bptd: BundledProgramTestData) -> MethodTestSuite:
        method_test_case = MethodTestCase(
            inputs=bptd.inputs,
            expected_outputs=bptd.expected_outputs,
        )
        return MethodTestSuite(
            method_name=bptd.method,
            test_cases=[method_test_case],
        )

    def _serialize(
        self,
        executorch_program: Union[
            ExecutorchProgram,
            ExecutorchProgramManager,
        ],
        method_test_suites: Sequence[MethodTestSuite],
        bptd: BundledProgramTestData,
    ) -> bytes:
        bundled_program = BundledProgram(
            executorch_program=executorch_program, method_test_suites=method_test_suites
        )
        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )
        return bundled_program_buffer
