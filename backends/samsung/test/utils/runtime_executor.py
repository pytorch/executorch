import logging
import os
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time

from functools import cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@cache
def get_runner_path() -> Path:
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return Path(git_root) / "build_samsung_android/examples/samsung/enn_executor_runner"


class EDBTestManager:
    COMMAND_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        pte_file,
        work_directory,
        input_files: List[str],
    ):
        self.pte_file = pte_file
        self.work_directory = work_directory
        self.input_files = input_files
        self.artifacts_dir = Path(self.pte_file).parent.absolute()
        self.output_folder = f"{self.work_directory}/output"
        self.runner = str(get_runner_path())
        self.devicefarm = "devicefarm-cli"
        self.bundle_path = self.artifacts_dir / "inputs.tar"

    def _edb(self, cmd):
        command = [self.devicefarm, *cmd]
        logging.info(f"[EDB] Run: {shlex.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.COMMAND_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            # The packaged CLI spawns children that inherit its output pipes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise RuntimeError(
                f"EDB command timed out after {self.COMMAND_TIMEOUT_SECONDS}s: "
                f"{shlex.join(command)}"
            ) from error

        stdout_msg = stdout.decode("utf-8").strip()
        if stdout_msg:
            logging.info(f"[EDB] stdout: {stdout_msg}")

        if process.returncode != 0:
            stderr_msg = stderr.decode("utf-8").strip()
            if stderr_msg:
                logging.error(f"[EDB] stderr: {stderr_msg}")
            raise RuntimeError("edb(Exynos device bridge) command execute failed")
        else:
            logging.info("[EDB] Command succeeded")

    def push(self):
        files = [Path(self.pte_file), Path(self.runner)]
        for input_file in self.input_files:
            path = Path(input_file)
            if path.name == input_file and (self.artifacts_dir / path).is_file():
                path = self.artifacts_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"Invalid input file path: {input_file}")
            files.append(path)
        with tarfile.open(self.bundle_path, "w") as bundle:
            for path in files:
                bundle.add(path, arcname=path.name, recursive=False)
        directory = shlex.quote(self.work_directory)
        self._edb(["-E", f"rm -rf {directory} && mkdir -p {directory}"])
        self._edb(["-U", str(self.bundle_path), self.work_directory])

    def execute(self):
        input_files_list = " ".join([os.path.basename(x) for x in self.input_files])
        enn_executor_runner_args = " ".join(
            [
                f"--model {shlex.quote(os.path.basename(self.pte_file))}",
                f'--input "{input_files_list}"',
                f"--output_path {shlex.quote(self.output_folder)}",
            ]
        )
        enn_executor_runner_cmd = " ".join(
            [
                f"cd {shlex.quote(self.work_directory)} &&",
                f"tar -xf {shlex.quote(self.bundle_path.name)} &&",
                f"mkdir -p {shlex.quote(self.output_folder)} &&",
                f"chmod 777 {shlex.quote(self.output_folder)} &&",
                "chmod +x ./enn_executor_runner &&",
                f"./enn_executor_runner {enn_executor_runner_args}",
            ]
        )

        self._edb(["-E", f"{enn_executor_runner_cmd}"])

    def pull(self, output_path):
        self._edb(["-D", self.output_folder, output_path])


class RuntimeExecutor:
    def __init__(self, executorch_program, inputs):
        self.executorch_program = executorch_program
        self.inputs = inputs

    def run_on_device(self) -> Tuple[torch.Tensor]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_filename, input_files = self._save_model_and_inputs(tmp_dir)
            test_manager = EDBTestManager(
                pte_file=os.path.join(tmp_dir, pte_filename),
                work_directory="/data/local/tmp/enn-executorch-test",
                input_files=input_files,
            )
            test_manager.push()
            test_manager.execute()

            host_output_save_dir = os.path.join(tmp_dir, "output")
            test_manager.pull(host_output_save_dir)

            model_outputs = self._get_model_outputs()

            attempts = 0
            while attempts < 3:
                if not os.path.isdir(host_output_save_dir):
                    attempts += 1
                    time.sleep(0.5)
                else:
                    break

            target_output_save_dir = os.path.join(host_output_save_dir, "output")
            num_of_output_files = len(os.listdir(target_output_save_dir))
            assert num_of_output_files == len(
                model_outputs
            ), f"Number of outputs is invalid, expect {len(model_outputs)} while got {num_of_output_files}"

            result = []
            for idx in range(num_of_output_files):
                output_array = np.fromfile(
                    os.path.join(target_output_save_dir, f"output_{idx}.bin"),
                    dtype=np.uint8,
                )
                output_tensor = (
                    torch.from_numpy(output_array)
                    .view(dtype=model_outputs[idx].dtype)
                    .reshape(model_outputs[idx].shape)
                )
                result.append(output_tensor)

            return tuple(result)

    def _get_model_outputs(self):
        output_node = self.executorch_program.exported_program().graph.output_node()
        output_fake_tensors = []
        for ori_output in output_node.args[0]:
            output_fake_tensors.append(ori_output.meta["val"])

        return tuple(output_fake_tensors)

    def _save_model_and_inputs(self, save_dir):
        pte_file_name = "program.pte"
        file_path = os.path.join(save_dir, f"{pte_file_name}")
        with open(file_path, "wb") as file:
            self.executorch_program.write_to_file(file)

        inputs_files = []
        for idx, input in enumerate(self.inputs):
            input_file_name = f"input_{idx}.bin"
            input.detach().numpy().tofile(os.path.join(save_dir, input_file_name))
            inputs_files.append(input_file_name)

        return pte_file_name, inputs_files
