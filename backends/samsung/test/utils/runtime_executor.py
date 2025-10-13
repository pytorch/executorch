import logging
import os
import subprocess
import tempfile

from functools import cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


@cache
def get_runner_path() -> Path:
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return Path(git_root) / "build_samsung_android/backends/samsung/enn_executor_runner"


class ADBTestManager:
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

    def _adb(self, cmd):
        cmds = ["adb"]

        assert self._is_adb_connected, "Fail to get available device to execute."

        cmds.extend(cmd)
        command = " ".join(cmds)
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            logging.info(result.stdout.decode("utf-8").strip())
            logging.error(result.stderr.decode("utf-8").strip())
            raise RuntimeError("adb command execute failed")

    def push(self):
        self._adb(["shell", f"rm -rf {self.work_directory}"])
        self._adb(["shell", f"mkdir -p {self.work_directory}"])
        self._adb(["push", self.pte_file, self.work_directory])
        self._adb(["push", self.runner, self.work_directory])

        for input_file in self.input_files:
            input_file_path = os.path.join(self.artifacts_dir, input_file)
            if Path(input_file).name == input_file and os.path.isfile(input_file_path):
                # default search the same level directory with pte
                self._adb(["push", input_file_path, self.work_directory])
            elif os.path.isfile(input_file):
                self._adb(["push", input_file, self.work_directory])
            else:
                raise FileNotFoundError(f"Invalid input file path: {input_file}")

    def execute(self):
        self._adb(["shell", f"rm -rf {self.output_folder}"])
        self._adb(["shell", f"mkdir -p {self.output_folder}"])
        # run the delegation
        input_files_list = " ".join([os.path.basename(x) for x in self.input_files])
        enn_executor_runner_args = " ".join(
            [
                f"--model {os.path.basename(self.pte_file)}",
                f'--input "{input_files_list}"',
                f"--output_path {self.output_folder}",
            ]
        )
        enn_executor_runner_cmd = " ".join(
            [
                f"'cd {self.work_directory} &&",
                f"./enn_executor_runner {enn_executor_runner_args}'",
            ]
        )

        self._adb(["shell", f"{enn_executor_runner_cmd}"])

    def pull(self, output_path):
        self._adb(["pull", "-a", self.output_folder, output_path])

    @staticmethod
    def _is_adb_connected():
        try:
            output = subprocess.check_output(["adb", "devices"])
            devices = output.decode("utf-8").splitlines()[1:]
            return [device.split()[0] for device in devices if device.strip() != ""]
        except subprocess.CAlledProcessError:
            return False


class RuntimeExecutor:
    def __init__(self, executorch_program, inputs):
        self.executorch_program = executorch_program
        self.inputs = inputs

    def run_on_device(self) -> Tuple[torch.Tensor]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_filename, input_files = self._save_model_and_inputs(tmp_dir)
            test_manager = ADBTestManager(
                pte_file=os.path.join(tmp_dir, pte_filename),
                work_directory="/data/local/tmp/enn-executorch-test",
                input_files=input_files,
            )
            test_manager.push()
            test_manager.execute()
            host_output_save_dir = os.path.join(tmp_dir, "output")
            test_manager.pull(host_output_save_dir)

            model_outputs = self._get_model_outputs()
            num_of_output_files = len(os.listdir(host_output_save_dir))
            assert num_of_output_files == len(
                model_outputs
            ), f"Number of outputs is invalid, expect {len(model_outputs)} while got {num_of_output_files}"

            result = []
            for idx in range(num_of_output_files):
                output_array = np.fromfile(
                    os.path.join(host_output_save_dir, f"output_{idx}.bin"),
                    dtype=np.uint8,
                )
                output_tensor = (
                    torch.from_numpy(output_array)
                    .view(dtype=model_outputs[idx].dtype)
                    .view(*model_outputs[idx].shape)
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
