# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import shutil
import subprocess
import tempfile

from typing import List, Optional, Tuple

import numpy as np
import torch

from executorch.backends.arm.test.test_models import TosaProfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QuantizationParams:
    __slots__ = ["node_name", "zp", "scale"]

    # todo: zps and scales can be per tensors or per channel => a list??
    def __init__(self, node_name: str, zp: int, scale: float):
        self.node_name = node_name  # not need I think, but good for error check
        self.zp = zp
        self.scale = scale


"""
This class is used to work with TOSA artifacts.
"""


class TosaTestUtils:
    def __init__(
        self,
        intermediate_path: Optional[str] = None,
        tosa_ref_model_path: Optional[str] = None,
        profile: Optional[TosaProfile] = None,
    ):
        self.intermediate_path = intermediate_path or tempfile.mkdtemp(
            prefix="arm_tosa_"
        )
        self.tosa_ref_model_path = tosa_ref_model_path or "tosa_reference_model"
        self.profile = profile or TosaProfile.MI
        assert os.path.exists(
            self.intermediate_path
        ), f"TOSA artifact path don't exist! Path: {self.intermediate_path}"

    def dbg_dump_readble_tosa_file(self) -> None:
        """
        This function is used to dump the TOSA buffer to a human readable
        format, using flatc.
        It requires the following files to be present on disk:
        1) output.tosa (in self.intermediate_path, produced by arm_backend.py)
        2) ./backends/arm/third-party/serialization_lib/schema/tosa.fbs.

        It is used for debugging purposes.

        Output from this is a file called output.json, located in
        self.intermediate_path.

        Todo:
            * I'd prefer if this function didn't use files on disk...
            * Check if we can move this function to dump_artificat() thingy...
        """

        tosa_input_file = self.intermediate_path + "/output.tosa"
        tosa_schema_file = (
            "./backends/arm/third-party/serialization_lib/schema/tosa.fbs"
        )

        assert os.path.exists(
            tosa_schema_file
        ), f"tosa_schema_file: {tosa_schema_file} does not exist"
        assert os.path.exists(
            tosa_input_file
        ), f"tosa_input_file: {tosa_input_file} does not exist"
        assert shutil.which("flatc") is not None

        cmd_flatc = [
            "flatc",
            "-o",
            self.intermediate_path,
            "--raw-binary",
            "-t",
            tosa_schema_file,
            "--",
            tosa_input_file,
        ]
        self._run_cmd(cmd_flatc)
        return

    def run_tosa_ref_model(
        self,
        params_input: Tuple[List[str], List[QuantizationParams]],
        param_output: Tuple[str, QuantizationParams],
        inputs: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run TOSA reference model using the tosa_refence_model program.

        In order to do that we need:
        1. desc.json, which points to files needed by tosa_refence_model.
        2. output.tosa, which is the TOSA buffer that describes the model we're
           trying to run.

        These two files are created by arm_backend.py as part of partition stage

        All these files are saved on disk in self.intermediate_path.

        Args:
            params_input (Tuple[List[str], List[QuantizationParams]]): A tuple
                containing a list of input node names and a list of their
                quantization parameters (if model is quantized).
            param_output (Tuple[str, QuantizationParams]): A tuple containing
                the output node name and its quantization parameters (if
                model is quantized).
            inputs (Tuple[torch.Tensor]): The input data to run the TOSA

        Returns:
            torch.Tensor: The output of the TOSA reference model, as a torch
                tensor.

        Here's a sample desc.json file:
        {
            "tosa_file": "output.tosa",
            "ifm_name": [
                "arg0_1"
            ],
            "ifm_file": [
                "arg0_1.npy"
            ],
            "ofm_name": [
                "quantized_decomposed_dequantize_per_tensor_default_1"
            ],
            "ofm_file": [
                "ref-quantized_decomposed_dequantize_per_tensor_default_1.npy"
            ],
            "expected_return_code": 0,
            "expected_failure": false
        }

        Todo:
            * It would be nice to not rely on files on disk. Should be possible
              as a next step. See:
              https://review.mlplatform.org/plugins/gitiles/tosa/reference_model/#executable-usage
        """

        desc_file_path = os.path.join(self.intermediate_path, "desc.json")
        assert os.path.exists(
            desc_file_path
        ), f"desc_file_path: {desc_file_path} does not exist"

        # Save the input data to disk as a .npy file, since that's what the TOSA
        # reference model expects. Name of the file must match the name in
        # desc.json, which is the tensor name from the graph + .npy
        for input_name, quant_param, data in zip(
            params_input[0], params_input[1], inputs
        ):
            data_np = data.detach().numpy()

            if self.profile is TosaProfile.BI:
                assert (
                    quant_param.node_name == input_name
                ), "These quantization params do not match the input tensor name"
                int8_max = np.iinfo(np.int8).max
                int8_min = np.iinfo(np.int8).min
                data_np = (
                    ((data_np / np.float32(quant_param.scale)) + quant_param.zp)
                    .round()
                    .clip(int8_min, int8_max)
                    .astype(np.int8)
                )
            file_path = os.path.join(self.intermediate_path, input_name + ".npy")
            np.save(file_path, data_np, allow_pickle=False)

        # Run the TOSA reference model via command line, this will produce a
        # .npy file with the result (aka OFM).
        assert (
            shutil.which(self.tosa_ref_model_path) is not None
        ), f"tosa_reference_model tool not found, did you run examples/arm/setup.sh? Path: {self.tosa_ref_model_path}"
        cmd_ref_model = [self.tosa_ref_model_path, "--test_desc", desc_file_path]
        self._run_cmd(cmd_ref_model)

        # Load desc.json, just to get the name of the output file above
        with open(desc_file_path) as f:
            desc_json = json.load(f)
        ofm_file_npy = os.path.join(self.intermediate_path, desc_json["ofm_file"][0])

        # Load the output file (OFM) and return it as a numpy array
        tosa_ref_output = np.load(ofm_file_npy)

        if self.profile is TosaProfile.BI:
            # Need to dequant back to FP32 for comparison with torch output
            quant_param = param_output[1]
            assert (
                quant_param is not None
            ), "There are no quantization parameters, check output parameters"
            tosa_ref_output = (tosa_ref_output - quant_param.zp) * quant_param.scale

        # tosa_output is a numpy array, convert to torch tensor for comparison
        tosa_ref_output = torch.from_numpy(tosa_ref_output.astype("float32"))

        return tosa_ref_output

    def _run_cmd(self, cmd: List[str]) -> None:
        """
        Run a command and check for errors.

        Args:
        cmd (List[str]): The command to run as a list.
        """
        try:
            subprocess.run(cmd, check=True)
        except:
            cmd_str = " ".join(cmd)
            raise RuntimeError(f"Failed to run: {cmd_str}")
