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

from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from executorch.backends.arm.test.arm_tosa_reference import (
    get_input_quantization_params,  # TODO: remove this dependecy
    get_output_quantization_param,  # TODO: remove this dependecy
    tosa_ref_dump_inputs,  # TODO: remove this dependecy
)

from executorch.backends.arm.test.test_models import TosaProfile
from executorch.backends.xnnpack.test.tester.tester import Partition, ToEdge

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QuantizationParams:
    __slots__ = ["zps", "scales"]

    def __init__(self, zps: Union[Dict, List[int]], scales: Union[Dict, List[float]]):
        self.zps = zps
        self.scales = scales


Quantization = namedtuple("Quantization", ["input", "output"])

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
        input_quant = QuantizationParams(zps={}, scales={})
        output_quant = QuantizationParams(zps={}, scales={})
        self.quantization = Quantization(input=input_quant, output=output_quant)
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

    def convert_inputs_to_tosa(
        self,
        partition_stage: Partition,
        toedge_stage: ToEdge,
        inputs_to_run: Tuple[torch.Tensor],
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Convert input tensors to numpy and save them to disk as .npy files. The
        TOSA reference model will use these files as input....

        Args:
        partition_stage (Partition): The partition stage.
        toedge_stage (ToEdge): The toedge stage.
        inputs_to_run (Tuple[torch.Tensor]): The input tensors to convert.

        Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples, where each tuple contain
            a numpy array and the name of the tensor.

        Todo:
            * I'd like to use some other common function instead of
              get_input_quantization_params() and
              get_output_quantization_param().
            * I'd like to get rid of the call to tosa_ref_dump_inputs as well.
              All this function is doing is to convert to numpy and save to disk
        """

        if self.profile == TosaProfile.BI:
            # TODO: Unclear to me why we need to pass toedge_stage here. Ideally
            # we shouldn't get the quantization params here at all, but rather
            # from the Quantizer...
            (
                self.quantization.input.scales,
                self.quantization.input.zps,
            ) = get_input_quantization_params(toedge_stage)
            (
                self.quantization.output.scales,
                self.quantization.output.zps,
            ) = get_output_quantization_param(toedge_stage)

            # TODO: I think it should be possible to get this input data from
            # somewhere else. Why do I need to call this just to get a npy file,
            # which is just a quantized version of the input..?
            np_data_and_tensor_names = tosa_ref_dump_inputs(
                partition_stage,
                inputs_to_run,
                self.intermediate_path,
                self.quantization.input.scales,
                self.quantization.input.zps,
                self.profile,
                save_on_disk=False,  # If True - this one produces arg0_1.npy, which is just a quant version of the input
                # inputs_to_run -> convert to numpy -> do "manual" quantization -> save to arg0_1.npy (TODO: remove this comment)
            )
        else:
            np_data_and_tensor_names = tosa_ref_dump_inputs(
                partition_stage,
                inputs_to_run,
                self.intermediate_path,
                {},
                {},
                save_on_disk=False,
            )

        return np_data_and_tensor_names

    def run_tosa_ref_model(
        self, tensor_names_and_inputs: List[Tuple[np.array, str]]
    ) -> torch.Tensor:
        """
        Run TOSA reference model using the tosa_refence_model program.

        In order to do that we need:
        1. desc.json, which points to files needed by tosa_refence_model.
        2. output.tosa, which is the TOSA buffer that describes the model we're
           trying to run.

        These two files are created by arm_backend.py as part of partition stage

        3. An IFM file containing input data, saved as .npy. This file is
           created by tosa_ref_dump_inputs()

        All these files are saved on disk in self.intermediate_path.

        Args:
        tensor_names_and_inputs (List[Tuple[np.array, str]]): A list of tuples
            where each tuple contains inputs (as numpy array) and the name of
            the tensor.

        Returns:
        torch.Tensor: The output of the TOSA reference model, as a torch tensor.

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
        # reference model expects. Name of the file is must match the name in
        # desc.json, which is the tensor name from the graph + .npy
        for tensor_name, data in tensor_names_and_inputs:
            file_path = os.path.join(self.intermediate_path, tensor_name + ".npy")
            np.save(file_path, data, allow_pickle=False)

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
            assert self.quantization.output.scales is not None
            assert self.quantization.output.zps is not None
            tosa_ref_output = (
                np.round(tosa_ref_output - self.quantization.output.zps)
                * self.quantization.output.scales
            )

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
