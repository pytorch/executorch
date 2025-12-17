import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import pandas as pd
import torch
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte

from graphviz import Digraph


class DrawGraph:
    def __init__(
        self,
        filename: str,
        directory: str,
        py_op_wrapper_list: [PyQnnManager.PyQnnOpWrapper],
        dot_string=False,
    ):
        self.filename = filename
        self.directory = directory
        self.py_op_wrapper_list = py_op_wrapper_list
        self.dot = Digraph(filename, format="svg")
        self.dot.attr(rankdir="TB")
        self.dot_string = dot_string
        self.draw()

    def dfs_add_edges(self, node_name, visited, node_list):
        if node_name in visited:
            return
        visited.add(node_name)

        input_list = node_list[node_name]["input_list"]
        for input_node_name in input_list:
            self.dot.edge(input_node_name, node_name)
            self.dfs_add_edges(input_node_name, visited, node_list)

    def get_dot_graph(self):
        visited = set()
        node_list = {}
        excel_data = []

        self.get_node(node_list)
        self.add_node(node_list, excel_data)
        self.to_excel(excel_data)

        # add edge
        for node_name, _ in node_list.items():
            if node_name not in visited:
                self.dfs_add_edges(node_name, visited, node_list)

        return self.dot

    def get_node(self, node_list):
        for py_op_wrapper in self.py_op_wrapper_list:
            op_wrapper = py_op_wrapper.GetOpWrapper()
            # TODO: multi output
            for i in range(op_wrapper.GetOpConfig()["numOfOutputs"]):
                if op_wrapper.GetOpConfig()["outputTensors"][0].version == 2:
                    node = op_wrapper.GetOpConfig()["outputTensors"][i].v2
                    node_name = node.name
                    input_list = []
                    for j in range(op_wrapper.GetOpConfig()["numOfInputs"]):
                        if op_wrapper.GetOpConfig()["inputTensors"][j].version == 2:
                            input_node = op_wrapper.GetOpConfig()["inputTensors"][j].v2
                            input_node_name = input_node.name
                            if input_node_name not in node_list:
                                node_list[input_node_name] = {
                                    "node": input_node,
                                    "input_list": [],
                                }
                            input_list.append(input_node_name)
                        else:
                            raise ValueError("Unsupported tensor version")
                    if node_name not in node_list:
                        node_list[node_name] = {"node": node, "input_list": input_list}
                    else:
                        node_list[node_name]["input_list"] = input_list
                else:
                    raise ValueError("Unsupported tensor version")

    def add_node(self, node_list, excel_data):
        for node_name, tensor in node_list.items():
            node = tensor["node"]
            name = node_name
            data_type = node.dataType
            tensor_type = node.type
            dims = node.dimensions
            quantization_encoding = node.quantizeParams.quantizationEncoding
            scale = []
            offset = []
            if (
                quantization_encoding
                == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            ):
                scale.append(node.quantizeParams.scaleOffsetEncoding.scale)
                offset.append(node.quantizeParams.scaleOffsetEncoding.offset)
            elif (
                quantization_encoding
                == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
            ):
                for i in range(
                    node.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets
                ):
                    scale.append(
                        node.quantizeParams.axisScaleOffsetEncoding.scaleOffset[i].scale
                    )
                    offset.append(
                        node.quantizeParams.axisScaleOffsetEncoding.scaleOffset[
                            i
                        ].offset
                    )
            excel_data.append(
                {
                    "name": name,
                    "tensor_type": tensor_type,
                    "scale": scale,
                    "offset": offset,
                }
            )
            # Default color for intermediate nodes
            bg_color = "white"
            if "input" in node_name or "output" in node_name:
                bg_color = "lightgreen"
            elif tensor_type == 4:
                bg_color = "lightpink"
            label = f"""<
                <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR><TD BGCOLOR="{bg_color}">name: {name}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">data_type: {data_type}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">tensor_type: {tensor_type}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">dims: {dims}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">quantization_encoding: {quantization_encoding}</TD></TR>
            """
            label += "</TABLE>>"
            self.dot.node(
                node_name,
                label,
                shape="box",
                style="rounded",
                fillcolor="transparent",
                color="black",
            )

    def to_excel(self, excel_data):
        param_rows = []
        activation_rows = []

        for entry in excel_data:
            name = entry["name"]
            scale = entry["scale"]
            offset = entry["offset"]
            if (
                entry["tensor_type"]
                == PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC
            ):
                param_rows.append({"name": name, "scale": scale, "offset": offset})
            else:
                activation_rows.append({"name": name, "scale": scale, "offset": offset})
        param_df = pd.DataFrame(param_rows)
        scale_df = pd.DataFrame(activation_rows)
        output_file = f"{self.filename}.xlsx"

        with pd.ExcelWriter(output_file) as writer:
            param_df.to_excel(writer, sheet_name="Parameters", index=False)
            scale_df.to_excel(writer, sheet_name="Scales", index=False)

    def draw(self):
        graph = self.get_dot_graph()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_directory = f"{tmp_dir}/outputs"
            graph.render(
                self.filename,
                directory=temp_directory,
                format="svg",
                cleanup=not self.dot_string,
            )
            source_file = os.path.join(temp_directory, f"{self.filename}.svg")
            destination_file = os.path.join(self.directory, f"{self.filename}.svg")
            shutil.move(source_file, destination_file)
            if self.dot_string:
                dot_file = os.path.join(temp_directory, f"{self.filename}")
                dot_dest_file = os.path.join(self.directory, f"{self.filename}.dot")
                shutil.move(dot_file, dot_dest_file)


class QnnTool:
    def __init__(
        self,
        tmp_dir,
        sample_input,
        soc_id,
        adb,
        build_folder,
        workspace="/data/local/tmp/qnn_executorch_test",
    ):
        self.qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
        self.ndk = os.environ.get("ANDROID_NDK_ROOT", None)
        assert self.qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        assert self.ndk, "ANDROID_NDK_ROOT was not found in environment variable"

        self.tmp_dir = tmp_dir
        self.workspace = workspace
        self.adb = adb
        self.sample_input = sample_input
        self.build_folder = build_folder
        self.root = str(Path(__file__).resolve().parents[3])
        self.config = {
            "backend_extension_config": {
                "backend_extensions": {
                    "config_file_path": "config.json",
                },
                "features": {
                    "qhas_json": True,
                },
            },
            "config": {
                "devices": [
                    {
                        "profiling_level": "linting",
                        "cores": [
                            {"perf_profile": "burst", "rpc_control_latency": 100}
                        ],
                        "soc_id": int(soc_id),
                    }
                ]
            },
        }

    def qnn_context_binary_generator(
        self,
        qnn_binary_file="forward_0.dlc",
        binary_name="forward.serialized",
    ):
        for file_name, data in self.config.items():
            with open(f"{self.tmp_dir}/{file_name}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

        target = "x86_64-linux-clang"
        cmds = [
            f"{self.qnn_sdk}/bin/{target}/qnn-context-binary-generator",
            "--backend",
            f"{self.qnn_sdk}/lib/{target}/libQnnHtp.so",
            "--model",
            f"{self.qnn_sdk}/lib/{target}/libQnnModelDlc.so",
            "--dlc_path",
            f"{self.tmp_dir}/{qnn_binary_file}",
            f"--config_file {self.tmp_dir}/backend_extension_config.json",
            f"--binary_file {binary_name}",
            f"--output_dir {self.tmp_dir}",
            "--profiling_level detailed",
            "--profiling_option optrace",
        ]
        result = subprocess.run(
            " ".join(cmds),
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        assert os.path.isfile(f"{self.tmp_dir}/{binary_name}.bin"), result.stderr

    def qnn_net_run(self, graph_name="forward.serialized"):

        self.config["backend_extension_config"]["backend_extensions"][
            "shared_library_path"
        ] = "./libQnnHtpNetRunExtensions.so"
        for file_name, data in self.config.items():
            with open(f"{self.tmp_dir}/{file_name}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

        target = "aarch64-android"
        files = [
            f"{self.qnn_sdk}/lib/{target}/libQnnHtpNetRunExtensions.so",
            f"{self.tmp_dir}/backend_extension_config.json",
            f"{self.tmp_dir}/config.json",
            f"{self.tmp_dir}/{graph_name}.bin",
            f"{self.qnn_sdk}/bin/{target}/qnn-net-run",
        ]
        cmds = [
            f"export LD_LIBRARY_PATH={self.workspace} &&",
            f"export ADSP_LIBRARY_PATH={self.workspace} &&",
            f"cd {self.workspace} &&",
            "./qnn-net-run",
            "--backend libQnnHtp.so",
            "--input_list input_list.txt",
            f"--retrieve_context {graph_name}.bin",
            "--use_native_input_files",
            "--use_native_output_files",
            "--config_file backend_extension_config.json",
            "--profiling_level detailed",
            "--profiling_option optrace",
        ]
        self.adb.push(
            inputs=self.sample_input,
            files=files,
        )
        self.adb.execute(custom_runner_cmd=" ".join(cmds))
        self.adb._adb(
            [
                "pull",
                "-a",
                f"{self.workspace}/output/qnn-profiling-data_0.log",
                self.tmp_dir,
            ]
        )

        assert os.path.isfile(
            f"{self.tmp_dir}/qnn-profiling-data_0.log"
        ), f"Error: qnn-profiling-data_0.log not found in {self.tmp_dir}"

    def qnn_profile_viewer(self, graph_name="forward_schematic", graph_idx=0):
        self.config["backend_extension_config"]["backend_extensions"][
            "shared_library_path"
        ] = "./libQnnHtpNetRunExtensions.so"
        self.config["backend_extension_config"] = {"features": {"qhas_json": True}}
        for file_name, data in self.config.items():
            with open(f"{self.tmp_dir}/{file_name}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

        target = "x86_64-linux-clang"
        cmds = [
            f"{self.qnn_sdk}/bin/{target}/qnn-profile-viewer",
            f"--config {self.tmp_dir}/backend_extension_config.json",
            f"--schematic {self.root}/{graph_name}.bin",
            f"--reader {self.qnn_sdk}/lib/{target}/libQnnHtpOptraceProfilingReader.so",
            f"--input_log {self.tmp_dir}/qnn-profiling-data_0.log",
            f"--output {self.tmp_dir}/optrace_{graph_idx}.json",
        ]
        result = subprocess.run(
            " ".join(cmds),
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        assert (
            result.returncode == 0
        ), f"Process failed with error: {result.stderr.decode('utf-8')}"

    def generate_optrace(
        self,
        qnn_binary_file="forward_0.dlc",
    ):
        """
        Generate Qnn HTP Optrace Profiling https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html#qnn-htp-optrace-profiling
        and QNN HTP Analysis Summary (QHAS) https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html#qnn-htp-analysis-summary-qhas
        . You can utilize the QAIRT Visualizer (https://pypi.org/project/qairt-visualizer/) to visualize the results from the files above.
        """
        graph_name, file_extension = os.path.splitext(qnn_binary_file)
        assert file_extension in [
            ".dlc",
            ".bin",
        ], f"Invalid file extension '{file_extension}'. Supported extensions are 'dlc' and 'bin'."

        # Attempt to extract a numeric index from the end of the graph name (e.g., "forward_123")
        match = re.match(r"^(.*)_(\d+)$", graph_name)
        graph_base_name = graph_name
        graph_idx = 0

        if match:
            graph_base_name = match.group(1)
            graph_idx = int(match.group(2))

        # Handle .dlc file extension by generating a serialized version of the graph
        if file_extension == ".dlc":
            self.qnn_context_binary_generator(
                qnn_binary_file, f"{graph_base_name}.serialized"
            )
            graph_name = f"{graph_base_name}.serialized"

        # Run the QNN graph and generate the schematic
        self.qnn_net_run(graph_name=graph_name)
        self.qnn_profile_viewer(
            graph_name=f"{graph_base_name}_schematic", graph_idx=graph_idx
        )

        # Clean up the schematic binary file if it exists
        schematic_bin_path = os.path.join(self.root, f"{graph_base_name}_schematic.bin")
        if os.path.isfile(schematic_bin_path):
            os.remove(schematic_bin_path)

        optrace_path = os.path.join(self.tmp_dir, f"optrace_{graph_idx}.json")
        qhas_path = os.path.join(
            self.tmp_dir, f"optrace_{graph_idx}_qnn_htp_analysis_summary.json"
        )
        assert os.path.isfile(optrace_path) and os.path.isfile(qhas_path), (
            "Error: Required files not found - either "
            f"{os.path.basename(optrace_path)} or {os.path.basename(qhas_path)} is missing."
        )

        return optrace_path, qhas_path


def generate_optrace(
    artifact, soc_id: QcomChipset, adb, pte_path: str, inputs: Tuple[torch.Tensor]
):
    """
    Generate optrace and QHAS (QNN HTP Analysis Summary) JSON files.

    Args:
        artifact (str): Path to the artifact folder.
        adb (SimpleADB): An object for communicating with Android device
        pte_path (str): The path to the generated PTE file, including the file extension (e.g., model.pte).
        inputs (Tuple[torch.Tensor]): The input tensors for the model.


    Returns:
        dict: A dictionary where keys are the dumped file paths and values are tuples containing the paths
        to the generated optrace and QHAS JSON files.
    """
    filename, _ = os.path.splitext(pte_path.split(os.sep)[-1])

    # Dump compiled binaries
    dumpfiles = dump_context_from_pte(pte_path)

    # Generate optrace and QHAS
    qnn_tool = QnnTool(
        artifact,
        inputs,
        soc_id,
        adb,
        build_folder=adb.build_path,
        workspace=adb.workspace,
    )

    binaries_trace = {}
    for file in dumpfiles:
        filename = file.split(os.sep)[-1]
        optrace, qhas = qnn_tool.generate_optrace(filename)
        binaries_trace[file] = (optrace, qhas)
    return binaries_trace
