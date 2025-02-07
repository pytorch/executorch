import os
import shutil
import tempfile

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import pandas as pd
from graphviz import Digraph


class DrawGraph:
    def __init__(
        self,
        filename: str,
        directory: str,
        py_op_wrapper_list: [PyQnnWrapper.PyQnnOpWrapper],
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
                if op_wrapper.GetOpConfig()["outputTensors"][0].version == 1:
                    node = op_wrapper.GetOpConfig()["outputTensors"][i].v1
                    node_name = node.name
                    input_list = []
                    for j in range(op_wrapper.GetOpConfig()["numOfInputs"]):
                        if op_wrapper.GetOpConfig()["inputTensors"][j].version == 1:
                            input_node = op_wrapper.GetOpConfig()["inputTensors"][j].v1
                            input_node_name = input_node.name
                            if input_node_name not in node_list:
                                node_list[input_node_name] = {
                                    "node": input_node,
                                    "input_list": [],
                                }
                            input_list.append(input_node_name)
                            # TODO: tensor v2
                        elif op_wrapper.GetOpConfig()["outputTensors"][j].version == 2:
                            raise ValueError("Unsupported tensor version: 2")
                    if node_name not in node_list:
                        node_list[node_name] = {"node": node, "input_list": input_list}
                    else:
                        node_list[node_name]["input_list"] = input_list
                # TODO: tensor v2
                elif op_wrapper.GetOpConfig()["outputTensors"][i].version == 2:
                    raise ValueError("Unsupported tensor version: 2")

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
                == PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            ):
                scale.append(node.quantizeParams.scaleOffsetEncoding.scale)
                offset.append(node.quantizeParams.scaleOffsetEncoding.offset)
            elif (
                quantization_encoding
                == PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
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
                == PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC
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
                self.filename, temp_directory, format="svg", cleanup=not self.dot_string
            )
            source_file = os.path.join(temp_directory, f"{self.filename}.svg")
            destination_file = os.path.join(".", f"{self.filename}.svg")
            shutil.move(source_file, destination_file)
            if self.dot_string:
                dot_file = os.path.join(temp_directory, f"{self.filename}")
                dot_dest_file = os.path.join(".", f"{self.filename}.dot")
                shutil.move(dot_file, dot_dest_file)
