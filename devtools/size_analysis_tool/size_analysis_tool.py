# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from executorch.devtools import parse_etrecord

from executorch.exir import ExportedProgram
from executorch.exir.backend.backend_api import LoweredBackendModule


def _get_tensor_data(node: torch.fx.Node, tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "name": node.name,
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype)[6:],  # Remove "torch." prefix
        "element_size": tensor.element_size(),
        "shape": list(tensor.shape),
        "num_bytes": tensor.element_size() * tensor.numel(),
        "nn_module_stack": (
            str(node.meta["nn_module_stack"])
            if "nn_module_stack" in node.meta
            else None
        ),
    }


def _get_delegate_blob_data(
    node: torch.fx.Node,
    lowered_backend_module: LoweredBackendModule,
    delegate_deserializers: Optional[
        Dict[str, Callable[[bytes], Dict[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    delegate_blob_data = {
        "name": node.name,
        "backend_id": lowered_backend_module.backend_id,
        "num_bytes": len(lowered_backend_module.processed_bytes),
    }
    if (
        delegate_deserializers is not None
        and lowered_backend_module.backend_id in delegate_deserializers
    ):
        delegate_blob_data.update(
            delegate_deserializers[lowered_backend_module.backend_id](
                lowered_backend_module.processed_bytes
            )
        )

    return delegate_blob_data


def _get_nested_model_data(
    graph_module: torch.fx.GraphModule,
    delegate_deserializers: Optional[
        Dict[str, Callable[[bytes], Dict[str, Any]]]
    ] = None,
    tensor_data: Optional[List[Dict[str, Any]]] = None,
    delegate_blob_data: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if tensor_data is None:
        tensor_data = []

    if delegate_blob_data is None:
        delegate_blob_data = []

    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            node_attr = getattr(node.graph.owning_module, node.target)
            if isinstance(node_attr, torch.Tensor):
                tensor_data.append(_get_tensor_data(node, node_attr))
            elif isinstance(node_attr, torch.fx.GraphModule):
                _get_nested_model_data(
                    node_attr, delegate_deserializers, tensor_data, delegate_blob_data
                )
            elif isinstance(node_attr, LoweredBackendModule):
                delegate_blob_data.append(
                    _get_delegate_blob_data(node, node_attr, delegate_deserializers)
                )

    return (tensor_data, delegate_blob_data)


def generate_model_size_information(
    model: ExportedProgram,
    delegate_deserializers: Optional[
        Dict[str, Callable[[bytes], Dict[str, Any]]]
    ] = None,
    flatbuffer: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Generate a json-serializable Dict containing information about a model's
    size. This includes data about individual tensors and delegate blobs.
    Optionally:
    - delegate_deserializers can be provided to manually specify additional
      information to include for delegate blobs for specific backends.
    - flatbuffer can be provided to include a comparison of total tensor data
      size to overall model size
    """

    tensor_and_delegate_blob_data = _get_nested_model_data(
        model.graph_module, delegate_deserializers
    )

    for data_list in tensor_and_delegate_blob_data:
        data_list.sort(key=lambda data: data["num_bytes"], reverse=True)

    (tensor_data, delegate_blob_data) = tensor_and_delegate_blob_data

    total_tensor_data_size = sum(data["num_bytes"] for data in tensor_data)
    total_delegate_blob_data_size = sum(
        data["num_bytes"] for data in delegate_blob_data
    )
    overview = {
        "total_tensor_data_size": total_tensor_data_size,
        "total_delegate_blob_data_size": total_delegate_blob_data_size,
    }
    if flatbuffer is not None:
        model_size = len(flatbuffer)
        overview.update(
            {
                "serialization_metadata_size": (
                    model_size - total_tensor_data_size - total_delegate_blob_data_size
                ),
                "model_size": model_size,
            }
        )

    return {
        "tensor_data": tensor_data,
        "delegate_blob_data": delegate_blob_data,
        "overview": overview,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--etrecord_path",
        required=True,
        help="The path to the ETRecord for the model to generate size information for",
    )

    parser.add_argument(
        "--output_path",
        default="model_size_information.json",
        help="The output path for the model size information as a json file",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    etrecord = parse_etrecord(args.etrecord_path)

    all_model_size_information = [
        generate_model_size_information(
            model=exported_program,
            delegate_deserializers=None,
            flatbuffer=None,
        )
        for (name, exported_program) in etrecord.graph_map.items()
    ]

    with open(args.output_path, "w") as f:
        f.write(json.dumps(all_model_size_information))


if __name__ == "__main__":
    main()
