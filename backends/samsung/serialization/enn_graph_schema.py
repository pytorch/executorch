# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Union

import executorch.backends.samsung.python.PyGraphWrapperAdaptor as PyGraphWrapper

import numpy as np

import torch


class EnnGraph:
    def __init__(self):
        # default
        self.version = "0.6.0"
        self.graph = PyGraphWrapper.PyEnnGraphWrapper()
        self.graph.Init()

        self.inputs = []
        self.outputs = []

    def define_op(
        self,
        name,
        type,
        input_ids: List[int],
        output_ids: List[int],
        params: Optional[Dict] = None,
    ):
        op = PyGraphWrapper.PyEnnOpWrapper(name, type, input_ids, output_ids)

        if params is not None:
            assert isinstance(params, dict), "Please pass op params as dict type."
            for key in params:
                py_param_wrapper = PyGraphWrapper.OpParamWrapper(key)
                if isinstance(params[key], (list, tuple)):
                    py_param_wrapper.SetVectorValue(params[key])
                elif isinstance(params[key], str):
                    py_param_wrapper.SetStringValue(params[key])
                elif isinstance(params[key], (int, float, bool)):
                    py_param_wrapper.SetScalarValue(params[key])
                else:
                    logging.error("Unsupported param type.")
                op.AddOpParam(py_param_wrapper)

        self.graph.DefineOpNode(op)

    def define_tensor(
        self,
        name: str,
        shape: List,
        data_type: str,
        tensor_type: str,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> int:
        layout = "NCHW" if len(shape) == 4 else "UNDEFINED"

        tensor = PyGraphWrapper.PyEnnTensorWrapper(name, shape, data_type, layout)

        if data is not None:
            if isinstance(data, torch.Tensor):
                data = data.detach().numpy()
            tensor.AddData(data)

        id = self.graph.DefineTensor(tensor)

        if tensor_type == "INPUT":
            self.inputs.append(id)
        elif tensor_type == "OUTPUT":
            self.outputs.append(id)

        return id

    def finish(self):
        self.graph.SetGraphInputTensors(self.inputs)
        self.graph.SetGraphOutputTensors(self.outputs)
        self.graph.FinishBuild()

    def serialize(self):
        return self.graph.Serialize()
