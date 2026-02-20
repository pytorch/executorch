# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Union

import executorch.backends.samsung.python.PyGraphWrapperAdaptor as PyGraphWrapper

import numpy as np

import torch
from executorch.backends.samsung.builders.utils import DATA_TYPE_STR_MAPPING
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.backends.samsung.utils.utils import quantize_tensor


class EnnGraph:
    def __init__(self):
        # default
        self.version = "0.6.0"
        self.graph = PyGraphWrapper.PyEnnGraphWrapper()
        self.graph.Init()

        self.inputs = []
        self.outputs = []

    def init(self, name: str, soc_name):
        self.name = name
        self.soc_name = soc_name

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
                # Set
                op.AddOpParam(py_param_wrapper)

        self.graph.DefineOpNode(op)

    def define_tensor(  # noqa: C901
        self,
        name: str,
        shape: List,
        data_type: str,
        tensor_type: str,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        quant_param: Optional[Dict[str, Any]] = None,
    ) -> int:
        layout = "NCHW" if len(shape) == 4 else "UNDEFINED"

        if quant_param is not None:
            data_type = DATA_TYPE_STR_MAPPING[
                quant_param[QuantConstants.QUANT_KEY.quant_dtype]
            ]

        tensor = PyGraphWrapper.PyEnnTensorWrapper(name, shape, data_type, layout)

        if quant_param is not None:
            need_quantize = True

            scales = self._affine_meta_param(
                quant_param[QuantConstants.QUANT_KEY.scale]
            )
            zero_points = self._affine_meta_param(
                quant_param[QuantConstants.QUANT_KEY.zero_point]
            )
            q_dtype = self._affine_meta_param(
                quant_param[QuantConstants.QUANT_KEY.quant_dtype]
            )
            tensor.AddQuantizeParam(q_dtype, scales, zero_points)

            if need_quantize and data is not None:
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data)
                data = quantize_tensor(
                    data,
                    scales,
                    zero_points,
                    quant_param[QuantConstants.QUANT_KEY.quant_dtype],
                    axis=quant_param.get("axis"),
                )

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

    @staticmethod
    def _affine_meta_param(param: Any) -> str:
        type_str_affine_table = {
            torch.int8: "AINT8",
        }
        if isinstance(param, str):
            return param
        if isinstance(param, (float, int)):
            return [param]
        if hasattr(param, "tolist"):
            return param.tolist()
        if isinstance(param, torch.dtype):
            # Convenient for debugging
            param = type_str_affine_table.get(param, "")

        return param
