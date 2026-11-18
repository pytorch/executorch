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

        # Only constant tensors that actually carry data get repacked to AINT4:
        # without data there are no bytes to pack, so the declared quant dtype
        # is kept as-is.
        is_int4 = (
            data is not None
            and quant_param is not None
            and self._is_int4_range(
                quant_param[QuantConstants.QUANT_KEY.quant_min],
                quant_param[QuantConstants.QUANT_KEY.quant_max],
            )
        )

        if is_int4:
            data_type = "INT4"
        elif quant_param is not None:
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
            q_dtype = (
                "AINT4"
                if is_int4
                else self._affine_meta_param(
                    quant_param[QuantConstants.QUANT_KEY.quant_dtype]
                )
            )
            tensor.AddQuantizeParam(q_dtype, scales, zero_points)

            if (
                need_quantize
                and data is not None
                and data.dtype in (torch.float16, torch.float32, np.float32, np.float16)
            ):
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data)
                data = quantize_tensor(
                    data,
                    scales,
                    zero_points,
                    quant_param[QuantConstants.QUANT_KEY.quant_dtype],
                    qrange=(
                        (
                            int(quant_param[QuantConstants.QUANT_KEY.quant_min]),
                            int(quant_param[QuantConstants.QUANT_KEY.quant_max]),
                        )
                        if is_int4
                        else None
                    ),
                    axis=quant_param.get("axis"),
                )
                if is_int4:
                    data = self._pack_int4(data)

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
    def _is_int4_range(quant_min: Any, quant_max: Any) -> bool:
        """True when both quant bounds are representable by 4-bit signed int."""
        try:
            qmin, qmax = int(quant_min), int(quant_max)
        except (TypeError, ValueError):
            return False
        return -8 <= qmin and qmax <= 7

    @staticmethod
    def _pack_int4(values: torch.Tensor) -> np.ndarray:
        """Pack int8 values (in [-8, 7]) into ENN AINT4 layout.

        Layout (GraphGen `core/graphgen_ir/graph_util/packed_int4.h`): the even
        element value[2k] goes to the low nibble and value[2k+1] to the high
        nibble of each byte. Total bytes == ceil(numel / 2).
        """
        flat = values.reshape(-1).to(torch.int64)
        if flat.numel() % 2 != 0:
            flat = torch.cat(
                [flat, torch.zeros(1, dtype=torch.int64, device=flat.device)]
            )
        low = (flat[0::2] & 0xF).to(torch.uint8)
        high = (flat[1::2] & 0xF).to(torch.uint8)
        packed = low | (high << 4)
        return packed.contiguous().numpy()

    @staticmethod
    def _affine_meta_param(param: Any) -> str:
        type_str_affine_table = {
            torch.int8: "AINT8",
            torch.int16: "AINT16",
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
