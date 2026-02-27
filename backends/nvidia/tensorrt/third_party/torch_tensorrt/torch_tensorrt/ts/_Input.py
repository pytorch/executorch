from typing import Any

from torch_tensorrt import _C
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input


class TorchScriptInput(Input):
    """
    Defines an input to a module in terms of expected shape, data type and tensor format.

    Attributes:
        shape_mode (torch_tensorrt.Input._ShapeMode): Is input statically or dynamically shaped
        shape (Tuple or Dict): Either a single Tuple or a dict of tuples defining the input shape.
            Static shaped inputs will have a single tuple. Dynamic inputs will have a dict of the form
            ``{
                "min_shape": Tuple,
                "opt_shape": Tuple,
                "max_shape": Tuple
            }``
        dtype (torch_tensorrt.dtype): The expected data type of the input tensor (default: torch_tensorrt.dtype.float32)
        format (torch_tensorrt.TensorFormat): The expected format of the input tensor (default: torch_tensorrt.TensorFormat.NCHW)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__ Method for torch_tensorrt.Input

        Input accepts one of a few construction patterns

        Args:
            shape (Tuple or List, optional): Static shape of input tensor

        Keyword Arguments:
            shape (Tuple or List, optional): Static shape of input tensor
            min_shape (Tuple or List, optional): Min size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implicitly this sets Input's shape_mode to DYNAMIC
            opt_shape (Tuple or List, optional): Opt size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implicitly this sets Input's shape_mode to DYNAMIC
            max_shape (Tuple or List, optional): Max size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implicitly this sets Input's shape_mode to DYNAMIC
            dtype (torch.dtype or torch_tensorrt.dtype): Expected data type for input tensor (default: torch_tensorrt.dtype.float32)
            format (torch.memory_format or torch_tensorrt.TensorFormat): The expected format of the input tensor (default: torch_tensorrt.TensorFormat.NCHW)
            tensor_domain (Tuple(float, float), optional): The domain of allowed values for the tensor, as interval notation: [tensor_domain[0], tensor_domain[1]).
                Note: Entering "None" (or not specifying) will set the bound to [0, 2)

        Examples:
            - Input([1,3,32,32], dtype=torch.float32, format=torch.channel_last)
            - Input(shape=(1,3,32,32), dtype=torch_tensorrt.dtype.int32, format=torch_tensorrt.TensorFormat.NCHW)
            - Input(min_shape=(1,3,32,32), opt_shape=[2,3,32,32], max_shape=(3,3,32,32)) #Implicitly dtype=torch_tensorrt.dtype.float32, format=torch_tensorrt.TensorFormat.NCHW
        """
        super().__init__(*args, **kwargs)

    def is_trt_dtype(self) -> bool:
        return bool(self.dtype != dtype.long)

    def _to_internal(self) -> _C.Input:
        internal_in = _C.Input()
        if self.shape_mode == Input._ShapeMode.DYNAMIC:
            if isinstance(self.shape, dict):
                if not TorchScriptInput._supported_input_size_type(
                    self.shape["min_shape"]
                ):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["min_shape"]))
                        + " for min_shape"
                    )
                else:
                    internal_in.min = self.shape["min_shape"]

                if not TorchScriptInput._supported_input_size_type(
                    self.shape["opt_shape"]
                ):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["opt_shape"]))
                        + " for opt_shape"
                    )
                else:
                    internal_in.opt = self.shape["opt_shape"]

                if not TorchScriptInput._supported_input_size_type(
                    self.shape["max_shape"]
                ):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["max_shape"]))
                        + " for max_shape"
                    )
                else:
                    internal_in.max = self.shape["max_shape"]
                internal_in.input_is_dynamic = True
        else:
            if not TorchScriptInput._supported_input_size_type(self.shape):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape))
                    + " for shape"
                )
            else:
                internal_in.opt = self.shape
            internal_in.input_is_dynamic = False

        internal_in.dtype = self.dtype.to(_C.dtype)
        internal_in._explicit_set_dtype = self._explicit_set_dtype
        internal_in.format = self.format.to(_C.TensorFormat)

        internal_in.tensor_domain = TorchScriptInput._parse_tensor_domain(
            self.tensor_domain
        )
        return internal_in
