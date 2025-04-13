#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import itertools
import logging
from typing import List, Optional

import executorch.backends.nxp.backend.ir.lib.tflite.Buffer as libBuffer
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.CustomOptionsFormat as libCustomOptionsFormat
import executorch.backends.nxp.backend.ir.lib.tflite.Model as libModel
import executorch.backends.nxp.backend.ir.lib.tflite.Operator as libOperator
import executorch.backends.nxp.backend.ir.lib.tflite.OperatorCode as libOperatorCode
import executorch.backends.nxp.backend.ir.lib.tflite.QuantizationDetails as libQuantizedDetails
import executorch.backends.nxp.backend.ir.lib.tflite.QuantizationParameters as libQuantizedParameters
import executorch.backends.nxp.backend.ir.lib.tflite.SubGraph as libSubGraphs
import executorch.backends.nxp.backend.ir.lib.tflite.Tensor as libTensor
import executorch.backends.nxp.backend.ir.lib.tflite.TensorType as libTensorType
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta

import flatbuffers as fb
import numpy as np
from executorch.backends.nxp.backend.ir import tensor_formatting
from executorch.backends.nxp.backend.ir.tflite_generator.meta import types
from executorch.backends.nxp.backend.ir.tflite_generator.meta.types import name_for_type

logger = logging.getLogger(__name__)


def _exactly_one_is_none(obj1: Optional, obj2: Optional):
    return (obj1 is not None and obj2 is None) or (obj1 is None and obj2 is not None)


class Buffer(meta.TFLiteObject):
    """'data' is an array of any type, but MUST have the correct 'dtype' specified!"""

    data: np.ndarray
    type: libTensorType.TensorType

    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more efficient. """

    """ Index to the 'buffers' vector. Used to assign the 'buffer' attribute of the 
        Tensor, this buffer belongs to."""
    tmp_index: int

    def __init__(
        self,
        data: np.ndarray = None,
        data_type: libTensorType.TensorType = libTensorType.TensorType.INT32,
    ) -> None:
        self.data = data
        self.type = data_type

    def __data_is_empty(self):
        """Determine if the buffer data is empty."""
        return (self.data is None) or (self.data.size == 0)

    def get_prepend_function(self, builder: fb.Builder):
        return types.prepend_function(builder, self.type)

    def gen_tflite(self, builder: fb.Builder):
        if self.__data_is_empty():
            # If there is no data, table is empty
            libBuffer.Start(builder)
            return libBuffer.End(builder)

        if self.data.dtype.itemsize != 1:
            # TFLite Buffer is an array of bytes. Larger datatypes must be reduced to bytes first.
            self.data = np.frombuffer(self.data.tobytes(), np.uint8)
        else:
            # Arrays of bytes must also be flattened.
            self.data = self.data.flatten()

        if self.data.dtype.kind in ["b", "i", "u", "f"]:  # flatbuffers.builder line 483
            tfl_data = builder.CreateNumpyVector(self.data)
            # In case of problems, see 'https://github.com/google/flatbuffers/issues/4668'.

        elif self.data.dtype.kind == "S":
            # String tensor. Not sure how to handle this case. I've played around with 'builder.CreateString()' but I
            #  couldn't quite make it work. As it is not a priority right now, just exit with error.
            logger.error(
                "Generating a TFLite static string tensor is not yet supported."
            )
            raise RuntimeError()

        else:
            # Cannot use the 'CreateNumpyVector' method -> use specific prepend functions.
            logger.warning(
                f"Creating a static TFLite tensor buffer for type '{name_for_type(self.type)}'. "
                "This is not a common case and it has not been tested!"
            )

            prepend = self.get_prepend_function(builder)

            # 'data' length has to be multiplied by item size, because tflite.Buffer is a vector of type 'UINT8'.
            #  So e.g. one 'INT32' item will take up 4 spaces in the vector.
            len_bytes = len(self.data) * types.type_size(self.type)

            libBuffer.StartDataVector(builder, len_bytes)
            # Flatbuffer is built in reverse, so for correct order, data must be iterated in reverse.
            for val in reversed(self.data):
                prepend(val)
            tfl_data = builder.EndVector()

        libBuffer.Start(builder)
        libBuffer.AddData(builder, tfl_data)

        return libBuffer.End(builder)


class Buffers(meta.TFLiteVector):
    vector: List[Buffer]

    def __init__(self, vector: List[Buffer] = None) -> None:
        super().__init__(vector, libModel.StartBuffersVector)


class OperatorCode(meta.TFLiteObject):
    """Represents an OperatorCode object, used in the vector 'operator_codes' in the model."""

    builtin_code: libBuiltinOperator.BuiltinOperator
    version: int
    custom_code: str

    def __init__(
        self,
        builtin_code: libBuiltinOperator.BuiltinOperator,
        version: int = 1,
        custom_code: str = None,
    ):
        """
        :param builtin_code: Operator code from the 'BuiltinOperator' enum.
        :param version: Operator version. Defaults to 1.
        :param custom_code: Custom code name. Parameter 'builtin_code' must be set to
                'BuiltinOperator.CUSTOM' when custom code is used.
        """
        self.version = version
        self.builtin_code = builtin_code
        self.custom_code = custom_code

        if (
            self.custom_code is not None
            and builtin_code != libBuiltinOperator.BuiltinOperator.CUSTOM
        ):
            logger.error(
                f"Attempt to use custom code with non-CUSTOM builtin code ({builtin_code})."
            )

    def gen_tflite(self, builder: fb.builder):
        """Generate TFLite representation for this OperatorCode"""
        if self.custom_code is not None:
            custom_code = builder.CreateString(self.custom_code)
        else:
            custom_code = None

        libOperatorCode.Start(builder)

        # The 'deprecated_builtin_code' is a byte. Make sure it doesn't overflow.
        # noinspection PyTypeChecker
        if self.builtin_code <= 127:
            libOperatorCode.AddDeprecatedBuiltinCode(builder, self.builtin_code)

        libOperatorCode.AddVersion(builder, self.version)
        libOperatorCode.AddBuiltinCode(builder, self.builtin_code)
        if custom_code is not None:
            libOperatorCode.AddCustomCode(builder, custom_code)
        return libOperatorCode.End(builder)


class OperatorCodes(meta.TFLiteVector):
    vector: List[OperatorCode]

    def __init__(self, operator_codes: List[OperatorCode] = None) -> None:
        super().__init__(operator_codes, libModel.StartOperatorCodesVector)


class Min(meta.FloatVector):
    def __init__(self, min: List[float] = None) -> None:
        super().__init__(min, libQuantizedParameters.StartMinVector, gen_empty=False)


class Max(meta.FloatVector):
    def __init__(self, max: List[float] = None) -> None:
        super().__init__(max, libQuantizedParameters.StartMaxVector, gen_empty=False)


class Scale(meta.FloatVector):
    def __init__(self, scale: List[float] = None) -> None:
        super().__init__(scale, libQuantizedParameters.StartScaleVector)


class ZeroPoint(meta.IntVector):
    def __init__(self, zero_point: List[int] = None) -> None:
        super().__init__(
            zero_point,
            libQuantizedParameters.StartZeroPointVector,
            lambda builder: builder.PrependInt64,
        )


class Quantization(meta.TFLiteObject):
    min: Min
    max: Max
    scale: Optional[Scale]
    zero_point: ZeroPoint
    quantized_dimension: int
    details_type: libQuantizedDetails.QuantizationDetails

    # TODO details

    def __init__(
        self,
        min: Min = Min(),  # noqa B008
        max: Max = Max(),  # noqa B008
        scale: Scale = None,
        zero_point: ZeroPoint = ZeroPoint([0]),  # noqa B008
        quantized_dimension: int = 0,
        details_type: libQuantizedDetails.QuantizationDetails = libQuantizedDetails.QuantizationDetails.NONE,
    ) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.zero_point = zero_point
        self.quantized_dimension = quantized_dimension
        self.details_type = details_type

    def __eq__(self, other):
        if self is None and other is None:
            return True
        elif _exactly_one_is_none(self, other):
            return False

        if _exactly_one_is_none(self.scale, other.scale):
            return False

        if self.scale is not None:
            if self.scale != other.scale:
                return False
        if self.zero_point != other.zero_point:
            return False
        if self.quantized_dimension != other.quantized_dimension:
            return False
        if self.min != other.min:
            return False
        if self.max != other.max:
            return False

        return True

    def is_per_channel(self) -> bool:
        """Determine if this quantization is per channel, instead of per tensor."""
        if (self.scale is not None and self.zero_point is not None) and (
            self.scale.len() == self.zero_point.len()
        ):
            return self.scale.len() > 1

        return False

    def is_per_tensor(self) -> bool:
        """Determine if this quantization is per tensor"""
        if (self.scale is not None and self.zero_point is not None) and (
            self.scale.len() == self.zero_point.len()
        ):
            return self.scale.len() == 1

        return False

    def gen_tflite(self, builder: fb.Builder):
        # Sometimes 1D per-tensor quantized tensors can have quantized_dimension != 0
        # (residue from badly defined ONNX models). This would cause TFLite inference to crash.
        if not self.is_per_channel():
            self.quantized_dimension = 0

        tfl_min = self.min.gen_tflite(builder)
        tfl_max = self.max.gen_tflite(builder)
        tfl_scale = self.scale.gen_tflite(builder)
        tfl_zero_point = self.zero_point.gen_tflite(builder)

        libQuantizedParameters.Start(builder)

        if tfl_min is not None:
            libQuantizedParameters.AddMin(builder, tfl_min)

        if tfl_max is not None:
            libQuantizedParameters.AddMax(builder, tfl_max)

        libQuantizedParameters.AddScale(builder, tfl_scale)

        libQuantizedParameters.AddZeroPoint(builder, tfl_zero_point)

        libQuantizedParameters.AddDetailsType(builder, self.details_type)

        libQuantizedParameters.AddQuantizedDimension(builder, self.quantized_dimension)

        return libQuantizedParameters.End(builder)


class Shape(meta.IntVector):
    __shape_offset: int

    __also_has_signature: bool
    __shape_signature_vector: List[int]
    __shape_signature_offset: int

    def __init__(self, shape: List[int]) -> None:
        super().__init__(shape, libTensor.StartShapeVector)
        self.__also_has_signature = False

    @property
    def flat_size(self):
        return np.prod(self.vector).item()

    def is_symbolic(self) -> bool:
        """Determine if the shape uses symbolic dimensions

        :return: True, if at least 1 dimension of the shape is not a positive integer.
        """

        return not all(isinstance(dim, int) and dim >= 0 for dim in self.vector)

    def is_well_defined(self) -> bool:
        """Determine if the shape is not empty and also is not symbolic.

        :return: True, if the shape contains just positive integers.
        """

        if self.len() == 0:
            return False

        return not self.is_symbolic()

    def __check_dims(self):
        """Check if all dimensions are integers. If not, transform this
        to 'shape_signature'."""

        self.__shape_signature_vector = []

        for val in self.vector:
            if (not isinstance(val, int)) or (val < 0):
                val = -1
                self.__also_has_signature = True

            self.__shape_signature_vector.append(val)

        if self.__also_has_signature:
            self.vector = [abs(val) for val in self.__shape_signature_vector]

    def gen_tflite(self, builder: fb.Builder, tensor):
        """Generates TFLite code for the Shape"""
        self.__check_dims()

        if self.__also_has_signature:
            tensor.has_rank = True

        self.__shape_offset = super().gen_tflite(builder)
        if self.__also_has_signature:
            self.vector = self.__shape_signature_vector
            self.__shape_signature_offset = super().gen_tflite(builder)

    def add_tf_lite(self, builder):
        libTensor.AddShape(builder, self.__shape_offset)

        if self.__also_has_signature:
            libTensor.AddShapeSignature(builder, self.__shape_signature_offset)


class Tensor(meta.TFLiteObject):
    is_variable: bool
    has_rank: bool
    type: libTensorType.TensorType
    buffer: int
    name: str
    shape: Shape
    quantization: Quantization
    # TODO sparsity
    # TODO shapeSignature
    # TODO variantTensors

    tensor_format: tensor_formatting.TensorFormat

    # TODO If 'hasRank' is false, "shape" must be [].

    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more efficient. """

    """ Reference to the 'Buffer' object holding this tensors data. 'tmpBuffer' MUST be 
        stored a 'Buffers' object and MUST be referenced using the index 'buffer'.  """
    tmp_buffer: Buffer

    """ Index to the 'tensors' vector for this tensor. """
    tmp_index: int

    # A boolean indicating, that this tensor should be considered as empty, by the TFLite inference engine.
    # Can only be used for optional tensors. Whether a tensor is optional is usually indicated in the comments of the
    #  corresponding TFLite kernel files.
    # If set to True, TFLite kernel modules will receive 'nullptr' as the returned value from the
    #  'GetOptionalInputTensor()' function.
    tmp_null_tensor: bool

    @property
    def rank(self):
        """Get the number of dimensions of this `Tensor`."""
        return self.shape.len()

    def __init__(
        self,
        shape: Shape = None,
        name: str = None,
        buffer: int = None,
        data_type: libTensorType.TensorType = libTensorType.TensorType.FLOAT32,
        quantization: Quantization = None,
        is_variable: bool = False,
        has_rank: bool = False,
    ) -> None:
        self.is_variable = is_variable
        self.has_rank = has_rank
        self.type = data_type
        self.buffer = buffer
        self.name = name
        self.shape = shape
        self.quantization = quantization

        self.tmp_null_tensor = False

        self.tensor_format = tensor_formatting.TensorFormat.NONE

    def gen_tflite(self, builder: fb.Builder):

        if self.shape is not None:
            self.shape.gen_tflite(builder, self)

        if self.name is not None:
            name = builder.CreateString(self.name)
        else:
            name = None

        if self.quantization is not None:
            tfl_quantization = self.quantization.gen_tflite(builder)
        else:
            tfl_quantization = None

        libTensor.Start(builder)

        if self.shape is not None:
            self.shape.add_tf_lite(builder)

        libTensor.AddType(builder, self.type)

        if self.buffer is not None:
            libTensor.AddBuffer(builder, self.buffer)

        if name is not None:
            libTensor.AddName(builder, name)

        if tfl_quantization is not None:
            libTensor.AddQuantization(builder, tfl_quantization)

        libTensor.AddIsVariable(builder, self.is_variable)

        libTensor.AddHasRank(builder, self.has_rank)

        return libTensor.End(builder)


class Tensors(meta.TFLiteVector):
    vector: List[Tensor]

    def __init__(self, tensors: List[Tensor] = None) -> None:
        super().__init__(tensors, libSubGraphs.StartTensorsVector)


class OperatorInputs(meta.IntVector):
    def __init__(self, inputs: List[int] = None):
        super().__init__(inputs, libOperator.StartInputsVector)


class OperatorOutputs(meta.IntVector):
    def __init__(self, outputs: List[int] = None):
        super().__init__(outputs, libOperator.StartOutputsVector)


class MutatingVariableInputs(meta.BoolVector):
    def __init__(self, mutating_variable_inputs: List[bool] = None) -> None:
        super().__init__(
            mutating_variable_inputs, libOperator.StartMutatingVariableInputsVector
        )


class Operator(meta.TFLiteObject):
    opcode_index: int
    custom_options_format: (
        libCustomOptionsFormat.CustomOptionsFormat
    )  # Only default value is possible
    mutating_variable_inputs: MutatingVariableInputs
    inputs: OperatorInputs
    outputs: OperatorOutputs
    builtin_options: meta.BuiltinOptions
    custom_options: meta.CustomOptions
    # TODO intermediates

    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more efficient. """

    """ Lists of references to 'Tensor' objects. Simpler to use when converting
        than 'inputs' and 'outputs'. """
    tmp_inputs: List[Tensor]
    tmp_outputs: List[Tensor]
    tmp_version: int  # OperatorConverter uses this to assign the corresponding operator code with correct version.

    # If `True`, this is an extra operator added during conversion. It was not present in the original ONNX model.
    tmp_added_extra: bool

    def __init__(
        self,
        inputs: OperatorInputs = None,
        outputs: OperatorOutputs = None,
        builtin_options: meta.BuiltinOptions = None,
        opcode_index: int = 0,
        mutating_variable_inputs: MutatingVariableInputs = MutatingVariableInputs(),  # noqa B008
        custom_options_format: libCustomOptionsFormat.CustomOptionsFormat = libCustomOptionsFormat.CustomOptionsFormat.FLEXBUFFERS,
        custom_options: meta.CustomOptions = None,
    ) -> None:
        self.opcode_index = opcode_index
        self.custom_options_format = custom_options_format
        self.mutating_variable_inputs = mutating_variable_inputs
        self.builtin_options = builtin_options
        if inputs is None:
            inputs = OperatorInputs()
        self.inputs = inputs
        if outputs is None:
            outputs = OperatorOutputs()
        self.outputs = outputs
        self.custom_options = custom_options

        self.tmp_inputs = []
        self.tmp_outputs = []
        self.tmp_version = 1
        self.tmp_added_extra = False

    def uses_per_channel_quantization(self) -> bool:
        """Determine if this operator uses per-channel quantization."""
        for tensor in itertools.chain(self.tmp_inputs, self.tmp_outputs):
            if tensor.quantization is None:
                continue

            if tensor.quantization.is_per_channel():
                return True

        return False

    def is_quantized_without_qdq(self) -> bool:
        """Determine if the Operator was quantized but not using the QDQ schema.

        ! This only works before quantization parameters are propagated !
        """
        y = self.tmp_outputs[0]

        if y.type not in {
            libTensorType.TensorType.INT8,
            libTensorType.TensorType.UINT8,
        }:
            return False

        inputs_quantized = any(x.quantization is not None for x in self.tmp_inputs)

        # Inputs are quantized and output isn't.
        return inputs_quantized and y.quantization is None

    def is_qdq_quantized(self) -> bool:
        """Determine if the Operator was quantized using the QDQ schema.

        ! This only works before quantization parameters are propagated !
        """
        y = self.tmp_outputs[0]
        output_quantized = y.quantization is not None
        output_8b_int = y.type in {
            libTensorType.TensorType.INT8,
            libTensorType.TensorType.UINT8,
        }

        if not output_quantized and output_8b_int:
            # (U)INT8 but not quantized -> not QDQ
            return False
        elif output_quantized and not output_8b_int:
            # Non-(U)INT8 output, but quantized -> not supported
            return False

        # Output quantized + INT8/UINT8 or different type (bool etc.)

        # Check if any of the inputs is quantized
        return any(x.quantization is not None for x in self.tmp_inputs)

    def gen_tflite(self, builder: fb.Builder):
        if self.inputs is not None:
            tfl_inputs = self.inputs.gen_tflite(builder)
        else:
            tfl_inputs = None

        if self.outputs is not None:
            tfl_outputs = self.outputs.gen_tflite(builder)
        else:
            tfl_outputs = None

        if self.custom_options is not None:
            tfl_custom_options = builder.CreateByteVector(self.custom_options)
        else:
            tfl_custom_options = None

        if self.builtin_options is not None:
            tfl_builtin_options = self.builtin_options.gen_tflite(builder)
        else:
            tfl_builtin_options = None

        if self.mutating_variable_inputs is not None:
            tfl_mutating_variable_inputs = self.mutating_variable_inputs.gen_tflite(
                builder
            )
        else:
            tfl_mutating_variable_inputs = None

        libOperator.Start(builder)

        libOperator.AddOpcodeIndex(builder, self.opcode_index)

        if tfl_inputs is not None:
            libOperator.AddInputs(builder, tfl_inputs)

        if tfl_outputs is not None:
            libOperator.AddOutputs(builder, tfl_outputs)

        if tfl_builtin_options is not None:
            libOperator.AddBuiltinOptions(builder, tfl_builtin_options)
            libOperator.AddBuiltinOptionsType(
                builder, self.builtin_options.builtin_options_type
            )

        if tfl_custom_options is not None:
            libOperator.AddBuiltinOptionsType(builder, 0)
            libOperator.AddCustomOptionsFormat(builder, self.custom_options_format)
            libOperator.AddCustomOptions(builder, tfl_custom_options)

        if tfl_mutating_variable_inputs is not None:
            libOperator.AddMutatingVariableInputs(builder, tfl_mutating_variable_inputs)

        return libOperator.End(builder)


class Operators(meta.TFLiteVector):
    vector: List[Operator]

    def __init__(self, operators: List[Operator] = None) -> None:
        super().__init__(operators, libSubGraphs.StartOperatorsVector)


class SubGraphInputs(meta.IntVector):
    """List of 'Tensor' objects. Easier to use while converting."""

    tmp_inputs: List[Tensor]

    def __init__(self, inputs: List[int] = None):
        """'inputs' is a list of indices into the 'tensors' vector."""
        super().__init__(inputs, libSubGraphs.StartInputsVector)
        self.tmp_inputs = []


class SubGraphOutputs(meta.IntVector):
    """List of 'Tensor' objects. Easier to use while converting."""

    tmp_outputs: List[Tensor]

    def __init__(self, outputs: List[int] = None):
        """'outputs' is a list of indices into the 'tensors' vector."""
        super().__init__(outputs, libSubGraphs.StartOutputsVector)
        self.tmp_outputs = []


class SubGraph(meta.TFLiteObject):
    inputs: SubGraphInputs
    outputs: SubGraphOutputs
    tensors: Tensors
    operators: Operators

    # TODO name

    def __init__(
        self,
        inputs: SubGraphInputs = None,
        outputs: SubGraphOutputs = None,
        tensors: Tensors = None,
        operators: Operators = None,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.operators = operators

    def gen_tflite(self, builder: fb.Builder):
        if self.tensors is not None:
            tfl_tensors = self.tensors.gen_tflite(builder)
        else:
            tfl_tensors = None

        if self.inputs is not None:
            tfl_inputs = self.inputs.gen_tflite(builder)
        else:
            tfl_inputs = None

        if self.outputs is not None:
            tfl_outputs = self.outputs.gen_tflite(builder)
        else:
            tfl_outputs = None

        if self.operators is not None:
            tfl_operators = self.operators.gen_tflite(builder)
        else:
            tfl_operators = None

        libSubGraphs.Start(builder)

        if tfl_tensors is not None:
            libSubGraphs.AddTensors(builder, tfl_tensors)

        if tfl_inputs is not None:
            libSubGraphs.AddInputs(builder, tfl_inputs)

        if tfl_outputs is not None:
            libSubGraphs.AddOutputs(builder, tfl_outputs)

        if tfl_operators is not None:
            libSubGraphs.AddOperators(builder, tfl_operators)

        return libSubGraphs.End(builder)


class SubGraphs(meta.TFLiteVector):
    vector: List[SubGraph]

    def __init__(self, sub_graphs: List[SubGraph] = None) -> None:
        super().__init__(sub_graphs, libModel.StartSubgraphsVector)


class Model(meta.TFLiteObject):
    version: int
    description: str
    operator_codes: OperatorCodes
    sub_graphs: SubGraphs
    buffers: Buffers
    # TODO signatureDefs
    # TODO metadata
    # TODO metadataBuffer

    __fileIdentifier = "TFL3"  # file_identifier from the used TFLite schema

    @classmethod
    def __gen_file_identifier(cls):
        """Generate byte-like object representing the TFLite format"""
        return cls.__fileIdentifier.encode("ascii")

    def __init__(
        self,
        version: int = 1,
        description: str = None,
        buffers: Buffers = None,
        operator_codes: OperatorCodes = None,
        sub_graphs: SubGraphs = None,
    ) -> None:
        self.version = version
        self.description = description
        self.operator_codes = operator_codes
        self.sub_graphs = sub_graphs
        self.buffers = buffers

    def gen_tflite(self, builder: fb.Builder):
        if self.operator_codes is not None:
            tfl_operator_codes = self.operator_codes.gen_tflite(builder)
        else:
            tfl_operator_codes = None

        if self.sub_graphs is not None:
            tfl_sub_graphs = self.sub_graphs.gen_tflite(builder)
        else:
            tfl_sub_graphs = None

        if self.description is not None:
            tfl_description = builder.CreateString(self.description)
        else:
            tfl_description = None

        if self.buffers is not None:
            tfl_buffers = self.buffers.gen_tflite(builder)
        else:
            tfl_buffers = None

        libModel.Start(builder)

        libModel.AddVersion(builder, self.version)

        if tfl_operator_codes is not None:
            libModel.AddOperatorCodes(builder, tfl_operator_codes)

        if tfl_sub_graphs is not None:
            libModel.AddSubgraphs(builder, tfl_sub_graphs)

        if tfl_description is not None:
            libModel.AddDescription(builder, tfl_description)

        if tfl_buffers is not None:
            libModel.AddBuffers(builder, tfl_buffers)

        builder.Finish(libModel.End(builder), Model.__gen_file_identifier())
