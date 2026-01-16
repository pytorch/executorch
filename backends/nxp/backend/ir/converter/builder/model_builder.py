#
# Copyright 2023 Martin Pavella
# Copyright 2023-2026 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from copy import deepcopy
from itertools import chain
from typing import List, Optional, Union

import executorch.backends.nxp.backend.ir.converter.conversion.translator as translator
import executorch.backends.nxp.backend.ir.logger as logger
import executorch.backends.nxp.backend.ir.tflite_generator.tflite_model as tflite_model
import numpy as np
from executorch.backends.nxp.backend.edge_helper import is_channels_last_dim_order
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder import (
    quantization_verification,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    uses_shape_broadcasting,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    propagate_quantization,
)
from executorch.backends.nxp.backend.ir.converter.tensor_utils import (
    _buffer_has_data,
    all_tensors_are_static,
    tensor_has_data,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing import optimizer
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    cast_options,
    dequantize_options,
    gather_options,
    pad_options,
    pad_v2_options,
    quantize_options,
    reshape_options,
    slice_options,
    transpose_options,
)
from executorch.backends.nxp.backend.ir.tflite_generator.custom_options.flex_transpose_options import (
    FlexTranspose,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec


class ModelBuilder:
    """
    Class encapsulates a TFLite object model defined in '/src/tflite_generator/'.
    Provides methods to create and modify the TFLite model.
    At the end call 'finish()' to finalize and optimise the model.
    """

    _tfl_model: tflite_model.Model

    _tensor_name_map: dict  # Mapping 'str' to 'tflT.Tensor'

    # Maps BuiltinOperator to a dict, mapping version to index. Operators of type 'BuiltinOperator.CUSTOM'
    # have their 'version' prepended with its name, for example "FlexErf_1".
    op_code_type_index_map: dict[BuiltinOperator, dict[Union[str, int], int]]

    _nchw_tensor_version: dict  # Mapping 'tflT.Tensor' to 'tflT.Tensor' which is
    # equal, but in NCHW format

    _skipped_output_map: dict  # Mapping 'tflT.Tensor' objects that were outputs
    # of skipped operators, to 'tflT.Tensor' outputs of
    # previous operators

    _zeros_tensor_map: dict  # Mapping 'string' shapes to 'tflT.Tensor' objects

    neutron_target_spec: NeutronTargetSpec

    dim_order_map: dict  # Mapping tensor names to their ExecuTorch `dim_order`.

    conversion_config: ConversionConfig

    _default_conversion_config = ConversionConfig()

    def __init__(
        self,
        model_version: int,
        model_description: str,
        neutron_target_spec: NeutronTargetSpec,
        dim_order_map: dict[str, ...],
        conversion_config: ConversionConfig = _default_conversion_config,
    ) -> None:
        self._tfl_model = tflite_model.Model(model_version, model_description)
        self.neutron_target_spec = neutron_target_spec
        self.conversion_config = conversion_config
        self.dim_order_map = dim_order_map

        self.op_code_type_index_map = {}
        self._tensor_name_map = {}
        self._nchw_tensor_version = {}
        self._skipped_output_map = {}
        self._zeros_tensor_map = {}

    def create_zeros_tensor(
        self, dims: List[int], name: str, dtype: np.dtype, can_reuse: bool = False
    ) -> tflite_model.Tensor:
        """Create and return a Tensor with given shape, name and dtype that only contains zeros.
        If 'can_reuse' is True, created tensor can be shared with other operators.
        """

        def _dims_to_string(dims: List[int]):
            """Convert a list of integers to a string."""
            tmp = [str(dim) for dim in dims]
            return "_".join(tmp)

        if can_reuse:
            # The zeros tensor can be shared with other operators
            str_dims = _dims_to_string(dims)
            tensor_as_string = str_dims + dtype.name

            # Check if such tensor already exists
            if tensor_as_string in self._zeros_tensor_map.keys():
                logger.d(
                    f"REUSING zero tensor of size {str_dims} with type {dtype.name}."
                )
                return self._zeros_tensor_map[tensor_as_string]

            else:
                # Create a new one and register it for potential future use.
                logger.d(
                    f"ADDING zero tensor of size {str_dims} with type {dtype.name}."
                )
                data = np.zeros(dims, dtype)
                new_tensor = self.create_tensor_for_data(data, name)

                self._zeros_tensor_map[tensor_as_string] = new_tensor

                return new_tensor

        # Tensor cannot be shared. Just create one and return it
        data = np.zeros(dims, dtype)

        return self.create_tensor_for_data(data, name)

    def create_pad_operator_before(
        self,
        before_op: tflite_model.Operator,
        on_input_index: int,
        explicit_padding: List[List[int]],
        constant_value: np.ndarray = None,
    ) -> tflite_model.Operator:
        """Create a TFLite 'Pad' operator before the 'before_op' operator. The input of 'before_op' on index
             'on_input_index' is where the 'Pad' operator will connect.

        :param before_op: TFLite operator that will consume the output of the new 'Pad' operator.
        :param on_input_index: Index of an input tensor of the 'before_op' operator, which will serve as the new input
                                for the 'Pad' operator.
        :param explicit_padding: TFLite style explicit padding compatible with the TFLite 'Pad' operator.
        :param constant_value: The scalar array used as pad value. Must be same type as input tensor at
                                index 'on_input_index'.
        :return: The TFLite 'Pad' operator.
        """
        if on_input_index >= len(before_op.tmp_inputs):
            logger.e(
                logger.Code.INTERNAL_ERROR,
                f"ModelBuilder.create_pad_operator_before(): input index '{on_input_index}' is out of range!",
            )

        input_tensor = before_op.tmp_inputs[on_input_index]

        # New shape of the tensor after padding
        padded_shape = translator.get_tflite_tensor_shape_with_explicit_padding(
            input_tensor.shape.vector, explicit_padding
        )

        # Create the output tensor of the 'Pad' operator.
        padded_tensor = self.duplicate_tensor(
            input_tensor, input_tensor.name + "_padded", empty_buffer=True
        )
        padded_tensor.shape = tflite_model.Shape(padded_shape)

        # Create the second input of the 'Pad' operator.
        explicit_padding_tensor = self.create_tensor_for_data(
            np.asarray(explicit_padding, dtype=np.int32), "padding"
        )

        # Create the 'Pad' operator
        pad_operator = tflite_model.Operator(builtin_options=pad_options.Pad())
        pad_operator.tmp_inputs = [input_tensor, explicit_padding_tensor]
        pad_operator.tmp_outputs = [padded_tensor]

        # Add tensor with constant values (scalar array)
        if constant_value is not None:
            # Only PadV2 supports the constant value tensor. (Seems that regular Pad does too, but it's not documented.)
            pad_operator.builtin_options = pad_v2_options.PadV2()

            constant_value_tensor = self.create_tensor_for_data(
                constant_value, "constant_values"
            )

            # Input tensor 'constant_values' must have same quantization params as 'input'
            propagate_quantization(input_tensor, constant_value_tensor)
            pad_operator.tmp_inputs.append(constant_value_tensor)

        # Connect the operators
        before_op.tmp_inputs[on_input_index] = padded_tensor

        return pad_operator

    def channels_first_version_of(self, t_tensor: tflite_model.Tensor):
        """Get the channels first version of non-static 't_tensor'. If one is not
        available in the graph yet, add transpose operator to create it."""
        if t_tensor in self._nchw_tensor_version.keys():
            return self._nchw_tensor_version[t_tensor]

        # Need to add Transpose operator to transform 't_tensor' to NCHW.

        new_tensor = self.duplicate_tensor(
            t_tensor, t_tensor.name + "_channels_first", empty_buffer=True
        )
        new_tensor.shape = translator.channels_last_shape_to_channels_first(
            t_tensor.shape
        )
        new_tensor.tensor_format = TensorFormat.CHANNELS_FIRST

        perm = translator.create_channels_last_to_channels_first_permutation(
            t_tensor.rank
        )
        transpose = self._create_transpose_operator(t_tensor, new_tensor, perm)

        self.check_and_append_operator(transpose)

        self._nchw_tensor_version[t_tensor] = new_tensor

        return new_tensor

    def redirect_tensor(
        self, from_tensor: tflite_model.Tensor, to_tensor: tflite_model.Tensor
    ):
        """Create a mapping of 'from_tensor' to 'to_tensor', which will ensure that when 'check_and_append_operator()'
             is called with an operator that references 'from_tensor', it will be replaced by 'to_tensor'. This ensures
             that future operators will not use output tensors of operators, which are not actually in the model.

            This method should be explicitly used when an operator is skipped during conversion, so that other operators
             which used the output tensors of the skipped operator will be redirected to valid tensors, such as an
             appropriate input tensor of the skipped operator.

        :param from_tensor: Tensor which will be replaced by 'to_tensor'.
        :param to_tensor: Valid tensor, that the future operators will use instead of the 'from_operator'.
        """

        old_replacement = self._skipped_output_map.get(from_tensor, None)
        if old_replacement is not None:
            if old_replacement != to_tensor:
                logger.e(
                    logger.Code.INTERNAL_ERROR,
                    "redirect_tensor(): Tensor has already been redirected before!",
                )
            else:
                # Tensor has already been mapped to 'to_tensor'.
                return

        # 'to_tensor' might have been redirected too (and so on) -> find the root of the redirection.
        while to_tensor in self._skipped_output_map.keys():
            to_tensor = self._skipped_output_map[to_tensor]

        # Map 'from_tensor' to 'to_tensor'.
        self._skipped_output_map[from_tensor] = to_tensor

        # Swap the names of the tensors to preserve the model IO interface.
        self.swap_tensor_names(from_tensor, to_tensor)

    def check_and_append_operator(self, t_op: tflite_model.Operator):
        """Append the new TFLite operator to the model."""

        self.get_operators().append(t_op)

    def create_transposed_tensor(
        self, tflite_tensor: tflite_model.Tensor, axes: list[int] | None = None
    ) -> tflite_model.Tensor:
        """Create a transposed version of given static TFLite tensor using numpy.transpose().

        :param tflite_tensor: Static TFLite tensor to create the transposed version for.
        :param axes: Permutation applied during transposition. If None, current axes in reversed order are used.
        :return: The new transposed TFLite tensor.
        """

        if not tensor_has_data(tflite_tensor):
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "ModelBuilder.create_transposed_tensor() requires a static tensor!",
            )

        new_tensor = self.duplicate_tensor(
            tflite_tensor, tflite_tensor.name + "_transposed"
        )

        new_tensor.tmp_buffer.data = np.transpose(new_tensor.tmp_buffer.data, axes)
        new_tensor.shape = tflite_model.Shape(list(new_tensor.tmp_buffer.data.shape))

        return new_tensor

    def duplicate_tensor(
        self,
        tensor: tflite_model.Tensor,
        new_name: Optional[str] = None,
        name_suffix: str = "",
        empty_buffer: bool = False,
    ) -> tflite_model.Tensor:
        """Create a new TFLite tensor, which is an identical copy of 'tensor', with a new name.
             If 'new_name' is given, it will be used as the name for the new tensor.
             If instead the 'name_suffix' is given, it will be appended to the name of 'tensor'.
             If neither is given, the new tensor will have a similar name as 'tensor'.

            The final name may be altered automatically, to ensure uniqueness.

        :param tensor: TFLite tensor to duplicate.
        :param new_name: Optional name for the new tensor.
        :param name_suffix: Optional suffix for the name of the new tensor.
        :param empty_buffer: If `True`, the new copied tensor will have its own new empy buffer with no data.
                              If `False`, the new copied tensor will also have a copy of the buffer (data) of the
                              original tensor.
        :return: A copy of 'tensor'.
        """

        new_tensor = deepcopy(tensor)

        new_name = new_name or tensor.name + name_suffix
        new_tensor.name = self._validate_new_tensor_name(new_name)

        self.append_new_buffer(new_tensor.tmp_buffer)
        if empty_buffer:
            new_tensor.tmp_buffer.data = None

        self.append_new_tensor(new_tensor)

        return new_tensor

    def swap_tensor_names(self, t1: tflite_model.Tensor, t2: tflite_model.Tensor):
        """Correctly swap the names of the 2 provided tensors."""

        logger.internal_assert(
            self._tensor_name_map.get(t1.name, t1) == t1
            and self._tensor_name_map.get(t2.name, t2) == t2,
            "ModelBuilder.swap_tensor_names(): The name to tensor mapping is not valid.",
        )

        self._tensor_name_map[t1.name] = t2
        self._tensor_name_map[t2.name] = t1

        t1.name, t2.name = t2.name, t1.name

    def _make_inputs_channels_first(self):
        new_inputs = []

        for input_tensor in self.get_sub_graph().inputs.tmp_inputs:

            if input_tensor.tensor_format.is_channels_last():
                # The input must be permuted.

                if is_channels_last_dim_order(
                    self.dim_order_map.get(input_tensor.name, [])
                ):
                    # Do NOT insert a Transpose, as the input will already be provided in the channels last format
                    #  during runtime.
                    new_inputs.append(input_tensor)
                    continue

                # Create a Transpose operator and replace the graph input

                new_input_shape = translator.channels_last_shape_to_channels_first(
                    input_tensor.shape
                )
                perm = translator.create_channels_first_to_channels_last_permutation(
                    input_tensor.rank
                )

                if not transposition_is_supported_on_neutron(
                    new_input_shape.vector, list(perm), self.neutron_target_spec
                ):
                    new_inputs.append(input_tensor)
                    continue

                if input_tensor.rank > 6:
                    msg = (
                        f"Couldn't preserve the shape of input tensor '{input_tensor.name}', because it has "
                        f"'{input_tensor.rank}' dimensions. TFLite Transpose only supports up to 6 dimensions."
                    )
                    logger.e(logger.Code.IO_PRESERVATION_ERROR, msg)

                new_input = self.duplicate_tensor(
                    input_tensor, input_tensor.name + "_channels_first"
                )
                new_input.shape = new_input_shape
                new_input.tensor_format = TensorFormat.CHANNELS_FIRST

                transpose = self._create_transpose_operator(
                    new_input, input_tensor, perm
                )

                self.get_operators().vector.insert(0, transpose)

                # Swap the names of `new_input` and `input_tensor`.
                self.swap_tensor_names(new_input, input_tensor)

                new_inputs.append(new_input)

            else:
                # Keep the input
                new_inputs.append(input_tensor)

        self.get_sub_graph().inputs.tmp_inputs = new_inputs

    def _make_outputs_channels_first(self):
        new_outputs = []

        for output_tensor in self.get_sub_graph().outputs.tmp_outputs:
            if output_tensor.tensor_format.is_channels_last():
                # The output must be permuted.

                if is_channels_last_dim_order(
                    self.dim_order_map.get(output_tensor.name, [])
                ):
                    # Do NOT insert a Transpose, as the output will be required to be in the channels last format
                    #  during runtime.
                    new_outputs.append(output_tensor)
                    continue

                # Add a Transpose operator, to make the output channels first

                shape = output_tensor.shape.vector
                perm = translator.create_channels_last_to_channels_first_permutation(
                    len(shape), True
                )
                if not transposition_is_supported_on_neutron(
                    shape, perm, self.neutron_target_spec
                ):
                    new_outputs.append(output_tensor)
                    continue

                if output_tensor.rank > 6:
                    logger.e(
                        logger.Code.IO_PRESERVATION_ERROR,
                        f"Couldn't preserve the shape of output tensor '{output_tensor.name}', because it has "
                        f"'{output_tensor.rank}' dimensions. TFLite Transpose only supports up to 6 "
                        "dimensions.",
                    )

                new_output = self.channels_first_version_of(output_tensor)

                # Swap the names of `new_output` and `output_tensor`.
                self.swap_tensor_names(new_output, output_tensor)

                new_outputs.append(new_output)

            else:
                new_outputs.append(output_tensor)

        self.get_sub_graph().outputs.tmp_outputs = new_outputs

    def _keep_one_empty_buffer(self):
        """Create a single empty `Buffer` object and assign it to all tensors in the model that don't have static data."""
        empty_buffer = self.get_first_empty_buffer()

        for t in self.get_tensors().vector:
            if tensor_has_data(t):
                # The buffer of `t` is not empty.
                continue

            if t.tmp_buffer == empty_buffer:
                # Already optimized.
                continue

            if t.is_variable:
                # The data of the tensor will change at runtime, so it shouldn't share the buffer with other tensors.
                continue

            # It's safe to replace the buffer.
            t.tmp_buffer = empty_buffer

    def replace_io_tensor_format_with_node_format(self):
        for t in chain(
            self.get_sub_graph().inputs.tmp_inputs,
            self.get_sub_graph().outputs.tmp_outputs,
        ):
            if isinstance(t.tensor_format, TensorFormat):
                t.tensor_format = t.tensor_format.to_equal_node_format()

    def finish(self) -> tflite_model.Model:
        """Finalize and optimize the converted TFLite model. Then return it.

        At least one of 'optimization_whitelist' and 'optimization_blacklist' must be 'None'.
        :return: The final TFLite model.
        """

        if self.conversion_config.use_neutron_for_format_conversion:
            # If the input or output is channels last, add a Transpose operator, to make is channels first.
            self._make_inputs_channels_first()
            self._make_outputs_channels_first()

        # Apply optimizations to the internal TFLite model.
        optimizer.Optimizer(
            self, self.conversion_config, self.neutron_target_spec
        ).optimize(
            self.conversion_config.optimization_whitelist,
            self.conversion_config.optimization_blacklist,
        )

        self._keep_one_empty_buffer()

        self.replace_io_tensor_format_with_node_format()

        # Remove outputs, which are not produced by any node. Otherwise, there would be errors after inference.
        operator_outputs = []
        for op in self.get_operators().vector:
            operator_outputs.extend(op.tmp_outputs)
        graph_outputs = self.get_sub_graph().outputs.tmp_outputs.copy()
        for output in graph_outputs:
            if output not in operator_outputs:
                self.get_sub_graph().outputs.tmp_outputs.remove(output)

        # Switch from using 'tmp' references to 'index' references in tensors and buffers.
        self._assign_tensor_and_buffer_indices(
            self.conversion_config.allow_inputs_stripping
        )

        if self.conversion_config.tflite_quantization_integrity_check:
            quantization_verification.verify_quantization_integrity(self._tfl_model)

        return self._tfl_model

    def _assign_io_tensor_indices(self, inputs, outputs, allow_inputs_stripping: bool):
        for tensor in outputs.tmp_outputs:
            try:
                outputs.append(tensor.tmp_index)
            except Exception:
                logger.e(
                    logger.Code.GENERATED_MODEL_INVALID,
                    f"The tensor '{tensor.name}' is among the model outputs, but does NOT appear in the graph!",
                )

        for tensor in inputs.tmp_inputs:
            try:
                inputs.append(tensor.tmp_index)
            except Exception:
                if allow_inputs_stripping:
                    logger.i(
                        f"The input tensor '{tensor.name}' will not be present in generated TFLite graph."
                    )
                else:
                    logger.e(
                        logger.Code.GENERATED_MODEL_INVALID,
                        f"The tensor '{tensor.name}' is among the model inputs, but does NOT appear in the graph!",
                    )

    def _assign_operators_io_tensor_indices(self, operators):
        for operator in operators.vector:
            for inputTensor in operator.tmp_inputs:
                operator.inputs.append(inputTensor.tmp_index)

            for outputTensor in operator.tmp_outputs:
                operator.outputs.append(outputTensor.tmp_index)

    def _assign_tensor_and_buffer_indices(self, allow_inputs_stripping: bool):
        """Correctly initialize all references via indices in all tensors and buffers."""

        # Assign each buffer its index
        for i, buffer in enumerate(self.get_buffers().vector):
            buffer.tmp_index = i

        # Assign each tensor its index and its buffer index
        for i, tensor in enumerate(self.get_tensors().vector):
            if tensor.tmp_null_tensor:
                # Using -1 as the index to the 'tensors' vector is way of telling the TFLite inference engine, that
                #  this tensor should not be used.
                # https://github.com/tensorflow/tensorflow/blob/05404d959119d41a8ffb8a75c6f232cfd8540d45/tensorflow/lite/kernels/kernel_util.cc#L79-L98
                tensor.tmp_index = -1
            else:
                tensor.tmp_index = i

            tensor.buffer = tensor.tmp_buffer.tmp_index

        # TODO Remove inputs and outputs that are not in the tensors collection

        subgraph = self.get_sub_graph()

        # Assign 'Outputs' and 'Inputs' their tensor indices
        self._assign_io_tensor_indices(
            inputs=subgraph.inputs,
            outputs=subgraph.outputs,
            allow_inputs_stripping=allow_inputs_stripping,
        )
        # Assign each operator its inputs and outputs indices
        self._assign_operators_io_tensor_indices(operators=subgraph.operators)

    def _build_operator_code(
        self, op_type: BuiltinOperator, version, custom_code: str = None
    ):
        """Add a new OperatorCode for given 'op_type' and 'version' to the 'operator_codes' vector."""
        op_code = tflite_model.OperatorCode(op_type, version, custom_code)

        self.get_operator_codes().append(op_code)

    def build_empty_buffer(self) -> tflite_model.Buffer:
        """Create, register and return a new empty 'Buffer' object."""
        buffer = tflite_model.Buffer()

        self.get_buffers().append(buffer)

        return buffer

    def create_tensor_for_data(self, data: np.ndarray, name: str):
        data_type = translator.numpy_type_to_tf_lite(data.dtype)

        buffer = tflite_model.Buffer(data, data_type)
        self.append_new_buffer(buffer)

        shape = translator.shape_from_numpy(data)
        name = self._validate_new_tensor_name(name)

        tensor = tflite_model.Tensor(shape, name, data_type=data_type)

        tensor.tmp_buffer = buffer

        self.append_new_tensor(tensor)

        return tensor

    def create_empty_tensor(
        self, name: str, tensor_type: TensorType, shape: Optional[List[int]] = None
    ):
        name = self._validate_new_tensor_name(name)

        if shape is not None:
            shape = tflite_model.Shape(list(shape))

        tensor = tflite_model.Tensor(shape, name, data_type=tensor_type)
        tensor.tmp_buffer = self.build_empty_buffer()

        self.append_new_tensor(tensor)

        return tensor

    def create_null_tensor(self, name: str = "null_"):
        """Create and return a TFLite tensor, which will be recognized by the TFLite inference engine as an empty
             tensor. Internal TFLite kernel functions will return 'nullptr' when accessing this tensor.

        :param name: Optional name for the null tensor.
        :return: The new TFLite null tensor.
        """

        tensor = self.create_empty_tensor(name, TensorType.FLOAT32)
        tensor.tmp_null_tensor = True
        return tensor

    """ -------------------- 'quality of life' functions. -------------------- """

    def operator_can_be_skipped(self, t_op: tflite_model.Operator) -> bool:
        """Determine whether operator 't_op' uses both a graph input and a graph output tensor. If it does, it cannot
             be skipped.

        :param t_op: TFLite operator to check.
        :return: True, if 't_op' doesn't use both a graph input and a graph output.
        """
        sub_graph = self.get_sub_graph()
        graph_inputs = sub_graph.inputs.tmp_inputs
        graph_outputs = sub_graph.outputs.tmp_outputs

        produces_graph_output = any(
            op_output in graph_outputs for op_output in t_op.tmp_outputs
        )

        consumes_graph_input = False
        for op_input in t_op.tmp_inputs:
            root = self._skipped_output_map.get(op_input, op_input)
            if root in graph_inputs:
                consumes_graph_input = True

        if produces_graph_output and consumes_graph_input:
            # The input and output would be disconnected.
            return False

        input_data_is_known = all_tensors_are_static(*t_op.tmp_inputs)

        if produces_graph_output and input_data_is_known:
            # If the operator is skipped, the output tensor would be assigned static data, which is not allowed for
            #  model outputs.
            return False

        return True

    def turn_operator_to_identity(self, t_op: tflite_model.Operator):
        """Turn the operator 't_op' to a Transpose operator, which does nothing.
            't_op' MUST have exactly 1 input tensor.

        :param t_op: TFLite operator to turn into Transpose.
        """
        if len(t_op.tmp_inputs) != 1:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "turn_operator_to_identity(): Operator doesn't have 1 input!",
            )

        if len(t_op.tmp_outputs) != 1:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "turn_operator_to_identity(): Operator doesn't have 1 output!",
            )

        if t_op.tmp_inputs[0].rank <= 6:
            # Create regular `Transpose`.
            t_op.builtin_options = transpose_options.Transpose()
        else:
            # 6D and bigger require the Flex delegate `Transpose`.
            if t_op.tmp_inputs[0].quantization is not None:
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    "Conversion requires the addition of a `Transpose` operator with more than 6 dimensions, "
                    "which doesn't support quantization.",
                )

            if not self.conversion_config.allow_select_ops:
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    "Conversion requires the addition of a `Transpose` operator with more than 6 dimensions, "
                    "which requires the use of Flex delegate. "
                    + logger.Message.ALLOW_SELECT_OPS,
                )

            t_op.custom_options = FlexTranspose()

        rank = t_op.tmp_inputs[0].rank
        identity = np.asarray(range(rank), np.int32)
        identity_tensor = self.create_tensor_for_data(identity, "identity")
        t_op.tmp_inputs.append(identity_tensor)

    def _validate_new_tensor_name(self, name: str) -> str:
        """Take tensor name 'name' and make it unique in the model. Returns a unique tensor name."""

        # Try adding numbers to the 'name' until it is unique
        suffix = 0
        new_name = name
        while self.tensor_exists(new_name):
            new_name = name + str(suffix)
            suffix += 1

        return new_name

    def op_code_index_for_op_type(
        self, op_type: BuiltinOperator, version: int = 1, custom_code: str = None
    ):
        """
        Return the index to the 'operator_codes' vector in the TFLite model for the operator
        with given 'op_type' and 'version'. If corresponding OperatorCode doesn't exist, add
        it and create a new mapping.

        :param op_type: Operator type. One of BuiltinOperator enum.
        :param version: Operator version. Defaults to 1.
        :param custom_code: Custom code name. Must be used with 'op_type' equal to 'BuiltinOperator.CUSTOM'.
        :return: Index of the operator in 'operator_codes' vector.
        """

        version_name = version
        if custom_code is not None:
            version_name = f"{custom_code}_{version}"

        if op_type not in self.op_code_type_index_map.keys():
            self.op_code_type_index_map[op_type] = {}

        if version_name not in self.op_code_type_index_map[op_type].keys():
            self.op_code_type_index_map[op_type][
                version_name
            ] = self.operator_codes_size()
            self._build_operator_code(op_type, version, custom_code)

        return self.op_code_type_index_map[op_type][version_name]

    def tensor_exists(self, name: str):
        """Determine if a tensor with 'name' already exists or not."""
        return name in self._tensor_name_map.keys()

    def _remove_tensor_with_name_from_collection(self, name, collection):
        """Find and remove a tensor with given 'name' from given 'collection'."""
        to_remove = None

        for t in collection:
            if t.name == name:
                to_remove = t
                break

        if to_remove is not None:
            collection.remove(to_remove)

    def _tensors_similar(
        self, t_tensor1: tflite_model.Tensor, t_tensor2: tflite_model.Tensor
    ) -> bool:
        """Determine if the given TFLite tensors have the same shape and
        datatype."""

        if t_tensor1.type != t_tensor2.type:
            return False

        return translator.collections_equal(
            t_tensor1.shape.vector, t_tensor2.shape.vector
        )

    def tensor_for_name(self, name: str) -> tflite_model.Tensor:
        """
        Get an existing TFLite tensor with given 'name'. If such tensor does NOT exist, function will
        create and register a new tensor with shape '[]', which will be returned. If the tensor was
        redirected, destination tensor is returned instead.

        :param name: Name of the tensor.
        :return: Tensor instance.
        """
        if name not in self._tensor_name_map.keys():
            logger.d(f"Tensor '{name}' is not yet in the tensors. Adding it!")

            new_tensor = tflite_model.Tensor(tflite_model.Shape([]), name)
            new_tensor.tmp_buffer = self.build_empty_buffer()

            self.append_new_tensor(new_tensor)
        else:
            tensor = self._tensor_name_map[name]
            if new_tensor := self._skipped_output_map.get(tensor, None):
                # Tensor was redirected - return destination tensor
                if not self._tensors_similar(tensor, new_tensor):
                    logger.e(
                        logger.Code.INTERNAL_ERROR,
                        "Attempt to return non-matching tensor after redirect!",
                    )

                return new_tensor

        return self._tensor_name_map[name]

    def buffers_size(self):
        """Return the number of buffers that are currently in the model."""
        return self.get_buffers().len()

    def operator_codes_size(self):
        """Return the number of operator codes that are currently in the model."""
        return self.get_operator_codes().len()

    def _remove_input_with_name(self, name):
        """Find and remove a tensor in the sub_graph 'inputs' with given 'name'."""
        self._remove_tensor_with_name_from_collection(
            name, self.get_sub_graph().inputs.tmp_inputs
        )

    def _remove_output_with_name(self, name):
        """Find and remove a tensor in the sub_graph 'outputs' with given 'name'."""
        self._remove_tensor_with_name_from_collection(
            name, self.get_sub_graph().outputs.tmp_outputs
        )

    def _remove_tensor_with_name(self, name):
        """Find and remove a tensor in the graph with given 'name'."""
        self._remove_tensor_with_name_from_collection(name, self.get_tensors().vector)

    def append_new_tensor(self, t_tensor: tflite_model.Tensor, overwrite: bool = False):
        """Append the TFLite tensor 't_tensor' to the 'SubGraph.tensors' and register it."""
        self._tensor_name_map[t_tensor.name] = t_tensor
        self.get_tensors().append(t_tensor)

    def append_new_buffer(self, buffer: tflite_model.Buffer):
        """Append the 'buffer' to the 'model.buffers'."""
        self.get_buffers().append(buffer)

    def get_first_empty_buffer(self) -> tflite_model.Buffer:
        """Return the first empty buffer in the model. It should be the one on index 0."""
        for b in self.get_buffers().vector:
            if not _buffer_has_data(b):
                return b

        # No empty buffers in the model -> create one. This is uncommon, but can happen in weird models.
        return self.build_empty_buffer()

    def get_operator_with_output(
        self, t_tensor: tflite_model.Tensor
    ) -> Optional[tflite_model.Operator]:
        """Get the first operator from the graph, that has 't_tensor' in its 'tmp_outputs' list.
        If such operator doesn't exist, return None.
        """

        for op in self.get_operators().vector:
            if t_tensor in op.tmp_outputs:
                return op

        return None

    def _create_transpose_operator(
        self,
        input_tensor: tflite_model.Tensor,
        output_tensor: tflite_model.Tensor,
        permutation: list[int] | np.ndarray,
    ):
        """Create a `Transpose` operator with given input, output and permutation."""
        if isinstance(permutation, list):
            permutation = np.asarray(permutation, np.int32)
        elif isinstance(permutation, np.ndarray):
            logger.internal_assert(
                permutation.dtype == np.int32,
                "model_builder._create_transpose_operator(): "
                "permutation doesn't have type int32.",
            )
        else:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "model_builder._create_transpose_operator(): permutation is not "
                "a list or a numpy array.",
            )

        permutation_tensor = self.create_tensor_for_data(permutation, "perm")

        if input_tensor.rank <= 6:
            # Create regular `Transpose`.
            transpose = tflite_model.Operator(
                builtin_options=transpose_options.Transpose(),
                opcode_index=self.op_code_index_for_op_type(BuiltinOperator.TRANSPOSE),
            )
        else:
            # 7D and bigger require the Flex delegate `Transpose`.

            if input_tensor.quantization is not None:
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    "Conversion requires the addition of a `Transpose` operator with more than 6 dimensions, "
                    "which doesn't support quantization.",
                )

            if not self.conversion_config.allow_select_ops:
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    "Conversion requires the addition of a `Transpose` operator with more than 6 dimensions, "
                    "which requires the use of Flex delegate. "
                    + logger.Message.ALLOW_SELECT_OPS,
                )

            transpose = tflite_model.Operator(
                custom_options=FlexTranspose(),
                opcode_index=self.op_code_index_for_op_type(
                    FlexTranspose.operator_type, 1, FlexTranspose.custom_code
                ),
            )

        transpose.tmp_inputs = [input_tensor, permutation_tensor]
        transpose.tmp_outputs = [output_tensor]
        transpose.tmp_added_extra = True

        return transpose

    def create_transpose_operator_before(
        self,
        before_operator: tflite_model.Operator,
        on_input_index: int,
        permutation: list[int] | np.ndarray,
    ):
        """
            Create a TFLite 'Transpose' operator before the 'before_operator'.
            The input of 'before_operator' at index 'on_input_index', is where the Transpose operator will connect to
            the graph.

        :param before_operator: Create the Transpose operator in front of this operator.
        :param on_input_index: Attach the output of the Transpose op to the input of 'before_operator' on this index.
        :param permutation: The permutation that will be applied by the Transpose operator.
        """

        input_tensor = before_operator.tmp_inputs[on_input_index]
        output_tensor = self.duplicate_tensor(
            input_tensor, name_suffix="_transposed_", empty_buffer=True
        )
        permuted_shape = translator.apply_permutation_to(
            output_tensor.shape.vector, permutation
        )
        output_tensor.shape = tflite_model.Shape(permuted_shape)

        transpose = self._create_transpose_operator(
            input_tensor, output_tensor, permutation
        )

        before_operator.tmp_inputs[on_input_index] = output_tensor

        return transpose

    def create_transpose_operator_after(
        self,
        after_operator: tflite_model.Operator,
        on_output_index: int,
        permutation: list[int] | np.ndarray,
        keep_output_shape: bool = True,
    ):
        """
            Create a TFLite 'Transpose' operator after the 'after_operator'.
            The output of 'after_operator' at index 'on_output_index' is where the Transpose operator will be connected.

            The original output tensor of 'after_operator' will be used as the output of the Transpose operator.
            Meaning that operators which use that output of 'after_operator' will now use the output of the Transpose
            operator instead!

            If 'keep_output_shape' is True, the output of the Transpose operator will have the same shape as the
             original output of the 'after_operator', and the 'after_operator' will have its output shape changed to
             match the permutation.
            If 'keep_output_shape' is False, the output of the Transpose operator will have the new permuted shape, and
             the shape of the output of 'after_operator' will stay the same.

        :param after_operator: Create the Transpose operator right after this operator.
        :param on_output_index: Attach the input of the Transpose op to the output of 'after_operator' on this index.
        :param permutation: The permutation that will be applied by the Transpose operator.
        :param keep_output_shape: If True, the output of the Transpose will have the same shape as the original output
                                   of 'after_operator', and 'after_operator' will have its output modified to match.
                                  If False, the output of the Transpose operator will have the new permuted shape, and
                                  the output of 'after_operator' will remain unchanged.
        """

        # Input and output tensors of the Transpose operator
        output_tensor = after_operator.tmp_outputs[on_output_index]
        input_tensor = self.duplicate_tensor(
            output_tensor, output_tensor.name, empty_buffer=True
        )

        if keep_output_shape:
            # The output of Transpose keeps its shape. Input of Transpose must be changed
            inverse_permutation = translator.create_inverse_permutation(permutation)
            pre_permuted_shape = translator.apply_permutation_to(
                input_tensor.shape.vector, inverse_permutation
            )
            input_tensor.shape = tflite_model.Shape(pre_permuted_shape)

        else:
            # Set the shape of the Transpose output
            permuted_shape = translator.apply_permutation_to(
                output_tensor.shape.vector, permutation
            )
            output_tensor.shape = tflite_model.Shape(permuted_shape)

        transpose = self._create_transpose_operator(
            input_tensor, output_tensor, permutation
        )

        after_operator.tmp_outputs[on_output_index] = input_tensor

        return transpose

    def create_quantize_operator_before(
        self,
        before_operator: tflite_model.Operator,
        on_input_index: int,
        new_input_data_type: TensorType,
        new_input_scale: Optional[List[float]] = None,
        new_input_zero_point: Optional[List[int]] = None,
    ):
        """
            Create a TFLite 'Quantize' operator before the 'before_operator'.
            The input of 'before_operator' at index 'on_input_index', is where the Quantize operator will connect to the
            graph.
            The input of 'before_operator' will now have a new data type and quantization parameters.

        :param before_operator: Create the Quantize operator in front of this operator.
        :param on_input_index: Attach the output of the Quantize op to the input of 'before_operator' on this index.
        :param new_input_data_type: New input TFLite data type of the 'before_operator' operator.
        :param new_input_scale: New input scale of the 'before_operator' operator.
        :param new_input_zero_point: New input zero point of the 'before_operator' operator.
        """

        input_tensor = before_operator.tmp_inputs[on_input_index]
        output_tensor = self.duplicate_tensor(
            input_tensor, input_tensor.name, empty_buffer=True
        )

        quantized_dimension = input_tensor.quantization.quantized_dimension

        if new_input_scale is None:
            new_input_scale = input_tensor.quantization.scale.vector.copy()
        if new_input_zero_point is None:
            new_input_zero_point = input_tensor.quantization.zero_point.vector.copy()

        output_tensor.type = new_input_data_type
        output_tensor.quantization = tflite_model.Quantization(
            scale=tflite_model.Scale(new_input_scale),
            zero_point=tflite_model.ZeroPoint(new_input_zero_point),
            quantized_dimension=quantized_dimension,
        )
        quantize = tflite_model.Operator(
            builtin_options=quantize_options.Quantize(),
            opcode_index=self.op_code_index_for_op_type(BuiltinOperator.QUANTIZE),
        )
        quantize.tmp_inputs = [input_tensor]
        quantize.tmp_outputs = [output_tensor]
        quantize.tmp_added_extra = True

        before_operator.tmp_inputs[on_input_index] = output_tensor

        return quantize

    def create_quantize_operator_after(
        self,
        after_operator: tflite_model.Operator,
        on_output_index: int,
        new_output_data_type: TensorType,
        new_output_scale: Optional[List[float]] = None,
        new_output_zero_point: Optional[List[int]] = None,
    ) -> tflite_model.Operator:
        """
            Create a TFLite 'Quantize' operator after the 'after_operator'.
            The output of 'after_operator' at index 'on_output_index', is where the Quantize operator will connect to
            the graph.
            The output of 'after_operator' will now have a new data type and quantization parameters.

        :param after_operator: Create the Quantize operator behind this operator.
        :param on_output_index: Attach the input of the Quantize op to the output of 'before_operator' on this index.
        :param new_output_data_type: New output TFLite data type of the 'after_operator' operator.
        :param new_output_scale: New output scale of the 'after_operator' operator.
        :param new_output_zero_point: New output zero point of the 'after_operator' operator.
        """

        output_tensor = after_operator.tmp_outputs[on_output_index]
        input_tensor = self.duplicate_tensor(
            output_tensor, output_tensor.name, empty_buffer=True
        )

        quantized_dimension = output_tensor.quantization.quantized_dimension

        if new_output_scale is None:
            new_output_scale = input_tensor.quantization.scale.vector.copy()
        if new_output_zero_point is None:
            new_output_zero_point = input_tensor.quantization.zero_point.vector.copy()

        input_tensor.type = new_output_data_type
        input_tensor.quantization = tflite_model.Quantization(
            scale=tflite_model.Scale(new_output_scale),
            zero_point=tflite_model.ZeroPoint(new_output_zero_point),
            quantized_dimension=quantized_dimension,
        )

        quantize = tflite_model.Operator(
            builtin_options=quantize_options.Quantize(),
            opcode_index=self.op_code_index_for_op_type(BuiltinOperator.QUANTIZE),
        )
        quantize.tmp_inputs = [input_tensor]
        quantize.tmp_outputs = [output_tensor]
        quantize.tmp_added_extra = True

        after_operator.tmp_outputs[on_output_index] = input_tensor

        return quantize

    def create_dequantize_operator_after(
        self,
        after_operator: tflite_model.Operator,
        on_output_index: int,
        new_output_data_type: TensorType,
        new_output_scale: list[float],
        new_output_zero_point: list[int],
        quantized_dimension: int,
    ) -> tflite_model.Operator:
        """
            Create a TFLite 'Dequantize' operator after the 'after_operator'.
            The output of 'after_operator' at index 'on_output_index', is where the Quantize operator will connect to
            the graph.
            The output of 'after_operator' will now have a new quantized data type.
            This method was designed for the use case, where 'after_operator' has a FLOAT32 output tensor, but the
             operator will produce a quantized output. The following operators however expect the float data.
             This is in line with other similar methods.

        :param after_operator: Create the Dequantize operator behind this operator.
        :param on_output_index: Attach the input of the Dequantize op to the output of 'before_operator' on this index.
        :param new_output_data_type: New output TFLite data type of the 'after_operator' operator.
        :param new_output_scale: New output scale of the 'after_operator' operator.
        :param new_output_zero_point: New output zero point of the 'after_operator' operator.
        :param quantized_dimension: The quantized dimension parameter of the new output tensor of 'after_operator'.
        :return: The Dequantize operator.
        """

        output_tensor = after_operator.tmp_outputs[on_output_index]
        input_tensor = self.duplicate_tensor(
            output_tensor, output_tensor.name, empty_buffer=True
        )

        input_tensor.type = new_output_data_type
        input_tensor.quantization = tflite_model.Quantization(
            scale=tflite_model.Scale(new_output_scale),
            zero_point=tflite_model.ZeroPoint(new_output_zero_point),
            quantized_dimension=quantized_dimension,
        )

        dequantize = tflite_model.Operator(
            builtin_options=dequantize_options.Dequantize(),
            opcode_index=self.op_code_index_for_op_type(BuiltinOperator.DEQUANTIZE),
        )
        dequantize.tmp_inputs = [input_tensor]
        dequantize.tmp_outputs = [output_tensor]
        dequantize.tmp_added_extra = True

        after_operator.tmp_outputs[on_output_index] = input_tensor

        return dequantize

    def create_reshape_before(
        self,
        before_op: tflite_model.Operator,
        on_input_index: int,
        new_shape: List[int],
    ) -> tflite_model.Operator:
        """
        Create a TFLite 'Reshape' operator before the 'before_op' operator. The input of 'before_op' on index
        'on_input_index' is where the 'Reshape' operator will connect. With this function it is expected
        to change input shape of 'before_op' operator on index 'on_input_index'.

        :param before_op: TFLite operator that will consume the output of the new 'Reshape' operator.
        :param on_input_index: Index of an input tensor of the 'before_op' operator, which will serve as the new input
                                for the 'Reshape' operator.
        :param new_shape: Shape of the new tensor that will serve as an output of 'Reshape' operator.
        :return: The TFLite 'Reshape' operator.
        """

        input_tensor = before_op.tmp_inputs[on_input_index]

        reshape_output = self.duplicate_tensor(
            input_tensor, input_tensor.name + "_reshaped", empty_buffer=True
        )
        reshape_output.shape = tflite_model.Shape(new_shape)

        reshape_op = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(new_shape)
        )

        reshape_op.tmp_inputs = [input_tensor]
        reshape_op.tmp_outputs = [reshape_output]

        before_op.tmp_inputs[on_input_index] = reshape_output

        return reshape_op

    def create_reshape_after(
        self,
        after_op: tflite_model.Operator,
        on_output_index: int,
        new_shape: List[int],
    ) -> tflite_model.Operator:
        """
        Create a TFLite 'Reshape' operator after the 'after_op' operator. The output of 'after_op' on index
        'on_output_index' is where the 'Reshape' operator will connect. This function will preserve output
        shape of 'after_op' operator on index 'on_output_index'.

        :param after_op: TFLite operator that will produce the input of the new 'Reshape' operator.
        :param on_output_index: Index of an output tensor of the 'after_op' operator, which will serve as the new input
                                for the 'Reshape' operator.
        :param new_shape: Shape of the new tensor that will serve as an output of 'Reshape' operator.
        :return: The TFLite 'Reshape' operator.
        """

        output_tensor = after_op.tmp_outputs[on_output_index]

        reshape_input = self.duplicate_tensor(
            output_tensor, output_tensor.name + "_reshaped", empty_buffer=True
        )
        output_tensor.shape = tflite_model.Shape(new_shape)

        reshape_op = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(new_shape)
        )

        reshape_op.tmp_inputs = [reshape_input]
        reshape_op.tmp_outputs = [output_tensor]
        reshape_op.tmp_added_extra = True

        after_op.tmp_outputs[on_output_index] = reshape_input

        return reshape_op

    def create_cast_before(
        self,
        before_op: tflite_model.Operator,
        on_input_index: int,
        new_type: TensorType,
    ) -> tflite_model.Operator:
        """
        Create a TFLite 'Cast' operator before the 'before_op' operator. The input of 'before_op' on index
        'on_input_index' is where the 'Cast' operator will connect.

        :param before_op: TFLite operator that will consume the output of the new 'Cast' operator.
        :param on_input_index: Index of an input tensor of the 'before_op' operator, which will serve as the new input
                                for the 'Cast' operator.
        :param new_type: Type of output tensor of 'Cast' operator.
        :return: The TFLite 'Cast' operator.
        """

        input_tensor = before_op.tmp_inputs[on_input_index]

        cast_output = self.duplicate_tensor(
            input_tensor, input_tensor.name + "_casted", empty_buffer=True
        )
        cast_output.type = new_type

        cast_op = tflite_model.Operator(
            builtin_options=cast_options.Cast(input_tensor.type, new_type)
        )
        cast_op.tmp_inputs = [input_tensor]
        cast_op.tmp_outputs = [cast_output]
        cast_op.tmp_added_extra = True

        before_op.tmp_inputs[on_input_index] = cast_output

        return cast_op

    def create_cast_after(
        self,
        after_op: tflite_model.Operator,
        on_output_index: int,
        new_type: TensorType,
    ) -> tflite_model.Operator:
        """
        Create a TFLite 'Cast' operator after the 'after_op' operator. The output of 'after_op' on index
        'on_output_index' is where the 'Cast' operator will connect. This function will change output
        type of 'after_op' operator on index 'on_output_index' to 'new_type'.

        :param after_op: TFLite operator that will produce the input of the new 'Cast' operator.
        :param on_output_index: Index of an output tensor of the 'after_op' operator, which will serve as the new input
                                for the 'Cast' operator.
        :param new_type: Type of the new tensor that will serve as an input of 'Cast' operator.
        :return: The TFLite 'Cast' operator.
        """

        output_tensor = after_op.tmp_outputs[on_output_index]

        cast_input = self.duplicate_tensor(
            output_tensor, output_tensor.name + "_casted", empty_buffer=True
        )
        cast_input.type = new_type

        cast_builtin_options = cast_options.Cast(
            in_data_type=new_type, out_data_type=output_tensor.type
        )
        cast_op = tflite_model.Operator(builtin_options=cast_builtin_options)

        cast_op.tmp_inputs = [cast_input]
        cast_op.tmp_outputs = [output_tensor]
        cast_op.tmp_added_extra = True

        after_op.tmp_outputs[on_output_index] = cast_input

        return cast_op

    def create_slice_after(
        self,
        after_op: tflite_model.Operator,
        on_output_index: int,
        begin: list[int],
        size: list[int],
    ):
        """
        Create a TFLite 'Slice' operator after the 'after_op' operator. The output of 'after_op' on index
        'on_output_index' is where the 'Slice' operator will connect. This function will preserve output
        shape of 'after_op' operator on index 'on_output_index'.

        :param after_op: TFLite operator that will produce the input of the new 'Slice' operator.
        :param on_output_index: Index of an output tensor of the 'after_op' operator, which will serve as the new input
                                for the 'Slice' operator.
        :param begin: List of indices where slicing begins. Must have same length as sliced tensor.
        :param size: List of sliced sizes. Defines how many items is sliced per dimension. Must
                        have same length as sliced tensor.
        :return: The TFLite 'Slice' operator.
        """

        output_tensor = after_op.tmp_outputs[on_output_index]

        logger.internal_assert(
            len(begin) == len(size),
            "create_slice_after(): Rank of 'begin' tensor and 'size' tensor don't match.",
        )
        logger.internal_assert(
            len(begin) == len(output_tensor.shape.vector),
            "create_slice_after(): Rank of 'begin' tensor and sliced tensor don't match.",
        )

        slice_input = self.duplicate_tensor(
            output_tensor, output_tensor.name + "_sliced", empty_buffer=True
        )
        output_tensor.shape = tflite_model.Shape(size)

        begin_tensor = self.create_tensor_for_data(np.asarray(begin, np.int32), "begin")
        size_tensor = self.create_tensor_for_data(np.asarray(size, np.int32), "size")

        slice_op = tflite_model.Operator(builtin_options=slice_options.Slice())
        slice_op.tmp_inputs = [slice_input, begin_tensor, size_tensor]
        slice_op.tmp_outputs = [output_tensor]
        slice_op.tmp_added_extra = True

        after_op.tmp_outputs[on_output_index] = slice_input

        return slice_op

    def create_gather_before(
        self,
        before_op: tflite_model.Operator,
        on_input_index: int,
        indices: list[int],
        output_shape: list[int],
        axis: int = 0,
    ) -> tflite_model.Operator:
        """
        Create a TFLite 'Gather' operator before the 'before_op' operator. The input of 'before_op' on index
        'on_input_index' is where the 'Gather' operator will connect.

        :param before_op: TFLite operator that will consume the output of the new 'Gather' operator.
        :param on_input_index: Index of an input tensor of the 'before_op' operator, which will serve as the new output
                                for the 'Gather' operator.
        :param indices: The `indices` operand of the TFLite 'Gather' operator.
        :param output_shape: The shape of the output of the 'Gather' operator.
        :param axis: The `axis` attribute of the TFLite 'Gather' operator.
        :return: The TFLite 'Gather' operator.
        """

        input_tensor = before_op.tmp_inputs[on_input_index]

        gather_output = self.duplicate_tensor(input_tensor, empty_buffer=True)
        gather_output.shape = tflite_model.Shape(output_shape)

        indices_tensor = self.create_tensor_for_data(
            np.array(indices, np.int32), "indices"
        )

        gather_op = tflite_model.Operator(builtin_options=gather_options.Gather(axis))

        gather_op.tmp_inputs = [input_tensor, indices_tensor]
        gather_op.tmp_outputs = [gather_output]
        gather_op.tmp_added_extra = True

        before_op.tmp_inputs[on_input_index] = gather_output

        return gather_op

    def ensure_correct_broadcasting(
        self, t_op: tflite_model.Operator, main_output: tflite_model.Tensor
    ) -> List[tflite_model.Operator]:
        """Make sure that all input tensors of 't_op' can have their shape broadcasted correctly.
             Static input tensors will be altered statically and for dynamic tensors, Reshape and Transpose operators
             will be added to ensure a valid shape.
            Note: The TFLite 't_op' operator still has to support shape broadcasting! This function just makes sure, the
             shapes are broadcastable correctly. it doesn't eliminate the need for broadcasting.

        :param t_op: TFLite operator with input tensors that need to be made broadcastable.
        :param main_output: The TFLite tensor, that is the main output of the operation carried out by 't_op'.
        :return: A list of TFLite operators Reshape and Transpose, that need to be added to the model before 't_op'.
        """

        if main_output not in t_op.tmp_outputs:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "ModelBuilder.ensure_correct_broadcasting(): 'main_output' is not among the outputs of 't_op'!",
            )

        if not uses_shape_broadcasting(t_op):
            # Operator doesn't use shape broadcasting
            return []

        if not main_output.tensor_format.is_channels_last() and not any(
            input_tensor.tensor_format.is_channels_last()
            for input_tensor in t_op.tmp_inputs
        ):
            # Operator uses only formatless tensors
            return []

        # -- Operator uses channels last tensors and shape broadcasting --

        ops_to_add = []
        new_tmp_inputs = []
        output_shape = main_output.shape
        output_rank = output_shape.len()

        for input_tensor in t_op.tmp_inputs:

            if input_tensor.shape != main_output.shape:
                if tensor_has_data(input_tensor):
                    # Replace the static input with one with a corrected shape.
                    x = self.prepare_static_tensor_for_correct_broadcasting_with_channels_first_tensors(
                        input_tensor, output_rank
                    )
                    new_tmp_inputs.append(x)
                else:
                    # Prepend Reshape and Transpose
                    ops = self.prepare_dynamic_tensor_for_correct_broadcasting_with_channels_first_tensors(
                        input_tensor, output_rank
                    )

                    if len(ops) != 0:
                        # The output of the 'Transpose' (last returned op) will be the new input of the operator
                        new_tmp_inputs.append(ops[-1].tmp_outputs[0])
                    else:
                        new_tmp_inputs.append(input_tensor)

                    ops_to_add.extend(ops)

            else:
                # Keep the original input as is
                new_tmp_inputs.append(input_tensor)

        t_op.tmp_inputs = new_tmp_inputs

        return ops_to_add

    def prepare_dynamic_tensor_for_correct_broadcasting_with_channels_first_tensors(
        self, tensor: tflite_model.Tensor, output_rank: int
    ) -> List[tflite_model.Operator]:
        """Create Reshape and Transpose operators, to make sure the shape of the dynamic 'tensor' can be correctly
             broadcasted with other TFLite channels last tensors.
            The assumption is that the 'tensor' needs to be broadcasted with channels last tensors with a greater or
             equal rank. And due to its smaller rank, the shapes will not line up.
            The output tensor of the last returned operator is new, and must be set as a new input of the original
             operator.

        :param tensor: Dynamic TFLite tensor, that needs to be broadcastable with channels last tensors, but the shape
                        doesn't line up, due to prior (possibly incorrect) conversion.
        :param output_rank: The rank of the output tensor of the operator.
        :return: A list of Reshape and Transpose operators, which need to be added to the model before 't_op'.
        """
        input_rank = tensor.shape.len()
        rank_diff = output_rank - input_rank

        if rank_diff < 0:
            logger.e(
                logger.Code.INTERNAL_ERROR, "'tensor' rank must be <= output_rank!"
            )

        if rank_diff == 0:
            # The tensor is already broadcastable
            return []

        ops_to_add = []

        # -- Add a Reshape operator to extend the rank --

        extended_shape = [1] * rank_diff + tensor.shape.vector
        transpose_input = self.duplicate_tensor(tensor)
        transpose_input.shape = tflite_model.Shape(extended_shape)

        reshape = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(extended_shape)
        )
        reshape.tmp_inputs = [tensor]
        reshape.tmp_outputs = [transpose_input]

        ops_to_add.append(reshape)

        # Add Transpose operator
        if tensor.tensor_format.is_channels_last():
            # The 'tensor' was incorrectly converted from channels first before. Revert it and then convert properly.

            revert_perm = translator.create_channels_last_to_channels_first_permutation(
                input_rank
            )

            # The indices refer to dimensions according to the rank of the input. But the Reshape may have increased the
            #  rank by prepending 1s. Therefore, we need to increment these indices according to the rank difference, to
            #  still refer to the same dimensions from the right.
            revert_perm += rank_diff

            # Prepend a partial identity, to keep leading dimensions unchanged.
            revert_perm = list(range(rank_diff)) + list(revert_perm)

            # Now add a permutation to convert the extended ExecuTorch shape to a TFLite shape
            to_tflite_perm = (
                translator.create_channels_first_to_channels_last_permutation(
                    output_rank
                )
            )

            perm = translator.combine_permutations(revert_perm, to_tflite_perm)

        else:
            # The 'tensor' was NOT incorrectly converted earlier. Just convert the extended shape to TFLite.
            perm = translator.create_channels_first_to_channels_last_permutation(
                output_rank
            )

        transpose_output = self.duplicate_tensor(transpose_input)
        transpose_output.shape = tflite_model.Shape(
            translator.apply_permutation_to(transpose_output.shape.vector, perm)
        )
        transpose_output.tensor_format = TensorFormat.CHANNELS_LAST

        transpose = self._create_transpose_operator(
            transpose_input, transpose_output, perm
        )
        ops_to_add.append(transpose)

        return ops_to_add

    def prepare_static_tensor_for_correct_broadcasting_with_channels_first_tensors(
        self, tensor: tflite_model.Tensor, output_rank: int
    ) -> tflite_model.Tensor:
        """Create a TFLite tensor based on the static 'tensor', so that it can be correctly broadcasted with channels
             last tensors, and return it.
            The assumption is that the 'tensor' needs to be broadcasted with channels last tensors with a greater or
             equal rank. And due to its smaller rank, the shapes will not line up.

        :param tensor: Static TFLite tensor, that needs to be broadcastable with channels last tensors, but the shape
                        doesn't line up, due to prior incorrect conversion.
        :param output_rank: The rank of the output tensor of the operator.
        :return: A new static tensor, with a corrected shape for TFLite broadcasting.
        """
        if not tensor_has_data(tensor):
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "ModelBuilder._reshape_static_tensor_to_be_broadcastable(): 'tensor' is not static!",
            )

        tensor = self.duplicate_tensor(
            tensor
        )  # Work with a clean copy, in case the tensor is also used elsewhere.
        data = tensor.tmp_buffer.data
        shape = tensor.shape.vector

        rank_diff = output_rank - len(shape)
        if rank_diff < 0:
            logger.e(
                logger.Code.INTERNAL_ERROR, "'tensor' rank must be <= output_rank!"
            )

        if tensor.tensor_format.is_channels_last():
            # The tensor was incorrectly converted to channels last. Extend it with 1s and convert properly.

            original_shape = translator.dims_to_channels_first(
                shape
            )  # Same shape as in the ExecuTorch model

            # Prepend 1s to the shape
            extended_executorch_shape = [1] * rank_diff + original_shape

            # Convert the full shape to TFLite format
            tflite_shape = translator.dims_to_channels_last(extended_executorch_shape)
            tensor.shape = tflite_model.Shape(tflite_shape)

            # Statically transpose the data
            data = translator.convert_data_to_channels_first(
                data
            )  # To the same shape as in the ExecuTorch model
            data = data.reshape(extended_executorch_shape)  # Extend with leading 1s
            tensor.tmp_buffer.data = translator.convert_data_to_channels_last(
                data
            )  # Convert to TFLite format

            assert tflite_shape == list(tensor.tmp_buffer.data.shape)

        else:
            # The tensor is the same as in the ExecuTorch model.

            extended_executorch_shape = [1] * rank_diff + shape

            # Convert the full shape to TFLite format
            tflite_shape = translator.dims_to_channels_last(extended_executorch_shape)
            tensor.shape = tflite_model.Shape(tflite_shape)

            # Statically transpose the data
            data = data.reshape(extended_executorch_shape)  # Extend with leading 1s
            tensor.tmp_buffer.data = translator.convert_data_to_channels_last(
                data
            )  # Convert to TFLite format

            assert tflite_shape == list(tensor.tmp_buffer.data.shape)

        return tensor

    def operator_produces_graph_output(self, t_op: tflite_model.Operator) -> bool:
        """Determine whether any output tensor of the operator 't_op' is also an output of the entire graph.

        :param t_op: TFLite operator to check,
        :return: True, if at least 1 output of 't_op' is also an output of the graph.
        """
        graph_outputs = self.get_sub_graph().outputs.tmp_outputs
        return any(output_tensor in graph_outputs for output_tensor in t_op.tmp_outputs)

    """ ---------------- Functions to get an element of the TFLite model. ----------------
    If the element doesn't exist, it is created. So functions always return a valid object. """

    def get_sub_graphs(self) -> tflite_model.SubGraphs:
        if self._tfl_model.sub_graphs is None:
            self._tfl_model.sub_graphs = tflite_model.SubGraphs()

        return self._tfl_model.sub_graphs

    def get_sub_graph(self) -> tflite_model.SubGraph:
        sub_graphs = self.get_sub_graphs()
        if sub_graphs.len() == 0:
            sub_graphs.append(tflite_model.SubGraph())

        return sub_graphs.get(0)

    def get_tensors(self) -> tflite_model.Tensors:
        sub_graph = self.get_sub_graph()
        if sub_graph.tensors is None:
            sub_graph.tensors = tflite_model.Tensors()

        return sub_graph.tensors

    def get_buffers(self) -> tflite_model.Buffers:
        if self._tfl_model.buffers is None:
            self._tfl_model.buffers = tflite_model.Buffers()

        return self._tfl_model.buffers

    def get_operators(self) -> tflite_model.Operators:
        sub_graph = self.get_sub_graph()
        if sub_graph.operators is None:
            sub_graph.operators = tflite_model.Operators()

        return sub_graph.operators

    def get_operator_codes(self) -> tflite_model.OperatorCodes:
        if self._tfl_model.operator_codes is None:
            self._tfl_model.operator_codes = tflite_model.OperatorCodes()

        return self._tfl_model.operator_codes
