# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import cast, Iterator, Tuple

import executorch.backends.nxp.backend.ir.converter.builder.model_builder as model_builder
from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.graph_utils import (
    builtin_operator_for_op_type,
    create_tensor_to_operator_dictionaries,
    InputTensorToOpsMap,
    NameToTensorMap,
    operator_is_type,
    OutputTensorToOpMap,
)
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.operator_rules import (
    OpRule,
)
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.tensor_rules import (
    TensorRule,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model


class OperatorBlock(ABC):
    @abstractmethod
    def validate(self):
        pass


@dataclass
class OpLikeBlock(OperatorBlock):
    ops: list[str] | None = None
    inputs: list[str | None] | None = None
    outputs: list[str | None] | None = None
    op_rules: list[OpRule] | None = None

    def validate(self):
        """Check if the `Op` follows the limitations of the PatternMatcher.
        If it doesn't exit with error and a corresponding message.
        """

        # `...` can only be used at the start or end of the inputs/outputs.
        if len(self.inputs_as_list()) > 2:
            logger.internal_assert(
                ... not in self.inputs_as_list()[1:-1],
                "PatternMatcher: The `...` can only be used "
                "at the start and/or end of the inputs.",
            )
        if len(self.outputs_as_list()) > 2:
            logger.internal_assert(
                ... not in self.outputs_as_list()[1:-1],
                "PatternMatcher: The `...` can only be used"
                " at the start and/or end of the outputs.",
            )

    def inputs_as_list(self) -> list[str | None]:
        """Return the `inputs` attribute. If it's `None`, return `[]`."""
        if self.inputs is None:
            return []
        return self.inputs

    def outputs_as_list(self) -> list[str | None]:
        """Return the `outputs` attribute. If it's `None`, return `[]`."""
        if self.outputs is None:
            return []
        return self.outputs

    def io_as_list(self) -> list[str | None]:
        """Return the `inputs` and `outputs` attributes combined into 1. If they are `None`, return `[]`."""
        return self.inputs_as_list() + self.outputs_as_list()


@dataclass
class Op(OpLikeBlock):
    """Class represents 1 operator."""

    def match(
        self,
        real_op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        """Try to match the `Op` with a real TFLite Operator. If the match is successful, add new mappings for matched
         tensors into the tensor_map`.
        :return: True, if the `Op` was successfully matched. Otherwise, return False.
        """

        # noinspection PyBroadException
        try:
            if not self._op_type_matches(real_op, builder):
                return False

            tensor_map_copy = tensor_map.copy()  # Use a copy in case the match fails.

            if not self._match_inputs(real_op, tensor_map_copy):
                return False

            if not self._match_outputs(real_op, tensor_map_copy):
                return False

            if not self._op_rules_satisfied(
                real_op, tensor_map_copy, input_to_ops_map, output_to_op_map, builder
            ):
                return False

            # Operator matched.
            tensor_map.update(tensor_map_copy)
            return True

        except Exception:
            # Unexpected failure.
            return False

    def _op_type_matches(
        self, real_op: tflite_model.Operator, builder: "model_builder.ModelBuilder"
    ) -> bool:
        """Check if the type of the TFLite operator `real_op` matches the types defined in this `Op`."""
        if self.ops is None:
            return True

        return any(operator_is_type(real_op, op_type, builder) for op_type in self.ops)

    def _op_rules_satisfied(
        self,
        real_op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        """Check if all operator rules defined for this `Op` are satisfied."""
        if self.op_rules is None:
            return True

        return all(
            rule(real_op, tensor_map, input_to_ops_map, output_to_op_map, builder)
            for rule in self.op_rules
        )

    def _match_inputs(  # noqa C901
        self, real_op: tflite_model.Operator, tensor_map: NameToTensorMap
    ) -> bool:
        """Check if it is possible to match the input tensors of the TFLite operator `real_op` with the ones
         defined for this `Op`.
        New mappings may be added into the `tensor_map`.
        """
        if self.inputs is None:
            return True

        num_real_inputs = len(real_op.tmp_inputs)

        step = 1
        real_input_index = 0
        inputs = self.inputs
        if inputs[0] is ... and inputs[-1] is not ...:
            # The `...` is used only at the start. In this case, iterate through the inputs from the end.
            step = -1
            real_input_index = num_real_inputs - 1
            inputs = reversed(inputs)

        def _checked_all_inputs(real_input_idx: int) -> bool:
            if step == 1:
                return real_input_idx >= num_real_inputs
            elif step == -1:
                return real_input_idx < 0
            else:
                raise ValueError

        can_skip = False
        for inpt in inputs:
            if _checked_all_inputs(real_input_index) and (inpt is not ...):
                return False  # The inputs don't match

            if inpt is ...:
                can_skip = True
                continue

            elif inpt is None:
                # The tensor is not named, but must be there.
                real_input_index += step
            else:
                # A tensor name is specified.
                real_in = real_op.tmp_inputs[real_input_index]
                if inpt in tensor_map.keys():
                    # Tensor has already been mapped.
                    logger.internal_assert(
                        type(tensor_map[inpt]) is tflite_model.Tensor,
                        f"PatternMatcher: consuming a set of tensors `{inpt}` is not supported right now.",
                    )
                    if tensor_map[inpt] != real_in:
                        # The tensor doesn't match the mapped one.
                        if can_skip:
                            real_input_index += step
                            continue

                        return False

                    else:
                        # The tensor has been mapped and matches.
                        real_input_index += step
                        continue

                # Map the matched tensor.
                can_skip = (
                    False  # Matched a tensor, so the `...` does not apply anymore.
                )
                tensor_map[inpt] = real_in
                real_input_index += step

        return True

    def _match_outputs(  # noqa C901
        self, real_op: tflite_model.Operator, tensor_map: NameToTensorMap
    ) -> bool:
        """Check if it is possible to match the output tensors of the TFLite operator `real_op` with the ones
         defined for this `Op`.
        New mappings may be added into the `tensor_map`.
        """
        if self.outputs is None:
            return True

        num_real_outputs = len(real_op.tmp_outputs)
        step = 1
        real_output_index = 0
        outputs = self.outputs
        if outputs[0] is ... and outputs[-1] is not ...:
            # The `...` is used only at the start. In this case, iterate through the outputs from the end.
            step = -1
            real_output_index = num_real_outputs - 1
            outputs = reversed(outputs)

        def _checked_all_outputs(real_output_idx: int) -> bool:
            if step == 1:
                return real_output_idx >= num_real_outputs
            elif step == -1:
                return real_output_idx < 0
            else:
                raise ValueError

        can_skip = False
        for out in outputs:
            if _checked_all_outputs(real_output_index) and (out is not ...):
                return False  # The outputs don't match

            if out is ...:
                can_skip = True
                continue

            elif out is None:
                # The tensor is not named, but must be there.
                real_output_index += step
            else:
                # A tensor name is specified.
                real_out = real_op.tmp_outputs[real_output_index]
                if out in tensor_map.keys():
                    # Tensor has already been mapped.
                    if tensor_map[out] != real_out:
                        # The tensor doesn't match.
                        if can_skip:
                            real_output_index += step
                            continue

                        return False
                    else:
                        # The tensor has been mapped and matches.
                        real_output_index += step
                        continue

                # Map the matched tensor.
                can_skip = (
                    False  # Matched a tensor, so the `...` does not apply anymore.
                )
                tensor_map[out] = real_out
                real_output_index += step

        return True


@dataclass
class MultipleSameOps(OpLikeBlock):
    """Class represents multiple occurrences of similar operators with the same op type, inputs and outputs."""

    def match(
        self,
        real_ops: list[tflite_model.Operator],
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        """Try to match the `MultipleSameOps` with real TFLite operators. If the match is successful, add new mappings
         for matched tensors into the tensor_map`.
        :return: True, if the `MultipleSameOps` was successfully matched. Otherwise, return False.
        """
        # noinspection PyBroadException
        try:
            if len(real_ops) == 0:
                return False

            if not self._op_types_match(real_ops, builder):
                return False

            tensor_map_copy = tensor_map.copy()  # Use a copy in case the match fails.

            if not self._match_inputs(real_ops, tensor_map_copy):
                return False

            if not self._match_outputs(real_ops, tensor_map_copy):
                return False

            if not self._op_rules_satisfied(
                real_ops, tensor_map_copy, input_to_ops_map, output_to_op_map, builder
            ):
                return False

            # Operator matched.
            tensor_map.update(tensor_map_copy)
            return True

        except Exception:
            # Unexpected failure.
            return False

    def validate(self):
        super().validate()
        logger.internal_assert(
            self.ops is not None,
            "PatternMatcher: `MultipleSameOps` doesn't support `ops=None` yet.",
        )

    def _op_types_match(
        self,
        real_ops: list[tflite_model.Operator],
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        """Check if the types of the TFLite operators `real_ops` match the types defined in this `MultipleSameOps`."""
        for real_op in real_ops:
            if not any(
                operator_is_type(real_op, op_type, builder) for op_type in self.ops
            ):
                return False

        return True

    def _op_rules_satisfied(
        self,
        real_ops: list[tflite_model.Operator],
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        """Check if all operator rules defined for this `MultipleSameOps` are satisfied for all operators."""
        if self.op_rules is None:
            return True

        for real_op in real_ops:
            if not all(
                rule(real_op, tensor_map, input_to_ops_map, output_to_op_map, builder)
                for rule in self.op_rules
            ):
                return False

        return True

    def _match_inputs(  # noqa C901
        self, real_ops: list[tflite_model.Operator], tensor_map: NameToTensorMap
    ) -> bool:
        """Check if it is possible to match the input tensors of the TFLite operators `real_ops` with the ones
         defined for this `MultipleSameOps`.
        New mappings may be added into the `tensor_map`.
        """
        if self.inputs is None:
            return True

        set_of_tensors_map = defaultdict(lambda: [])

        for real_op in real_ops:
            num_real_inputs = len(real_op.tmp_inputs)

            step = 1
            real_input_index = 0
            inputs = self.inputs
            if inputs[0] is ... and inputs[-1] is not ...:
                # The `...` is used only at the start. In this case, iterate through the inputs from the end.
                step = -1
                real_input_index = num_real_inputs - 1
                inputs = reversed(inputs)

            def _checked_all_inputs(real_input_idx: int) -> bool:
                if step == 1:  # noqa B036
                    return real_input_idx >= num_real_inputs  # noqa B036
                elif step == -1:  # noqa B036
                    return real_input_idx < 0
                else:
                    raise ValueError

            can_skip = False
            for inpt in inputs:
                if _checked_all_inputs(real_input_index) and (inpt is not ...):
                    return False  # The inputs don't match

                if inpt is ...:
                    can_skip = True
                    continue

                elif inpt is None:
                    # The tensor is not named, but must be there.
                    real_input_index += step
                else:
                    # A tensor name is specified.
                    real_in = real_op.tmp_inputs[real_input_index]
                    if inpt in tensor_map.keys():
                        # Tensor has already been mapped.
                        logger.internal_assert(
                            type(tensor_map[inpt]) is tflite_model.Tensor,
                            f"PatternMatcher: consuming a set of tensors `{inpt}` is not supported right now.",
                        )
                        if tensor_map[inpt] != real_in:
                            # The tensor doesn't match the mapped one.
                            if can_skip:
                                real_input_index += step
                                continue

                            return False

                        else:
                            # The tensor has been mapped and matches.
                            real_input_index += step
                            continue

                    # Map the matched tensor.
                    can_skip = (
                        False  # Matched a tensor, so the `...` does not apply anymore.
                    )
                    set_of_tensors_map[inpt].append(real_in)

                    real_input_index += step

        # The `MultipleSameOps` were matched with `real_ops`. Add the new tensor mappings to the `tensor_map`.
        tensor_map.update(set_of_tensors_map)

        return True

    def _match_outputs(
        self, real_ops: list[tflite_model.Operator], tensor_map: NameToTensorMap
    ) -> bool:
        """Check if it is possible to match the output tensors of the TFLite operators `real_ops` with the ones
         defined for this `MultipleSameOps`.
        New mappings may be added into the `tensor_map`.
        """
        if self.outputs is None:
            return True

        set_of_tensors_map = defaultdict(lambda: [])

        for real_op in real_ops:
            num_real_outputs = len(real_op.tmp_outputs)

            step = 1
            real_output_index = 0
            outputs = self.outputs
            if outputs[0] is ... and outputs[-1] is not ...:
                # The `...` is used only at the start. In this case, iterate through the outputs from the end.
                step = -1
                real_output_index = num_real_outputs - 1
                outputs = reversed(outputs)

            def _checked_all_outputs(real_output_idx: int) -> bool:
                if step == 1:  # noqa B036
                    return real_output_idx >= num_real_outputs  # noqa B036
                elif step == -1:  # noqa B036
                    return real_output_idx < 0
                else:
                    raise ValueError

            for output in outputs:
                if _checked_all_outputs(real_output_index) and (output is not ...):
                    return False  # The outputs don't match

                if output is ...:
                    continue

                elif output is None:
                    # The tensor is not named, but must be there.
                    real_output_index += step
                else:
                    # A tensor name is specified.
                    real_out = real_op.tmp_outputs[real_output_index]
                    if output in tensor_map.keys():
                        # Tensor has already been mapped. This isn't supported right now.
                        logger.e(
                            logger.Code.INTERNAL_ERROR,
                            "PatternMatcher: MultipleSameOps is producing an already "
                            f"defined tensor `{output}`, which is not yet supported.",
                        )

                    # Map the matched tensor.
                    set_of_tensors_map[output].append(real_out)

                    real_output_index += step

        # The `MultipleSameOps` were matched with `real_ops`. Add the new tensor mappings to the `tensor_map`.
        tensor_map.update(set_of_tensors_map)

        return True


@dataclass()
class OneOf(OperatorBlock):
    """Class represents 1 operator, which matches at least 1 of the specified `Op` objects."""

    # For now, limited to `Op` objects.
    one_of_ops: list[Op]

    def validate(self):
        for op in self.one_of_ops:
            op.validate()


# noinspection PyMethodMayBeStatic
class PatternMatcher:
    builder: "model_builder.ModelBuilder"
    pattern: list[OperatorBlock]
    tensor_rules: list[TensorRule] | None

    def __init__(
        self,
        builder: "model_builder.ModelBuilder",
        pattern: list[OperatorBlock],
        tensor_rules: list[TensorRule] | None = None,
    ):
        self.builder = builder
        self.pattern = pattern
        self.tensor_rules = tensor_rules

        self._validate_pattern()

    def _tensor_rules_satisfied(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
    ) -> bool:
        """Check if all currently applicable tensor rules are satisfied."""
        if self.tensor_rules is None:
            return True

        for rule in self.tensor_rules:
            if rule.is_applicable(tensor_map) and not rule(
                tensor_map, input_to_ops_map, output_to_op_map, self.builder
            ):
                return False  # Rule is not satisfied.

        return True

    def _get_opcode_indices_for(self, op_type: str) -> int | None:
        builtin_op = builtin_operator_for_op_type(op_type)
        return self.builder.op_code_type_index_map.get(builtin_op, None)

    def _validate_pattern(self):
        """Make sure the `pattern` is valid according to the limitations of the `PatternMatcher`.
        If it isn't, exit with error and a corresponding message.
        """
        if len(self.pattern) == 0:
            logger.e(logger.Code.INTERNAL_ERROR, "PatternMatcher: empty pattern.")

        if type(self.pattern[0]) is not Op:
            logger.e(
                logger.Code.INTERNAL_ERROR,
                "PatternMatcher: invalid pattern. The first block must be an `Op`.",
            )

        for block in self.pattern:
            block.validate()

    def _all_ops_are_in_the_model(self):
        """Determine if it is even possible to find a match for the pattern, based on whether the ops in the pattern
        are in the model.
        """

        for block in self.pattern:
            match block:
                case Op():
                    op = cast(Op, block)
                    if op.ops is not None:
                        if all(
                            self._get_opcode_indices_for(op_type) is None
                            for op_type in op.ops
                        ):
                            return False

                case MultipleSameOps():
                    multiple_same_ops = cast(MultipleSameOps, block)
                    if all(
                        self._get_opcode_indices_for(op_type) is None
                        for op_type in multiple_same_ops.ops
                    ):
                        return False

                case OneOf():
                    one_of = cast(OneOf, block)
                    valid = False
                    for op in one_of.one_of_ops:
                        if any(
                            self._get_opcode_indices_for(op_type) is not None
                            for op_type in op.ops
                        ):
                            valid = True

                    if not valid:
                        return False

        return True

    def _extend_pattern_with_op(
        self,
        op: Op,
        real_pattern: list,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
    ) -> bool:
        """Extend the currently matched pattern in `real_pattern` with an operator represented by `op`.
        This function finds a suitable TFLite operator in the model, and adds it to `real_pattern`.
        :return: True, if a matching operator was found. Otherwise, False.
        """
        if all(tensor not in tensor_map.keys() for tensor in op.io_as_list()):
            # The operator is not connected to the already matched part of the pattern. This is not supported.
            logger.e(
                logger.Code.INTERNAL_ERROR,
                f"PatternMatcher: Op on index {len(real_pattern)} is not connected "
                "to the preceding operators in the pattern.",
            )

        # The Op is somehow connected to the matched part.

        tensor_map_copy = tensor_map.copy()

        # Check if it is connected via the inputs.
        for inpt in op.inputs_as_list():
            if inpt not in tensor_map_copy.keys():
                continue

            # Found connecting input.
            connecting_input = tensor_map_copy[inpt]
            logger.internal_assert(
                type(connecting_input) is tflite_model.Tensor,
                f"PatternMatcher: consuming a set of tensors `{inpt}` is not yet supported.",
            )

            following_ops = input_to_ops.get(connecting_input.name, [])
            for following_op in following_ops:
                if following_op in real_pattern:
                    continue  # This operator has already been matched.

                if op.match(
                    following_op,
                    tensor_map_copy,
                    input_to_ops,
                    output_to_op,
                    self.builder,
                ) and self._tensor_rules_satisfied(
                    tensor_map_copy, input_to_ops, output_to_op
                ):
                    # Successful match.
                    real_pattern.append(following_op)
                    tensor_map.update(tensor_map_copy)
                    return True

                else:
                    tensor_map_copy = (
                        tensor_map.copy()
                    )  # Erase any potential invalid mappings.

        # Try operators connected via the outputs.
        for out in op.outputs_as_list():
            if out not in tensor_map_copy.keys():
                continue

            # Found connecting output.
            connecting_output = tensor_map_copy[out]
            preceding_op = output_to_op.get(connecting_output.name, None)
            if preceding_op is None:
                continue
            if preceding_op in real_pattern:
                continue  # This operator has already been matched.
            if op.match(
                preceding_op, tensor_map_copy, input_to_ops, output_to_op, self.builder
            ) and self._tensor_rules_satisfied(
                tensor_map_copy, input_to_ops, output_to_op
            ):
                # Successful match.
                real_pattern.append(preceding_op)
                tensor_map.update(tensor_map_copy)
                return True

            else:
                tensor_map_copy = (
                    tensor_map.copy()
                )  # Erase any potential invalid mappings.

        return False

    def _extend_pattern_with_multiple_same_ops(
        self,
        multiple_same_ops: MultipleSameOps,
        real_pattern: list,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
    ) -> bool:
        """Extend the currently matched pattern in `real_pattern` with multiple operators represented by
         `multiple_same_ops`.
        This function finds suitable TFLite operators in the model, and adds them to `real_pattern`.
        :return: True, if a matching operators were found. Otherwise, False.
        """
        if all(
            tensor not in tensor_map.keys() for tensor in multiple_same_ops.io_as_list()
        ):
            # The `MultipleSameOps` is not connected to the already matched part of the pattern. This is not supported.
            logger.e(
                logger.Code.INTERNAL_ERROR,
                f"PatternMatcher: MultipleSameOps on index {len(real_pattern)} is not "
                "connected to any preceding Ops in the pattern.",
            )

        # ---- The MultipleSameOps is somehow connected to the matched part. ----

        tensor_map_copy = tensor_map.copy()

        # Check if it is connected via the inputs.
        for inpt in multiple_same_ops.inputs_as_list():
            if inpt not in tensor_map_copy.keys():
                continue

            # Found connecting input.
            connecting_input = tensor_map_copy[inpt]
            following_ops = input_to_ops.get(connecting_input.name, [])
            logger.internal_assert(
                type(connecting_input) is tflite_model.Tensor,
                f"PatternMatcher: consuming a set of tensors `{inpt}` is not yet supported.",
            )

            # All following ops have to match.
            if any(following_op in real_pattern for following_op in following_ops):
                continue  # This operator has already been matched.

            if multiple_same_ops.match(
                following_ops, tensor_map_copy, input_to_ops, output_to_op, self.builder
            ) and self._tensor_rules_satisfied(
                tensor_map_copy, input_to_ops, output_to_op
            ):
                # Successful match.
                real_pattern.append(following_ops)
                tensor_map.update(tensor_map_copy)
                return True

            else:
                tensor_map_copy = (
                    tensor_map.copy()
                )  # Erase any potential invalid mappings.

        # `MultipleSameOps` cannot be connected via the outputs.
        return False

    def _extend_pattern_with_one_of(
        self,
        one_of: OneOf,
        real_pattern: list,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
    ) -> bool:
        """Extend the currently matched pattern in `real_pattern` with an operator represented by `one_of`.
        This function finds a suitable TFLite operator in the model, and adds it to `real_pattern`.
        :return: True, if a matching operator was found. Otherwise, False.
        """
        for op in one_of.one_of_ops:
            tensor_map_copy = tensor_map.copy()
            if self._extend_pattern_with_op(
                op, real_pattern, tensor_map_copy, input_to_ops, output_to_op
            ):
                # Successfully matched the `OneOf`.
                tensor_map.update(tensor_map_copy)
                return True

        return False

    def _match_rest_of_pattern(
        self,
        real_pattern: list,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        pattern_idx: int,
    ):
        """Provided that a part of the pattern has been matched with operators in the TFLite model, extend this matched
         `real_pattern` with new TFLite operators that match the rest of the pattern.
        :param pattern_idx: Index into the `self.patter`, with the first block that has not yet been matched.
        """
        if pattern_idx >= len(self.pattern):
            # Successfully matched full pattern.
            return True

        tensor_map_copy = tensor_map.copy()

        match self.pattern[pattern_idx]:
            case Op():
                op = cast(Op, self.pattern[pattern_idx])
                if self._extend_pattern_with_op(
                    op, real_pattern, tensor_map_copy, input_to_ops, output_to_op
                ):
                    # Successful match.
                    pattern_idx += 1
                    tensor_map.update(tensor_map_copy)

                else:
                    # Failed to match the Op.
                    return False

            case MultipleSameOps():
                multiple_same_ops = cast(MultipleSameOps, self.pattern[pattern_idx])
                if self._extend_pattern_with_multiple_same_ops(
                    multiple_same_ops,
                    real_pattern,
                    tensor_map_copy,
                    input_to_ops,
                    output_to_op,
                ):
                    # Successful match.
                    pattern_idx += 1
                    tensor_map.update(tensor_map_copy)

                else:
                    # Failed to match the MultipleSameOps.
                    return False

            case OneOf():
                one_of = cast(OneOf, self.pattern[pattern_idx])
                if self._extend_pattern_with_one_of(
                    one_of, real_pattern, tensor_map_copy, input_to_ops, output_to_op
                ):
                    # Successful match.
                    pattern_idx += 1
                    tensor_map.update(tensor_map_copy)

                else:
                    # Failed to match the Op.
                    return False

            case _:
                logger.e(
                    logger.Code.INTERNAL_ERROR,
                    f"PatternMatcher: pattern contains unexpected block `{self.pattern[pattern_idx]}`.",
                )

        # Matched a block. Recursively match the rest of the pattern.
        return self._match_rest_of_pattern(
            real_pattern, tensor_map, input_to_ops, output_to_op, pattern_idx
        )

    def match_patterns(
        self,
    ) -> Iterator[
        Tuple[
            list[tflite_model.Operator | list[tflite_model.Operator]],
            NameToTensorMap,
            InputTensorToOpsMap,
            OutputTensorToOpMap,
        ]
    ]:
        """Iterate over the model and yield matched patterns of operators."""

        if not self._all_ops_are_in_the_model():
            # The model doesn't contain sufficient operators to satisfy the pattern.
            return

        input_to_ops, output_to_op = create_tensor_to_operator_dictionaries(
            self.builder
        )

        real_pattern: list[tflite_model.Operator] = (
            []
        )  # List of matched TFLite operators in the TFLite model.
        tensor_map: NameToTensorMap = {}

        # The first block of a pattern is always an `Op`.
        first_pattern_op = cast(Op, self.pattern[0])

        for first_real_op in self.builder.get_operators():
            if first_pattern_op.match(
                first_real_op, tensor_map, input_to_ops, output_to_op, self.builder
            ) and self._tensor_rules_satisfied(tensor_map, input_to_ops, output_to_op):
                # Successful first match.
                real_pattern.append(first_real_op)

            else:
                # Mismatch.
                real_pattern = []
                tensor_map = {}
                continue

            # Matched the first `Op`. Now try to match the rest of the pattern.
            if self._match_rest_of_pattern(
                real_pattern, tensor_map, input_to_ops, output_to_op, 1
            ):  # Start from index 1 in the pattern.
                # Successfully matched full pattern.
                yield real_pattern, tensor_map, input_to_ops, output_to_op

                # The underlying TFLite model may have been changed. Re-compute the tensor to operator maps to be safe.
                input_to_ops, output_to_op = create_tensor_to_operator_dictionaries(
                    self.builder
                )

            real_pattern = []
            tensor_map = {}
