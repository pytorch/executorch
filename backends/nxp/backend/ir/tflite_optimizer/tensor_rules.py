# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import executorch.backends.nxp.backend.ir.converter.builder.model_builder as model_builder

import numpy as np
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import (
    InputTensorToOpsMap,
    OutputTensorToOpMap,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.pattern_matcher import (
    NameToTensorMap,
    operator_is_type,
)


class TensorRule(ABC):
    @abstractmethod
    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        pass

    @abstractmethod
    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        """Determine if the rule can be tested, based on whether the required tensors have already been mapped."""
        pass


class MultipleTensorRule(TensorRule):
    @property
    @abstractmethod
    def rules(self) -> list[TensorRule]:
        """The individual tensor rules."""
        pass

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        return all(
            rule(tensor_map, input_to_ops, output_to_op, builder) for rule in self.rules
        )

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(rule.is_applicable(tensor_map) for rule in self.rules)


@dataclass
class TensorHasRank(TensorRule):
    tensor: str
    rank: int

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].rank == self.rank
            case list():
                return all(t.rank == self.rank for t in tensor_map[self.tensor])
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorHasData(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].tmp_buffer.data is not None
            case list():
                return all(
                    t.tmp_buffer.data is not None for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorsHaveData(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorHasData(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


@dataclass
class TensorHasStaticValue(TensorRule):
    # Rule assures that the tensor has a single static value, which is equal to the provided `value`.

    tensor: str
    value: int | float

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                data = tensor_map[self.tensor].tmp_buffer.data
                if data is None or data.size > 1:
                    return False

                return np.allclose(data, np.asarray([self.value], data.dtype))

            case list():
                for t in tensor_map[self.tensor]:
                    data = t.tmp_buffer.data
                    if data is None or data.size > 1:
                        return False

                    if not np.allclose(data, np.asarray([self.value], data.dtype)):
                        return False

                return True

            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorHasNConsumers(TensorRule):
    tensor: str
    n: int

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        model_outputs = builder.get_sub_graph().outputs.tmp_outputs
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                num_consumers = len(input_to_ops.get(tensor_map[self.tensor].name, []))
                if tensor_map[self.tensor] in model_outputs:
                    num_consumers += 1
                return num_consumers == self.n

            case list():
                for t in tensor_map[self.tensor]:
                    num_consumers = len(input_to_ops.get(t.name, []))
                    if t in model_outputs:
                        num_consumers += 1
                    if num_consumers != self.n:
                        return False

                return True

            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


class TensorHasOneConsumer(TensorHasNConsumers):
    def __init__(self, tensor: str):
        super().__init__(tensor, 1)


class TensorsHaveOneConsumer(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorHasOneConsumer(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


@dataclass
class TensorConsumedOnlyBy(TensorRule):
    tensor: str
    consuming_operator_type: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return all(
                    operator_is_type(op, self.consuming_operator_type, builder)
                    for op in input_to_ops.get(tensor_map[self.tensor].name, [])
                )
            case list():
                for t in tensor_map[self.tensor]:
                    if not all(
                        operator_is_type(op, self.consuming_operator_type, builder)
                        for op in input_to_ops.get(t.name, [])
                    ):
                        return False
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorDimensionsMatch(TensorRule):
    tensor_1: str
    dim_idx_1: int

    tensor_2: str
    dim_idx_2: int

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        t1 = tensor_map[self.tensor_1]
        t2 = tensor_map[self.tensor_2]

        if (type(t1), type(t2)) != (tflite_model.Tensor, tflite_model.Tensor):
            raise NotImplementedError(
                "Tensor rule `TensorDimensionsMatch` is not implemented for sets of tensors."
            )

        if (not t1.shape.is_well_defined()) or (not t2.shape.is_well_defined()):
            return False

        return t1.shape[self.dim_idx_1] == t2.shape[self.dim_idx_2]

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor_1 in tensor_map.keys() and self.tensor_2 in tensor_map.keys()


@dataclass
class TensorHasDimensionOfSize(TensorRule):
    tensor: str
    dim_index: int
    dim_size: int

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].shape[self.dim_index] == self.dim_size

            case list():
                return all(
                    t.shape[self.dim_index] == self.dim_size
                    for t in tensor_map[self.tensor]
                )

            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorsHaveSameShape(TensorRule):
    tensors: list[str]

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        mapped_tensors = [tensor_map[tensor] for tensor in self.tensors]
        if any(type(t) is not tflite_model.Tensor for t in mapped_tensors):
            raise NotImplementedError(
                "Tensor rule `TensorsHaveSameShape` is not implemented for sets of tensors."
            )

        if not all(t.shape.is_well_defined() for t in mapped_tensors):
            # Not all shapes are known.
            return False

        if len(self.tensors) == 0:
            return True

        first_shape = mapped_tensors[0].shape
        return all(t.shape == first_shape for t in mapped_tensors)

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(tensor in tensor_map.keys() for tensor in self.tensors)


@dataclass
class TensorsHaveSameType(TensorRule):
    tensors: list[str]

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        if len(self.tensors) == 0:
            return True

        mapped_tensors = [tensor_map[tensor] for tensor in self.tensors]
        if any(type(t) is not tflite_model.Tensor for t in mapped_tensors):
            raise NotImplementedError(
                "Tensor rule `TensorsHaveSameType` is not implemented for sets of tensors."
            )

        first_type = mapped_tensors[0].type
        return all(t.type == first_type for t in mapped_tensors)

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(tensor in tensor_map.keys() for tensor in self.tensors)


@dataclass
class RuleIf(TensorRule):
    condition_rule: TensorRule
    body_rule: TensorRule

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        if self.condition_rule(tensor_map, input_to_ops, output_to_op, builder):
            return self.body_rule(tensor_map, input_to_ops, output_to_op, builder)

        return True

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.condition_rule.is_applicable(
            tensor_map
        ) and self.body_rule.is_applicable(tensor_map)


class RuleOr(TensorRule):

    def __init__(self, *rules: TensorRule):
        self.rules = list(rules)

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        return any(
            rule(tensor_map, input_to_ops, output_to_op, builder) for rule in self.rules
        )

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(rule.is_applicable(tensor_map) for rule in self.rules)


class RuleAnd(TensorRule):

    def __init__(self, *rules: TensorRule):
        self.rules = list(rules)

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        return all(
            rule(tensor_map, input_to_ops, output_to_op, builder) for rule in self.rules
        )

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(rule.is_applicable(tensor_map) for rule in self.rules)


@dataclass
class TensorHasType(TensorRule):
    tensor: str
    type_: TensorType

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].type == self.type_
            case list():
                return all(t.type == self.type_ for t in tensor_map[self.tensor])
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorsHaveType(MultipleTensorRule):
    def __init__(self, tensors: list[str], type_: TensorType):
        self._rules = [TensorHasType(t, type_) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


@dataclass
class TensorIsChannelsLast(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].tensor_format.is_channels_last()
            case list():
                return all(
                    t.tensor_format.is_channels_last() for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorIsChannelsFirst(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].tensor_format.is_channels_first()
            case list():
                return all(
                    t.tensor_format.is_channels_first() for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorIsFormatless(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].tensor_format == TensorFormat.FORMATLESS
            case list():
                return all(
                    t.tensor_format == TensorFormat.FORMATLESS
                    for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorIsQuantized(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].quantization is not None
            case list():
                return all(t.quantization is not None for t in tensor_map[self.tensor])
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorIsNotQuantized(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return tensor_map[self.tensor].quantization is None
            case list():
                return all(t.quantization is None for t in tensor_map[self.tensor])
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


@dataclass
class TensorIsPerTensorQuantized(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                tensor = tensor_map[self.tensor]
                return (
                    tensor.quantization is not None
                ) and tensor.quantization.is_per_tensor()
            case list():
                return all(
                    (t.quantization is not None) and t.quantization.is_per_tensor()
                    for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


class TensorsAreQuantized(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorIsQuantized(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


class TensorsAreNotQuantized(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorIsNotQuantized(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


class TensorsArePerTensorQuantized(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorIsPerTensorQuantized(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules


@dataclass
class TensorsHaveSameQuantization(TensorRule):
    tensors: list[str]

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        if len(self.tensors) == 0:
            return True

        all_tensors: list[tflite_model.Tensor] = []
        for mapped_tensor in (tensor_map[tensor] for tensor in self.tensors):
            match mapped_tensor:
                case tflite_model.Tensor():
                    all_tensors.append(mapped_tensor)
                case list():
                    all_tensors.extend(mapped_tensor)
                case _:
                    raise ValueError

        first_quantization = all_tensors[0].quantization
        first_type = all_tensors[0].type
        return all(t.quantization == first_quantization for t in all_tensors) and all(
            t.type == first_type for t in all_tensors
        )

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return all(tensor in tensor_map.keys() for tensor in self.tensors)


@dataclass
class TensorIsNotModelOutput(TensorRule):
    tensor: str

    def __call__(
        self,
        tensor_map: NameToTensorMap,
        input_to_ops: InputTensorToOpsMap,
        output_to_op: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        match tensor_map[self.tensor]:
            case tflite_model.Tensor():
                return (
                    tensor_map[self.tensor]
                    not in builder.get_sub_graph().outputs.tmp_outputs
                )
            case list():
                return all(
                    t not in builder.get_sub_graph().outputs.tmp_outputs
                    for t in tensor_map[self.tensor]
                )
            case _:
                raise ValueError

    def is_applicable(self, tensor_map: NameToTensorMap) -> bool:
        return self.tensor in tensor_map.keys()


class TensorsAreNotModelOutputs(MultipleTensorRule):
    def __init__(self, tensors: list[str]):
        self._rules = [TensorIsNotModelOutput(t) for t in tensors]

    @property
    def rules(self) -> list[TensorRule]:
        return self._rules
