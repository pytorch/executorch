from typing import Any, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch

from .types import Shape, ShapeRange
from .utils import get_dynamic_dims


def generate_input_specs(inputs, lower_setting, additional_inputs=None):
    # dynamic_batch is TRT only flag.
    if (
        not lower_setting.explicit_batch_dimension
        or lower_setting.dynamic_batch is False
    ):
        return InputTensorSpec.from_tensors(inputs)

    # If we don't have additional inputs, we assume the first dimension
    # is the dynamic batch dimension. Otherwise, we use the additional
    # inputs to determine the batch dimension.
    if additional_inputs is None:
        batch_dims = None
        if not isinstance(inputs, torch.Tensor) and len(inputs) > 1:
            bs = inputs[0].size(0)
            batch_dims = None
            if not all(x.size(0) == bs for x in inputs):
                batch_dims = InputTensorSpec.find_batch_size_dim(inputs)
        return InputTensorSpec.from_tensors_with_dynamic_batch_size(
            inputs,
            (
                0,
                lower_setting.max_batch_size,
                lower_setting.max_batch_size,
            ),
            lower_setting.opt_profile_replica,
            batch_dims,
        )
    else:
        batch_dims = []

        for i, j in zip(inputs, additional_inputs):
            found_batch_dim = False

            for idx, values in enumerate(zip(i.shape, j.shape)):
                if values[0] != values[1]:
                    assert (
                        found_batch_dim is False
                    ), f"We've already found a batch dim, {i.shape}, {j.shape}."
                    batch_dims.append(idx)
                    found_batch_dim = True

            if not found_batch_dim:
                raise RuntimeError(
                    f"Failed to find batch dimension because shapes are the same, {i.shape}"
                )

        return InputTensorSpec.from_tensors_with_dynamic_batch_size(
            inputs,
            (
                0,
                lower_setting.max_batch_size,
                lower_setting.max_batch_size,
            ),
            lower_setting.opt_profile_replica,
            batch_dims,
        )


class InputTensorSpec(NamedTuple):
    """
    This class contains the information of a input tensor.

    shape: shape of the tensor.

    dtype: dtyep of the tensor.

    device: device of the tensor. This is only used to generate inputs to the given model
        in order to run shape prop. For TensorRT engine, inputs have to be on cuda device.

    shape_ranges: If dynamic shape is needed (shape has dimensions of -1), then this field
        has to be provided (default is empty list). Every shape_range is a tuple of three
        tuples ((min_input_shape), (optimized_input_shape), (max_input_shape)). Each shape_range
        is used to populate a TensorRT optimization profile.
        e.g. If the input shape varies from (1, 224) to (100, 224) and we want to optimize
        for (25, 224) because it's the most common input shape, then we set shape_ranges to
        ((1, 224), (25, 225), (100, 224)).

    has_batch_dim: Whether the shape includes batch dimension. Batch dimension has to be provided
        if the engine want to run with dynamic shape.
    """

    shape: Shape
    dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    shape_ranges: List[ShapeRange] = []
    has_batch_dim: bool = True

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "InputTensorSpec":
        """
        Produce an InputTenosrSpec named tuple which contains the
        information of the given PyTorch tensor.

        Args:
            tensor (torch.Tensor): A PyTorch tensor.

        Returns:
            An InputTensorSpec named tuple.
        """
        return cls(tensor.shape, tensor.dtype, tensor.device)

    @classmethod
    def from_tensors(cls, tensors: Sequence[torch.Tensor]) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which contain
        the information of all the given PyTorch tensors.

        Args:
            tensors (Iterable[torch.Tensor]): A list of PyTorch tensors.

        Returns:
            A list of InputTensorSpec named tuples.
        """
        assert isinstance(tensors, (list, tuple))
        return [cls.from_tensor(t) for t in tensors]

    @classmethod
    def from_tensors_with_dynamic_batch_size(
        cls,
        tensors: Sequence[torch.Tensor],
        batch_size_range: Tuple[int, int, int],
        opt_profile_replica: int = 1,
        batch_dims: Optional[List[int]] = None,
    ) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which would contain
        the information of all the given PyTorch tensors. The produced input
        tensor specs will treat all tensors' first dimension as batch dimension
        and mark them as dynmaic.

        Args:
            tensors (Sequence[torch.Tensor]): A list of PyTorch tensors.
            batch_size_range (Tuple[int, int, int]): The first integer indicates
                the smallest batch size allowed. The second integer indiceates
                the batch size that we'll optimize for. The third integer indicates
                the largest batch size allowed.
            opt_profile_replica (int): If dynamic shape is enabled, each execution
                context requires a different optimization profile. This arg determines
                how many optimization profile replicas we want to produce.
            batch_dims (Optional[List[int]]): The batch dim might not be the leading dim
                and allow user to specify the batch dims using this arg. Default we treat
                dim 0 as the batch dim.

        Returns:
            A list of InputTensorSpec named tuples with dynamic ranges.
        """
        if batch_dims is None:
            batch_dims = cls.find_batch_size_dim(tensors)

        input_specs = []
        batch_size = tensors[0].size(batch_dims[0])

        for i, tensor in enumerate(tensors):
            batch_dim = batch_dims[i]
            if batch_dim == -1:
                input_specs.append(cls.from_tensor(tensor))
            else:
                shape = list(tensor.shape)
                assert batch_size == tensor.size(
                    batch_dim
                ), f"The {i}th tensor (shape: {tensor.shape}) doesn't have the correct batch size: {batch_size}."
                shape[batch_dim] = -1
                shape_ranges: List[ShapeRange] = [tuple(tuple(shape[0:batch_dim] + [bs] + shape[batch_dim + 1 :]) for bs in batch_size_range)] * opt_profile_replica  # type: ignore[list-item]
                input_specs.append(
                    cls(tuple(shape), tensor.dtype, tensor.device, shape_ranges)
                )

        return input_specs

    @classmethod
    # pyre-ignore [2]: Parameter `sample_input` must have a type other than `Any`
    def find_batch_size_dim(cls, inputs: Any) -> []:
        if isinstance(inputs, torch.Tensor) or len(inputs) <= 1:
            return [0]
        shapes = [i.shape for i in inputs]
        frequency_map = {}
        first_dims = set()
        for shape in shapes:
            if len(shape) < 2:
                # By pass for rank-1 tensors. MRS model has rank-1 tensor carry no batch_size info
                continue
            # Dedup shape value for single tensor
            first_dims.add(shape[0])
            shape = set(shape)
            for i in shape:
                frequency_map[i] = frequency_map.get(i, 0) + 1

        if len(first_dims) == 1:
            # first dim is the same in every input: we use it as batch_size
            batch_size = first_dims.pop()
        elif frequency_map:
            # first dims are different: we use the most frequent dim as batch_size
            sorted_frequency = sorted(frequency_map.items(), key=lambda x: -x[1])
            batch_size = sorted_frequency[0][0]
        else:
            # no dims to sort: no batch_size
            batch_size = -1

        bs_dim = []
        for i in inputs:
            # Default batch size dim = -1, indicate no batch_size
            dim = -1
            for index, val in enumerate(i.shape):
                if val == batch_size:
                    dim = index
                    break
            bs_dim.append(dim)

        return bs_dim

    def to_random_tensor(self, id=1):
        shape = tuple(self.shape)
        if len(get_dynamic_dims(shape)):
            # id=0 -> min shape
            # id=1 -> optimal shape
            # id=2 -> max shape
            shape = tuple(self.shape_ranges[0][id])
        elif not self.has_batch_dim:
            shape = (1,) + tuple(shape)

        return torch.randn(shape).to(dtype=self.dtype, device=self.device)

    @staticmethod
    def create_inputs_from_specs(input_specs: Iterable["InputTensorSpec"]):
        inputs = []
        for spec in input_specs:
            inputs.append(spec.to_random_tensor())

        return inputs

    @staticmethod
    def create_inputs_from_max_specs(input_specs: Iterable["InputTensorSpec"]):
        inputs = []
        for spec in input_specs:
            inputs.append(spec.to_random_tensor(2))

        return inputs
