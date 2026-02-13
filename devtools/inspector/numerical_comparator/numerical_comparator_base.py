# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

from executorch.devtools.inspector._inspector_utils import DebugHandle

if TYPE_CHECKING:
    from executorch.devtools.inspector._inspector import Inspector

# Type alias for the mapping used in preprocessing
# Maps (aot_debug_handle, aot_output) -> (runtime_debug_handle, runtime_output)
IntermediateOutputMapping = Dict[Tuple[DebugHandle, Any], Tuple[DebugHandle, Any]]


class NumericalComparatorBase(ABC):
    """Base class for numerical comparison with optional preprocessing.

    This class provides a framework for comparing intermediate outputs between
    AOT (Ahead-of-Time) and runtime execution. Subclasses can override the
    `preprocessing` method to transform tensors before comparison (e.g., layout
    conversion, dequantization) and must implement `element_compare` for
    element-wise comparison logic.

    The `compare` method is the main entry point called by Inspector, which
    orchestrates the full comparison pipeline: preprocess -> element-wise compare
    -> aggregate results into a DataFrame.

    Attributes:
        _inspector: Optional reference to the Inspector instance, which provides
            access to the reference graph and other metadata needed for preprocessing.
    """

    def __init__(self, inspector: Optional["Inspector"] = None) -> None:
        """Initialize the comparator.

        Args:
            inspector: Optional Inspector instance that provides access to the
                reference graph and other metadata. Can be set later via the
                `inspector` property.
        """
        self._inspector: Optional["Inspector"] = inspector

    @property
    def inspector(self) -> Optional["Inspector"]:
        """Get the Inspector instance."""
        return self._inspector

    @inspector.setter
    def inspector(self, value: Optional["Inspector"]) -> None:
        """Set the Inspector instance."""
        self._inspector = value

    def preprocessing(
        self, mapping: IntermediateOutputMapping
    ) -> IntermediateOutputMapping:
        """Transform the mapping before comparison.

        Override this method to apply custom preprocessing to the intermediate
        outputs before comparison. This is useful for backends like Qualcomm that
        require tensor transformations (e.g., dequantization, layout conversion)
        before accurate numeric discrepancy measurement.

        The default implementation returns the mapping unchanged.

        Args:
            mapping: Dictionary mapping AOT (debug_handle, intermediate_output) pairs
                to runtime (debug_handle, intermediate_output) pairs.

                - Key: Tuple[DebugHandle, Any]
                    - DebugHandle: Tuple[int, ...] - debug handle(s) from AOT graph
                    - Any: torch.Tensor or sequence - AOT intermediate output

                - Value: Tuple[DebugHandle, Any]
                    - DebugHandle: Tuple[int, ...] - debug handle(s) from runtime
                    - Any: torch.Tensor or sequence - runtime intermediate output

        Returns:
            The transformed mapping, ready for element-wise comparison.

        Note:
            When implementing custom preprocessing, you can access the reference
            graph via `self._inspector.get_reference_graph()` to retrieve node
            metadata such as quantization parameters or layout information.
        """
        return mapping

    @abstractmethod
    def element_compare(self, a: Any, b: Any) -> float:
        """Compare two tensors and return a scalar distance.

        This method should be overridden by subclasses to provide custom
        element-wise comparison logic (e.g., MSE, L1, SNR).

        Args:
            a: The first intermediate output to compare (typically AOT output).
            b: The second intermediate output to compare (typically runtime output).

        Returns:
            A numerical result indicating the comparison outcome (e.g., distance,
            error metric). Lower values typically indicate better agreement.
        """
        pass

    def compare(
        self,
        mapping: IntermediateOutputMapping,
        aot_debug_handle_to_op_names: Dict[DebugHandle, List[str]],
        runtime_debug_handle_to_op_names: Dict[DebugHandle, List[str]],
    ) -> pd.DataFrame:
        """Full comparison pipeline: preprocess -> element-wise compare -> aggregate.

        This is the main entry point called by Inspector.calculate_numeric_gap().
        It orchestrates the full comparison pipeline and returns a DataFrame
        with the results.

        Args:
            mapping: Dictionary mapping AOT (debug_handle, intermediate_output) pairs
                to runtime (debug_handle, intermediate_output) pairs.
            aot_debug_handle_to_op_names: Mapping from AOT debug handles to operator names.
            runtime_debug_handle_to_op_names: Mapping from runtime debug handles to operator names.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - aot_ops: List of AOT operator names
                - aot_intermediate_output: AOT intermediate output tensor
                - runtime_ops: List of runtime operator names
                - runtime_intermediate_output: Runtime intermediate output tensor
                - gap: List of numerical gap values
        """
        from executorch.devtools.inspector._inspector_utils import find_op_names

        def _validate_preprocessing_output(
            processed_mapping: IntermediateOutputMapping,
        ) -> None:
            """Validate the output format of preprocessing().

            Ensures the preprocessed mapping follows the expected format:
            Dict[Tuple[DebugHandle, Any], Tuple[DebugHandle, Any]]

            Args:
                processed_mapping: The mapping returned by preprocessing().

            Raises:
                TypeError: If processed_mapping is not a dict.
                ValueError: If any key or value in the mapping has an invalid format.
            """
            if not isinstance(processed_mapping, dict):
                raise TypeError(
                    f"preprocessing() must return a dict, got {type(processed_mapping).__name__}. "
                    "Expected format: Dict[Tuple[DebugHandle, Any], Tuple[DebugHandle, Any]]"
                )

            for key, value in processed_mapping.items():
                # Validate key format: Tuple[DebugHandle, Any]
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(
                        f"Invalid key format in preprocessed mapping: {key}. "
                        "Expected Tuple[DebugHandle, Any] where DebugHandle is Tuple[int, ...]"
                    )
                aot_debug_handle, _ = key
                if not isinstance(aot_debug_handle, tuple) or not all(
                    isinstance(x, int) for x in aot_debug_handle
                ):
                    raise ValueError(
                        f"Invalid AOT debug handle in key: {aot_debug_handle}. "
                        "Expected Tuple[int, ...]"
                    )

                # Validate value format: Tuple[DebugHandle, Any]
                if not isinstance(value, tuple) or len(value) != 2:
                    raise ValueError(
                        f"Invalid value format in preprocessed mapping: {value}. "
                        "Expected Tuple[DebugHandle, Any] where DebugHandle is Tuple[int, ...]"
                    )
                runtime_debug_handle, _ = value
                if not isinstance(runtime_debug_handle, tuple) or not all(
                    isinstance(x, int) for x in runtime_debug_handle
                ):
                    raise ValueError(
                        f"Invalid runtime debug handle in value: {runtime_debug_handle}. "
                        "Expected Tuple[int, ...]"
                    )

        def _compare_intermediate_outputs(a: Any, b: Any) -> List[float]:
            """Compare two outputs, handling both sequence and non-sequence cases.

            Args:
                a: The first intermediate output to compare.
                b: The second intermediate output to compare.

            Returns:
                List[float]: A list of comparison results.

            Raises:
                ValueError: If one input is a sequence and the other is not,
                    or if sequences have different lengths.
            """
            is_a_sequence = isinstance(a, Sequence)
            is_b_sequence = isinstance(b, Sequence)
            if is_a_sequence and is_b_sequence:
                if len(a) != len(b):
                    raise ValueError(
                        f"Sequences 'a' ({a}) and 'b' ({b}) must have the same length "
                        f"for comparison. len(a): {len(a)} len(b): {len(b)}."
                    )
                return [self.element_compare(x, y) for x, y in zip(a, b)]
            elif not is_a_sequence and not is_b_sequence:
                return [self.element_compare(a, b)]
            else:
                raise ValueError(
                    f"Both inputs 'a' ({a}) and 'b' ({b}) must be sequences "
                    f"or both must be non-sequences."
                )

        # Step 1: Apply preprocessing
        processed_mapping = self.preprocessing(mapping)

        # Validate the preprocessed mapping format
        _validate_preprocessing_output(processed_mapping)

        # Step 2: Element-wise comparison and aggregation
        rows = []
        for (aot_debug_handle, aot_intermediate_output), (
            runtime_debug_handle,
            runtime_intermediate_output,
        ) in processed_mapping.items():
            if aot_intermediate_output is None or runtime_intermediate_output is None:
                continue
            # If aot outputs length is > 1 then comparison fails since we don't really have
            # any instances where runtime intermediate output is a tuple or list.
            # This does not happen when edge dialect program is reference for comparison
            # but happens in aten graph where ops like unbind remain undecomposed.
            if (
                isinstance(aot_intermediate_output, Sequence)
                and len(aot_intermediate_output) > 1
            ):
                continue
            rows.append(
                {
                    "aot_ops": find_op_names(
                        aot_debug_handle, aot_debug_handle_to_op_names
                    ),
                    "aot_intermediate_output": aot_intermediate_output,
                    "runtime_ops": find_op_names(
                        runtime_debug_handle, runtime_debug_handle_to_op_names
                    ),
                    "runtime_intermediate_output": runtime_intermediate_output,
                    "gap": _compare_intermediate_outputs(
                        aot_intermediate_output, runtime_intermediate_output
                    ),
                }
            )

        # Step 3: Build and return DataFrame
        return pd.DataFrame(rows)
