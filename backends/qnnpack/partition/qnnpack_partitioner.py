from executorch.backends.qnnpack.partition.support_patterns import (
    get_dynamic_quant_addmm_with_view_copy_graph,
    get_dynamic_quant_addmm_without_view_copy_graph,
    get_dynamic_quant_mm_with_view_copy_graph,
    get_dynamic_quant_mm_without_view_copy_graph,
)
from executorch.backends.qnnpack.qnnpack_preprocess import QnnpackBackend
from executorch.backends.transforms.addmm_mm_to_linear import (
    apply_addmm_mm_to_linear_transform,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    _SingleOpDelegatePartitioner,
)


class QnnpackPartitioner(_SingleOpDelegatePartitioner):
    def __init__(self) -> None:
        qnnp_patterns = [
            get_dynamic_quant_addmm_with_view_copy_graph(),
            get_dynamic_quant_addmm_without_view_copy_graph(),
            get_dynamic_quant_mm_with_view_copy_graph(),
            get_dynamic_quant_mm_without_view_copy_graph(),
            # Maybe there is a better way to handle dynamic shape
            # However, if we want to decouple partitioner from how the
            # graph was generated we need to capture all the ways in
            # which graph is generated _that_ can affect partitioner.
            get_dynamic_quant_addmm_with_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_addmm_without_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_mm_with_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_mm_without_view_copy_graph(dynamic_shape=True),
        ]
        super().__init__(
            QnnpackBackend.__name__, qnnp_patterns, [apply_addmm_mm_to_linear_transform]
        )
