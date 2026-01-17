from torch.export import ExportedProgram
import torch

def _should_transform(tensor: torch.Tensor) -> bool:
    """
    Returns whether the given tensor should be transformed by the pass to
    contiguous dim order.
    """
    return not tensor.is_contiguous() and not tensor.is_contiguous(memory_format=torch.channels_last)

def convert_constant_dim_order_pass(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    """
    Normalize the dim order of constant tensors, ensuring that all constant tensors
    have either default or channels_last dim order. Tensors with other dim orders or
    striding are converted to contiguous tensors. This pass acts in-place on the
    unlifted exported program.

    Args:
        exported_program: The ExportedProgram to transform.

    Returns:
        The modified ExportedProgram with normalized constant dim order.
    """

    for key, const in exported_program.constants.items():
        if isinstance(const, torch.Tensor) and _should_transform(const):
            exported_program.constants[key] = const.contiguous()

            # Also update the corresponding placeholder node meta value. This doesn't
            # get automatically updated during retracing as ExportPass uses the placeholder
            # meta as the source of truth. TODO(?)
            input_spec = next((spec for spec in exported_program.graph_signature.input_specs if spec.kind == torch.export.graph_signature.InputKind.CONSTANT_TENSOR and spec.target == key), None)
            if input_spec is None:
                raise RuntimeError(f"Missing input spec for constant tensor {key}.")

            placeholder_node = next((n for n in exported_program.graph.nodes if n.op == 'placeholder' and n.name == input_spec.arg.name), None)
            if input_spec is None:
                raise RuntimeError(f"Missing placeholder for constant tensor {input_spec.arg.name}.")

            placeholder_node.meta["val"] = placeholder_node.meta["val"].contiguous()

    return exported_program
