import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


def _should_transform(tensor: torch.Tensor) -> bool:
    """
    Returns whether the given tensor should be transformed by the pass to
    contiguous dim order.
    """
    return not tensor.is_contiguous() and not tensor.is_contiguous(
        memory_format=torch.channels_last
    )


def _update_placeholder_meta(
    exported_program: ExportedProgram,
    target: str,
    kind: InputKind,
):
    input_spec = next(
        (
            spec
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == kind and spec.target == target
        ),
        None,
    )
    if input_spec is None:
        raise RuntimeError(f"Missing input spec for lifted tensor {target}.")

    placeholder_node = next(
        (
            n
            for n in exported_program.graph.nodes
            if n.op == "placeholder" and n.name == input_spec.arg.name
        ),
        None,
    )
    if input_spec is None:
        raise RuntimeError(f"Missing placeholder for {input_spec.arg.name}.")

    placeholder_node.meta["val"] = placeholder_node.meta["val"].contiguous()


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
            _update_placeholder_meta(exported_program, key, InputKind.CONSTANT_TENSOR)

    # Convert buffers.
    non_persistent_buffer_names = set(
        exported_program.graph_signature.non_persistent_buffers
    )
    for key, buffer in exported_program.named_buffers():
        if (
            key not in non_persistent_buffer_names
            and isinstance(buffer, torch.Tensor)
            and _should_transform(buffer)
        ):
            # Persistent buffers are stored in the state dict.
            exported_program.state_dict[key] = buffer.contiguous()

            _update_placeholder_meta(exported_program, key, InputKind.BUFFER)

    return exported_program
