import torch
from torch.export import ExportedProgram


class KeepCondPredicateOnCpuPass:
    """
    A pass that locates torch.cond in the graph and makes sure the predicate stays on CPU
    if the predicate is a buffer (placeholder).
    """

    requires_exported_program = True

    def __call__(self, exported_program: ExportedProgram):
        graph_module = exported_program.graph_module
        state_dict = exported_program.state_dict

        # Map input names to buffer names
        inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers

        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                pred_node = node.args[0]
                if pred_node.op == "placeholder":
                    # Found a placeholder used as predicate
                    # Check if it corresponds to a buffer
                    if pred_node.name in inputs_to_buffers:
                        buffer_name = inputs_to_buffers[pred_node.name]

                        # Move the buffer in state_dict to CPU
                        if buffer_name in state_dict:
                            # We modify the tensor in place or replace it?
                            # Replacing it is safer.
                            tensor = exported_program.state_dict[buffer_name]
                            if tensor.device.type != "cpu":
                                if isinstance(tensor, torch.nn.Parameter):
                                    exported_program._state_dict[buffer_name] = (
                                        torch.nn.Parameter(
                                            tensor.to("cpu"),
                                            tensor.requires_grad,
                                        )
                                    )
                                else:
                                    exported_program._state_dict[buffer_name] = (
                                        tensor.to("cpu")
                                    )

                        if buffer_name in exported_program.constants:
                            tensor = exported_program._constants[buffer_name]
                            if tensor.device.type != "cpu":
                                exported_program._constants[buffer_name] = tensor.to(
                                    "cpu"
                                )

                        # Also update the placeholder metadata
                        if "val" in pred_node.meta:
                            fake_tensor = pred_node.meta["val"]
                            if isinstance(fake_tensor, torch.Tensor):
                                pred_node.meta["val"] = fake_tensor.to("cpu")
        exported_program.validate()
