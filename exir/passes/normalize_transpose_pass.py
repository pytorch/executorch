import torch
from executorch.exir.pass_base import ExportPass


class NormalizeTransposePass(ExportPass):
    """
    Even with functionalization on, we still get graph with
    torch.ops.aten.t.default op. Ideally we should fix functionalization.
    TODO: once we have that, we should remove this pass.
    Check test_normalize_transpose_op in test_passes.py for more details
    """

    def call_operator(self, op, args, kwargs, meta):
        if op == torch.ops.aten.t.default:
            return super().call_operator(
                torch.ops.aten.t_copy.default, (args[0],), kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
