# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from abc import ABC, abstractmethod

from executorch.exir._warnings import experimental
from torch._export.verifier import _verify_exported_program_signature
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature


@experimental("This API is experimental and subject to change without notice.")
class OpBackend(ABC):
    """A backend that lowers operators in place rather than delegating them.

    A ``Partitioner`` may only tag nodes; ``to_backend`` asserts the graph it
    returns is identical. An operator backend rewrites the graph instead,
    replacing operators with its own kernels and adding the constants those
    kernels need, so it runs as a peer to a delegate rather than through one.

    Its kernels are outside the edge dialect, so the program has to reach it
    with ``_check_ir_validity`` off. Programs keep the verifier they were built
    with, so that is a property of the ``to_edge`` call, not of anything the
    backend can do: with it on, the backend's own ``_transform`` rejects the
    operators it just installed. ``to_backend`` clears the flag for delegates;
    an operator backend has no equivalent hook and its caller must.
    """

    @abstractmethod
    def lower(
        self, exported_program: ExportedProgram, method_name: str
    ) -> ExportedProgram:
        """Return this method's program, rewritten.

        The program handed in is a private copy, so rewriting it in place is
        fine and returning it is fine.

        An input added here needs both a spec in
        ``graph_signature.input_specs`` and its tensor stored -- in
        ``state_dict`` for a parameter or persistent buffer, in ``constants``
        otherwise. It must also be placed before every user input, which the
        emitter does not require but both in-tree helpers do
        (``backends.transforms.utils.create_constant_placeholder``, and
        ``lift_constant_tensor_pass`` for a ``get_attr`` node).

        Raise, naming the node, for an operator this backend claimed but
        cannot lower: preserved operators are not decomposed, so there is no
        portable fallback and it would reach the runtime unlowered.

        ``method_name`` is for diagnostics.
        """


def _lower_and_verify(
    exported_program: ExportedProgram,
    op_backend: OpBackend,
    method_name: str,
) -> ExportedProgram:
    """Lower one method's program, checking what the backend hands back.

    The backend gets a private copy to rewrite, so the caller's program is
    left alone whatever the backend does -- including raising part-way through.

    An added input easily leaves the graph signature no longer describing the
    graph; unchecked, that surfaces later as a message-less assertion or a bare
    ``KeyError`` from the emitter, naming neither the backend nor the input.

    Deliberately not ``ExportedProgram.validate()``: that re-enters the edge
    verifier, which rejects the operators an operator backend exists to
    install.
    """
    # A backend is free to rewrite in place, and the passes most of them are
    # built from do exactly that, so it gets a copy: the caller still holds the
    # original as an earlier stage's artifact. The graph is duplicated, the
    # weights it refers to are not.
    private = copy.copy(exported_program)
    private._graph_module = copy.deepcopy(exported_program.graph_module)
    private._state_dict = dict(exported_program.state_dict)
    private._constants = dict(exported_program.constants)
    # The signature needs its own spec objects, not just its own lists:
    # `create_constant_placeholder`, which the contract above points at, inserts
    # into `input_specs` before rebinding the signature, and `InputSpec` is a
    # mutable dataclass, so a backend editing one in place would reach back into
    # the caller's program even if lowering then raises.
    signature = exported_program.graph_signature
    private._graph_signature = ExportGraphSignature(
        copy.deepcopy(signature.input_specs), copy.deepcopy(signature.output_specs)
    )
    private._range_constraints = dict(exported_program.range_constraints)
    private._module_call_graph = list(exported_program.module_call_graph)

    lowered = op_backend.lower(private, method_name)

    name = f"{type(op_backend).__name__}.lower() on '{method_name}'"
    if not isinstance(lowered, ExportedProgram):
        raise TypeError(f"{name} must return an ExportedProgram, got {type(lowered)}")

    try:
        _verify_exported_program_signature(lowered)
    except Exception as e:
        raise ValueError(f"{name} returned an inconsistent program: {e}") from e
    return lowered
