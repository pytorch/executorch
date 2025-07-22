import facto.specdb.function as fn
import torch

from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import ConstraintProducer as cp, InPosArg, OutArg, Spec

"""
This file contains FACTO operator specs for ops not in the standard FACTO db. This mainly
includes ops not in the Core ATen op set and preserved by a backend, such as linear.
"""

LINEAR_DEFAULT_SPEC = Spec(
    op="linear.default",  # (Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    inspec=[
        InPosArg(
            ArgType.Tensor,
            name="input",
            deps=[1, 2],
            constraints=[
                cp.Dtype.Eq(lambda deps: deps[0].dtype),
                cp.Rank.Ge(lambda deps: 2),
                cp.Size.In(
                    lambda deps, r, d: fn.broadcast_to(
                        (fn.safe_size(deps[0], 0), fn.safe_size(deps[1], 1)), r, d
                    )
                ),
            ],
        ),
        InPosArg(
            ArgType.Tensor,
            name="weight",
            constraints=[
                cp.Dtype.Ne(lambda deps: torch.bool),
                cp.Rank.Eq(lambda deps: 2),
            ],
        ),
        InPosArg(
            ArgType.Tensor,
            name="bias",
            deps=[1],
            constraints=[
                cp.Dtype.Eq(lambda deps: deps[0].dtype),
                cp.Rank.Eq(lambda deps: 2),
                cp.Size.Eq(
                    lambda deps, r, d: fn.safe_size(deps[0], 1) if d == 0 else None
                ),
            ],
        ),
    ],
    outspec=[
        OutArg(ArgType.Tensor),
    ],
)

_extra_specs = [
    LINEAR_DEFAULT_SPEC,
]

ExtraSpecDB: dict[str, Spec] = {s.op: s for s in _extra_specs}
