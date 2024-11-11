# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from typing import Callable, Optional, Set, Union

import torch
from executorch.backends.cadence.aot.utils import get_edge_overload_packet

from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket

from executorch.exir.pass_base import ExportPass
from torch._ops import OpOverloadPacket


# Is an overlap in tensor lifetime and storage allowed at the current opt level?
# We allow overlap at opt level >= 2.
def allow_lifetime_and_storage_overlap(opt_level: int) -> bool:
    return opt_level >= 2


# A dataclass that stores the attributes of an ExportPass.
@dataclass
class CadencePassAttribute:
    opt_level: Optional[int] = None
    debug_pass: bool = False


# A dictionary that maps an ExportPass to its attributes.
ALL_CADENCE_PASSES: dict[ExportPass, CadencePassAttribute] = {}


def get_cadence_pass_attribute(p: ExportPass) -> CadencePassAttribute:
    return ALL_CADENCE_PASSES[p]


# A decorator that registers a pass.
def register_cadence_pass(
    pass_attribute: CadencePassAttribute,
) -> Callable[[ExportPass], ExportPass]:
    def wrapper(cls: ExportPass) -> ExportPass:
        ALL_CADENCE_PASSES[cls] = pass_attribute
        return cls

    return wrapper


def get_all_available_cadence_passes() -> Set[ExportPass]:
    return set(ALL_CADENCE_PASSES.keys())


# Create a new filter to filter out relevant passes from all Jarvis passes.
def create_cadence_pass_filter(
    opt_level: int, debug: bool = False
) -> Callable[[ExportPass], bool]:
    def _filter(p: ExportPass) -> bool:
        pass_attribute = get_cadence_pass_attribute(p)
        return (
            pass_attribute.opt_level is not None
            and pass_attribute.opt_level <= opt_level
            and (not pass_attribute.debug_pass or debug)
        )

    return _filter


# Return the overload packet for the edge or torch op.
def get_overload_packet(
    op: Union[Callable[..., str], str],
) -> Union[OpOverloadPacket, EdgeOpOverloadPacket, None]:
    return (
        get_edge_overload_packet(op)
        if isinstance(op, EdgeOpOverload)
        else getattr(op, "overloadpacket", None)
    )


# Get the list of node names in a graph module (only for "call_function" ops and
# EdgeOpOverload targets). This should be used only after to_edge is called.
def get_node_names_list_from_gm(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    graph_nodes = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, EdgeOpOverload):
            continue
        graph_nodes.append(node.name)
    return graph_nodes
