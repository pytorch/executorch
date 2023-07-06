# pyre-strict

import copy
import dataclasses
import json
from typing import Any, Dict, Optional, Tuple

import executorch.exir as exir
import torch
import torch._export.exported_program as ep
import torch._export.serde.schema as schema
import torch._export.serde.serialize as export_serialize
from torch.fx.experimental import symbolic_shapes


class GraphModuleSerializer(export_serialize.GraphModuleSerializer):
    def __init__(
        self, graph_signature: ep.ExportGraphSignature, call_spec: ep.CallSpec
    ) -> None:
        super().__init__(graph_signature, call_spec)
        self.state_dict: Dict[str, torch.Tensor] = {}  # TODO(T157676982)

    # pyre-ignore
    def serialize_input(self, arg) -> schema.Argument:
        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                attr = getattr(self.original_graph_module, arg.target)  # pyre-ignore
                if isinstance(attr, torch.Tensor):
                    self.state_dict[arg.name] = copy.deepcopy(attr)
                    return schema.Argument.create(
                        as_tensor=schema.TensorArgument(name=arg.name)
                    )
        return super().serialize_input(arg)

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> schema.Graph:
        self.original_graph_module: torch.fx.GraphModule = graph_module  # pyre-ignore
        return super().serialize_graph(graph_module)


class ExportedProgramSerializer(export_serialize.ExportedProgramSerializer):
    def serialize(
        self, exported_program: exir.ExportedProgram
    ) -> Tuple[schema.ExportedProgram, bytes]:
        gm_serializer = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.call_spec
        )
        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)

        serialized_range_constraints = export_serialize.serialize_range_constraints(
            exported_program.range_constraints
        )
        serialized_equality_constraints = (
            export_serialize.serialize_equality_constraints(
                exported_program.equality_constraints
            )
        )

        return (
            schema.ExportedProgram(
                graph_module=serialized_graph_module,
                opset_version=self.opset_version,
                range_constraints=serialized_range_constraints,
                equality_constraints=serialized_equality_constraints,
            ),
            export_serialize.serialize_state_dict(gm_serializer.state_dict),
        )


class GraphModuleDeserializer(export_serialize.GraphModuleDeserializer):
    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.state_dict: Dict[str, Any] = state_dict  # TODO(T157676982)

    # TODO(angelayi): implement for delegation

    # pyre-ignore
    def deserialize_input(self, inp: schema.Argument) -> Any:
        value = inp.value
        if isinstance(value, schema.TensorArgument):
            if value.name in self.state_dict:  # TODO(T157676982)
                val = self.state_dict[value.name]
                setattr(self.module, value.name, val)
                return self.graph.create_node(
                    "get_attr",
                    value.name,
                    name=value.name,
                )

        return super().deserialize_input(inp)


class ExportedProgramDeserializer(export_serialize.ExportedProgramDeserializer):
    def deserialize(
        self,
        serialized_exported_program: schema.ExportedProgram,
        serialized_state_dict: bytes,
    ) -> exir.ExportedProgram:
        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                export_serialize._int_to_sympy_int(v.min_val),
                export_serialize._int_to_sympy_int(v.max_val),
            )
            for k, v in serialized_exported_program.range_constraints.items()
        }

        state_dict = export_serialize.deserialize_state_dict(serialized_state_dict)
        (
            graph_module,
            sig,
            call_spec,
            symbol_name_to_symbol,
        ) = GraphModuleDeserializer(state_dict).deserialize(
            serialized_exported_program.graph_module,
            symbol_name_to_range,
        )
        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range,
            symbol_name_to_symbol,
        )

        # Update the state dict any attributes accessed in the new graph. We need
        # this because we stored the delegate module directly in the new graph
        # module.
        for node in graph_module.graph.nodes:
            if node.op == "get_attr":
                state_dict[node.target] = getattr(graph_module, node.target)

        equality_constraints = export_serialize.deserialize_equality_constraints(
            serialized_exported_program.equality_constraints
        )

        return exir.ExportedProgram(
            state_dict,
            graph_module.graph,
            sig,
            call_spec,
            {},  # TODO(T157676982)
            range_constraints,
            equality_constraints,
        )


def serialize(
    exported_program: exir.ExportedProgram,
    opset_version: Optional[Dict[str, int]] = None,
) -> Tuple[bytes, bytes]:
    serialized_exported_program, serialized_state_dict = ExportedProgramSerializer(
        opset_version
    ).serialize(exported_program)
    json_program = json.dumps(
        dataclasses.asdict(serialized_exported_program),
        cls=export_serialize.EnumEncoder,
    )
    json_bytes = json_program.encode("utf-8")
    return json_bytes, serialized_state_dict


def deserialize(
    exported_program_bytes: bytes,
    state_dict: bytes,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> exir.ExportedProgram:
    exported_program_str = exported_program_bytes.decode("utf-8")
    exported_program_dict = json.loads(exported_program_str)
    serialized_exported_program = export_serialize._dict_to_dataclass(
        schema.ExportedProgram, exported_program_dict
    )
    return ExportedProgramDeserializer(expected_opset_version).deserialize(
        serialized_exported_program, state_dict
    )
