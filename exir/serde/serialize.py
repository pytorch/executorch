# pyre-strict

import dataclasses
import json
from typing import Dict, Optional, Tuple

import executorch.exir as exir
import torch._export.serde.schema as schema
import torch._export.serde.serialize as export_serialize
from torch.fx.experimental import symbolic_shapes


class GraphModuleSerializer(export_serialize.GraphModuleSerializer):
    pass


class ExportedProgramSerializer(export_serialize.ExportedProgramSerializer):
    # pyre-ignore
    def serialize(
        self, exported_program: exir.ExportedProgram
    ) -> Tuple[schema.ExportedProgram, bytes]:
        serialized_graph_module = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.call_spec
        ).serialize(exported_program.graph_module)
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
            export_serialize.serialize_state_dict(exported_program.state_dict),
        )


class GraphModuleDeserializer(export_serialize.GraphModuleDeserializer):
    # TODO(angelayi): implement for delegation
    pass


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

        (
            graph_module,
            sig,
            call_spec,
            symbol_name_to_symbol,
        ) = GraphModuleDeserializer().deserialize(
            serialized_exported_program.graph_module,
            symbol_name_to_range,
        )
        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range,
            symbol_name_to_symbol,
        )
        state_dict = export_serialize.deserialize_state_dict(serialized_state_dict)

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
            state_dict,
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
