# pyre-strict

import copy
import dataclasses
import json
import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import executorch.exir as exir
import executorch.exir.memory as memory
import torch
import torch._export.exported_program as ep
import torch._export.serde.schema as schema
import torch._export.serde.serialize as export_serialize
from torch.fx.experimental import symbolic_shapes


log: logging.Logger = logging.getLogger(__name__)


class GraphModuleSerializer(export_serialize.GraphModuleSerializer):
    def __init__(
        self, graph_signature: ep.ExportGraphSignature, call_spec: ep.CallSpec
    ) -> None:
        super().__init__(graph_signature, call_spec)
        self.state_dict: Dict[str, torch.Tensor] = {}  # TODO(T157676982)

    def handle_call_function(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"

        if node.target is memory.alloc:
            ex_node = schema.Node(
                target="memory.alloc",
                inputs=self.serialize_alloc_inputs(node.args),
                outputs=self.serialize_arbitrary_outputs(node),
                metadata=self.serialize_metadata(node),
            )
            self.graph_state.nodes.append(ex_node)
            return

        super().handle_call_function(node)

    def serialize_alloc_inputs(
        self, inputs  # pyre-ignore
    ) -> List[schema.NamedArgument]:
        """
        Serialize the inputs to the memory.alloc function. Since there's no
        specific spec, we jut serialize the inputs with a dummy name.
        We serialize the AllocSpec into a string "size;dtype"
        """
        assert len(inputs) == 1

        def serialize_alloc_spec(alloc_spec: memory.AllocSpec) -> schema.Argument:
            return schema.Argument.create(
                as_string=f"{alloc_spec[0]};{export_serialize._TORCH_TO_SERIALIZE_DTYPE[alloc_spec[1]].value}"
            )

        if isinstance(inputs[0], list):
            # Singleton list
            assert len(inputs[0]) == 1
            return [
                schema.NamedArgument(
                    name="alloc_list", arg=serialize_alloc_spec(inputs[0][0])
                )
            ]
        else:
            # Single value
            return [
                schema.NamedArgument(
                    name="alloc_arg", arg=serialize_alloc_spec(inputs[0])
                )
            ]

    def serialize_arbitrary_outputs(self, node: torch.fx.Node) -> List[schema.Argument]:
        meta_val = node.meta["val"]

        # Check single value return
        if isinstance(meta_val, torch.Tensor):
            return [
                schema.Argument.create(
                    as_tensor=self.serialize_tensor_output(node.name, meta_val)
                )
            ]

        # There are a two possibilities at this point:
        # - This operator returns a list of Tensors.
        # - This operator returns multiple Tensors.
        #
        # Either way, start by gathering a list of TensorArguments with the correct names.
        # For consistent naming with FX, consult the downstream `getitem` node and
        # make sure our outputs have the same name.
        idx_to_name = {}
        for user in node.users:
            if user.target is not operator.getitem:
                continue
            idx_to_name[user.args[1]] = user.name

        for idx, _ in enumerate(meta_val):
            # FX does not emit a getitem node for any outputs that are unused.
            # However, we need a name for them so that the number of outputs will
            # correctly match the schema. Just assign a dummy name.
            if idx not in idx_to_name:
                idx_to_name[idx] = f"{node.name}_unused_{idx}"

        arg_list = []
        for i, element_meta_val in enumerate(meta_val):
            arg_list.append(
                self.serialize_tensor_output(idx_to_name[i], element_meta_val)
            )

        if len(meta_val) == 1:
            # The operator returns a list of tensors
            return [schema.Argument.create(as_tensors=arg_list)]
        else:
            # The operator returns multiple tensors
            return [schema.Argument.create(as_tensor=arg) for arg in arg_list]

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

    # pyre-ignore
    def deserialize_node(self, serialized_node: schema.Node, target: Callable) -> None:
        if target == "memory.alloc":
            args = self.deserialize_alloc_inputs(serialized_node.inputs)
            fx_node = self.graph.create_node(
                "call_function", memory.alloc, args, {}, "alloc"
            )

            self.deserialize_arbitrary_outputs(serialized_node, fx_node)

            fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))
            return

        elif isinstance(target, str):
            # Create a dummy fake op if the target does not exist
            # because we cannot create a call_function node w/o a
            # callable target
            log.warning(
                f"Could not find operator {target}. Returning fake operator."
            )  # noqa: G004

            # pyre-ignore
            def fake_op(x):
                raise NotImplementedError("Fake op is not meant to be run.")

            fake_op.__name__ = target
            target = fake_op
            return

        super().deserialize_node(serialized_node, target)

    # pyre-ignore
    def deserialize_alloc_inputs(self, serialized_inputs: List[schema.NamedArgument]):
        def deserialize_alloc_spec(serialized_alloc_spec: str) -> memory.AllocSpec:
            serialized_alloc_spec_elems = serialized_alloc_spec.split(";")
            assert len(serialized_alloc_spec_elems) == 2
            serialized_size_elems = (
                serialized_alloc_spec_elems[0].strip("()").split(",")
            )

            size = tuple(int(x) for x in serialized_size_elems if x != "")
            dtype = export_serialize._SERIALIZE_TO_TORCH_DTYPE[
                int(serialized_alloc_spec_elems[1])
            ]
            return (size, dtype)

        assert serialized_inputs[0].arg.type == "as_string"

        # Single value
        if len(serialized_inputs) == 1 and serialized_inputs[0].name == "alloc_arg":
            res = (deserialize_alloc_spec(serialized_inputs[0].arg.value),)
            return res

        # Singleton list value
        assert len(serialized_inputs) == 1
        alloc_specs = [deserialize_alloc_spec(serialized_inputs[0].arg.value)]
        return (alloc_specs,)

    def deserialize_arbitrary_outputs(
        self, serialized_node: schema.Node, fx_node: torch.fx.Node
    ) -> None:
        # Single tensor return
        if (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_tensor"
        ):
            return self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)

        self.deserialize_multiple_outputs(serialized_node, fx_node)

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
