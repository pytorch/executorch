# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import base64
import copy
import dataclasses
import io
import json
import logging
import operator
import os
import zipfile
from typing import Any, Callable, Dict, List, Optional, Union

import executorch.exir as exir
import executorch.exir.memory as memory
import executorch.exir.serde.export_serialize as export_serialize
import torch
import torch._export.serde.schema as schema
import torch.export.exported_program as ep
from executorch.exir import delegate
from executorch.exir.backend.compile_spec_schema import (
    CompileSpec as delegate_CompileSpec,
)
from executorch.exir.dialects._ops import _DialectNamespace, ops as exir_ops
from executorch.exir.dialects.backend._ops import BackendOpOverload
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.lowered_backend_module import (
    LoweredBackendModule as ExirLoweredBackendModule,
)
from executorch.exir.serde.export_serialize import SerializedArtifact
from executorch.exir.serde.schema import (
    CompileSpec,
    LoweredBackendModule as SerdeLoweredBackendModule,
    SCHEMA_VERSION,
)
from torch._export.serde.schema import SchemaVersion
from torch._export.serde.serialize import SerializeError
from torch._export.serde.union import _Union
from torch._export.verifier import load_verifier
from torch.fx.experimental import symbolic_shapes

log: logging.Logger = logging.getLogger(__name__)


class GraphModuleSerializer(export_serialize.GraphModuleSerializer):
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,
        module_call_graph: List[ep.ModuleCallEntry],
    ) -> None:
        super().__init__(graph_signature, module_call_graph)
        self.state_dict: Dict[str, torch.Tensor] = {}  # TODO(T157676982)

    def serialize_operator(
        self,
        target: Union[
            str,
            EdgeOpOverload,
            BackendOpOverload,
            torch._ops.OpOverload,
            torch._ops.HigherOrderOperator,
        ],
    ) -> str:
        if isinstance(target, str):
            return target
        elif target.__module__.startswith("executorch.exir.dialects.edge"):
            # TODO(zhxchen17) Maybe provide a function name helper in FX.
            # From torch.fx.node._get_qualified_name
            module = target.__module__.replace(
                "executorch.exir.dialects.edge._ops",
                "executorch.exir.dialects.edge.ops",
            )
            return f"{module}.{target.__name__}"
        elif target.__module__.startswith("executorch.exir.dialects.backend"):
            module = target.__module__.replace(
                "executorch.exir.dialects.backend._ops",
                "executorch.exir.dialects.backend.ops",
            )
            return f"{module}.{target.__name__}"

        return super().serialize_operator(target)

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
        elif isinstance(node.target, EdgeOpOverload):
            assert node.target._op is not None
            ex_node = schema.Node(
                target=self.serialize_operator(node.target),
                # pyre-ignore Undefined attribute [16]: Item `typing.Callable` of
                # `typing.Union[typing.Callable[..., typing.Any], str]` has no attribute `_op`.
                inputs=self.serialize_inputs(node.target._op, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                # TODO: create a new tensor_values here, meta might have faketensor info
                metadata=self.serialize_metadata(node),
            )
            self.graph_state.nodes.append(ex_node)
            return
        elif node.target is delegate.executorch_call_delegate:
            ex_node = schema.Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_call_delegate_inputs(node.args),
                outputs=self.serialize_arbitrary_outputs(node),
                metadata=self.serialize_metadata(node),
            )
            self.graph_state.nodes.append(ex_node)
            return

        super().handle_call_function(node)

    def serialize_outputs(self, node: torch.fx.Node) -> List[schema.Argument]:
        if isinstance(node.target, EdgeOpOverload):
            # Store the original edge op
            edge_op = node.target
            # Replace the edge op with the original ATen op so that we can just call into
            # the serialize_outputs implementation present in the parent class.
            node.target = edge_op._op
            ret = super().serialize_outputs(node)
            # Replace the edge op back.
            node.target = edge_op
        else:
            ret = super().serialize_outputs(node)
        return ret

    def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
        meta = super().serialize_metadata(node)

        if "debug_handle" in node.meta:
            debug_handle = node.meta["debug_handle"]
            meta["debug_handle"] = str(debug_handle)

        return meta

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
            return [
                schema.NamedArgument(name="alloc_list", arg=serialize_alloc_spec(arg))
                for arg in inputs[0]
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
        def handle_input_get_attr(arg: torch.fx.Node) -> bool:
            if arg.op == "get_attr":
                attr = getattr(self.original_graph_module, arg.target)  # pyre-ignore
                if isinstance(attr, torch.Tensor):
                    self.state_dict[arg.name] = copy.deepcopy(attr)
                    return True
            return False

        if isinstance(arg, torch.fx.Node):
            if handle_input_get_attr(arg):
                return schema.Argument.create(
                    as_tensor=schema.TensorArgument(name=arg.name)
                )
        elif isinstance(arg, (list, tuple)):
            if all(isinstance(a, torch.fx.Node) for a in arg) and any(
                (a.op == "get_attr" for a in arg)
            ):
                # list of tensors
                tensors = []
                for a in arg:
                    if a.op == "get_attr":
                        handle_input_get_attr(a)
                    tensors.append(schema.TensorArgument(name=a.name))

                return schema.Argument.create(
                    as_tensors=tensors,
                )

        return super().serialize_input(arg)

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> schema.Graph:
        self.original_graph_module: torch.fx.GraphModule = graph_module  # pyre-ignore
        return super().serialize_graph(graph_module)

    def serialize_call_delegate_inputs(
        self, args  # pyre-ignore
    ) -> List[schema.NamedArgument]:
        lowered_module_arg = args[0]
        delegate_args = args[1:]

        serialized_lowered_module = self.serialize_lowered_module(lowered_module_arg)
        serialized_lowered_module_arg = schema.NamedArgument(
            name=lowered_module_arg.target,
            arg=schema.Argument.create(as_string=serialized_lowered_module),
        )

        serialized_args = [serialized_lowered_module_arg]
        for i, arg in enumerate(delegate_args):
            serialized_args.append(
                schema.NamedArgument(
                    name=f"delegate_arg_{i}", arg=self.serialize_input(arg)
                )
            )
        return serialized_args

    def serialize_lowered_module(self, lowered_module_arg: torch.fx.Node) -> str:
        assert lowered_module_arg.op == "get_attr"
        assert isinstance(lowered_module_arg.target, str)

        def serialize_bytes(b: bytes) -> str:
            # We want to serialize the bytes to string because JSON cannot
            # serialize bytes.
            # Since the given bytes may be serialized with any encoding, so we
            # want to first encode with base64, and then decode it with
            # ascii. During deserialization we can just directly decode with b64
            # to get the original encoded bytes.
            return base64.b64encode(b).decode("ascii")

        lowered_module = getattr(
            lowered_module_arg.graph.owning_module, lowered_module_arg.target
        )
        assert isinstance(lowered_module, ExirLoweredBackendModule)

        serialized_compile_spec = [
            CompileSpec(cs.key, serialize_bytes(cs.value))
            for cs in lowered_module.compile_specs
        ]

        serialized_artifact = ExportedProgramSerializer().serialize(
            lowered_module.original_module
        )
        assert isinstance(serialized_artifact.exported_program, schema.ExportedProgram)

        serialized_processed_bytes = serialize_bytes(lowered_module.processed_bytes)

        serialized_lowered_module = SerdeLoweredBackendModule(
            original_module=serialized_artifact.exported_program,  # pyre-ignore
            original_state_dict=serialize_bytes(serialized_artifact.state_dict),
            processed_bytes=serialized_processed_bytes,
            compile_specs=serialized_compile_spec,
            backend_id=lowered_module.backend_id,
        )

        json_lowered_module = json.dumps(
            _dataclass_to_dict(serialized_lowered_module),
            cls=export_serialize.EnumEncoder,
        )
        return json_lowered_module


class ExportedProgramSerializer(export_serialize.ExportedProgramSerializer):
    def serialize(
        self, exported_program: ep.ExportedProgram
    ) -> export_serialize.SerializedArtifact:
        # This is a direct copy of torch.export's serializer

        assert isinstance(exported_program, ep.ExportedProgram)
        gm_serializer = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.module_call_graph
        )
        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)

        serialized_range_constraints = export_serialize.serialize_range_constraints(
            exported_program.range_constraints
        )

        # TODO: Directly serialize exported_program.constants once
        # CustomClassHolders get stored in the ExportedProgram rather than in
        # the graph
        constants = {}
        for n, c in gm_serializer.custom_objs.items():
            constants[n] = c
        for n, t in exported_program.constants.items():
            assert n not in constants
            constants[n] = t

        return export_serialize.SerializedArtifact(
            schema.ExportedProgram(
                graph_module=serialized_graph_module,
                opset_version=self.opset_version,
                range_constraints=serialized_range_constraints,
                schema_version=SchemaVersion(-1, -1),
                dialect=exported_program.dialect,
            ),
            export_serialize.serialize_torch_artifact(exported_program.state_dict),
            export_serialize.serialize_torch_artifact(constants),
        )


class GraphModuleDeserializer(export_serialize.GraphModuleDeserializer):
    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.state_dict: Dict[str, Any] = state_dict  # TODO(T157676982)

    def deserialize_operator(self, serialized_target: str) -> str:
        def find_operator(module: _DialectNamespace, serialized_target: str) -> str:
            serialized_target_names = serialized_target.split(".")[5:]

            target = module
            for name in serialized_target_names:
                if not hasattr(target, name):
                    return serialized_target
                else:
                    target = getattr(target, name)
            return target

        if serialized_target.startswith("executorch.exir.dialects.edge.ops"):
            return find_operator(exir_ops.edge, serialized_target)
        elif serialized_target.startswith("executorch.exir.dialects.backend.ops"):
            return find_operator(exir_ops.backend, serialized_target)

        return super().deserialize_operator(serialized_target)

    # pyre-ignore
    def deserialize_inputs_no_schema(self, serialized_node) -> Any:
        return tuple(
            self.deserialize_input(input.arg) for input in serialized_node.inputs
        )

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

        elif target is delegate.executorch_call_delegate:
            if (
                len(serialized_node.outputs) == 1
                and serialized_node.outputs[0].type == "as_tensor"
            ):
                # If it's a single tensor return then we can use the name of the
                # node itself
                name = serialized_node.outputs[0].value.name
            else:
                # Otherwise FX will make a name for us, and we'll have `getitem`
                # nodes pointed to that
                name = None

            args = self.deserialize_call_delegate_inputs(serialized_node.inputs)
            fx_node = self.graph.create_node("call_function", target, args, {}, name)

            self.deserialize_arbitrary_outputs(serialized_node, fx_node)

            fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))
            return
        elif isinstance(target, EdgeOpOverload):
            # For convenience: if this node returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            name = (
                serialized_node.outputs[0].value.name
                if export_serialize._is_single_tensor_return(target._op)
                else None  # FX will generate a name for us.
            )
            args, kwargs = self.deserialize_inputs(target._op, serialized_node)
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
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

            args = self.deserialize_inputs_no_schema(serialized_node)
            fx_node = self.graph.create_node("call_function", target, args, None, None)
            self.deserialize_arbitrary_outputs(serialized_node, fx_node)

            return

        super().deserialize_node(serialized_node, target)

    def deserialize_outputs(
        self, serialized_node: schema.Node, fx_node: torch.fx.Node
    ) -> None:
        if isinstance(fx_node.target, EdgeOpOverload):
            # Store the original edge op
            edge_op = fx_node.target
            # Replace the edge op with the original ATen op so that we can just call into
            # node deserialize_outputs implementation present in the parent class.
            fx_node.target = edge_op._op
            super().deserialize_outputs(serialized_node, fx_node)
            # Replace the edge op back.
            fx_node.target = edge_op
        else:
            super().deserialize_outputs(serialized_node, fx_node)

    def deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        res = super().deserialize_metadata(metadata)

        if debug_handle := metadata.get("debug_handle"):
            res["debug_handle"] = int(debug_handle)

        return res

    def deserialize_graph_output(self, output: schema.Argument) -> torch.fx.Node:
        if isinstance(output.value, schema.TensorArgument):
            if output.value.name in self.state_dict:  # TODO(T157676982)
                val = self.state_dict[output.value.name]
                setattr(self.module, output.value.name, val)
                node = self.graph.create_node(
                    "get_attr",
                    output.value.name,
                    name=output.value.name,
                )
                node.meta = {"val": ""}
                return node
            return self.serialized_name_to_node[output.value.name]
        elif isinstance(output.value, (schema.SymIntArgument, schema.SymBoolArgument)):
            return self.serialized_name_to_node[output.value.as_name]
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")

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

        alloc_specs = [
            deserialize_alloc_spec(serialized_input.arg.value)
            for serialized_input in serialized_inputs
        ]
        return (alloc_specs,)

    def deserialize_arbitrary_outputs(
        self, serialized_node: schema.Node, fx_node: torch.fx.Node
    ) -> None:
        if len(serialized_node.outputs) == 0:
            return
        # Single tensor return
        elif (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_tensor"
        ):
            return self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
        elif len(serialized_node.outputs) == 1 and isinstance(
            serialized_node.outputs[0].value,
            (schema.SymIntArgument, schema.SymBoolArgument),
        ):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return

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
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], schema.TensorArgument):
                result = []
                for arg in value:
                    if arg.name in self.state_dict:  # TODO(T157676982)
                        val = self.state_dict[arg.name]
                        setattr(self.module, arg.name, val)
                        result.append(
                            self.graph.create_node(
                                "get_attr",
                                arg.name,
                                name=arg.name,
                            )
                        )
                    else:
                        result.append(self.serialized_name_to_node[arg.name])
                return result

        return super().deserialize_input(inp)

    # pyre-ignore
    def deserialize_call_delegate_inputs(
        self, serialized_inputs: List[schema.NamedArgument]
    ):
        serialized_lowered_module = serialized_inputs[0]
        lowered_module_node = self.deserialize_lowered_module(serialized_lowered_module)
        serialized_delegate_inputs = serialized_inputs[1:]
        args = tuple(
            self.deserialize_input(input.arg) for input in serialized_delegate_inputs
        )
        return (lowered_module_node,) + args

    def deserialize_lowered_module(
        self, serialized_lowered_module_arg: schema.NamedArgument
    ) -> torch.fx.Node:
        assert serialized_lowered_module_arg.arg.type == "as_string"
        lowered_module_str = serialized_lowered_module_arg.arg.value
        json_lowered_module = json.loads(lowered_module_str)
        serialized_lowered_module = export_serialize._dict_to_dataclass(
            SerdeLoweredBackendModule, json_lowered_module
        )

        backend_id = serialized_lowered_module.backend_id
        processed_bytes = base64.b64decode(serialized_lowered_module.processed_bytes)
        compile_specs = [
            delegate_CompileSpec(key=cs.key, value=base64.b64decode(cs.value))
            for cs in serialized_lowered_module.compile_specs
        ]

        original_module = ExportedProgramDeserializer().deserialize(
            export_serialize.SerializedArtifact(
                serialized_lowered_module.original_module,
                base64.b64decode(serialized_lowered_module.original_state_dict),
                b"",
            )
        )

        lowered_module = ExirLoweredBackendModule(
            original_module,
            backend_id,
            processed_bytes,
            compile_specs,
        )
        self.module.register_module(serialized_lowered_module_arg.name, lowered_module)
        return self.graph.get_attr(serialized_lowered_module_arg.name)


class ExportedProgramDeserializer(export_serialize.ExportedProgramDeserializer):
    def deserialize(
        self,
        serialized_artifact: export_serialize.SerializedArtifact,
    ) -> ep.ExportedProgram:
        assert isinstance(serialized_artifact.exported_program, schema.ExportedProgram)

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                export_serialize._int_to_sympy_int(v.min_val),
                export_serialize._int_to_sympy_int(v.max_val),
            )
            for k, v in serialized_artifact.exported_program.range_constraints.items()
        }
        state_dict = export_serialize.deserialize_torch_artifact(
            serialized_artifact.state_dict
        )

        constants = export_serialize.deserialize_torch_artifact(
            serialized_artifact.constants
        )

        # TODO: No need to do this once CustomClassHolders are lifted to the ExportedProgram
        constants = {k: v for k, v in constants.items() if isinstance(v, torch.Tensor)}

        res = GraphModuleDeserializer(state_dict).deserialize(
            serialized_artifact.exported_program.graph_module,  # pyre-ignore
            symbol_name_to_range,
            constants,
        )

        graph_module = res.graph_module
        module_call_graph = res.module_call_graph
        symbol_name_to_symbol = res.names_to_symbols

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

        dummy_g = torch.fx.Graph()
        dummy_g.output(())
        exported_program = exir.ExportedProgram(
            root=state_dict,
            graph=dummy_g,
            graph_signature=ep.ExportGraphSignature(input_specs=[], output_specs=[]),
            state_dict={},  # TODO(T157676982)
            range_constraints=range_constraints,
            module_call_graph=module_call_graph,
            verifier=load_verifier(
                serialized_artifact.exported_program.dialect  # pyre-ignore
            ),
        )
        exported_program.graph_module.graph = graph_module.graph
        exported_program._graph_signature = res.signature
        for node in graph_module.graph.nodes:
            if node.op == "get_attr":
                setattr(
                    exported_program.graph_module, node.target, state_dict[node.target]
                )
        return exported_program


# pyre-ignore
def _dataclass_to_dict(obj):
    if isinstance(obj, _Union):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name, None))
            for f in dataclasses.fields(obj)
        }
    elif dataclasses.is_dataclass(obj):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: Optional[Dict[str, int]] = None,
) -> export_serialize.SerializedArtifact:
    serialized_artifact = ExportedProgramSerializer(opset_version).serialize(
        exported_program
    )
    assert isinstance(serialized_artifact.exported_program, schema.ExportedProgram)
    json_program = json.dumps(
        _dataclass_to_dict(serialized_artifact.exported_program),
        cls=export_serialize.EnumEncoder,
    )
    json_bytes = json_program.encode("utf-8")
    artifact = export_serialize.SerializedArtifact(
        json_bytes, serialized_artifact.state_dict, serialized_artifact.constants
    )
    return artifact


def deserialize(
    artifact: export_serialize.SerializedArtifact,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ep.ExportedProgram:
    assert isinstance(artifact.exported_program, bytes)
    exported_program_str = artifact.exported_program.decode("utf-8")
    exported_program_dict = json.loads(exported_program_str)
    serialized_exported_program = export_serialize._dict_to_dataclass(
        schema.ExportedProgram, exported_program_dict
    )
    return ExportedProgramDeserializer(expected_opset_version).deserialize(
        export_serialize.SerializedArtifact(
            serialized_exported_program, artifact.state_dict, artifact.constants
        )
    )


def save(
    ep_save: ep.ExportedProgram,
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    if not isinstance(ep_save, ep.ExportedProgram):
        raise TypeError(f"save() expects an ExportedProgram but got {type(ep)}")

    artifact: SerializedArtifact = serialize(ep_save, opset_version)

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    with zipfile.ZipFile(f, "w") as zipf:
        # Save every field in the SerializedArtifact to a file.
        assert isinstance(artifact.exported_program, bytes)
        zipf.writestr("serialized_exported_program.json", artifact.exported_program)
        zipf.writestr("serialized_state_dict.pt", artifact.state_dict)
        zipf.writestr("serialized_constants.pt", artifact.constants)

        zipf.writestr("version", ".".join(map(str, SCHEMA_VERSION)))

        # Add extra files if provided
        if extra_files:
            for extra_file_name, content in extra_files.items():
                encoded_content = content.encode("utf-8")
                zipf.writestr(f"extra_files/{extra_file_name}", encoded_content)


def load(
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ep.ExportedProgram:
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    extra_files = extra_files or {}

    with zipfile.ZipFile(f, "r") as zipf:
        # Check the version
        version = zipf.read("version").decode().split(".")

        assert len(version) == len(SCHEMA_VERSION)
        if version[0] != str(SCHEMA_VERSION[0]):
            raise RuntimeError(
                f"Serialized version {version} does not match our current "
                f"schema version {SCHEMA_VERSION}."
            )

        # Load serialized_ep and serialized_state_dict from the zip file

        serialized_exported_program: Optional[bytes] = None
        serialized_state_dict: Optional[bytes] = None
        serialized_constants: Optional[bytes] = None

        for file_info in zipf.infolist():
            file_content = zipf.read(file_info.filename)

            if file_info.filename == "serialized_exported_program.json":
                serialized_exported_program = file_content
            elif file_info.filename == "serialized_state_dict.json":
                print("This version of file is deprecated")
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.json":
                print("This version of file is deprecated")
                serialized_constants = file_content
            elif file_info.filename == "serialized_state_dict.pt":
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.pt":
                serialized_constants = file_content
            elif file_info.filename.startswith("extra_files"):
                filename = file_info.filename.split("/", 1)[1]
                extra_files[filename] = file_content.decode("utf-8")

        assert serialized_exported_program is not None
        assert serialized_state_dict is not None
        assert serialized_constants is not None
        artifact: SerializedArtifact = SerializedArtifact(
            serialized_exported_program,
            serialized_state_dict,
            serialized_constants,
        )

        # Deserialize ExportedProgram
        ep = deserialize(artifact, expected_opset_version)

        return ep
