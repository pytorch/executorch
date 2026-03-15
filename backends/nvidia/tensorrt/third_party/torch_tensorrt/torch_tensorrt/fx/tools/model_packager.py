from pathlib import Path
from typing import BinaryIO, Sequence, TextIO, Union

import torch
from torch.fx.passes.split_utils import getattr_recursive
from torch.package import PackageExporter

"""
A tool to package acc submodule as a torch package. The packaged model can be loaded
with just PyTorch library.
"""


def flatten_model(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Remove all original modules with an attr holder module so that all original modules
    and names are not present.
    """
    holder_module = torch.nn.Module()
    model._holder = holder_module
    attr_id = 0

    for node in model.graph.nodes:
        assert node.op != "call_module"
        if node.op == "get_attr":
            attr = getattr_recursive(model, node.target)
            setattr(holder_module, f"_attr_{attr_id}", attr)
            with model.graph.inserting_before(node):
                new_node = model.graph.get_attr(f"_holder._attr_{attr_id}")
                node.replace_all_uses_with(new_node)
            attr_id += 1

    model.graph.eliminate_dead_code()
    model.recompile()
    model.delete_all_unused_submodules()
    return model


def generate_standalone_repro(
    model: torch.fx.GraphModule, output: Union[str, Path, TextIO], prelude: str = ""
) -> None:
    """
    Generate a standalone python file for the model where weights are randomized
    and the model flattened.
    This only works if leaf nodes are only torch.nn modules.
    """
    model = flatten_model(model)

    INDENT = "    "
    lines = [
        "",
        "import torch",
        "from torch import nn",
    ]
    code = str(model.code)

    import_modules = set()
    import_map = {
        "torch_tensorrt_fx_tracer_acc_tracer_acc_ops": "torch_tensorrt.fx.tracer.acc_tracer.acc_ops",
        "torch_tensorrt_fx_passes_lower_basic_pass": "torch_tensorrt.fx.passes.lower_basic_pass",
    }
    for line in code.split("\n"):
        for k, v in import_map.items():
            if k in line:
                sub_string = line.split("(")[0].split()[-1]
                if sub_string.startswith(k):
                    mod = sub_string.replace(k + "_", "")
                    import_modules.add(
                        "from " + v + " import " + mod + " as " + sub_string
                    )
    for mod in sorted(import_modules):
        lines.append(mod)

    lines += [
        "",
        "",
        "class ExportedModule(nn.Module):",
        f"{INDENT}def __init__(self):",
        f"{INDENT * 2}super().__init__()",
    ]

    for k, v in model._holder.named_parameters():
        shape = ", ".join([str(i) for i in v.shape])
        rand_func = "randn" if torch.is_floating_point(v) else "randint"
        int_range = "" if torch.is_floating_point(v) else "0, 5, "
        lines.append(
            f"{INDENT * 2}self.{k} = nn.Parameter(torch.{rand_func}({int_range}{shape}, dtype={v.dtype}))"
        )

    def dump(f):
        f.write(prelude)
        f.write("\n".join(lines))
        f.write(
            "\n".join(
                [
                    INDENT + line.replace("self._holder.", "self.")
                    for line in code.split("\n")
                ]
            )
        )
        f.write("\n")

    if isinstance(output, (Path, str)):
        with open(str(output), "w") as f:
            dump(f)
    else:
        dump(output)


class ModelPackager:
    @classmethod
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        pe.extern(
            [
                "builtins",
                "sys",
                "torch.**",
            ]
        )

    @classmethod
    def set_mocked_modules(cls, pe: PackageExporter):
        pe.mock(
            "**",
            exclude=[
                "torch_tensorrt.fx.tracer.acc_tracer.acc_ops",
                "torch_tensorrt.fx.tracer.acc_tracer.acc_normalizer",
                "torch_tensorrt.fx.tracer.acc_tracer.acc_op_properties",
            ],
        )

    @classmethod
    def package_model(
        cls,
        model: torch.nn.Module,
        model_inputs: Sequence[torch.Tensor],
        output: Union[str, Path, BinaryIO],
        preserve_model_structure: bool = False,
    ) -> None:
        if not preserve_model_structure:
            model = flatten_model(model)
        with PackageExporter(output) as pe:
            cls.set_extern_modules(pe)
            cls.set_mocked_modules(pe)
            pe.intern("**")
            pe.save_pickle("repro", "model", model)
            pe.save_pickle("repro", "inputs", model_inputs)
