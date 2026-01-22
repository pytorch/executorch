# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json

from dataclasses import dataclass
from pathlib import Path

from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, VgfQuantizer
from executorch.backends.arm.vgf.partitioner import VgfCompileSpec, VgfPartitioner


@dataclass
class DocumentationJob:
    template_path: Path
    placeholder: str
    replacement_text: str
    output_path: Path


def get_docstring(obj) -> str:
    """
    Returns the docstring of an object, formatted in markdown to resemble pytorch-style
    docs, e.g. an argument list like:

    Args:
        arg1: description

    is converted to

    Args:
    - **arg1**: description
    - **arg2**: description
    """

    docstring = inspect.getdoc(obj)
    if docstring is None:
        print(
            f"WARNING: Docstring not found for object '{obj.__name__}'. Please document all user facing components of the Arm Backend properly."
        )
        docstring = ""

    lines = docstring.split("\n")
    for line in lines:
        if ":" in line and line.startswith(" "):
            new_line = line.strip()
            pos = new_line.index(":")
            new_line = f"- **{new_line[:pos]}**" + new_line[pos:]
            docstring = docstring.replace(line, new_line)

    return docstring


def get_function_docstring(cls, func) -> str:
    """
    Returns a function's signature and docstring formatted in markdown.
    """
    return f"```python\ndef {cls.__name__}.{func.__name__}{inspect.signature(func)}:\n```\n{get_docstring(func)}\n\n"


def get_class_docstring(cls, filter_funcs=()) -> str:
    """
    Returns a class signature and docstring, as well as documentation for all its public
    methods with names not matching strings listed in filter_funcs.
    """
    header = f"```python\nclass {cls.__name__}{inspect.signature(cls)}\n```\n{get_docstring(cls)}\n\n"

    class_functions = [
        getattr(cls, name)
        for name in dir(cls)
        if callable(getattr(cls, name))
        and not name.startswith("_")
        and not any(f in name for f in filter_funcs)
    ]

    function_docstrings = [
        get_function_docstring(cls, func) for func in class_functions
    ]

    return header + "".join(function_docstrings)


def get_jupyter_code(path, get_bash, which_cells: list[int] | None = None) -> str:
    """
    Returns all code cells from the jupyter notebook at 'path'. If get_bash is True,
    only bash cells are returned, otherwise only python cells are returned.
    which_cells lets you supply a list of cell indicies to return.
    """
    output = f"```{'bash' if get_bash else 'python'}\n"
    with open(path, "r") as f:
        j = json.load(f)
        i = -1
        for cell in j["cells"]:
            is_code = cell["cell_type"] == "code"
            if len(cell["source"]) == 0:
                continue
            is_bash = "bash" in cell["source"][0]
            is_copyright = "Copyright" in cell["source"][0]
            if is_code and is_bash == get_bash and not is_copyright:
                i += 1
                if which_cells is not None:
                    if i not in which_cells:
                        continue
                for line in cell["source"]:
                    is_print = "print_readable" in line
                    is_bash_line = "bash" in line
                    is_setup_path = "setup_path.sh" in line
                    if not (is_print or is_bash_line or is_setup_path):
                        output += line
                output += "\n"
    output += "```\n"
    return output


def generate_document(job: DocumentationJob):
    """Generates a markdown document based on a DocumentationJob."""
    with open(job.template_path, "r") as f:
        content = f.read()

    content = content.replace(job.placeholder, job.replacement_text)

    # Remove multiple new lines at end of document if it exists
    if content.endswith("\n\n"):
        content = content.removesuffix("\n")

    with open(job.output_path, "w") as f:
        f.write(content)


def generate_ethos_u_docs():
    """Generates documentation for the Ethos-U components in the backend."""
    compilespec_string = get_class_docstring(
        EthosUCompileSpec,
        ("DebugMode", "to_list", "from_list", "from_list_hook", "validate"),
    )
    partitioner_string = get_class_docstring(EthosUPartitioner)
    quantizer_string = get_class_docstring(
        EthosUQuantizer, ("prepare_obs_or_fq_callback", "annotate", "validate")
    )
    example_string = get_jupyter_code(
        "./examples/arm/ethos_u_minimal_example.ipynb", get_bash=False
    )

    documentation_jobs = [
        DocumentationJob(
            Path(
                "backends/arm/scripts/docgen/ethos-u/backends-arm-ethos-u-overview.md.in"
            ),
            "$COMPILE_SPEC",
            compilespec_string,
            Path("docs/source/backends/arm-ethos-u/arm-ethos-u-overview.md"),
        ),
        DocumentationJob(
            Path(
                "backends/arm/scripts/docgen/ethos-u/backends-arm-ethos-u-partitioner.md.in"
            ),
            "$PARTITIONER",
            partitioner_string,
            Path("docs/source/backends/arm-ethos-u/arm-ethos-u-partitioner.md"),
        ),
        DocumentationJob(
            Path(
                "backends/arm/scripts/docgen/ethos-u/backends-arm-ethos-u-quantization.md.in"
            ),
            "$QUANTIZER",
            quantizer_string,
            Path("docs/source/backends/arm-ethos-u/arm-ethos-u-quantization.md"),
        ),
        DocumentationJob(
            Path(
                "backends/arm/scripts/docgen/ethos-u/ethos-u-getting-started-tutorial.md.in"
            ),
            "$MINIMAL_EXAMPLE",
            example_string,
            Path(
                "docs/source/backends/arm-ethos-u/tutorials/ethos-u-getting-started.md"
            ),
        ),
    ]

    for job in documentation_jobs:
        generate_document(job)


def generate_vgf_docs():
    """Generates documentation for the VGF components in the backend."""
    compilespec_string = get_class_docstring(
        VgfCompileSpec,
        ("DebugMode", "to_list", "from_list", "from_list_hook", "validate"),
    )
    partitioner_string = get_class_docstring(VgfPartitioner)
    quantizer_string = get_class_docstring(
        VgfQuantizer, ("prepare_obs_or_fq_callback", "annotate", "validate")
    )
    example_string = get_jupyter_code(
        "./examples/arm/vgf_minimal_example.ipynb",
        get_bash=False,
        which_cells=[0, 2, 3],
    )

    documentation_jobs = [
        DocumentationJob(
            Path("backends/arm/scripts/docgen/vgf/backends-arm-vgf-overview.md.in"),
            "$COMPILE_SPEC",
            compilespec_string,
            Path("docs/source/backends/arm-vgf/arm-vgf-overview.md"),
        ),
        DocumentationJob(
            Path("backends/arm/scripts/docgen/vgf/backends-arm-vgf-partitioner.md.in"),
            "$PARTITIONER",
            partitioner_string,
            Path("docs/source/backends/arm-vgf/arm-vgf-partitioner.md"),
        ),
        DocumentationJob(
            Path("backends/arm/scripts/docgen/vgf/backends-arm-vgf-quantization.md.in"),
            "$QUANTIZER",
            quantizer_string,
            Path("docs/source/backends/arm-vgf/arm-vgf-quantization.md"),
        ),
        DocumentationJob(
            Path("backends/arm/scripts/docgen/vgf/vgf-getting-started-tutorial.md.in"),
            "$MINIMAL_EXAMPLE",
            example_string,
            Path("docs/source/backends/arm-vgf/tutorials/vgf-getting-started.md"),
        ),
    ]

    for job in documentation_jobs:
        generate_document(job)


def generate_ethosu_tutorial():
    """Generates the tutorial for the Ethos-U minimal example."""
    ethosu_example = get_jupyter_code(
        "./examples/arm/ethos_u_minimal_example.ipynb", get_bash=False
    )
    doc = "tutorial-arm-ethos-u.md"
    with open(f"backends/arm/scripts/docgen/{doc}.in", "r") as f:
        content = f.read()
        content = content.replace("$MINIMAL_EXAMPLE", ethosu_example)

    with open(f"docs/source/{doc}", "w") as f:
        f.write(content)


def generate_vgf_tutorial():
    """Generates the tutorial for the VGF minimal example."""
    vgf_example = get_jupyter_code(
        "./examples/arm/vgf_minimal_example.ipynb",
        get_bash=False,
        which_cells=[0, 2, 3],
    )

    doc = "tutorial-arm-vgf.md"
    with open(f"backends/arm/scripts/docgen/{doc}.in", "r") as f:
        content = f.read()
        content = content.replace("$MINIMAL_EXAMPLE", vgf_example)

    with open(f"docs/source/{doc}", "w") as f:
        f.write(content)


if __name__ == "__main__":
    generate_ethos_u_docs()
    generate_vgf_docs()
