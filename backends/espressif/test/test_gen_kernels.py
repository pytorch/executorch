# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import executorch.version
import pytest
import torch
from executorch.codegen.tools import gen_oplist as gen_oplist_tool
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import to_edge
from executorch.exir.backend.canonical_partitioners.all_node_partitioner import (
    AllNodePartitioner,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (  # noqa: F401
    BackendWithCompilerDemo,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "examples/espressif/idf_component_template/tools/gen_kernels.py"
LIBRARIES = [
    ("portable", "kernels/portable/functions.yaml"),
    ("quantized", "kernels/quantized/quantized.yaml"),
    ("cadence", "backends/cadence/aot/functions.yaml"),
    ("esp_nn", "backends/espressif/ops/operators.yaml"),
]
REGISTRATION = "RegisterCodegenUnboxedKernelsEverything.cpp"
MAX_KERNEL_HEADER = "include/executorch/runtime/kernel/selected_max_kernel_num.h"

_spec = importlib.util.spec_from_file_location("gen_kernels", TOOL)
gk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gk)


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class AddMul(torch.nn.Module):
    def forward(self, x, y):
        return (x + y) * y


ARGS = (torch.ones(2), torch.ones(2))


def export(module=None):
    return to_edge(torch.export.export(module or AddMul(), ARGS)).to_executorch()


def save_pte(path, module=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export(module).buffer)
    return path


def gen_oplist(path, **kwargs):
    gen_oplist_tool.gen_oplist(str(path), **kwargs)
    return path


@pytest.fixture
def component(tmp_path):
    comp = tmp_path / "component"
    shutil.copytree(REPO / "codegen/templates", comp / "tools/codegen/templates")
    libraries = {
        "schema": 1,
        "prim_ops_source": str(REPO / "kernels/prim_ops/register_prim_ops.cpp"),
        "libraries": [{"name": n, "yaml": str(REPO / y)} for n, y in LIBRARIES],
    }
    (comp / "tools/kernel_libraries.json").write_text(json.dumps(libraries))
    version = executorch.version.__version__.split("+")[0]
    (comp / "idf_component.yml").write_text(f'version: "{version}"\n')
    (comp / "VERSION").write_text(
        f"executorch_commit: {executorch.version.git_version}\n"
    )
    return comp


def run(component, out, *args):
    return gk.main(["--component-dir", str(component), "--out", str(out), *args])


def read_bundle(out):
    return json.loads((out / "kernels.json").read_text())


def test_canonical_op_name():
    assert gk.canonical_op_name({"op": "add.out"}) == "aten::add.out"
    assert (
        gk.canonical_op_name(
            {"func": "quantized_decomposed::add.out(Tensor a) -> Tensor"}
        )
        == "quantized_decomposed::add.out"
    )


def test_assign_ops_priority():
    libraries = [
        ("portable", {"aten::add.out"}),
        ("cadence", {"aten::add.out", "cadence::quantized_relu.per_tensor_out"}),
    ]
    ops = {
        "aten::add.out",
        "cadence::quantized_relu.per_tensor_out",
        "aten::sym_size.int",
        "aten::addd.out",
    }
    by_library, prims, unknown = gk.assign_ops(ops, libraries, {"aten::sym_size.int"})
    assert by_library == {
        "portable": ["aten::add.out"],
        "cadence": ["cadence::quantized_relu.per_tensor_out"],
    }
    assert prims == ["aten::sym_size.int"]
    assert unknown == ["aten::addd.out"]


def test_differing_duplicate_declarations_fail(component, tmp_path, capsys):
    entry = "- op: add.out\n  kernels:\n    - arg_meta: null\n      kernel_name: {}\n"
    libraries = []
    for name, kernel in (("first", "torch::executor::add_out"), ("second", "x::add")):
        (tmp_path / f"{name}.yaml").write_text(entry.format(kernel))
        libraries.append({"name": name, "yaml": str(tmp_path / f"{name}.yaml")})
    config = json.loads((component / "tools/kernel_libraries.json").read_text())
    config["libraries"] = libraries
    (component / "tools/kernel_libraries.json").write_text(json.dumps(config))
    assert run(component, tmp_path / "kernels", "--ops", "aten::add.out") == 1
    err = capsys.readouterr().err
    assert "aten::add.out" in err
    assert "first" in err and "second" in err


def test_add_mul_model(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--pte", str(save_pte(tmp_path / "m.pte"))) == 0
    assert read_bundle(out)["libraries"] == {
        "portable": ["aten::add.out", "aten::mul.out"]
    }
    registration = (out / "portable" / REGISTRATION).read_text()
    assert '"aten::add.out"' in registration
    assert "native::add_out(" in registration


def test_bundle_records_the_component_commit(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out") == 0
    assert read_bundle(out)["component_commit"] == executorch.version.git_version


def test_ops_route_to_three_libraries(component, tmp_path):
    out = tmp_path / "kernels"
    ops = (
        "aten::add.out,quantized_decomposed::quantize_per_tensor.out,"
        "cadence::quantized_relu.per_tensor_out"
    )
    assert run(component, out, "--ops", ops) == 0
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "cadence",
        "include",
        "portable",
        "quantized",
    ]
    assert '"aten::add.out"' not in (out / "cadence" / REGISTRATION).read_text()


def test_unknown_op_fails(component, tmp_path, capsys):
    assert run(component, tmp_path / "kernels", "--ops", "aten::addd.out") == 2
    err = capsys.readouterr().err
    assert "aten::addd.out" in err
    assert "--ops" in err


def test_fully_delegated_model(component, tmp_path):
    edge = to_edge(torch.export.export(Add(), ARGS)).to_backend(
        AllNodePartitioner(
            "BackendWithCompilerDemo", [CompileSpec("max_value", bytes([4]))]
        )
    )
    pte = tmp_path / "delegated.pte"
    pte.write_bytes(edge.to_executorch().buffer)
    out = tmp_path / "kernels"
    assert run(component, out, "--pte", str(pte)) == 0
    bundle = read_bundle(out)
    assert bundle["libraries"] == {}
    assert bundle["prim_ops"] == []
    assert bundle["backends"] == ["BackendWithCompilerDemo"]
    # gen_max_kernel_num counts every prim; the parse must find them all.
    _, _, prims = gk._load_libraries(component)
    assert f"MAX_KERNEL_NUM {len(prims)}\n" in (out / MAX_KERNEL_HEADER).read_text()


def test_prim_ops_selected(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out,aten::sym_size.int") == 0
    assert read_bundle(out)["prim_ops"] == ["aten::sym_size.int"]
    assert (
        "#define INCLUDE_ATEN_SYM_SIZE_INT\n"
        in (out / "selected_prim_ops.h").read_text()
    )


def test_records_model_hash(component, tmp_path):
    pte = save_pte(tmp_path / "models" / "a.pte")
    out = tmp_path / "kernels"
    assert run(component, out, "--pte", str(pte)) == 0
    assert read_bundle(out)["models"] == [
        {"name": "a.pte", "sha256": hashlib.sha256(pte.read_bytes()).hexdigest()}
    ]


def test_records_requested_ops(component, tmp_path):
    pte = str(save_pte(tmp_path / "m.pte"))
    oplist = gen_oplist(tmp_path / "ops.yaml", root_ops="aten::div.out")
    out = tmp_path / "kernels"
    assert run(component, out, "--pte", pte) == 0
    assert read_bundle(out)["ops"] == []
    args = ["--pte", pte, "--ops", "aten::sub.out", "--oplist", str(oplist)]
    assert run(component, out, *args) == 0
    assert read_bundle(out)["ops"] == ["aten::div.out", "aten::sub.out"]


def test_deterministic(component, tmp_path):
    pte = str(save_pte(tmp_path / "m.pte"))
    for out in ("a", "b"):
        assert (
            run(component, tmp_path / out, "--pte", pte, "--ops", "aten::sub.out") == 0
        )

    def tree(root):
        return {
            p.relative_to(root): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    assert tree(tmp_path / "a") == tree(tmp_path / "b")


def test_regenerate_removes_stale_library(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "cadence::quantized_relu.per_tensor_out") == 0
    assert (out / "cadence").is_dir()
    assert run(component, out, "--ops", "aten::add.out") == 0
    assert not (out / "cadence").exists()


def test_refuses_non_bundle_out_dir(component, tmp_path):
    out = tmp_path / "src"
    out.mkdir()
    (out / "main.c").write_text("int main;\n")
    assert run(component, out, "--ops", "aten::add.out") == 2
    assert [p.name for p in out.iterdir()] == ["main.c"]


def test_refuses_bundle_with_foreign_files(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out") == 0
    (out / "notes.txt").write_text("mine\n")
    assert run(component, out, "--ops", "aten::mul.out") == 2
    assert (out / "notes.txt").read_text() == "mine\n"


@pytest.mark.parametrize("name", ["include/application.h", "portable/notes.txt"])
def test_refuses_bundle_with_nested_foreign_files(component, tmp_path, name):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out") == 0
    (out / name).write_text("mine\n")
    assert run(component, out, "--ops", "aten::mul.out") == 2
    assert (out / name).read_text() == "mine\n"


@pytest.mark.parametrize(
    "name", [".DS_Store", "portable/.DS_Store", "Thumbs.db", "desktop.ini"]
)
def test_regenerate_replaces_os_metadata(component, tmp_path, name):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out") == 0
    (out / name).write_text("")
    assert run(component, out, "--ops", "aten::mul.out") == 0
    assert not (out / name).exists()


def test_refuses_model_inside_out(component, tmp_path):
    out = tmp_path / "kernels"
    assert run(component, out, "--ops", "aten::add.out") == 0
    pte = save_pte(out / "m.pte")
    assert run(component, out, "--pte", str(pte)) == 2
    assert pte.is_file()


def test_out_is_a_file(component, tmp_path, capsys):
    out = tmp_path / "kernels"
    out.write_text("")
    assert run(component, out, "--ops", "aten::add.out") == 2
    assert "error:" in capsys.readouterr().err


def test_requested_ops_from_buck_shaped_oplist(component, tmp_path):
    # gen_all_oplist labels a file under this layout with its Buck target.
    buck_out = (
        tmp_path / "x/buck-out/v2/gen/fbsource/0123abcd/app/__ops_et_oplist__/out"
    )
    buck_out.mkdir(parents=True)
    oplist = gen_oplist(buck_out / "selected_operators.yaml", root_ops="aten::div.out")
    out = tmp_path / "kernels"
    assert run(component, out, "--oplist", str(oplist)) == 0
    assert read_bundle(out)["ops"] == ["aten::div.out"]


def test_oplist_input(component, tmp_path):
    oplist = gen_oplist(tmp_path / "ops.yaml", root_ops="aten::mul.out")
    out = tmp_path / "kernels"
    assert run(component, out, "--oplist", str(oplist)) == 0
    assert read_bundle(out)["libraries"] == {"portable": ["aten::mul.out"]}


def test_include_all_operators_refused(component, tmp_path, capsys):
    oplist = gen_oplist(tmp_path / "all.yaml", include_all_operators=True)
    assert run(component, tmp_path / "kernels", "--oplist", str(oplist)) == 2
    assert "all operators" in capsys.readouterr().err


def test_espnn_ci_program_ops_route(component, tmp_path):
    pytest.importorskip("torchao")
    programs = tmp_path / "programs"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "examples/espressif/export_espnn_test_programs.py"),
            "--output-dir",
            str(programs),
            "--only",
            "tiny-transformer,lstm-keywords,cpu-island-between-delegates,"
            "cumsum-on-cpu,transpose-select-stack",
        ],
        check=True,
        capture_output=True,
    )
    ops = (programs / "operators.txt").read_text().strip()
    assert run(component, tmp_path / "kernels", "--ops", ops) == 0


def test_bundled_program_rejected(component, tmp_path, capsys):
    bpte = tmp_path / "m.bpte"
    test_case = MethodTestCase(list(ARGS), AddMul()(*ARGS))
    bundle = BundledProgram(export(), [MethodTestSuite("forward", [test_case])])
    bpte.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))
    assert run(component, tmp_path / "kernels", "--pte", str(bpte)) == 2
    err = capsys.readouterr().err
    assert "bundled program" in err
    assert ".pte" in err


def test_paths_with_spaces(component, tmp_path, monkeypatch):
    scratch = tmp_path / "my tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    pte = save_pte(tmp_path / "my models" / "m.pte")
    out = tmp_path / "my kernels"
    assert run(component, out, "--pte", str(pte)) == 0
    assert read_bundle(out)["models"][0]["name"] == "m.pte"


def test_version_mismatch_fails(component, tmp_path, capsys):
    (component / "idf_component.yml").write_text('version: "0.0.1"\n')
    (component / "VERSION").write_text("executorch_commit: 0123456789ab\n")
    assert run(component, tmp_path / "kernels", "--ops", "aten::add.out") == 2
    err = capsys.readouterr().err
    assert "0.0.1" in err
    assert executorch.version.__version__ in err


def test_version_mismatch_without_git_sha(component, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(executorch.version, "git_version", None)
    (component / "idf_component.yml").write_text('version: "0.0.1"\n')
    assert run(component, tmp_path / "kernels", "--ops", "aten::add.out") == 2
    assert "0.0.1" in capsys.readouterr().err


def test_versions_match():
    assert gk.versions_match("1.6.0", "s", "1.6.0", "t")
    assert gk.versions_match("1.6.0+cpu", "s", "1.6.0~2", "t")
    assert gk.versions_match("1.6.0+9bf7e25", "abc", "1.6.0~1", "abc")
    assert not gk.versions_match("1.6.0+9bf7e25", "s", "1.6.0", "t")
    assert not gk.versions_match("1.6.0+9bf7e25", "", "1.6.0", "")
    assert gk.versions_match("1.7.0.dev20261104", "abc", "1.7.0-dev.20261104", "abc")
    assert not gk.versions_match("1.6.1", "s", "1.6.0", "t")
    assert not gk.versions_match("1.7.0.dev1", "abc", "1.7.0-dev.1", "def")
