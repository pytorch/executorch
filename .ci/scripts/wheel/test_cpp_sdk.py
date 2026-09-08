# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that a standalone C++ application can use the wheel as an SDK.

The wheel ships prebuilt runtime, kernel, delegate, thread pool and profiler
libraries, plus headers and a CMake package config. A Python test can exercise none
of that: the Python extension links those libraries itself, so it passes whether or
not the package config names them correctly, whether or not the headers are complete,
and whether or not an application that links them can find them at run time.

So these checks build and run a real application from outside the wheel. Nothing here
uses the source tree, because a user has only the installed package.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Exports the model to a .pte and prints the reference outputs, so the C++ side can
# be compared against eager PyTorch rather than merely checked for not crashing.
#
# The same network as the Python parity check: several operator kinds, so a run
# exercises the merged CPU kernels rather than a single add.
_EXPORT_SCRIPT = """
import json
import sys
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, x, image):
        a = torch.relu(self.linear(x))
        b = self.conv(image).flatten(1)
        return a.sum(dim=1, keepdim=True) + b.mean(dim=1, keepdim=True)


destination, mode = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
model = Net().eval()
example = (torch.randn(2, 8), torch.randn(2, 1, 6, 6))
with torch.no_grad():
    expected = model(*example)

if mode == "quantized":
    # Quantize with the same flow the documentation shows, so the exported program
    # references the quantized operator set rather than the plain one.
    # Importing this loads the ahead-of-time library, which is what registers the out
    # variants of the quantized operators with torch. Without it the export fails with
    # "Missing out variants: quantized_decomposed::quantize_per_tensor", because the
    # lowering step has no out variant to select.
    # Loaded directly rather than through executorch.kernels.quantized, whose __init__
    # swallows every exception, so a load failure would otherwise appear much later as
    # "Missing out variants" with no indication of why.
    import executorch as _executorch

    _root = Path(list(_executorch.__path__)[0]) / "kernels" / "quantized"
    _libs = sorted(_root.glob("*quantized_ops_aot_lib.*"))
    assert len(_libs) == 1, f"expected one ahead-of-time library, found {_libs}"
    torch.ops.load_library(str(_libs[0]))
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(torch.export.export(model, example).module(), quantizer)
    prepared(*example)
    model = convert_pt2e(prepared)

partitioners = []
if mode == "delegate":
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    partitioners = [XnnpackPartitioner()]

program = to_edge_transform_and_lower(
    torch.export.export(model, example), partitioner=partitioners
).to_executorch()
buffer = program.buffer
with open(destination, "wb") as handle:
    handle.write(buffer)

# The inputs travel with the model so the C++ side feeds identical values. Written as
# plain text rather than a tensor format, because reading one is not what is under
# test here and a dependency on one would be a second thing that can fail.
print(
    json.dumps(
        {
            "inputs": [
                {"shape": list(t.shape), "data": t.flatten().tolist()}
                for t in example
            ],
            "expected": expected.flatten().tolist(),
            "delegated": mode == "delegate",
            "has_xnnpack": b"XnnpackBackend" in bytes(buffer),
            # Whether the program actually carries quantized operators. The numeric comparison alone
            # cannot tell: an unquantized export of the same model produces a closer match than the
            # tolerance a quantized one needs, so it would pass while proving nothing about the
            # quantized kernels.
            "has_quantized": b"quantized_decomposed" in bytes(buffer),
        }
    )
)
"""


_CONSUMER_SOURCE = r"""
// A standalone application. It includes only what the wheel installs and links only
// the wheel's imported targets, so it fails if the shipped headers are incomplete or
// the package config does not make the libraries findable at run time.
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace executorch::extension;

namespace {

// A minimal reader for the numbers the export step printed. Deliberately not a JSON
// library: adding a dependency here would mean a failure could come from the parser
// rather than from the SDK.
std::vector<float> read_floats(const std::string& path) {
  std::ifstream file(path);
  std::vector<float> values;
  float value = 0.0f;
  while (file >> value) {
    values.push_back(value);
  }
  return values;
}

std::vector<int> read_ints(const std::string& path) {
  std::ifstream file(path);
  std::vector<int> values;
  int value = 0;
  while (file >> value) {
    values.push_back(value);
  }
  return values;
}

} // namespace

int main(int argc, char** argv) {
  // Seven, because argv[6] is read below: the program name plus six arguments. A guard of six let a
  // caller that passed one too few read past the end of argv.
  if (argc < 7) {
    std::printf("usage: consumer <pte> <shape0> <data0> <shape1> <data1> <expected>\n");
    return 2;
  }
  executorch::runtime::runtime_init();

  const auto shape_a = read_ints(argv[2]);
  auto data_a = read_floats(argv[3]);
  const auto shape_b = read_ints(argv[4]);
  auto data_b = read_floats(argv[5]);
  const auto expected = read_floats(argv[6]);

  std::vector<executorch::aten::SizesType> sizes_a(shape_a.begin(), shape_a.end());
  std::vector<executorch::aten::SizesType> sizes_b(shape_b.begin(), shape_b.end());

  // The documented entry points, not the lower-level runtime API. Constructing these
  // needs real definitions at link time, so it checks that the shipped headers and
  // the shipped libraries agree rather than only that the headers parse.
  module::Module module(argv[1]);
  const auto load_error = module.load();
  if (load_error != executorch::runtime::Error::Ok) {
    std::printf("load failed: 0x%x\n", (unsigned)load_error);
    return 1;
  }

  auto input_a = make_tensor_ptr(sizes_a, data_a.data());
  auto input_b = make_tensor_ptr(sizes_b, data_b.data());

  const auto result = module.forward({input_a, input_b});
  if (!result.ok()) {
    std::printf("forward failed: 0x%x\n", (unsigned)result.error());
    return 1;
  }

  const auto output = result->at(0).toTensor();
  if ((size_t)output.numel() != expected.size()) {
    std::printf(
        "output has %zu values, expected %zu\n",
        (size_t)output.numel(),
        expected.size());
    return 1;
  }

  // Compared against eager PyTorch, not merely produced. A model that returns wrong
  // numbers without erroring would satisfy every other check here.
  const float* actual = output.const_data_ptr<float>();
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double diff = std::fabs((double)actual[i] - (double)expected[i]);
    // Rejected here rather than through the comparison below, because fmax treats a
    // NaN as a missing value and returns the other operand, so an all-NaN output
    // would leave the running maximum at zero and pass at any tolerance.
    if (!std::isfinite(diff)) {
      std::printf("output value %zu is not comparable: %g\n", i, (double)actual[i]);
      return 1;
    }
    worst = std::fmax(worst, diff);
  }
  // Passed in rather than fixed, because the acceptable difference depends on the
  // model. A float32 model should match to within rounding, while an int8 quantized one
  // legitimately differs by about one quantization step, and using the looser number
  // for both would stop the float path catching a real regression.
  const double tolerance = argc > 7 ? std::atof(argv[7]) : 1e-4;
  if (worst > tolerance) {
    std::printf(
        "output differs from eager PyTorch by %g, tolerance %g\n", worst, tolerance);
    return 1;
  }

  std::printf(
      "ok backends=%zu maxdiff=%g\n",
      (size_t)executorch::runtime::get_num_registered_backends(),
      worst);
  return 0;
}
"""


def _consumer_cmake(components) -> str:
    """A consumer project that links the given components by their public names.

    REQUIRED COMPONENTS rather than a bare find_package, because that is the form the
    documentation shows and it has to fail loudly when the wheel does not ship what
    it advertises.
    """
    requested = " ".join(components)
    links = "\n".join(
        f"target_link_libraries(consumer PRIVATE executorch::{name})"
        for name in components
    )
    return f"""cmake_minimum_required(VERSION 3.28)
project(consumer CXX)
find_package(executorch REQUIRED COMPONENTS {requested})
add_executable(consumer consumer.cpp)
{links}
"""


def _mach_o_runtime_paths(binary) -> list:
    """The runtime search path entries a Mach-O binary records.

    Mach-O keeps one entry per LC_RPATH load command, where ELF keeps a single colon joined
    string, so they are read rather than split.
    """
    otool = _tool("otool")
    listing = subprocess.run(
        [otool, "-l", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    entries = []
    lines = listing.splitlines()
    for index, line in enumerate(lines):
        if "LC_RPATH" not in line:
            continue
        for following in lines[index + 1 : index + 4]:
            stripped = following.strip()
            if stripped.startswith("path "):
                entries.append(stripped.split(" (offset", 1)[0][len("path ") :])
                break
    return entries


def _dynamic_lib_suffix() -> str:
    """The loadable library suffix on this platform, including the dot."""
    return ".dylib" if sys.platform == "darwin" else ".so"


def _library_file_name(base_name: str) -> str:
    """The file name a library has on this platform."""
    return f"{base_name}{_dynamic_lib_suffix()}"


def _recorded_dependencies(binary) -> str:
    """What a built binary records about its dependencies and search paths.

    readelf prints the ELF dynamic section, otool -l the Mach-O load commands. Both
    carry the same facts: a dependency entry and a runtime search path entry, named
    NEEDED and RUNPATH on ELF, LC_LOAD_DYLIB and LC_RPATH on Mach-O.
    """
    if sys.platform == "darwin":
        tool, args = _tool("otool"), ["-l"]
        needed = "otool"
    else:
        tool, args = _tool("readelf"), ["-d"]
        needed = "readelf"
    assert tool is not None, f"{needed} is needed to read the runtime search path"
    return subprocess.run(
        [tool, *args, str(binary)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _tool(name: str) -> str:
    """Locate a build tool, including one pip installed beside this interpreter.

    `shutil.which` searches PATH only, and a virtual environment's bin is on PATH only
    when the environment is activated. These checks run by invoking the interpreter
    directly, so a tool installed into that environment is present on disk and
    invisible to a PATH search.
    """
    # _tool returns the bare name when a PATH search fails, so it never returns None and an
    # `is not None` assert on its result can never fire. Callers check the subprocess result
    # instead, which is what actually surfaces a missing tool.
    found = shutil.which(name)
    if found:
        return found
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else name


def _loader_clean_environment() -> dict:
    """The environment with every loader override removed.

    Making the shipped libraries findable is the package config's job, and a
    search path or an injected library inherited from the environment does that
    job instead, hiding a failure to do it. Both spellings go on both platforms:
    the one that does not apply is absent rather than harmful, and naming only
    the ELF variable is what left the macOS runs honouring DYLD_LIBRARY_PATH.
    """
    overrides = (
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    )
    return {key: value for key, value in os.environ.items() if key not in overrides}


def _installed_package_dir() -> Path:
    """Where the wheel installed itself, found without importing it.

    Imported from the source tree, `executorch.__file__` points at the checkout rather
    than at the installed package, so a check would read the wrong files and pass while
    the wheel was broken.
    """
    for entry in sys.path:
        candidate = Path(entry) / "executorch"
        if (candidate / "share" / "cmake").is_dir():
            return candidate
    raise AssertionError(
        "no installed executorch package with share/cmake on sys.path; these checks "
        "must run against an installed wheel, not the source tree"
    )


def _write_tensor(directory: Path, stem: str, tensor) -> tuple:
    """Write one tensor's shape and values as whitespace-separated text."""
    shape_file = directory / f"{stem}.shape"
    data_file = directory / f"{stem}.data"
    shape_file.write_text(" ".join(str(n) for n in tensor["shape"]))
    data_file.write_text(" ".join(repr(v) for v in tensor["data"]))
    return shape_file, data_file


def _export(work_dir: Path, mode: str) -> tuple:
    """Export the model to a .pte and return it with the reference numbers."""
    script = work_dir / "export.py"
    script.write_text(_EXPORT_SCRIPT)
    model = work_dir / f"model_{mode}.pte"
    result = subprocess.run(
        [sys.executable, str(script), str(model), mode],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"exporting the {mode} model failed, so the C++ side cannot be checked "
        f"against it:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    reference = json.loads(result.stdout.strip().splitlines()[-1])
    assert model.is_file(), f"the export step produced no {model}"
    return model, reference


def _build_consumer(work_dir: Path, name: str, components) -> Path:
    """Configure and build the consumer application against the installed package."""
    package_dir = _installed_package_dir()
    config = package_dir / "share" / "cmake" / "executorch-config.cmake"
    assert config.is_file(), f"the wheel ships no CMake package config at {config}"

    source_dir = work_dir / name
    build_dir = work_dir / f"{name}-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(_CONSUMER_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_consumer_cmake(components))

    configured = subprocess.run(
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config.parent}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configured.returncode == 0, (
        f"a consumer requesting {list(components)} could not configure against the "
        f"installed package:\n{configured.stdout[-2000:]}\n{configured.stderr[-2000:]}"
    )
    built = subprocess.run(
        [_tool("cmake"), "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, (
        f"a consumer requesting {list(components)} compiled against the shipped "
        f"headers but did not link:\n{built.stdout[-3000:]}\n{built.stderr[-3000:]}"
    )
    consumer = build_dir / "consumer"
    assert consumer.is_file(), f"the build produced no {consumer}"
    return consumer


def _run_consumer(
    consumer: Path, model: Path, reference, work_dir: Path, tolerance: float = 1e-4
) -> str:
    """Run the application and require it to match eager PyTorch within `tolerance`.

    The tolerance is a parameter because the acceptable difference depends on the model.
    A float32 model should match to within rounding, while an int8 quantized one
    legitimately differs by about one quantization step, and using the looser number for
    both would stop the float path catching a real regression.
    """
    inputs = reference["inputs"]
    shape_a, data_a = _write_tensor(work_dir, "a", inputs[0])
    shape_b, data_b = _write_tensor(work_dir, "b", inputs[1])
    expected = work_dir / "expected.data"
    expected.write_text(" ".join(repr(v) for v in reference["expected"]))

    environment = _loader_clean_environment()
    result = subprocess.run(
        [
            str(consumer),
            str(model),
            str(shape_a),
            str(data_a),
            str(shape_b),
            str(data_b),
            str(expected),
            str(tolerance),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, (
        "the C++ application built against the installed wheel did not run "
        f"correctly:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    return result.stdout.strip()


def test_runtime_alone_links_but_cannot_compute(work_dir: Path) -> None:
    """Linking only the runtime builds and loads, and cannot execute.

    Measured rather than assumed: libexecutorch.so defines only primitive operators, such
    as aten::sym_size.int, and none of the kernels a model computes with, so an
    application linking it alone loads a program and then reports the operators it needs
    as missing. That is the intended split, and stating it here documents why the kernels
    are a separate component instead of leaving a reader to guess.

    The value of the check is the boundary. It fails if the runtime silently starts
    carrying model kernels again, which would mean the split had regressed, and it fails
    if the runtime cannot even load a program.
    """
    model, reference = _export(work_dir, "plain")
    consumer = _build_consumer(work_dir, "runtime-only", ["runtime"])

    inputs = reference["inputs"]
    shape_a, data_a = _write_tensor(work_dir, "ra", inputs[0])
    shape_b, data_b = _write_tensor(work_dir, "rb", inputs[1])
    expected = work_dir / "r_expected.data"
    expected.write_text(" ".join(repr(v) for v in reference["expected"]))
    environment = _loader_clean_environment()
    result = subprocess.run(
        [
            str(consumer),
            str(model),
            str(shape_a),
            str(data_a),
            str(shape_b),
            str(data_b),
            str(expected),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        "an application linking only executorch::runtime executed a model, so the "
        "runtime is carrying operator kernels that are supposed to live in their own "
        "component"
    )
    assert "Missing operator" in combined, (
        "an application linking only the runtime failed for some reason other than "
        f"absent kernels, which is the documented behaviour:\n{combined[-1500:]}"
    )
    print("✓ executorch::runtime alone links and loads, and has no model kernels")


def test_kernels_component_runs_a_model(work_dir: Path) -> None:
    """Adding the CPU kernels component keeps the model running and correct."""
    model, reference = _export(work_dir, "plain")
    consumer = _build_consumer(
        work_dir, "with-kernels", ["runtime", "kernels_optimized"]
    )
    output = _run_consumer(consumer, model, reference, work_dir)
    print(f"✓ a C++ app linking executorch::kernels_optimized runs a model ({output})")


def test_delegated_model_needs_the_delegate_component(work_dir: Path) -> None:
    """A delegated model runs when the delegate is linked, and fails when it is not.

    Both halves matter. Only running the positive case would pass even if the delegate
    target did nothing, because the runtime falls back to portable kernels for
    anything a backend does not claim. The negative case is what shows the delegate is
    actually doing the work, and that the retention options on the target are what
    make its registration reach the registry.
    """
    model, reference = _export(work_dir, "delegate")
    assert reference["has_xnnpack"], (
        "the exported program contains no XnnpackBackend payload, so this check would "
        "prove nothing about the delegate"
    )

    # The kernels come too. A partitioner claims only what its backend supports, so a
    # delegated program still has ordinary operators in it, and an application without
    # the kernels fails on those rather than on anything to do with the delegate.
    # Measured: this model keeps aten::mean.out outside the XNNPACK partition.
    consumer = _build_consumer(
        work_dir,
        "with-delegate",
        ["runtime", "kernels_optimized", "backend_xnnpack"],
    )
    output = _run_consumer(consumer, model, reference, work_dir)
    print(
        f"✓ a C++ app linking executorch::backend_xnnpack runs a delegated model "
        f"({output})"
    )

    # The same program, run by an application that has the kernels but not the
    # delegate. Only the delegate is removed, so a failure can only be about the
    # missing backend rather than about absent operators.
    without = _build_consumer(work_dir, "no-delegate", ["runtime", "kernels_optimized"])
    environment = _loader_clean_environment()
    inputs = reference["inputs"]
    shape_a, data_a = _write_tensor(work_dir, "na", inputs[0])
    shape_b, data_b = _write_tensor(work_dir, "nb", inputs[1])
    expected = work_dir / "n_expected.data"
    expected.write_text(" ".join(repr(v) for v in reference["expected"]))
    result = subprocess.run(
        [
            str(without),
            str(model),
            str(shape_a),
            str(data_a),
            str(shape_b),
            str(data_b),
            str(expected),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0, (
        "a delegated program ran in an application that never linked the delegate, so "
        "either the delegate is reaching the registry without being asked for or the "
        f"program was not delegated at all. Output was:\n{result.stdout[-1000:]}"
    )
    # The exit code alone is not enough: this consumer also exits non-zero on a missing
    # file and on too few arguments, so a run that failed to start would pass this check
    # while proving nothing about the delegate.
    combined = result.stdout + result.stderr
    assert "not registered" in combined, (
        "the application failed for some reason other than the delegate being absent, "
        f"which is the documented behaviour:\n{combined[-1500:]}"
    )
    print(
        "✓ the same delegated model fails without executorch::backend_xnnpack, "
        "so the component is what registers it"
    )


def test_consumer_is_relocatable(work_dir: Path) -> None:
    """The application still runs after being moved away from the wheel.

    Building in place leaves the wheel's absolute lib directory on the link line, which
    resolves the runtime whatever $ORIGIN says. Copying the application next to a copy
    of the libraries, with the original package hidden, is what actually shows the
    package is relocatable rather than only working where it was built.
    """
    model, reference = _export(work_dir, "plain")
    consumer = _build_consumer(work_dir, "relocate", ["runtime", "kernels_optimized"])

    dynamic = _recorded_dependencies(consumer)
    assert _library_file_name("libexecutorch") in dynamic, (
        "the application records no dependency on the shipped runtime, so it is not "
        f"linking what the wheel ships:\n{dynamic}"
    )
    token = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    assert token in dynamic, (
        f"the application has no {token} relative runtime search path, so it cannot "
        f"work anywhere but where it was built:\n{dynamic}"
    )
    # The newer tag specifically, not just any search path. DT_RPATH is searched
    # ahead of LD_LIBRARY_PATH and applies to a dependency's own dependencies, so
    # a consumer given DT_RPATH cannot point an instrumented or locally built
    # runtime at their application. Both tags satisfy the check above, so without
    # this the package could silently go back to the older one.
    # ELF only. Mach-O records one LC_RPATH with no weaker older variant, so there is no
    # equivalent preference to check there.
    if sys.platform != "darwin":
        assert "(RUNPATH)" in dynamic, (
            "the application's runtime search path is recorded as DT_RPATH rather than "
            "DT_RUNPATH. DT_RPATH outranks LD_LIBRARY_PATH and is inherited by "
            "dependencies, so a consumer could not override a packaged library with "
            f"their own build:\n{dynamic}"
        )

    package_dir = _installed_package_dir()
    deployed = work_dir / "deployed"
    deployed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(consumer, deployed / "consumer")
    # Every directory the wheel ships a library in, not just lib/. The CUDA delegate records a
    # dependency on a library under backends/cuda/, so copying lib/ alone produced a deployment
    # that cannot start, and this check could not see it.
    for source in ("lib", "backends/cuda"):
        directory = package_dir / source
        if not directory.is_dir():
            continue
        for library in sorted(directory.glob(_library_file_name("lib*") + "*")):
            if library.is_file() and not library.is_symlink():
                shutil.copy2(library, deployed / library.name)

    moved = deployed / "consumer"
    # Strip the absolute entry the build left behind, so only the loader-relative token can
    # resolve the libraries. Without this the application would find the original wheel and
    # the check would pass for the wrong reason.
    if sys.platform == "darwin":
        entries = _mach_o_runtime_paths(moved)
    else:
        # Fatal, not a skip, and only on the ELF side, which is the only one that reads or
        # rewrites the search path with patchelf. A skip here is indistinguishable from a pass
        # in the log, which is the shape of failure this suite exists to avoid.
        patchelf = _tool("patchelf")
        if shutil.which("patchelf") is None and not Path(patchelf).is_file():
            print("- patchelf not present, installing it so this check can run")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "patchelf"],
                capture_output=True,
                text=True,
                check=True,
            )
            patchelf = _tool("patchelf")
        assert shutil.which("patchelf") or Path(patchelf).is_file(), (
            "patchelf is required to prove the application is relocatable, and could not "
            "be installed. Without it the relocated application resolves the original "
            "package and the check would pass without testing anything."
        )
        entries = [
            entry
            for entry in subprocess.run(
                [patchelf, "--print-rpath", str(moved)],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split(":")
            if entry
        ]
    kept = [entry for entry in entries if not entry.startswith(str(package_dir))]

    if sys.platform == "darwin":
        # One entry per load command, so each unwanted one is deleted individually and the
        # fallback is added only when stripping emptied the list.
        install_name_tool = _tool("install_name_tool")
        for entry in entries:
            if entry in kept:
                continue
            subprocess.run(
                [install_name_tool, "-delete_rpath", entry, str(moved)],
                capture_output=True,
                check=True,
            )
        if not kept:
            subprocess.run(
                [install_name_tool, "-add_rpath", "@loader_path", str(moved)],
                capture_output=True,
                check=True,
            )
    else:
        subprocess.run(
            [patchelf, "--set-rpath", ":".join(kept) or "$ORIGIN", str(moved)],
            check=True,
        )

    output = _run_consumer(moved, model, reference, work_dir)
    print(f"✓ the application still runs deployed away from the wheel ({output})")


def test_one_registry_in_the_cpp_process(work_dir: Path) -> None:
    """An application linking several components gets one registry, not one each.

    This is the property the split exists to create, checked in a C++ process rather
    than only by inspecting symbol tables. Every component resolves the registry from
    the one runtime library, so the count a consumer observes grows by exactly what it
    links that registers a backend, rather than resetting or doubling.
    """
    model, reference = _export(work_dir, "plain")

    def backends_seen(name, components) -> int:
        consumer = _build_consumer(work_dir, name, components)
        output = _run_consumer(consumer, model, reference, work_dir)
        for field in output.split():
            if field.startswith("backends="):
                return int(field.split("=", 1)[1])
        raise AssertionError(f"the application printed no backend count: {output}")

    # Both cases have to be able to run the model, so both link the kernels. The
    # variable under test is how many further component libraries are linked, not
    # whether the program executes.
    lean = backends_seen("registry-lean", ["runtime", "kernels_optimized"])
    full = backends_seen(
        "registry-full",
        ["runtime", "kernels_optimized", "threadpool", "etdump", "backend_xnnpack"],
    )
    # The delegate genuinely adds one backend, so the counts differ by exactly that.
    # What must not happen is the count resetting or doubling, which is what a second
    # registry in the process looks like.
    assert full == lean + 1, (
        f"an application linking two components sees {lean} registered backends while "
        f"one linking five, of which exactly one registers a backend, sees {full}. A "
        "component is carrying its own registry rather than resolving the shared one."
    )
    print(
        f"✓ one shared registry: {lean} backends with two components, {full} with five"
    )


def test_find_package_honours_a_version_request(work_dir: Path) -> None:
    """`find_package(executorch <version>)` must accept and reject correctly.

    Without a version file CMake rejects every versioned request, whatever version is
    actually installed, so a consumer pinning a minimum cannot configure at all.
    The wheel generates the file at packaging time because the version is only known
    then: the base comes from version.txt and a nightly overrides it.
    """
    package_dir = _installed_package_dir()
    version_file = package_dir / "share" / "cmake" / "executorch-config-version.cmake"
    assert version_file.is_file(), (
        f"the wheel ships no CMake version file at {version_file}, so find_package "
        "rejects every versioned request"
    )

    installed = None
    build_version = None
    for line in version_file.read_text().splitlines():
        if line.startswith("set(PACKAGE_VERSION"):
            installed = line.split('"')[1]
        elif line.startswith("set(EXECUTORCH_BUILD_VERSION"):
            build_version = line.split('"')[1]
    assert installed, f"could not read PACKAGE_VERSION from {version_file}"
    assert not installed.startswith("@"), (
        f"the version file still holds an unsubstituted placeholder, {installed}, so "
        "packaging copied the template instead of filling it in"
    )

    # The two variables report different things and are filled separately. Only checking the numeric one
    # would pass on a file where the full version was truncated to it, or where its placeholder was never
    # substituted, and the full version is what a consumer compares to pin an exact build.
    assert build_version, f"could not read EXECUTORCH_BUILD_VERSION from {version_file}"
    assert not build_version.startswith(
        "@"
    ), f"the build version still holds an unsubstituted placeholder, {build_version}"
    assert build_version.startswith(installed), (
        f"the build version {build_version} does not start with the numeric release {installed}, "
        "so they describe different builds"
    )
    from executorch.version import __version__ as installed_version

    assert build_version == installed_version, (
        f"the version file says {build_version} but the installed package says {installed_version}, "
        "so a consumer pinning an exact build would compare against the wrong one"
    )

    # CMake compares dotted integers only, and find_package rejects a REQUESTED version
    # that is not one, so the numeric release part is what a consumer can ask for. The
    # wheel's own version can carry more: a dev segment for a nightly and a local part
    # such as +cpu or a commit hash. CMake truncates the stored version at the first
    # non-numeric segment, which makes those compare equal to the release, so pinning
    # the release is the behaviour a consumer actually gets.
    release = re.match(r"\d+(?:\.\d+)*", installed)
    assert release, f"no numeric release part in the installed version {installed}"
    release = release.group(0)
    major = release.split(".")[0]
    too_new = f"{int(major) + 1}.0"
    source_dir = work_dir / "version-probe"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text("int main() { return 0; }\n")

    cases = [(release, True), (too_new, False)]
    # A request one major below the installed package is rejected in the same way as one above it.
    # ExecuTorch promises nothing across that boundary, so a consumer written against the older
    # major must not be handed this one: it would configure against a package it has never seen and
    # fail later, at compile or run time, instead of here. Only meaningful once there is a major
    # below this one to ask for.
    if int(major) >= 1:
        cases.append((f"{int(major) - 1}.1", False))
    for requested, must_accept in cases:
        # Deliberately the older floor: this probe never links an imported target, so it also checks
        # that version acceptance answers correctly below the version those targets need.
        (source_dir / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(probe CXX)\n"
            f"find_package(executorch {requested} REQUIRED)\n"
            "add_executable(consumer consumer.cpp)\n"
        )
        build_dir = work_dir / f"version-probe-build-{requested}"
        result = subprocess.run(
            [
                _tool("cmake"),
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                f"-DCMAKE_PREFIX_PATH={version_file.parent}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        accepted = result.returncode == 0
        assert accepted == must_accept, (
            f"find_package(executorch {requested}) against an installed {installed} "
            f"{'was rejected' if must_accept else 'was accepted'}, which is wrong:\n"
            f"{result.stdout[-800:]}{result.stderr[-800:]}"
        )
    print(
        f"✓ find_package honours a version request (installed {installed}, "
        f"accepts {release}, rejects every other major)"
    )


def test_profiler_component_is_usable(work_dir: Path) -> None:
    """A C++ application must be able to construct the profiler the etdump component represents.

    Linking a component proves the library resolves. It does not prove a consumer can call anything in it.
    A library whose only exported symbol is an internal helper still links, so the component could be
    requested and linked but not used.
    """
    package_dir = _installed_package_dir()
    # Globbed, not an exact name: the library carries a version suffix outside a wheel build, and an exact
    # match would silently skip this check there. The profiler is required elsewhere in this suite, so its
    # absence is a fault rather than a reason to skip.
    shipped = sorted(
        (package_dir / "lib").glob(_library_file_name("libexecutorch_etdump") + "*")
    )
    assert shipped, (
        f"the wheel ships no profiler library under {package_dir / 'lib'}, so the etdump component it "
        "advertises cannot be linked"
    )

    source_dir = work_dir / "with-etdump"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(
        "#include <executorch/devtools/etdump/etdump_flatcc.h>\n"
        "#include <memory>\n"
        "int main() {\n"
        "  auto tracer = std::make_unique<executorch::etdump::ETDumpGen>();\n"
        "  return tracer == nullptr ? 1 : 0;\n"
        "}\n"
    )
    (source_dir / "CMakeLists.txt").write_text(_consumer_cmake(["runtime", "etdump"]))

    build_dir = work_dir / "with-etdump-build"
    configured = subprocess.run(
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={package_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configured.returncode == 0, (
        "configuring an application that uses the profiler failed:\n"
        f"{configured.stdout[-1500:]}\n{configured.stderr[-1500:]}"
    )
    built = subprocess.run(
        [_tool("cmake"), "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, (
        "an application that constructs the profiler failed to build against the installed wheel, so the "
        f"component cannot be used by a consumer:\n{built.stdout[-2000:]}\n{built.stderr[-2000:]}"
    )
    print("✓ a C++ app linking executorch::etdump constructs the profiler")


def test_every_shipped_header_compiles(work_dir: Path) -> None:
    """Each installed header must compile on its own against the installed wheel.

    A header that cannot be included is worse than one that is absent, because the failure arrives in
    someone else's project at compile time. This caught a profiler header that includes a regular
    expression library the wheel does not carry, and whose implementation the shipped library does not
    define either.

    Compiled one at a time rather than all together, so the message names the header at fault.
    """
    package_dir = _installed_package_dir()
    include_root = package_dir / "include"
    headers = sorted(include_root.rglob("*.h"))
    assert (
        headers
    ), f"no headers found under {include_root}, so this check would prove nothing"

    # The same include directories the CMake package exports, since that is what a consumer gets.
    includes = [
        f"-I{include_root}",
        f"-I{include_root / 'executorch' / 'runtime' / 'core' / 'portable_type' / 'c10'}",
        # The same definition every imported target carries. Without it the vendored c10 headers reach for
        # a header generated inside a PyTorch build, which no wheel can carry, so compiling without it
        # tests a configuration no consumer of this package is ever in.
        "-DC10_USING_CUSTOM_GENERATED_MACROS",
    ]
    # Deliberately no CUDA toolkit include directory. A CUDA wheel's own headers have to compile
    # against nothing but the wheel, the same as every other header here. Adding the builder's toolkit
    # would measure the build machine rather than the consumer, and a header that only compiles that way
    # fails in the consumer's project instead of here.
    # Headers a wheel-only consumer cannot compile and is not expected to. Each needs something outside the
    # package: a platform that is not the one being built for, or a third-party library the wheel does not
    # carry. They ship because a source build includes them, and holding them to this rule would report a
    # defect with no available fix.
    needs_more_than_the_wheel = (
        # These ship because other shipped headers include them, so they cannot be left out, and they do
        # not compile on their own: each needs a third-party library the wheel links but publishes no
        # headers for, or a platform other than the one being built for.
        "mman_windows.h",  # a Windows compatibility shim, needs the MinGW headers
        # These say in their own text that they must not be included directly, and name the header to
        # include instead. Including one anyway is a use error rather than a packaging defect.
        "c10/util/complex_math.h",
        "torch/headeronly/util/complex_utils.h",
    )

    source = work_dir / "header_probe.cpp"
    broken = []
    skipped_but_fine = []
    compiled = 0
    skipped_names = []
    for header in headers:
        relative = header.relative_to(include_root)
        skipped = relative.as_posix().endswith(needs_more_than_the_wheel)
        source.write_text(
            f"#include <{relative.as_posix()}>\nint main() {{ return 0; }}\n"
        )
        result = subprocess.run(
            [_tool("c++"), "-std=c++20", *includes, "-fsyntax-only", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if skipped:
            skipped_names.append(relative.as_posix())
            if result.returncode == 0:
                skipped_but_fine.append(str(relative))
            continue
        compiled += 1
        if result.returncode != 0:
            missing = re.search(r"fatal error: ([^:]+): No such file", result.stderr)
            broken.append(
                f"{relative}: {missing.group(1) if missing else 'does not compile'}"
            )

    # A skip list quietly loses value as the code changes: an entry that starts compiling stays skipped and
    # nobody notices the coverage was given up for nothing. So the skipped ones are compiled too, and an
    # entry that now works is reported rather than left in place.
    assert not skipped_but_fine, (
        "these headers are on the skip list but compile now, so the list is stale and is giving up "
        f"coverage for no reason. Remove them from it: {skipped_but_fine}"
    )

    assert not broken, (
        "the wheel ships headers that cannot be included from the installed package, so a consumer "
        "following the documentation would fail to compile:\n  " + "\n  ".join(broken)
    )
    # The number actually compiled, and the exemptions named. Reporting the total
    # selected would claim coverage of headers this never compiled.
    print(
        f"✓ {compiled} of {len(headers)} shipped headers compile against the installed "
        f"wheel; {len(skipped_names)} need something the wheel does not carry "
        f"({', '.join(sorted(skipped_names))})"
    )


def test_shipped_headers_have_implementations(work_dir: Path) -> None:
    """A header that compiles but has no implementation in any shipped library is unusable.

    Compiling proves only that the declarations parse. Two headers once shipped whose implementation
    lived in a component no shipped library links, so a consumer got an undefined reference at link
    time. A syntax check cannot see that, so this links a real program against the shipped libraries.

    A sample, not a sweep: one entry point per header listed below, chosen because there is no way to
    guess a callable declaration from a header alone. It catches a component whose library stops being
    linked, which is the failure that shipped. It does not catch a newly added declaration that nobody
    implements; that needs a symbol scan over every shipped header, which is worth doing separately.
    """
    package = _installed_package_dir()
    include_root = package / "include"

    # One small program per header, calling the declaration a consumer would call first. Kept as source
    # rather than derived, because there is no way to guess a usable call from a header alone.
    probes = {
        "extension/memory_allocator/malloc_memory_allocator.h": (
            "#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>\n"
            "using namespace executorch::extension;\n"
            "int main() { MallocMemoryAllocator allocator; return allocator.allocate(16) == nullptr; }\n"
        ),
        "extension/module/module.h": (
            "#include <executorch/extension/module/module.h>\n"
            "using namespace executorch::extension;\n"
            'int main() { Module module("none.pte"); return module.method_names().ok() ? 0 : 1; }\n'
        ),
        "extension/tensor/tensor.h": (
            "#include <executorch/extension/tensor/tensor.h>\n"
            "using namespace executorch::extension;\n"
            "int main() { float data[4] = {}; auto tensor = make_tensor_ptr({2, 2}, data); "
            "return tensor->numel() == 4 ? 0 : 1; }\n"
        ),
        # The profiler, which lives in its own library rather than in the runtime. Included because a
        # dead declaration shipped on this class for a while and the probes above could not reach it:
        # they link the runtime only, so no etdump symbol was ever resolved here.
        "devtools/etdump/etdump_flatcc.h": (
            "#include <executorch/devtools/etdump/etdump_flatcc.h>\n"
            "using namespace executorch::etdump;\n"
            'int main() { ETDumpGen generator; generator.create_event_block("probe"); '
            "return 0; }\n"
        ),
        # The thread pool, which is the other component backing declarations in a shipped header.
        # Compiled with ET_USE_THREADPOOL, which is what the package puts on the runtime target when
        # the thread pool ships, so this probe sees the same declaration a consumer does. Without it
        # the header supplies a local inline definition instead and the probe linked identically with
        # and without the library, which made it unable to detect the component being dropped.
        # Measured both ways: with the definition, linking the runtime alone fails on
        # executorch::extension::parallel_for.
        "runtime/kernel/thread_parallel_interface.h": (
            "#define ET_USE_THREADPOOL\n"
            "#include <executorch/runtime/kernel/thread_parallel_interface.h>\n"
            "using namespace executorch::extension;\n"
            "int main() { return parallel_for(0, 1, 1, [](int64_t, int64_t) {}) ? 0 : 1; }\n"
        ),
    }

    includes = [
        f"-I{include_root}",
        f"-I{include_root / 'executorch' / 'runtime' / 'core' / 'portable_type' / 'c10'}",
        "-DC10_USING_CUSTOM_GENERATED_MACROS",
    ]
    library_dir = package / "lib"
    unresolved = []
    for header, program in probes.items():
        assert (
            include_root / "executorch" / header
        ).is_file(), f"{header} is not shipped, so this probe is checking nothing"
        source = work_dir / "link_probe.cpp"
        source.write_text(program)
        result = subprocess.run(
            [
                _tool("c++"),
                "-std=c++17",
                *includes,
                str(source),
                "-o",
                str(work_dir / "link_probe"),
                f"-L{library_dir}",
                "-lexecutorch",
                # The component libraries too, not only the runtime. Linking the runtime alone left
                # every declaration outside it unreachable, so a probe for one of those headers passed
                # without resolving anything. Missing ones are skipped by the loop below.
                *[
                    f"-l{name}"
                    for name in (
                        "executorch_etdump",
                        "executorch_kernels_optimized",
                        "executorch_threadpool",
                    )
                    if (library_dir / (f"lib{name}" + _dynamic_lib_suffix())).is_file()
                ],
                f"-Wl,-rpath,{library_dir}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            missing = re.findall(r"undefined reference to `([^']+)'", result.stderr)
            unresolved.append(f"{header}: {sorted(set(missing))[:3] or 'did not link'}")

    assert not unresolved, (
        "the wheel ships headers whose implementation is in no shipped library, so a consumer compiles "
        "and then fails to link:\n  " + "\n  ".join(unresolved)
    )
    print(f"✓ all {len(probes)} probed headers link against the shipped libraries")


def test_documented_example_compiles(work_dir: Path) -> None:
    """The C++ example in the documentation must compile against the installed wheel.

    Extracted from the documentation rather than copied here, so the two cannot drift. A
    reader who follows the documentation gets code that builds, and a dangling include or
    a renamed entry point fails this check instead of shipping.
    """
    here = Path(__file__).resolve()
    root = here.parents[3] if len(here.parents) > 3 else here.parent
    documentation = root / "docs" / "source" / "using-executorch-cpp.md"
    if not documentation.is_file():
        print("- the documentation is not present, skipping the example check")
        return

    # The first fenced cpp block after the prebuilt-package heading. Anchored on the
    # heading so an unrelated example elsewhere on the page is not picked up.
    text = documentation.read_text()
    marker = "### Using the prebuilt libraries from the pip package"
    assert marker in text, (
        f"{documentation.name} no longer documents the prebuilt package, so a reader has "
        "no instructions for the libraries this wheel ships"
    )
    section = text[text.index(marker) :]
    blocks = re.findall(r"```cpp\n(.*?)```", section, re.S)
    assert blocks, (
        f"{documentation.name} documents the prebuilt package but shows no C++ example, "
        "so nothing proves the documented usage compiles"
    )

    source_dir = work_dir / "documented"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "main.cpp").write_text(blocks[0])
    # The components the documentation itself tells a reader to ask for.
    (source_dir / "CMakeLists.txt").write_text(
        _consumer_cmake(["runtime", "kernels_optimized"]).replace(
            "consumer.cpp", "main.cpp"
        )
    )

    package_dir = _installed_package_dir()
    config = package_dir / "share" / "cmake" / "executorch-config.cmake"
    build_dir = work_dir / "documented-build"
    configured = subprocess.run(
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config.parent}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configured.returncode == 0, (
        "the documented example does not configure against the installed package:\n"
        f"{configured.stdout[-1500:]}{configured.stderr[-1500:]}"
    )
    built = subprocess.run(
        [_tool("cmake"), "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, (
        "the documented example does not build against the installed package, so a "
        f"reader following the documentation gets code that fails:\n"
        f"{built.stdout[-2500:]}{built.stderr[-2500:]}"
    )
    print("✓ the C++ example in the documentation compiles against the wheel")


def _provision_pre_328_cmake(work_dir: Path) -> str:
    """Return a path to a cmake older than 3.28, or "" if one cannot be had.

    The route this check exercises is only reachable with such a binary, and a
    release builder is not obliged to carry one, so fetch it rather than leaving
    the check permanently skipped. A failed fetch returns empty and the caller
    decides what that means, which differs between a local run and a release job.
    """
    override = os.environ.get("EXECUTORCH_PRE_328_CMAKE", "")
    if override and Path(override).is_file():
        return override

    venv_dir = work_dir / "pre-328-cmake"
    binary = venv_dir / "bin" / "cmake"
    if not binary.is_file():
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    str(venv_dir / "bin" / "pip"),
                    "install",
                    "--quiet",
                    "cmake==3.24.*",
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as error:
            print(f"- could not provision a pre-3.28 cmake: {error}")
            return ""
    return str(binary) if binary.is_file() else ""


def test_pre_3_28_route_builds_a_consumer_through_variables(work_dir: Path) -> None:
    """The pre-3.28 route offers only variables, and those variables have to work.

    CMake before 3.28 writes the $ORIGIN token in a runtime search path incorrectly, so
    the config file skips its imported targets on those versions and exposes plain
    variables instead. The whole modern-CMake test set never enters that branch, because
    cmake_minimum_required(3.28) does not lower CMAKE_VERSION and the tests above run
    with whatever cmake is on PATH. So a consumer stuck on an older cmake would find at
    run time that the wheel produced no usable link.

    Uses the binary EXECUTORCH_PRE_328_CMAKE names, or fetches one. A local run with no
    old cmake and no package index says so and moves on. A release job does not: the
    fetch is the only thing standing between this route and no coverage at all, and a
    transient index failure that turns into a green run is how the route would be
    published unexercised.
    """
    old_cmake = os.environ.get("EXECUTORCH_PRE_328_CMAKE", "")
    if not old_cmake or not Path(old_cmake).is_file():
        old_cmake = _provision_pre_328_cmake(work_dir)
    if not old_cmake:
        assert os.environ.get("GITHUB_ACTIONS") != "true", (
            "no cmake older than 3.28 could be provisioned, so the route the wheel "
            "offers every consumer below that version went unchecked. Point "
            "EXECUTORCH_PRE_328_CMAKE at one, or restore the job's access to the "
            "package index."
        )
        print(
            "- no cmake older than 3.28 is available, skipping the pre-3.28 route "
            "check"
        )
        return
    version = subprocess.run(
        [old_cmake, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[0]
    match = re.search(r"(\d+)\.(\d+)", version)
    assert match, f"could not read a version from {old_cmake}: {version!r}"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) < (3, 28), (
        f"EXECUTORCH_PRE_328_CMAKE={old_cmake} is version {major}.{minor}, but this "
        "check needs a binary older than 3.28 to enter the fallback route"
    )

    package_dir = _installed_package_dir()
    config = package_dir / "share" / "cmake" / "executorch-config.cmake"
    assert config.is_file(), f"the wheel ships no CMake package config at {config}"

    source_dir = work_dir / "pre-328"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(_CONSUMER_SOURCE)
    # No COMPONENTS and no named target, which is the shape the older-CMake route
    # forces. The variables have to carry everything a consumer needs: the runtime and
    # its components, the include directories, the compile definitions and the C++
    # standard.
    (source_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\n"
        "project(consumer CXX)\n"
        "find_package(executorch REQUIRED)\n"
        "add_executable(consumer consumer.cpp)\n"
        "target_include_directories(consumer PRIVATE ${EXECUTORCH_INCLUDE_DIRS})\n"
        "target_compile_definitions(consumer PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})\n"
        "target_link_libraries(consumer PRIVATE ${EXECUTORCH_LIBRARIES})\n"
        "set_target_properties(consumer PROPERTIES CXX_STANDARD ${EXECUTORCH_CXX_STANDARD})\n"
    )
    build_dir = work_dir / "pre-328-build"
    for command in (
        [
            old_cmake,
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config.parent}",
        ],
        [old_cmake, "--build", str(build_dir)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (
            "a consumer on CMake older than 3.28 could not build against the wheel "
            f"through the documented variables:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )

    consumer = build_dir / "consumer"
    assert consumer.is_file(), f"the build produced no {consumer}"
    # Run it. Linking proves the variables name the right files; only executing proves
    # they also leave the program able to find them at run time, which is the half of
    # this route that no imported target is there to supply.
    model, reference = _export(work_dir, "plain")
    _run_consumer(consumer, model, reference, work_dir)
    # The runtime and CPU kernels have to be on the link line, since the variables are
    # the only thing that carries them on this route.
    dependencies = _recorded_dependencies(consumer)
    assert _library_file_name("libexecutorch") in dependencies, (
        "a consumer built through EXECUTORCH_LIBRARIES on pre-3.28 CMake does not "
        f"depend on the runtime:\n{dependencies}"
    )
    assert "libexecutorch_kernels_optimized" in dependencies, (
        "the pre-3.28 aggregate does not carry the CPU kernels, so a consumer built "
        "through it would fail at run time with operators reported missing"
    )
    print(
        f"✓ a consumer on CMake {major}.{minor} builds through EXECUTORCH_LIBRARIES, "
        "links the runtime plus the CPU kernels, and runs"
    )


def test_quantized_kernels_component_runs_a_model(work_dir: Path) -> None:
    """A C++ application can run a quantized model using the shipped quantized kernels.

    Before the quantized kernels became their own library they existed only inside the
    ahead-of-time extension beside the Python bindings, so a C++ application loading a
    quantized program had nothing to link and failed at run time with the operators
    reported missing.

    A missing library is a failure rather than a skip. The preset that builds the wheel
    always enables the quantized kernels, so their absence is a regression in packaging
    or in the build, not a configuration this suite has to tolerate. Skipping there
    reported the whole check as coverage while running none of it.
    """
    package_dir = _installed_package_dir()
    # Globbed for the same reason the profiler check is: the library carries a version suffix outside a
    # wheel build, and an exact name would skip this silently there rather than running it.
    shipped = sorted(
        (package_dir / "lib").glob(
            _library_file_name("libexecutorch_kernels_quantized") + "*"
        )
    )
    assert shipped, (
        "the wheel ships no quantized kernels library. The preset that builds it enables "
        "them unconditionally, so this is a packaging or build regression rather than an "
        "unsupported configuration."
    )

    model, reference = _export(work_dir, "quantized")
    # The export has to have produced a quantized program, or the rest of this proves nothing about the
    # quantized kernels. The numeric comparison cannot tell the difference: an unquantized export of the
    # same model lands well inside the tolerance a quantized one needs, so it would pass while linking a
    # library it never exercised.
    assert reference["has_quantized"], (
        "the quantized export produced a program with no quantized operators, so this check would "
        "prove nothing about the quantized kernels"
    )
    consumer = _build_consumer(
        work_dir,
        "with-quantized",
        ["runtime", "kernels_optimized", "kernels_quantized"],
    )
    # One int8 quantization step over this model's output range is about 5e-3, so a
    # float32 tolerance cannot be met by a correct quantized run.
    output = _run_consumer(consumer, model, reference, work_dir, tolerance=2e-2)
    print(
        f"✓ a C++ app linking executorch::kernels_quantized runs a quantized model "
        f"({output})"
    )


def test_aggregate_variable_excludes_the_quantized_kernels(work_dir: Path) -> None:
    """`${EXECUTORCH_LIBRARIES}` must not drag in the quantized kernels.

    The export-time plugin that `executorch.kernels.quantized` loads carries its own
    copy of those kernels rather than depending on the shipped library, so a process
    holding both registers the same operators twice and the runtime stops on the
    second one. An application that links whatever the package offers by default
    would inherit that, so the component is defined but held out of the aggregate and
    a consumer that wants it names it.

    Checked by reading the link line rather than by running, because the failure is a
    process-wide abort that needs a Python interpreter in the same process to trigger.
    What this owns is the packaging decision: is the library on the link line at all.
    """
    package_dir = _installed_package_dir()
    # Fatal for the same reason the check above is: the preset that builds the wheel
    # always enables these kernels, so their absence is a regression rather than a
    # configuration to tolerate, and skipping would report this as coverage.
    assert sorted(
        (package_dir / "lib").glob(
            _library_file_name("libexecutorch_kernels_quantized") + "*"
        )
    ), "the wheel ships no quantized kernels library, so this check cannot run"

    source_dir = work_dir / "aggregate-only"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(_CONSUMER_SOURCE)
    # No COMPONENTS and no named target, which is the shape the older-CMake route
    # forces and the documentation offers as the general case.
    (source_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(consumer CXX)\n"
        "find_package(executorch REQUIRED)\n"
        "add_executable(consumer consumer.cpp)\n"
        "target_link_libraries(consumer PRIVATE ${EXECUTORCH_LIBRARIES})\n"
    )
    build_dir = work_dir / "aggregate-only-build"
    config = package_dir / "share" / "cmake" / "executorch-config.cmake"
    for command in (
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config.parent}",
        ],
        [_tool("cmake"), "--build", str(build_dir)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (
            "an application linking only ${EXECUTORCH_LIBRARIES} could not be built:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )

    consumer = build_dir / "consumer"
    dependencies = _recorded_dependencies(consumer)
    assert "libexecutorch_kernels_quantized" not in dependencies, (
        "an application that linked only ${EXECUTORCH_LIBRARIES} depends on the "
        "quantized kernels. That library collides with the export-time plugin, so it "
        "has to be opted into by name rather than handed to every consumer."
    )
    # The rest of the aggregate still has to be there, or this would pass by shipping
    # nothing at all.
    assert "libexecutorch_kernels_optimized" in dependencies, (
        "the aggregate no longer carries the CPU kernels, so an application linking it "
        "would fail at run time with the operators reported missing"
    )
    print(
        "✓ ${EXECUTORCH_LIBRARIES} carries the CPU kernels and not the quantized ones"
    )


def run_tests(work_dir: Path) -> None:
    test_find_package_honours_a_version_request(work_dir)
    test_profiler_component_is_usable(work_dir)
    test_every_shipped_header_compiles(work_dir)
    test_shipped_headers_have_implementations(work_dir)
    test_documented_example_compiles(work_dir)
    test_runtime_alone_links_but_cannot_compute(work_dir)
    test_kernels_component_runs_a_model(work_dir)
    test_pre_3_28_route_builds_a_consumer_through_variables(work_dir)
    test_quantized_kernels_component_runs_a_model(work_dir)
    test_aggregate_variable_excludes_the_quantized_kernels(work_dir)
    test_delegated_model_needs_the_delegate_component(work_dir)
    test_consumer_is_relocatable(work_dir)
    test_one_registry_in_the_cpp_process(work_dir)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        run_tests(Path(directory))
