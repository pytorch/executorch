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
    import executorch.kernels.quantized  # noqa: F401
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
#include <sstream>
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
  if (argc < 6) {
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
    worst = std::fmax(worst, std::fabs((double)actual[i] - (double)expected[i]));
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


def _tool(name: str) -> str:
    """Locate a build tool, including one pip installed beside this interpreter.

    `shutil.which` searches PATH only, and a virtual environment's bin is on PATH only
    when the environment is activated. These checks run by invoking the interpreter
    directly, so a tool installed into that environment is present on disk and
    invisible to a PATH search.
    """
    found = shutil.which(name)
    if found:
        return found
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else name


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

    # No LD_LIBRARY_PATH. Making the shipped libraries findable is the package
    # config's job, and inheriting one from the environment would hide a failure to
    # do it.
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }
    result = subprocess.run(
        [
            str(consumer),
            str(model),
            str(shape_a),
            str(data_a),
            str(shape_b),
            str(data_b),
            str(expected),
            repr(tolerance),
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


def test_runtime_alone_links_but_has_no_kernels(work_dir: Path) -> None:
    """Linking only the runtime builds and loads, and cannot execute.

    Measured rather than assumed: libexecutorch.so defines no operator kernels, so an
    application linking it alone loads a program and then reports every operator as
    missing. That is the intended split, and stating it here documents why the kernels
    are a separate component instead of leaving a reader to guess.

    The value of the check is the boundary. It fails if the runtime silently starts
    carrying kernels again, which would mean the split had regressed, and it fails if
    the runtime cannot even load a program.
    """
    model, reference = _export(work_dir, "plain")
    consumer = _build_consumer(work_dir, "runtime-only", ["runtime"])

    inputs = reference["inputs"]
    shape_a, data_a = _write_tensor(work_dir, "ra", inputs[0])
    shape_b, data_b = _write_tensor(work_dir, "rb", inputs[1])
    expected = work_dir / "r_expected.data"
    expected.write_text(" ".join(repr(v) for v in reference["expected"]))
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }
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
    print("✓ executorch::runtime alone links and loads, and has no kernels to execute")


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
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }
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

    assert shutil.which("readelf") is not None, "readelf is needed to read the RUNPATH"
    dynamic = subprocess.run(
        [_tool("readelf"), "-d", str(consumer)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "libexecutorch.so" in dynamic, (
        "the application records no dependency on the shipped runtime, so it is not "
        f"linking what the wheel ships:\n{dynamic}"
    )
    assert "$ORIGIN" in dynamic, (
        "the application has no $ORIGIN-relative runtime search path, so it cannot "
        f"work anywhere but where it was built:\n{dynamic}"
    )

    package_dir = _installed_package_dir()
    deployed = work_dir / "deployed"
    deployed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(consumer, deployed / "consumer")
    for library in sorted((package_dir / "lib").glob("lib*.so*")):
        if library.is_file() and not library.is_symlink():
            shutil.copy2(library, deployed / library.name)

    moved = deployed / "consumer"
    # Strip the absolute entry the build left behind, so only $ORIGIN can resolve the
    # libraries. Without this the application would find the original wheel and the
    # check would pass for the wrong reason.
    # Fatal, not a skip. Stripping the absolute entry is the whole point: without it the
    # relocated application finds the original package and this check passes for the
    # wrong reason. A skip here is indistinguishable from a pass in the log, which is
    # the shape of failure this suite exists to avoid.
    patchelf = _tool("patchelf")
    if shutil.which("patchelf") is None and not Path(patchelf).is_file():
        print("- patchelf not present, installing it so this check can run")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "patchelf"],
            capture_output=True,
            text=True,
            check=False,
        )
        patchelf = _tool("patchelf")
    assert shutil.which("patchelf") or Path(patchelf).is_file(), (
        "patchelf is required to prove the application is relocatable, and could not "
        "be installed. Without it the relocated application resolves the original "
        "package and the check would pass without testing anything."
    )
    current = subprocess.run(
        [patchelf, "--print-rpath", str(moved)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    kept = [
        entry
        for entry in current.split(":")
        if entry and not entry.startswith(str(package_dir))
    ]
    subprocess.run(
        [patchelf, "--set-rpath", ":".join(kept) or "$ORIGIN", str(moved)], check=True
    )

    output = _run_consumer(moved, model, reference, work_dir)
    print(f"✓ the application still runs deployed away from the wheel ({output})")


def test_one_registry_in_the_cpp_process(work_dir: Path) -> None:
    """An application linking several components gets one registry, not one each.

    This is the property the split exists to create, checked in a C++ process rather
    than only by inspecting symbol tables. Every component resolves the registry from
    the one runtime library, so the count a consumer observes must not grow with the
    number of components it links.
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
    # Equal, not merely non-zero. A second registry would show up as a different count
    # once more registering libraries are linked.
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

    Without a version file CMake reports the package version as "unknown" and accepts
    any request, so a consumer pinning a minimum silently gets whatever is installed.
    The wheel generates the file at packaging time because the version is only known
    then: the base comes from version.txt and a nightly overrides it.
    """
    package_dir = _installed_package_dir()
    version_file = package_dir / "share" / "cmake" / "executorch-config-version.cmake"
    assert version_file.is_file(), (
        f"the wheel ships no CMake version file at {version_file}, so find_package "
        "reports the version as unknown and accepts every request"
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

    for requested, must_accept in ((release, True), ("0.1", True), (too_new, False)):
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
        f"accepts {release}, rejects {too_new})"
    )


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
    assert headers, f"no headers found under {include_root}, so this check would prove nothing"

    # The same include directories the CMake package exports, since that is what a consumer gets.
    includes = [
        f"-I{include_root}",
        f"-I{include_root / 'executorch' / 'runtime' / 'core' / 'portable_type' / 'c10'}",
    ]

    source = work_dir / "header_probe.cpp"
    broken = []
    for header in headers:
        relative = header.relative_to(include_root)
        source.write_text(f"#include <{relative.as_posix()}>\nint main() {{ return 0; }}\n")
        result = subprocess.run(
            [_tool("c++"), "-std=c++20", *includes, "-fsyntax-only", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            missing = re.search(r"fatal error: ([^:]+): No such file", result.stderr)
            broken.append(f"{relative}: {missing.group(1) if missing else 'does not compile'}")

    assert not broken, (
        "the wheel ships headers that cannot be included from the installed package, so a consumer "
        "following the documentation would fail to compile:\n  " + "\n  ".join(broken)
    )
    print(f"✓ all {len(headers)} shipped headers compile against the installed wheel")


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


def test_quantized_kernels_component_runs_a_model(work_dir: Path) -> None:
    """A C++ application can run a quantized model using the shipped quantized kernels.

    Before the quantized kernels became their own library they existed only inside the
    ahead-of-time extension beside the Python bindings, so a C++ application loading a
    quantized program had nothing to link and failed at run time with the operators
    reported missing.

    Skipped rather than failed when the wheel ships no such library, because building
    without the quantized kernels is a supported configuration.
    """
    package_dir = _installed_package_dir()
    shipped = package_dir / "lib" / "libexecutorch_kernels_quantized.so"
    if not shipped.is_file():
        print("- this wheel ships no quantized kernels, skipping")
        return

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


def run_tests(work_dir: Path) -> None:
    test_find_package_honours_a_version_request(work_dir)
    test_every_shipped_header_compiles(work_dir)
    test_documented_example_compiles(work_dir)
    test_runtime_alone_links_but_has_no_kernels(work_dir)
    test_kernels_component_runs_a_model(work_dir)
    test_quantized_kernels_component_runs_a_model(work_dir)
    test_delegated_model_needs_the_delegate_component(work_dir)
    test_consumer_is_relocatable(work_dir)
    test_one_registry_in_the_cpp_process(work_dir)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        run_tests(Path(directory))
