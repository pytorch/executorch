# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile a Helion CuTe kernel launch into a C function AOTInductor links.

The CUDA backend cannot run Helion's CuTe launcher at inference: it needs Python,
Helion and the CuTe DSL. This module compiles one launch ahead of time instead.
The CuTe DSL exports the compiled launch as an object file with a plain C entry
point (``export_to_c``), and a generated C shim adapts that entry point to the
AOTInductor custom-op ABI. Both go into a static library the AOTInductor
``.so`` links, so the kernel ends up inside the ``.pte`` like the Triton kernels
AOTInductor compiles. The launch is registered as a custom op whose calls
AOTInductor lowers to the shim.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import re
import shutil
import string
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast, NamedTuple

import torch
from torch._inductor.cpp_builder import get_cpp_compiler
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.multiprocessing.reductions import StorageWeakRef

_OP_NAMESPACE = "helion_cute"
_DSL_REQUIREMENT = (
    "Helion with CuTe backend support and nvidia-cutlass-dsl 4.7 or newer"
)
_MIN_DSL_VERSION = (4, 7)
_DSL_RUNTIME_LIBRARY = "cuda_dialect_runtime_static"
_TVM_FFI_OPTION = "--enable-tvm-ffi"
# The kwargs a Helion CuTe launch site may pass; the export reproduces both.
_LAUNCH_KWARGS = frozenset({"block", "cute_compile_options"})
_DEFAULT_BLOCK = (256, 1, 1)

_SHIM_TEMPLATE = """\
#include <cstdint>
#include <cstdio>
#include <mutex>

#include <cuda_runtime.h>

// The exported header loads the kernel module through this macro; record the
// first failure instead of only printing it, so the call can return an error.
static cudaError_t module_load_status = cudaSuccess;
#define CUTE_DSL_CUDA_ERROR_CHECK(err)                                \\
  {{                                                                  \\
    const cudaError_t status_ = (err);                                \\
    if (status_ != cudaSuccess && module_load_status == cudaSuccess) {{ \\
      module_load_status = status_;                                   \\
    }}                                                                \\
  }}
#include "{prefix}.h"

struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;
using AOTITorchError = int32_t;
extern "C" AOTITorchError aoti_torch_get_data_ptr(AtenTensorHandle, void**);
extern "C" AOTITorchError aoti_torch_get_dim(AtenTensorHandle, int64_t*);
extern "C" AOTITorchError aoti_torch_get_sizes(AtenTensorHandle, int64_t**);
extern "C" AOTITorchError aoti_torch_get_strides(AtenTensorHandle, int64_t**);
extern "C" AOTITorchError aoti_torch_get_device_index(AtenTensorHandle, int32_t*);
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t, void**);

namespace {{
constexpr int kCount = {count};
constexpr int64_t kRanks[kCount] = {{{ranks}}};
{layout_arrays}
const int64_t* const kSizes[kCount] = {{{sizes}}};
const int64_t* const kStrides[kCount] = {{{strides}}};
constexpr uintptr_t kAlignments[kCount] = {{{alignments}}};

bool matches_export(AtenTensorHandle tensor, int i, void** pointer) {{
  int64_t rank = 0;
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  if (aoti_torch_get_data_ptr(tensor, pointer) != 0 ||
      aoti_torch_get_dim(tensor, &rank) != 0 ||
      aoti_torch_get_sizes(tensor, &sizes) != 0 ||
      aoti_torch_get_strides(tensor, &strides) != 0 || rank != kRanks[i]) {{
    return false;
  }}
  for (int64_t d = 0; d < rank; ++d) {{
    if (sizes[d] != kSizes[i][d] || strides[d] != kStrides[i][d]) {{
      return false;
    }}
  }}
  return reinterpret_cast<uintptr_t>(*pointer) % kAlignments[i] == 0;
}}
}} // namespace

extern "C" AOTITorchError {shim}({parameters}, AtenTensorHandle* /*unused*/) {{
  static {prefix}_Kernel_Module_t module;
  static std::once_flag loaded;
  std::call_once(loaded, [] {{ {prefix}_Kernel_Module_Load(&module); }});
  if (module_load_status != cudaSuccess) {{
    fprintf(stderr, "{prefix}: cannot load the kernel module: %s\\n",
            cudaGetErrorString(module_load_status));
    return 1;
  }}
  const AtenTensorHandle tensors[kCount] = {{{handles}}};
  void* pointers[kCount];
  for (int i = 0; i < kCount; ++i) {{
    if (!matches_export(tensors[i], i, &pointers[i])) {{
      fprintf(stderr,
              "{prefix}: argument %d does not have the layout or alignment "
              "the kernel was compiled for\\n", i);
      return 1;
    }}
  }}
  int32_t device = 0;
  void* stream = nullptr;
  if (aoti_torch_get_device_index(tensors[0], &device) != 0 ||
      aoti_torch_get_current_cuda_stream(device, &stream) != 0) {{
    fprintf(stderr, "{prefix}: cannot get the current CUDA stream\\n");
    return 1;
  }}
  const int32_t status = cute_dsl_{prefix}_wrapper(
      &module, {pointer_args}, {grid}, static_cast<cudaStream_t>(stream));
  if (status != 0) {{
    fprintf(stderr, "{prefix}: the CuTe launch failed with %d\\n", status);
  }}
  return status;
}}
"""


class _ExportedLaunch(NamedTuple):
    op: Any
    shim_declaration: str
    library: str
    library_directory: Path


class _ProgramLaunches(threading.local):
    def __init__(self) -> None:
        self.launches: dict[str, _ExportedLaunch] = {}


class _ExportState:
    """The launches exported in this process, and those of each thread's program.

    Contract with the CUDA backend: ``AotiBackend.preprocess`` runs
    ``LowerHelionPass`` and then, on the same thread, reads
    ``exported_fallback_kernels`` and enters ``link_exported_launches`` for the
    compile, so a thread's program launches are exactly the ones its next
    compile on that thread accepts and links. Calling ``LowerHelionPass``
    outside that flow leaves the thread's launches recorded until the next
    pass run (``begin_program``) or link on that thread.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.library: torch.library.Library | None = None
        self.exported: dict[str, _ExportedLaunch] = {}
        self.per_thread = _ProgramLaunches()

    def program_launches(self) -> dict[str, _ExportedLaunch]:
        return self.per_thread.launches


_STATE = _ExportState()


class _CapturedLaunch(NamedTuple):
    cute_kernel: Any
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    compile_options: str | None
    tensors: tuple[torch.Tensor, ...]


def _xyz(dims: Sequence[object]) -> tuple[int, int, int]:
    values = []
    for value in dims:
        if not isinstance(value, int):
            raise NotImplementedError(
                f"Helion CuTe lowering needs static launch dimensions, got {value!r}"
            )
        values.append(value)
    x, y, z = (values + [1, 1, 1])[:3]
    return x, y, z


def _capture_launch(
    entrypoint: Callable[..., Any],
    launcher_parameter: str,
    arguments: tuple[torch.Tensor, ...],
) -> _CapturedLaunch:
    launches: list[
        tuple[Any, Sequence[object], tuple[object, ...], dict[str, object]]
    ] = []

    def capture(
        cute_kernel: Any, grid: Sequence[object], *args: object, **kwargs: object
    ) -> None:
        launches.append((cute_kernel, grid, args, kwargs))

    with torch.no_grad():
        entrypoint(*arguments, **{launcher_parameter: capture})
    if len(launches) != 1:
        raise NotImplementedError("Helion CuTe lowering requires a single launch")
    cute_kernel, grid, args, kwargs = launches[0]
    unsupported = set(kwargs) - _LAUNCH_KWARGS
    if unsupported:
        raise NotImplementedError(
            f"Helion CuTe lowering does not support launch options {sorted(unsupported)}"
        )
    if not args or not all(isinstance(a, torch.Tensor) for a in args):
        raise NotImplementedError(
            "Helion CuTe lowering supports launches with tensor arguments only"
        )
    options = kwargs.get("cute_compile_options")
    if options is not None and not isinstance(options, str):
        raise NotImplementedError(f"Unexpected CuTe compile options {options!r}")
    block = kwargs.get("block", _DEFAULT_BLOCK)
    if not isinstance(block, (tuple, list)):
        raise NotImplementedError(f"Unexpected CuTe block {block!r}")
    grid_xyz = _xyz(grid)
    if min(grid_xyz) <= 0:
        raise NotImplementedError(f"Helion CuTe lowering found an empty grid {grid!r}")
    return _CapturedLaunch(
        cute_kernel,
        grid_xyz,
        _xyz(block),
        options,
        tuple(cast(torch.Tensor, a) for a in args),
    )


def _example_like(fake: torch.Tensor, aligned: bool) -> torch.Tensor:
    """A real tensor with ``fake``'s layout to compile the launch for.

    Helion derives each pointer's assumed alignment from the example tensors. A
    tensor that may be a view at run time, with a storage offset the exported
    metadata does not show, gets an example one element into its storage, so
    the kernel assumes only element alignment for it.
    """
    offset = 0 if aligned else 1
    extent = 1 + sum((n - 1) * s for n, s in zip(fake.shape, fake.stride()))
    base = torch.empty(offset + extent, dtype=fake.dtype, device=fake.device)
    return base.as_strided(fake.shape, fake.stride(), offset)


def _dsl_runtime_directory() -> Path:
    # The pip wheel keeps `cutlass` under nvidia_cutlass_dsl/dsl_packages while
    # other installs put it beside nvidia_cutlass_dsl, so locate the runtime
    # through the nvidia_cutlass_dsl package itself.
    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Helion CuTe lowering cannot locate the CuTe DSL package")
    roots = [Path(location) for location in spec.submodule_search_locations]
    if torch.version.cuda is None:
        raise RuntimeError("Helion CuTe lowering needs a CUDA build of PyTorch")
    cuda_major = int(torch.version.cuda.split(".")[0])
    # The wheel ships one runtime per CUDA major version: take the newest one
    # the CUDA version PyTorch was built with supports.
    pattern = f"cu*/lib/lib{_DSL_RUNTIME_LIBRARY}.a"
    libraries = [library for root in roots for library in root.glob(pattern)]
    flavors = []
    for library in libraries:
        flavor = library.parent.parent.name[2:]
        if flavor.isdigit() and int(flavor) <= cuda_major:
            flavors.append((int(flavor), library.parent))
    if not flavors:
        raise RuntimeError(
            f"Helion CuTe lowering cannot find lib{_DSL_RUNTIME_LIBRARY}.a for CUDA "
            f"{cuda_major} or older under {', '.join(map(str, roots))}"
        )
    return max(flavors)[1]


@functools.cache
def _cute_launcher_modules() -> tuple[Any, Any, Any, Any]:
    """Helion's private CuTe launcher modules, checked against what we call.

    Lowering builds on private Helion launcher functions and the CuTe DSL, so
    probe them up front: a Helion or DSL that renamed one must fail with the
    requirement error instead of a bare AttributeError or ImportError.
    """
    try:
        dsl_version: str | None = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        dsl_version = None
    if dsl_version is not None:
        parsed = tuple(
            int(part) for part in dsl_version.split(".")[:2] if part.isdigit()
        )
        if parsed and parsed < _MIN_DSL_VERSION:
            raise RuntimeError(
                f"Helion CuTe lowering needs {_DSL_REQUIREMENT}; "
                f"found nvidia-cutlass-dsl {dsl_version}"
            )
    try:
        launcher = importlib.import_module("helion.runtime.cute.launcher")
        arg_signature = importlib.import_module("helion.runtime.cute.arg_signature")
        sass_check = importlib.import_module("helion.runtime.cute.sass_check")
        cute = importlib.import_module("cutlass.cute")
    except ImportError as error:
        raise RuntimeError(f"Helion CuTe lowering needs {_DSL_REQUIREMENT}") from error
    required = (
        (launcher, "_cute_stage_under_aligned_operands"),
        (launcher, "_build_cute_schema_and_args"),
        (launcher, "_ensure_cute_dsl_arch_env"),
        (launcher, "_create_cute_wrapper"),
        (launcher, "_cute_num_sm_from_arch_args"),
        (launcher, "_cute_current_stream"),
        (arg_signature, "schema_pointer_alignment"),
        (sass_check, "verify_compiled"),
        (cute, "compile"),
    )
    missing = [
        f"{module.__name__}.{name}"
        for module, name in required
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"Helion CuTe lowering needs {_DSL_REQUIREMENT}; "
            f"this install lacks {', '.join(missing)}"
        )
    return launcher, arg_signature, sass_check, cute


def _compile_and_export(
    captured: _CapturedLaunch, prefix: str, workdir: Path
) -> tuple[Any, tuple[Any, ...], list[int]]:
    """Compile the launch with Helion's own wrapper and export it to C."""
    launcher, arg_signature, sass_check, cute = _cute_launcher_modules()
    tensors = captured.tensors
    if launcher._cute_stage_under_aligned_operands(captured.cute_kernel, tensors):
        raise NotImplementedError(
            "Helion CuTe lowering does not support launches that copy "
            "under-aligned operands"
        )
    launch = launcher._build_cute_schema_and_args(
        captured.cute_kernel, tensors, captured.grid
    )
    schema = tuple(launch.schema)
    if launch.owned_tensors or len(schema) != len(tensors):
        raise NotImplementedError(
            "Helion CuTe lowering does not support launches with workspace tensors"
        )
    alignments = []
    for entry in schema:
        if entry[0] != "tensor" or len(entry) not in (5, 6):
            raise NotImplementedError(
                f"Helion CuTe lowering supports static tensor arguments only, got {entry[0]!r}"
            )
        alignments.append(arg_signature.schema_pointer_alignment(entry))
    launch_args = tuple(launch.launch_args)
    if (
        len(launch_args) != len(tensors) + 3
        or tuple(int(a) for a in launch_args[-3:]) != captured.grid
    ):
        raise NotImplementedError("Helion CuTe lowering found an unexpected launch ABI")
    # The steps Helion's own compile path runs before and after cute.compile.
    launcher._ensure_cute_dsl_arch_env(tensors)
    jit_function = launcher._create_cute_wrapper(
        captured.cute_kernel,
        schema,
        captured.block,
        num_sm=launcher._cute_num_sm_from_arch_args(tensors),
        device=tensors[0].device,
    )
    stream = launcher._cute_current_stream()
    options = " ".join(
        o for o in (captured.compile_options or "").split() if o != _TVM_FFI_OPTION
    )
    if options:
        compiled = cute.compile(jit_function, *launch_args, stream, options=options)
    else:
        compiled = cute.compile(jit_function, *launch_args, stream)
    # Helion sets _helion_cute_sass_checks only on kernels that declared SASS
    # checks, so a missing attribute means none were requested.
    sass_checks = tuple(getattr(captured.cute_kernel, "_helion_cute_sass_checks", ()))
    sass_check.verify_compiled(compiled, sass_checks, prefix)
    compiled.export_to_c(
        file_path=str(workdir), file_name=prefix, function_prefix=prefix
    )
    header = (workdir / f"{prefix}.h").read_text()
    expected = len(tensors) + 3
    signature = re.search(rf"cute_dsl_{prefix}_wrapper\(([^)]*)\)", header)
    if signature is None or len(signature.group(1).split(",")) != expected + 2:
        raise NotImplementedError("Helion CuTe lowering found an unexpected C wrapper")
    return compiled, schema, alignments


def _build_library(
    prefix: str,
    workdir: Path,
    tensors: tuple[torch.Tensor, ...],
    alignments: list[int],
    grid: tuple[int, int, int],
) -> str:
    """Compile the C shim and archive it with the exported kernel object."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("Helion CuTe lowering needs a CUDA toolkit (CUDA_HOME)")
    shim = f"aoti_torch_cuda_{prefix}"
    count = len(tensors)
    parameters = ", ".join(f"AtenTensorHandle t{i}" for i in range(count))

    layout_arrays = "\n".join(
        f"constexpr int64_t k{name}{i}[] = {{{', '.join(map(str, values)) or '0'}}};"
        for i, t in enumerate(tensors)
        for name, values in (("Sizes", t.shape), ("Strides", t.stride()))
    )

    (workdir / "shim.cpp").write_text(
        _SHIM_TEMPLATE.format(
            prefix=prefix,
            shim=shim,
            parameters=parameters,
            count=count,
            ranks=", ".join(str(t.dim()) for t in tensors),
            layout_arrays=layout_arrays,
            sizes=", ".join(f"kSizes{i}" for i in range(count)),
            strides=", ".join(f"kStrides{i}" for i in range(count)),
            handles=", ".join(f"t{i}" for i in range(count)),
            alignments=", ".join(str(a) for a in alignments),
            pointer_args=", ".join(f"pointers[{i}]" for i in range(count)),
            grid=", ".join(str(g) for g in grid),
        )
    )
    try:
        subprocess.run(
            [
                get_cpp_compiler(),
                "-c",
                "-fPIC",
                "-O2",
                "-std=c++17",
                f"-I{workdir}",
                f"-I{CUDA_HOME}/include",
                str(workdir / "shim.cpp"),
                "-o",
                str(workdir / "shim.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                shutil.which("ar") or "ar",
                "rcs",
                str(workdir / f"lib{prefix}.a"),
                str(workdir / "shim.o"),
                str(workdir / f"{prefix}.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Helion CuTe lowering failed to build the launch library: "
            f"{error.stderr}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Helion CuTe lowering cannot run the C++ compiler or ar: {error}"
        ) from error
    return f"AOTITorchError {shim}({parameters}, AtenTensorHandle* ret)"


def _register_op(
    prefix: str,
    mutable: Sequence[bool],
    launch: Callable[[tuple[torch.Tensor, ...]], None],
) -> Any:
    if len(mutable) > len(string.ascii_lowercase):
        raise NotImplementedError(
            "Helion CuTe lowering supports at most 26 tensor arguments"
        )
    arguments = ", ".join(
        f"Tensor({string.ascii_lowercase[i]}!) t{i}" if m else f"Tensor t{i}"
        for i, m in enumerate(mutable)
    )
    if _STATE.library is None:
        _STATE.library = torch.library.Library(_OP_NAMESPACE, "FRAGMENT")
    library = _STATE.library
    # The kernel is compiled for the exported sizes and strides.
    library.define(
        f"{prefix}({arguments}) -> ()", tags=(torch.Tag.needs_exact_strides,)
    )

    def run(*tensors: torch.Tensor) -> None:
        launch(tensors)

    def fake(*tensors: torch.Tensor) -> None:
        return None

    library.impl(prefix, run, "CUDA")
    torch.library.register_fake(f"{_OP_NAMESPACE}::{prefix}", fake, lib=library)
    return getattr(getattr(torch.ops, _OP_NAMESPACE), prefix).default


def _export(
    prefix: str, captured: _CapturedLaunch, mutable: Sequence[bool]
) -> _ExportedLaunch:
    launcher = _cute_launcher_modules()[0]
    root = Path(cache_dir()) / "executorch_helion_cute"
    root.mkdir(parents=True, exist_ok=True)
    # A directory of its own per export, so a process never links an archive
    # another process wrote under the same name.
    library_directory = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=root))
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        compiled, schema, alignments = _compile_and_export(captured, prefix, workdir)
        declaration = _build_library(
            prefix, workdir, captured.tensors, alignments, captured.grid
        )
        shutil.copyfile(
            workdir / f"lib{prefix}.a", library_directory / f"lib{prefix}.a"
        )
    cute_kernel, grid = captured.cute_kernel, captured.grid
    compiled_args = captured.tensors

    def launch(tensors: tuple[torch.Tensor, ...]) -> None:
        # The eager twin of the C shim's check: same sizes, strides and dtype
        # as compiled for, and each pointer aligned at least as much as the
        # compiled alignment. An exactly matching Helion schema would also
        # reject a more-aligned pointer for an element-aligned operand, which
        # the shim (and the compiled kernel) accepts.
        if len(tensors) != len(compiled_args) or any(
            tensor.shape != expected.shape
            or tensor.stride() != expected.stride()
            or tensor.dtype != expected.dtype
            or tensor.data_ptr() % alignment != 0
            for tensor, expected, alignment in zip(
                tensors, compiled_args, alignments
            )
        ):
            raise RuntimeError(
                f"{prefix} was compiled for other argument layouts or alignments"
            )
        args = launcher._build_cute_schema_and_args(cute_kernel, tensors, grid)
        compiled(*args.launch_args, launcher._cute_current_stream())

    return _ExportedLaunch(
        _register_op(prefix, mutable, launch), declaration, prefix, library_directory
    )


def export_cute_launch(
    entrypoint: Callable[..., Any],
    launcher_parameter: str,
    source: str,
    fake_arguments: tuple[torch.Tensor, ...],
    mutated_arguments: frozenset[int],
    aligned_arguments: frozenset[int],
) -> tuple[Callable[..., None], Any]:
    """Export the single CuTe launch of ``entrypoint``.

    Returns a launcher with Helion's launcher signature that calls the custom
    op the launch was registered as, together with that op, so tracing the
    entrypoint with the launcher records the op in place of the launch and
    the caller can check for exactly that op.
    """
    with unset_fake_temporarily():
        arguments = tuple(
            _example_like(a, i in aligned_arguments)
            for i, a in enumerate(fake_arguments)
        )
        captured = _capture_launch(entrypoint, launcher_parameter, arguments)
        # The kernel writes the storages the host code allocated and those of
        # the inputs the captured call recorded as mutated.
        argument_of = {
            StorageWeakRef(a.untyped_storage()): i for i, a in enumerate(arguments)
        }
        mutable = []
        for tensor in captured.tensors:
            index = argument_of.get(StorageWeakRef(tensor.untyped_storage()))
            mutable.append(index is None or index in mutated_arguments)
        key = repr(
            (
                source,
                [
                    (t.shape, t.stride(), t.storage_offset(), t.dtype)
                    for t in captured.tensors
                ],
                captured.grid,
                captured.block,
                captured.compile_options,
                mutable,
                torch.cuda.get_device_capability(captured.tensors[0].device),
            )
        )
        prefix = f"helion_cute_{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        with _STATE.lock:
            exported = _STATE.exported.get(prefix)
            if exported is not None and not (
                exported.library_directory / f"lib{exported.library}.a"
            ).is_file():
                # The cached archive is gone (for example its cache directory
                # was cleared under a long-lived process); export it again
                # instead of linking a missing library.
                exported = None
            if exported is None:
                exported = _export(prefix, captured, mutable)
                _STATE.exported[prefix] = exported
    _STATE.program_launches()[prefix] = exported
    op = exported.op

    def launch_op(
        cute_kernel: Any, grid: Any, *args: torch.Tensor, **kwargs: Any
    ) -> None:
        op(*args)

    return launch_op, op


def begin_program() -> None:
    """Forget the launches of the previous program lowered on this thread."""
    _STATE.program_launches().clear()


def exported_fallback_kernels() -> dict[str, None]:
    """The custom ops the CUDA backend must accept as supported fallbacks."""
    return {f"{_OP_NAMESPACE}::{p}": None for p in _STATE.program_launches()}


def program_link_options() -> tuple[list[str], list[str]]:
    """Link directories and libraries for this thread's program launches.

    The CUDA backend adds these to the AOTInductor link of the calling
    thread only (through its compile patch), so no process-wide state
    changes while other threads lower.
    """
    launches = list(_STATE.program_launches().values())
    if not launches:
        return [], []
    directories = [str(e.library_directory) for e in launches]
    directories.append(str(_dsl_runtime_directory()))
    libraries = [e.library for e in launches] + [_DSL_RUNTIME_LIBRARY]
    return directories, libraries


@contextlib.contextmanager
def link_exported_launches() -> Iterator[None]:
    """Let AOTInductor call the CuTe launches of this thread's program.

    Declares the launches' C shims to AOTInductor. Inductor config patches
    are thread-local, and the archives themselves are added by the CUDA
    backend's thread-scoped compile patch reading ``program_link_options``.
    """
    launches = list(_STATE.program_launches().values())
    if not launches:
        yield
        return
    shims = dict(torch._inductor.config.aot_inductor.custom_ops_to_c_shims)
    shims.update({e.op: [e.shim_declaration] for e in launches})
    try:
        with torch._inductor.config.patch(
            {"aot_inductor.custom_ops_to_c_shims": shims}
        ):
            yield
    finally:
        begin_program()
