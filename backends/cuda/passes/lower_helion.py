# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower native Helion kernel calls to launches AOTInductor can compile.

AOTInductor cannot compile Helion's higher-order ops into a standalone C++
wrapper. This pass renders each Helion call with its single, explicitly selected
config, traces the generated host code with fake tensors and splices the result
into the exported program:

- Triton kernels become ordinary ``wrap_triton`` launches, which AOTInductor
  compiles like any user Triton kernel.
- CuTe kernels are compiled now and become calls to a custom op whose C
  implementation AOTInductor links into its ``.so`` (see ``helion_cute.py``).

Helion is imported only when a Helion call is found.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import importlib
import importlib.metadata
import inspect
import operator
import warnings
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, cast, NamedTuple

import torch

from executorch.backends.cuda.passes import helion_cute
from torch._guards import detect_fake_mode
from torch._inductor.codecache import PyCodeCache
from torch._library.triton import set_wrap_triton_enabled
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature, InputKind
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import map_arg
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from triton.runtime.jit import JITFunction

# Launch options AOTI reproduces; anything else would be silently dropped.
# wrap_triton records launch kwargs by name on the captured kernel call, and
# Inductor materializes them as backend options for the AOTI compilation.
_SUPPORTED_LAUNCH_OPTIONS = ("num_warps", "num_stages", "maxnreg", "enable_fp_fusion")

# Helion has no public API for a call that torch.export captured: the HOPs, their
# side table and their output spec belong to Helion's private torch.compile
# integration, and so does the launcher parameter of a standalone entrypoint. The
# pass reads that contract only through this section and otherwise uses Helion's
# public API (Kernel.bind, BoundKernel.to_code). The CuTe path additionally reads
# Helion's private CuTe launcher, checked in helion_cute._cute_launcher_modules.
# The one other private trait the pass relies on is the functional HOP's
# (outputs, {name: clone}) return layout, which _make_hop_replacement mirrors
# and _reroute_hop_outputs indexes into.
_HELION_HOP_NAMES = (
    "helion_kernel_wrapper_mutation",
    "helion_kernel_wrapper_functional",
)
_LAUNCHER_PARAMETER = "_launcher"
_TRITON_BACKEND = "triton"
_CUTE_BACKEND = "cute"
_HELION_RUNTIME_PREFIX = "helion."
_HELION_REQUIREMENT = "Helion 1.5.2 or newer"
_MIN_HELION_VERSION = (1, 5, 2)
_TESTED_HELION_VERSION = (1, 5, 2)


class _CapturedHelionCall(NamedTuple):
    kernel: Any
    is_functional: bool
    tensor_args: Mapping[str, Node]
    has_constant_args: bool
    tensors_to_clone: tuple[str, ...]
    tensor_output_count: int
    aliases_inputs: bool
    mutates_inputs: bool
    output_spec: Mapping[str, Any]


def _check_helion_version() -> None:
    try:
        version: str | None = importlib.metadata.version("helion")
    except importlib.metadata.PackageNotFoundError:
        # A Helion importable without distribution metadata (for example a
        # source checkout) may still name its version on the module. When no
        # version is discoverable, the attribute probes in
        # _helion_capture_modules are the version evidence available.
        helion = importlib.import_module("helion")
        version = getattr(helion, "__version__", None)
    if version is None:
        return
    parsed = []
    for part in version.split(".")[:3]:
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            raise RuntimeError(
                f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
                f"found Helion {version}"
            )
        parsed.append(int(digits))
    if tuple(parsed) < _MIN_HELION_VERSION or (
        tuple(parsed) == _MIN_HELION_VERSION
        and any(mark in version for mark in ("rc", "dev", ".a", ".b"))
    ):
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"found Helion {version}"
        )
    if tuple(parsed) > _TESTED_HELION_VERSION:
        warnings.warn(
            f"Helion {version} is newer than the newest tested version "
            f"({'.'.join(map(str, _TESTED_HELION_VERSION))}); the pass "
            "reads Helion's private capture state, whose meaning may "
            "have changed",
            stacklevel=2,
        )


@functools.cache
def _helion_capture_modules() -> tuple[ModuleType, ModuleType]:
    _check_helion_version()
    try:
        hops = importlib.import_module("helion._compiler._dynamo.higher_order_ops")
        variables = importlib.import_module("helion._compiler._dynamo.variables")
    except ImportError as error:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}"
        ) from error
    required = (
        (hops, "helion_kernel_side_table"),
        (hops, "helion_kernel_wrapper_functional"),
        (variables, "_get_flat_output"),
    )
    missing = [f"{m.__name__}.{name}" for m, name in required if not hasattr(m, name)]
    if missing:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"this Helion lacks {', '.join(missing)}"
        )
    return hops, variables


def _read_captured_call(node: Node) -> _CapturedHelionCall:
    hops, _ = _helion_capture_modules()
    kwargs = cast(Mapping[str, Any], node.kwargs)
    # Read the private fields strictly: a Helion that renames one must fail
    # with the requirement error instead of surfacing a bare KeyError. The
    # two exceptions are direct_aliases and mutated_inputs: Helion omits them
    # from a capture its functionalization rebuilt when they are empty, so a
    # missing entry means "none". For Triton launches the traced replacement
    # cross-checks both (torch's mutation analysis plus the placeholder and
    # aliasing checks below), with one exception: an output that aliases a
    # mutated input aliases the clone the trace launches on, which neither
    # post-trace check sees, so only direct_aliases rejects that case. CuTe
    # launches rely on Helion's report entirely: their op schema's writes
    # come from the same tensors_to_clone set, so an unreported CuTe kernel
    # write cannot be caught after tracing.
    try:
        spec = kwargs["output_spec"]
        is_functional = node.target is hops.helion_kernel_wrapper_functional
        return _CapturedHelionCall(
            kernel=hops.helion_kernel_side_table.get_kernel(kwargs["kernel_idx"]),
            is_functional=is_functional,
            tensor_args=kwargs["tensor_args"],
            has_constant_args=bool(kwargs["constant_args"]),
            tensors_to_clone=(
                tuple(kwargs["tensors_to_clone"]) if is_functional else ()
            ),
            tensor_output_count=sum(
                leaf["type"] == "tensor" for leaf in spec["leaf_specs"]
            ),
            aliases_inputs=bool(spec.get("direct_aliases")),
            mutates_inputs=bool(spec.get("mutated_inputs")),
            output_spec=spec,
        )
    except KeyError as error:
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
            f"this capture lacks the expected field {error}"
        ) from error


def _check_capture_semantics(bound: Any, captured: _CapturedHelionCall) -> None:
    """Reject a kernel whose host code or environment changed since export."""
    _, variables = _helion_capture_modules()
    expected = captured.output_spec.get("_host_semantic_fingerprint")
    if expected is None or not hasattr(bound, "_get_host_semantic_fingerprint"):
        raise RuntimeError(
            f"Lowering Helion calls needs {_HELION_REQUIREMENT}, which records "
            "host-semantic fingerprints"
        )
    leaves, _, _ = variables._get_flat_output(bound.host_function)
    normalization = captured.output_spec.get("_host_semantic_input_normalization", ())
    actual = bound._get_host_semantic_fingerprint(
        leaves, input_normalization=normalization
    )
    if actual != expected:
        raise RuntimeError(
            "Helion capture and CUDA lowering have different host semantics"
        )


# ---------------------------------------------------------------------------


class _HelionCall(NamedTuple):
    captured: _CapturedHelionCall
    argument_names: list[str]
    argument_nodes: tuple[Node, ...]
    fake_arguments: tuple[torch.Tensor, ...]


def _check_single_launch_site(source: str, entrypoint: str) -> None:
    module = ast.parse(source)
    entry = next(
        (
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == entrypoint
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Generated Triton source has no entrypoint {entrypoint!r}")
    launches = [
        n
        for n in ast.walk(entry)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == _LAUNCHER_PARAMETER
    ]
    if len(launches) != 1:
        raise NotImplementedError(
            "Helion lowering requires a single Triton launch site"
        )
    for keyword in launches[0].keywords:
        if keyword.arg in _SUPPORTED_LAUNCH_OPTIONS:
            continue
        if (
            keyword.arg == "launch_cooperative_grid"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
        ):
            continue
        raise NotImplementedError(
            f"Helion lowering does not support launch option {keyword.arg!r}; "
            "the CUDA/AOTI runtime must preserve its launch contract first"
        )


def _launch_through_wrap_triton(kernel, grid, *args, **kwargs):
    return torch.library.wrap_triton(kernel)[grid](*args, **kwargs)


def _parse_helion_call(captured: _CapturedHelionCall) -> _HelionCall:
    kernel = captured.kernel
    if kernel.settings.backend not in (_TRITON_BACKEND, _CUTE_BACKEND):
        raise NotImplementedError(
            "Helion lowering supports only the Triton and CuTe backends; this "
            f"kernel uses {kernel.settings.backend!r}"
        )
    if kernel.settings.backend == _CUTE_BACKEND and not kernel.settings.static_shapes:
        raise NotImplementedError(
            "Helion CuTe lowering requires a kernel defined with "
            "static_shapes=True; the CuTe DSL compiles one kernel for "
            "the exported sizes"
        )
    if len(kernel.configs) != 1:
        raise ValueError(
            "Select one Helion config offline with @helion.kernel(config=...) before CUDA lowering"
        )
    if captured.aliases_inputs:
        raise NotImplementedError(
            "Helion lowering does not yet support aliased outputs"
        )
    if not captured.is_functional and captured.mutates_inputs:
        raise NotImplementedError(
            "Mutating Helion calls must be functionalized before CUDA lowering"
        )
    names = list(kernel.signature.parameters)
    if (
        captured.has_constant_args
        or set(names) != set(captured.tensor_args)
        or not all(isinstance(v, Node) for v in captured.tensor_args.values())
    ):
        raise NotImplementedError(
            "Helion lowering currently requires tensor-only arguments"
        )
    nodes = tuple(captured.tensor_args[name] for name in names)
    call = _HelionCall(captured, names, nodes, tuple(n.meta["val"] for n in nodes))
    _check_arguments(call)
    return call


def _has_symbolic_sizes(value: torch.Tensor) -> bool:
    return any(not isinstance(size, int) for size in (*value.shape, *value.stride()))


def _check_arguments(call: _HelionCall) -> None:
    for value in call.fake_arguments:
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise NotImplementedError("Helion lowering requires strided tensor inputs")
    kernel = call.captured.kernel

    if any(_has_symbolic_sizes(fake) for fake in call.fake_arguments):
        if kernel.settings.backend == _CUTE_BACKEND:
            raise NotImplementedError(
                "Helion CuTe lowering does not support dynamic shapes; "
                "the CuTe DSL compiles one kernel for the exported sizes"
            )
        written = [
            fake
            for name, fake in zip(call.argument_names, call.fake_arguments)
            if name in call.captured.tensors_to_clone
        ]
        if any(_has_symbolic_sizes(fake) for fake in written):
            raise NotImplementedError(
                "Helion lowering does not support dynamic shapes for "
                "written arguments"
            )
        if kernel.settings.static_shapes:
            raise NotImplementedError(
                "Helion lowering of dynamic shapes needs a kernel defined "
                "with static_shapes=False, so its sizes stay runtime arguments"
            )
    for name, value in zip(call.argument_names, call.fake_arguments):
        # The kernel is specialized to the strides of the uncloned argument.
        if name in call.captured.tensors_to_clone and (
            value.clone().stride() != value.stride()
        ):
            raise NotImplementedError(
                "Helion lowering does not support mutation clones that change strides"
            )


def _bind_helion_jit_helpers(source: str) -> str:
    """Call Helion's in-kernel Triton helpers through global names.

    Generated kernels call helpers such as ``helion_triton_helpers.<helper>``.
    When AOTInductor compiles a user Triton kernel it copies the JIT functions
    the kernel references by global name, but not those reached through a
    module, so each such helper gets a global binding of its own name.
    """
    namespace = vars(PyCodeCache.load(source))
    tree = ast.parse(source)
    module_of: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        module = namespace.get(node.value.id)
        if not (
            inspect.ismodule(module)
            and module.__name__.startswith(_HELION_RUNTIME_PREFIX)
            and isinstance(getattr(module, node.attr, None), JITFunction)
        ):
            continue
        if node.attr in namespace or hasattr(builtins, node.attr):
            raise NotImplementedError(
                f"Helion helper {node.attr!r} clashes with a name in the generated module"
            )
        module_of[node.attr] = node.value.id
    if not module_of:
        return source

    class _Unqualify(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if (
                isinstance(node.value, ast.Name)
                and module_of.get(node.attr) == node.value.id
            ):
                return ast.copy_location(ast.Name(node.attr, ast.Load()), node)
            return self.generic_visit(node)

    tree = _Unqualify().visit(tree)
    for name, module_name in module_of.items():
        index = next(
            (
                i
                for i, stmt in enumerate(tree.body)
                if isinstance(stmt, (ast.Import, ast.ImportFrom))
                and any(
                    (alias.asname or alias.name) == module_name for alias in stmt.names
                )
            ),
            None,
        )
        if index is None:
            raise RuntimeError(
                f"Lowering Helion calls needs {_HELION_REQUIREMENT}; "
                f"generated source never imports {module_name}"
            )
        tree.body.insert(index + 1, ast.parse(f"{name} = {module_name}.{name}").body[0])
    return ast.unparse(ast.fix_missing_locations(tree))


def _render_entrypoint(call: _HelionCall) -> tuple[Callable[..., Any], str]:
    kernel = call.captured.kernel
    config = kernel.configs[0]
    if config.get("host_tensor_descriptors", False):
        raise NotImplementedError(
            "Helion lowering does not support host tensor descriptors"
        )
    dynamic = any(_has_symbolic_sizes(fake) for fake in call.fake_arguments)
    try:
        bound = kernel.bind(call.fake_arguments)
    except Exception as error:
        if not dynamic:
            raise
        # Helion cannot bind a kernel that specializes a symbolic size:
        # the specialized value would be baked into the compiled kernel
        # for the example size. Kernels that specialize only static
        # sizes bind fine, so keep Helion's error and name that cause.
        try:
            host_source = inspect.getsource(kernel.fn)
        except (OSError, TypeError):
            host_source = ""
        if "specialize(" in host_source:
            error.add_note(
                "this can happen when the kernel specializes a dynamic "
                "size with hl.specialize"
            )
        raise
    _check_capture_semantics(bound, call.captured)
    # The default output keeps Helion's runtime imports; with
    # allow_helion_deps=False Helion drops the import of its in-kernel helpers
    # without inlining them. AOTInductor compiles the kernels, so inference
    # still needs no Helion.
    source = bound.to_code(config)
    if kernel.settings.backend == _TRITON_BACKEND:
        _check_single_launch_site(source, kernel.name)
        source = _bind_helion_jit_helpers(source)
    return getattr(PyCodeCache.load(source), kernel.name), source


def _make_hop_replacement(
    call: _HelionCall,
    entrypoint: Callable[..., Any],
    launcher: Callable[..., Any],
) -> Callable[..., Any]:
    """Call the entrypoint and return outputs shaped like the Helion HOP's."""
    captured = call.captured
    # Placeholders at offset 0 start at their allocation, so a plain
    # clone is enough (and cheaper) for them; a view parameter keeps its
    # offset and goes through the gather like any other view.
    placeholders = {
        name
        for name, node in zip(call.argument_names, call.argument_nodes)
        if node.op == "placeholder"
    }

    def clone_for_write(name: str, tensor: torch.Tensor) -> torch.Tensor:
        # A plain clone of a view is materialized by Inductor as its own
        # kernel, which can drop the view's storage offset when the
        # consumer is an extern kernel call. Gathering through
        # index_select materializes the same copy with the load indexing
        # composed through the view. A dense tensor that is not
        # contiguous is gathered over its whole storage span and viewed
        # back with its own strides, so every dense layout is covered.
        if name not in placeholders or tensor.storage_offset() != 0:
            if tensor.is_contiguous():
                flat = tensor.reshape(-1)
                index = torch.arange(flat.numel(), device=flat.device)
                return flat.index_select(0, index).reshape(tensor.shape)
            if cast(Any, tensor).is_non_overlapping_and_dense():
                span = 1 + sum(
                    (size - 1) * stride
                    for size, stride in zip(tensor.shape, tensor.stride())
                )
                storage = tensor.as_strided((span,), (1,))
                index = torch.arange(span, device=tensor.device)
                copied = storage.index_select(0, index)
                return copied.as_strided(tensor.shape, tensor.stride())
        return tensor.clone()

    def run(*values):
        arguments = dict(zip(call.argument_names, values))
        cloned = {
            name: clone_for_write(name, arguments[name])
            for name in captured.tensors_to_clone
        }
        launch_args = (
            cloned.get(name, arguments[name]) for name in call.argument_names
        )
        outputs = entrypoint(*launch_args, **{_LAUNCHER_PARAMETER: launcher})
        flat = tuple(
            leaf
            for leaf in pytree.tree_leaves(outputs)
            if isinstance(leaf, torch.Tensor)
        )
        if len(flat) != captured.tensor_output_count:
            raise RuntimeError("Helion lowering changed the captured output structure")
        return (flat, cloned) if captured.is_functional else flat

    return run


def _check_no_placeholder_mutation(traced: GraphModule, kernel_name: str) -> None:
    placeholders = {n for n in traced.graph.nodes if n.op == "placeholder"}
    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        for index, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            if index < len(node.args):
                value = node.args[index]
            elif argument.name in node.kwargs:
                value = node.kwargs[argument.name]
            else:
                continue
            if any(leaf in placeholders for leaf in pytree.tree_leaves(value)):
                raise NotImplementedError(
                    f"Helion lowering traced a write to an input of kernel "
                    f"{kernel_name!r} that Helion did not report as mutated; "
                    "splicing it would mutate a graph input with no matching "
                    "graph-signature entry"
                )


def _weakest_aligned_arguments(
    calls: list[tuple[Node, frozenset[str]]],
) -> dict[Node, frozenset[int]]:
    """Weakest aligned argument indexes of each CuTe kernel across its calls.

    One kernel called first on a user input and then on an activation
    shares one launch exported at the intersection, instead of two
    launches: one compile and archive, at the cost of the aligned call
    losing its 16-byte alignment assumption. Calls that fail to parse are skipped here; tracing them
    raises the real error.
    """
    weakest: dict[Any, set[int]] = {}
    parsed: list[tuple[Node, Any, set[int]]] = []
    for node, aligned_names in calls:
        try:
            captured = _read_captured_call(node)
            if captured.kernel.settings.backend != _CUTE_BACKEND:
                continue
            call = _parse_helion_call(captured)
        except Exception:
            continue
        aligned = {
            i
            for i, n in enumerate(call.argument_nodes)
            if n.op == "placeholder" and n.name in aligned_names
        }
        # Only calls that can share a launch (same kernel and argument
        # shapes) may narrow each other's alignment. Strides are left
        # out: the refresh after each splice can change them, and a
        # wider group only narrows alignment, which stays correct.
        # Storage offsets are left out because the intersection gives
        # every call in a group the same example offsets, which is what
        # the prefix hashes. Two kernel objects built from the same
        # function and config (for example one per model block) export
        # the same launch, so group by name and configs, not by kernel
        # object.
        kernel = captured.kernel
        key = (
            (kernel.name, repr(kernel.configs)),
            tuple((tuple(t.shape), t.dtype) for t in call.fake_arguments),
        )
        parsed.append((node, key, aligned))
        if key in weakest:
            weakest[key] &= aligned
        else:
            weakest[key] = set(aligned)
    return {node: frozenset(weakest[key]) for node, key, _ in parsed}


def _trace_helion_call(
    node: Node,
    target_platform: str | None = None,
    aligned_arguments: frozenset[int] = frozenset(),
) -> tuple[GraphModule, tuple[Node, ...]]:
    captured = _read_captured_call(node)
    call = _parse_helion_call(captured)
    if (
        captured.kernel.settings.backend == _CUTE_BACKEND
        and target_platform == "windows"
    ):
        raise NotImplementedError(
            "Helion CuTe lowering is Linux-only; it cannot export a "
            "kernel for Windows"
        )
    entrypoint, source = _render_entrypoint(call)
    if call.captured.kernel.settings.backend == _CUTE_BACKEND:
        mutated = frozenset(
            i
            for i, name in enumerate(call.argument_names)
            if name in call.captured.tensors_to_clone
        )
        launcher, launch_op = helion_cute.export_cute_launch(
            entrypoint,
            _LAUNCHER_PARAMETER,
            source,
            call.fake_arguments,
            mutated,
            aligned_arguments,
        )

        def is_launch(n: Node) -> bool:
            return n.target is launch_op

    else:
        launcher = _launch_through_wrap_triton

        def is_launch(n: Node) -> bool:
            return "triton_kernel_wrapper" in str(n.target)

    run = _make_hop_replacement(call, entrypoint, launcher)
    if call.captured.kernel.settings.backend == _TRITON_BACKEND:
        run = torch.func.functionalize(run)
    # torch.func.functionalize rejects custom ops that write their arguments, so
    # a CuTe launch stays a write. It only writes tensors the replacement
    # allocates or clones, and AOTInductor functionalizes it when it compiles.
    with set_wrap_triton_enabled(True):
        traced = make_fx(run, tracing_mode="fake")(*call.fake_arguments)
    if not any(is_launch(n) for n in traced.graph.nodes):
        raise RuntimeError("Helion lowering lost the kernel launch during capture")
    _check_no_placeholder_mutation(traced, captured.kernel.name)
    _check_outputs_do_not_alias_inputs(traced)
    _check_index_range(
        call.captured.kernel,
        [
            leaf.meta["val"]
            for n in traced.graph.nodes
            if is_launch(n)
            for leaf in pytree.tree_leaves((n.args, n.kwargs))
            if isinstance(leaf, Node) and isinstance(leaf.meta.get("val"), torch.Tensor)
        ],
    )
    return traced, call.argument_nodes


def _record_shape_guards(
    fake_arguments: tuple[torch.Tensor, ...],
) -> tuple[Any | None, set[str], set[str]]:
    """The shape env of the call's symbolic sizes, its guards and symbols.

    The symbols are recorded before binding: an equality guard added
    while binding replaces its symbol with a constant in later
    expressions, so symbols re-read afterwards would miss it.
    """
    shape_env = None
    symbols: set[str] = set()
    for fake in fake_arguments:
        if not isinstance(fake, torch.Tensor) or fake.layout != torch.strided:
            continue
        for size in (*fake.shape, *fake.stride()):
            if isinstance(size, torch.SymInt):
                shape_env = size.node.shape_env
                symbols |= {str(s) for s in size.node.expr.free_symbols}
    if shape_env is None:
        return None, set(), symbols
    return shape_env, {str(guard[0]) for guard in shape_env.guards}, symbols


def _check_no_new_shape_guards(
    shape_env: Any | None,
    guards_before: set[str],
    symbols: set[str],
) -> None:
    """Reject guards binding or tracing added on exported sizes.

    A guard added here would specialize the compiled kernel for the
    example sizes, and nothing enforces it when the program runs at
    another size in the exported range.
    """
    if shape_env is None:
        return
    for guard in shape_env.guards:
        text = str(guard[0])
        if text in guards_before:
            continue
        if symbols & {str(s) for s in guard[0].free_symbols}:
            raise NotImplementedError(
                "Helion lowering does not support dynamic shapes for "
                f"this kernel: lowering it assumes {text}"
            )


def _check_index_range(kernel: Any, tensors: Any) -> None:
    """Reject dynamic ranges whose offsets can exceed 32-bit indexing.

    Helion indexes with 32-bit offsets unless told otherwise, and the
    compiled kernel is not rechecked against the exported range at run
    time, so the largest element offset of every tensor the launch
    indexes must stay in range for the whole exported range.
    """
    if kernel.settings.backend != _TRITON_BACKEND:
        return
    measures = []
    for fake in tensors:
        offset: Any = 0
        for size, stride in zip(fake.shape, fake.stride()):
            offset = offset + (size - 1) * stride
        measures.append((offset, "largest element offset"))
        measures.append((fake.numel(), "element count"))
    # An unbounded dimension is planned at the example size whatever
    # the index dtype, so it is rejected before the int64 exemption.
    for measure, what in measures:
        if isinstance(measure, torch.SymInt):
            upper = measure.node.shape_env.bound_sympy(measure.node.expr).upper
            if not upper <= 2**63:
                raise NotImplementedError(
                    "Helion lowering does not support dynamic shapes "
                    f"whose {what} is unbounded; give every dynamic "
                    "dimension a max="
                )
    if kernel.settings.index_dtype is torch.int64:
        return
    for measure, what in measures:
        if isinstance(measure, torch.SymInt):
            upper = measure.node.shape_env.bound_sympy(measure.node.expr).upper
            if upper > 2**31 - 1:
                raise NotImplementedError(
                    "Helion lowering does not support dynamic shapes "
                    f"whose {what} exceeds 2**31 - 1 without "
                    "index_dtype=torch.int64; for bounded ranges that "
                    "still exceed it, set index_dtype=torch.int64"
                )


def _check_outputs_do_not_alias_inputs(traced: GraphModule) -> None:
    inputs = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in traced.graph.nodes
        if n.op == "placeholder"
    }
    for output in pytree.tree_leaves(traced.graph.output_node().args[0]):
        if (
            isinstance(output, Node)
            and StorageWeakRef(output.meta["val"].untyped_storage()) in inputs
        ):
            raise NotImplementedError(
                "Helion lowering does not yet support outputs that alias inputs"
            )


def _splice_traced_graph(
    graph: Graph, before: Node, traced: GraphModule, inputs: tuple[Node, ...]
) -> Any:
    placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
    environment = dict(zip(placeholders, inputs))
    result = None
    with graph.inserting_before(before):
        for traced_node in traced.graph.nodes:
            if traced_node.op == "placeholder":
                continue
            if traced_node.op == "output":
                result = map_arg(traced_node.args[0], environment.__getitem__)
            elif traced_node.op == "call_function":
                copied = graph.node_copy(traced_node, environment.__getitem__)
                # Keep the Helion call's provenance on the replacement region;
                # node_copy alone keeps only the make_fx metadata.
                for key in ("stack_trace", "nn_module_stack", "from_node"):
                    if key in before.meta:
                        copied.meta[key] = before.meta[key]
                environment[traced_node] = copied
            else:
                raise NotImplementedError(
                    f"Helion lowering cannot inline {traced_node.op} nodes"
                )
    return result


def _reroute_hop_outputs(
    graph: Graph, node: Node, replacement: Any, signature: ExportGraphSignature | None
) -> None:
    """Replace ``node`` and its ``getitem`` users with the matching replacement nodes."""
    if isinstance(replacement, Node):
        _check_same_tensor_layout(node, replacement)
        node.replace_all_uses_with(replacement)
        if signature is not None:
            signature.replace_all_uses(node.name, replacement.name)
    else:
        for user in list(node.users):
            if user.op != "call_function" or user.target is not operator.getitem:
                raise NotImplementedError(
                    "Helion lowering requires captured outputs to be accessed by getitem"
                )
            _reroute_hop_outputs(graph, user, replacement[user.args[1]], signature)
    graph.erase_node(node)


def _check_same_tensor_layout(original: Node, replacement: Node) -> None:
    # The fake recorded for a Helion output can be stale in the same way
    # view fakes are (an empty_like kernel on a restored view produces
    # the view's layout, not the contiguous layout recorded for the
    # call), so strides may differ; downstream fakes are refreshed from
    # the replacement after splicing. Only a shape or dtype change
    # means the replacement is not the captured output.
    expected, actual = original.meta.get("val"), replacement.meta["val"]
    if not isinstance(expected, torch.Tensor):
        raise NotImplementedError(
            f"Helion lowering replaced node {original.name!r}, whose exported "
            "value is missing or not a tensor, with a tensor; the captured "
            "output structure does not match what the pass expected"
        )
    if (
        expected.dim() != actual.dim()
        or expected.dtype != actual.dtype
        or not all(
            statically_known_true(recorded == produced)
            for recorded, produced in zip(expected.shape, actual.shape)
        )
    ):
        raise NotImplementedError(
            f"Helion lowering produced {tuple(actual.shape)}/{actual.stride()}/{actual.dtype} for an output "
            f"recorded for the call as {tuple(expected.shape)}/{expected.stride()}/{expected.dtype}"
        )


def _touched(operands: Any, changed: set[Node], stale: set[Node]) -> bool:
    return any(
        isinstance(leaf, Node) and (leaf in changed or leaf in stale)
        for leaf in pytree.tree_leaves(operands)
    )


def _refresh_view_fakes(
    module: GraphModule,
    seeds: set[Node] | None = None,
    stale_seeds: set[Node] | None = None,
    replay_views: bool = True,
    reasons: dict[Node, str] | None = None,
) -> tuple[set[Node], set[int]]:
    """Recompute view nodes' fake tensors from their arguments' fakes.

    `ReplaceViewCopyWithViewPass` in `backends/aoti/passes` turns the
    edge dialect's `*_copy` ops back into views by swapping only
    `node.target`, so a restored view keeps the copy's fake: contiguous,
    at offset 0, whatever the view's real layout. Specializing a kernel
    on that fake compiles it for the wrong strides and silently reads
    the wrong elements. Replaying each view op on its arguments' fakes
    restores the real layout, including the pieces of a restored
    `split` reached through `getitem`. Ops computed from a restored
    view were specialized on the stale fake too, so their fakes are
    recomputed as the change propagates downstream, through `torch.cond`
    branches and outputs in graph order. Fakes of other ops are trusted
    as recorded: this pass only repairs layouts that restored views (or
    replacements it splices in itself) can invalidate.

    Returns the nodes whose layout could not be recomputed (stale) and
    the ids of every module the walk refreshed, so the caller can
    reject Helion calls that read stale fakes or sit in control flow
    this walk does not reach.
    """
    changed: set[Node] = set(seeds) if seeds else set()
    stale: set[Node] = set(stale_seeds) if stale_seeds else set()
    if reasons is None:
        reasons = {}
    changed |= stale
    reached = {id(module)}
    # Replaying a data-dependent op mints fresh unbacked symbols; keep
    # them from leaking into the program's shape environment, where
    # the next trace would trip over the pending symbols.
    mode = detect_fake_mode(
        [
            node.meta["val"]
            for node in module.graph.nodes
            if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor)
        ]
    )
    shape_env = mode.shape_env if mode is not None else None
    for node in module.graph.nodes:
        if node.op != "call_function" or "val" not in node.meta:
            continue
        # Helion calls themselves are replaced later, and their output
        # layouts are checked against the replacement then.
        if str(node.target) in _HELION_HOP_NAMES:
            continue
        if node.target is torch.ops.higher_order.cond:
            operands = node.args[3] if len(node.args) >= 4 else ()
            if not replay_views and not _touched(operands, changed, stale):
                continue
            cond_changed, cond_reached = _refresh_cond(module, node, stale, reasons)
            reached |= cond_reached
            if cond_changed:
                changed.add(node)
            continue
        _refresh_node_layout(node, changed, stale, reasons, replay_views, shape_env)
    return stale, reached


def _refresh_node_layout(
    node: Node,
    changed: set[Node],
    stale: set[Node],
    reasons: dict[Node, str],
    replay_views: bool,
    shape_env: Any | None,
) -> None:
    leaves = [
        leaf
        for leaf in pytree.tree_leaves((node.args, node.kwargs))
        if isinstance(leaf, Node)
    ]
    if any(leaf in stale for leaf in leaves):
        stale.add(node)
        changed.add(node)
        for leaf in leaves:
            if leaf in reasons:
                reasons[node] = reasons[leaf]
                break
        return
    is_view = getattr(node.target, "is_view", False) or (
        node.target is operator.getitem
    )
    downstream = any(leaf in changed for leaf in leaves)
    if not ((is_view and replay_views) or downstream):
        return
    try:
        args = map_arg(node.args, lambda n: n.meta["val"])
        kwargs = map_arg(node.kwargs, lambda n: n.meta["val"])
        with (
            shape_env.ignore_fresh_unbacked_symbols()
            if shape_env is not None
            else contextlib.nullcontext()
        ):
            value = node.target(*args, **kwargs)
    except Exception:
        stale.add(node)
        changed.add(node)
        return
    if _layout_differs(value, node.meta["val"]):
        node.meta["val"] = value
        changed.add(node)


def _layout_differs(value: Any, current: Any) -> bool:
    values = value if isinstance(value, (list, tuple)) else [value]
    currents = current if isinstance(current, (list, tuple)) else [current]
    try:
        matches = len(values) == len(currents) and all(
            isinstance(v, torch.Tensor)
            and isinstance(c, torch.Tensor)
            and v.dim() == c.dim()
            and all(
                statically_known_true(recorded == refreshed)
                for recorded, refreshed in zip(v.shape, c.shape)
            )
            and v.dtype == c.dtype
            for v, c in zip(values, currents)
        )
        return matches and any(
            not statically_known_true(v.storage_offset() == c.storage_offset())
            or any(
                not statically_known_true(a == b)
                for a, b in zip(v.stride(), c.stride())
            )
            for v, c in zip(values, currents)
        )
    except Exception:
        # Replaying a data-dependent op mints a fresh unbacked size,
        # and comparing it against the recorded one would guard on
        # data. Such an output has no fixed layout to refresh.
        return False


def _refresh_cond_branches(
    containing: GraphModule,
    node: Node,
    operands: Any,
    stale: set[Node],
    reasons: dict[Node, str],
) -> tuple[list[GraphModule] | None, set[int]]:
    """Refresh both branches of a cond; None if one cannot be resolved."""
    reached: set[int] = set()
    branch_modules = []
    for branch in node.args[1:3]:
        if not isinstance(branch, Node) or branch.op != "get_attr":
            stale.add(node)
            return None, reached
        try:
            branch_module = containing.get_submodule(cast(str, branch.target))
        except AttributeError:
            stale.add(node)
            return None, reached
        if not isinstance(branch_module, GraphModule):
            stale.add(node)
            return None, reached
        placeholders = [n for n in branch_module.graph.nodes if n.op == "placeholder"]
        stale_seeds = set()
        for placeholder, operand in zip(placeholders, operands):
            if isinstance(operand, Node) and "val" in operand.meta:
                placeholder.meta["val"] = operand.meta["val"]
                if operand in stale:
                    stale_seeds.add(placeholder)
                    if operand in reasons:
                        reasons[placeholder] = reasons[operand]
        branch_stale, branch_reached = _refresh_view_fakes(
            branch_module,
            seeds=set(placeholders),
            stale_seeds=stale_seeds,
            reasons=reasons,
        )
        stale |= branch_stale
        reached |= branch_reached
        branch_modules.append(branch_module)
        if any(
            n.op == "call_function" and str(n.target) in _HELION_HOP_NAMES
            for n in branch_module.graph.nodes
        ):
            # The branch output's real layout is only known once the
            # branch call is lowered, which happens after parent-graph
            # calls; a parent call reading this cond's output cannot be
            # specialized yet.
            stale.add(node)
            reasons[node] = "cond"
        output = branch_module.graph.output_node().args[0]
        if any(
            isinstance(leaf, Node) and leaf in branch_stale
            for leaf in pytree.tree_leaves(output)
        ):
            stale.add(node)
    return branch_modules, reached


def _refresh_cond(
    containing: GraphModule, node: Node, stale: set[Node], reasons: dict[Node, str]
) -> tuple[bool, set[int]]:
    """Refresh one cond's branches and output; True if its fake changed."""
    operands = node.args[3] if len(node.args) >= 4 else None
    if not isinstance(operands, (list, tuple)):
        stale.add(node)
        return True, set()
    branch_modules, reached = _refresh_cond_branches(
        containing, node, operands, stale, reasons
    )
    if branch_modules is None or node in stale:
        return True, reached
    try:
        outputs = [
            map_arg(
                branch_module.graph.output_node().args[0],
                lambda n: n.meta["val"],
            )
            for branch_module in branch_modules
        ]
        first = pytree.tree_leaves(outputs[0])
        second = pytree.tree_leaves(outputs[1])
        disagree = len(first) != len(second) or any(
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and (
                a.dim() != b.dim()
                or not all(
                    statically_known_true(x == y) for x, y in zip(a.shape, b.shape)
                )
                or not all(
                    statically_known_true(x == y)
                    for x, y in zip(a.stride(), b.stride())
                )
                or a.dtype != b.dtype
            )
            for a, b in zip(first, second)
        )
        # Keep the cond fake in this graph's mode: branch output fakes
        # belong to the branch graphs.
        converted = _in_recorded_mode(node.meta["val"], outputs[0])
    except Exception:
        stale.add(node)
        return True, reached
    if disagree:
        stale.add(node)
        return True, reached
    current = node.meta["val"]
    currents = current if isinstance(current, (list, tuple)) else [current]
    converted_values = (
        converted if isinstance(converted, (list, tuple)) else [converted]
    )
    changed = any(
        isinstance(v, torch.Tensor)
        and isinstance(c, torch.Tensor)
        and (
            not all(
                statically_known_true(a == b) for a, b in zip(v.stride(), c.stride())
            )
            or not statically_known_true(v.storage_offset() == c.storage_offset())
        )
        for v, c in zip(converted_values, currents)
    )
    node.meta["val"] = converted
    return changed, reached


def _in_recorded_mode(recorded: Any, refreshed: Any) -> Any:
    if isinstance(recorded, torch.Tensor) and isinstance(refreshed, torch.Tensor):
        return recorded.as_strided(
            tuple(refreshed.shape),
            refreshed.stride(),
            refreshed.storage_offset(),
        )
    if isinstance(recorded, (list, tuple)) and isinstance(refreshed, (list, tuple)):
        return type(recorded)(
            _in_recorded_mode(r, f) for r, f in zip(recorded, refreshed)
        )
    return refreshed


def _reject_if_stale(
    node: Node, stale: set[Node], reasons: dict[Node, str] | None = None
) -> None:
    leaves = [
        leaf
        for leaf in pytree.tree_leaves((node.args, node.kwargs))
        if isinstance(leaf, Node) and leaf in stale
    ]
    if not leaves:
        return
    if reasons is not None and any(reasons.get(leaf) == "cond" for leaf in leaves):
        raise NotImplementedError(
            "Helion lowering does not support a Helion call on the "
            "output of a torch.cond whose branches call Helion: the "
            "branch calls are lowered after parent-graph calls, so "
            "the output layout is not known when the parent call is "
            "specialized"
        )
    raise NotImplementedError(
        "Helion lowering cannot recompute the layout of "
        "an argument of the Helion call at "
        f"{node.name!r}, so it cannot specialize the "
        "kernel for its real strides"
    )


def _reject_stale_helion_calls(
    helion_modules: list[GraphModule],
    reached: set[int],
    stale: set[Node],
    reasons: dict[Node, str] | None = None,
) -> None:
    for module in helion_modules:
        if id(module) not in reached:
            raise NotImplementedError(
                "Helion lowering does not support Helion calls in "
                "nested control flow other than torch.cond branches"
            )
        for node in module.graph.nodes:
            if node.op == "call_function" and str(node.target) in _HELION_HOP_NAMES:
                _reject_if_stale(node, stale, reasons)


def _refresh_program_fakes(
    root: GraphModule, modules: list[GraphModule]
) -> tuple[set[Node], dict[Node, str]]:
    """Refresh view fakes for the Helion calls in one program.

    Rejects Helion calls in control flow the cond walk does not reach
    (for example a `while_loop` or `scan` body) instead of specializing
    them on stale fakes.
    """
    helion_modules = [
        module
        for module in modules
        if any(
            node.op == "call_function" and str(node.target) in _HELION_HOP_NAMES
            for node in module.graph.nodes
        )
    ]
    if not helion_modules:
        return set(), {}
    reasons: dict[Node, str] = {}
    stale, reached = _refresh_view_fakes(root, reasons=reasons)
    _reject_stale_helion_calls(helion_modules, reached, stale, reasons)
    return stale, reasons


class LowerHelionPass:
    """Replace native Helion calls with launches of their selected config.

    Supported: Triton and CuTe kernels with one explicit config, strided
    tensor arguments (dynamic shapes on the Triton backend for kernels
    defined with static_shapes=False, except for written arguments and
    kernels that specialize a dynamic size with hl.specialize, ranges
    whose largest element offset or element count can exceed 2**31 - 1
    without index_dtype=torch.int64, and kernels whose lowering adds a guard on
    an exported size; CuTe kernels must be defined with
    static_shapes=True), multiple tensor outputs and functionalized input mutation.
    Any number of distinct launches may be lowered: each distinct CuTe
    launch is exported under its own prefix and linked from its own
    archive, and calls with the same config and argument layouts share
    one launch.
    Config selection stays offline; the pass neither tunes nor launches kernels
    (CuTe kernels are compiled for the export GPU), and inference needs neither
    Python nor Helion. CuTe kernels are rejected when ``target_platform`` is
    ``"windows"``: their shim and archive are Linux artifacts.
    """

    requires_exported_program = True

    def __init__(self, target_platform: str | None = None) -> None:
        self.target_platform = target_platform

    def __call__(self, exported_program: ExportedProgram) -> None:
        helion_cute.begin_program()
        root = exported_program.graph_module
        modules = [m for m in root.modules() if isinstance(m, GraphModule)]
        stale, reasons = _refresh_program_fakes(root, modules)
        # A program an earlier pass run already lowered contains helion_cute
        # ops but no Helion calls; their archives still have to link.
        helion_cute.note_existing_launches(modules)
        if self.target_platform == "windows" and (
            helion_cute.current_program_launches()
        ):
            raise NotImplementedError(
                "Helion CuTe lowering is Linux-only; it cannot export a "
                "kernel for Windows"
            )
        signature = exported_program.graph_signature
        # The operands a CuTe kernel may assume start at their allocation:
        # the root graph's user inputs, parameters and buffers. One that is
        # itself a view at an offset that is not 16-byte aligned does not
        # start at its allocation, so it is compiled for element alignment
        # like every other operand; that also keeps two calls that differ
        # only in such an offset on one shared launch.
        placeholders = {
            node.name: node for node in root.graph.nodes if node.op == "placeholder"
        }

        def starts_aligned(name: str) -> bool:
            fake = placeholders[name].meta.get("val")
            return (
                not isinstance(fake, torch.Tensor)
                or (fake.storage_offset() * fake.element_size()) % 16 == 0
            )

        aligned_names = frozenset(
            spec.arg.name
            for spec in signature.input_specs
            if spec.kind
            in (InputKind.USER_INPUT, InputKind.PARAMETER, InputKind.BUFFER)
            and spec.arg.name in placeholders
            and starts_aligned(spec.arg.name)
        )
        weakest = _weakest_aligned_arguments(
            [
                (node, aligned_names if module is root else frozenset())
                for module in modules
                for node in module.graph.nodes
                if node.op == "call_function" and str(node.target) in _HELION_HOP_NAMES
            ]
        )
        for module in modules:
            is_root = module is root
            modified = False
            for node in list(module.graph.nodes):
                if (
                    node.op != "call_function"
                    or str(node.target) not in _HELION_HOP_NAMES
                ):
                    continue
                try:
                    _reject_if_stale(node, stale, reasons)
                    guard_fakes = tuple(
                        arg.meta["val"]
                        for arg in _read_captured_call(node).tensor_args.values()
                        if isinstance(arg, Node) and "val" in arg.meta
                    )
                    shape_env, guards_before, symbols = _record_shape_guards(
                        guard_fakes
                    )
                    traced, inputs = _trace_helion_call(
                        node,
                        self.target_platform,
                        weakest.get(node, frozenset()),
                    )
                    replacement = _splice_traced_graph(
                        module.graph, node, traced, inputs
                    )
                    _reroute_hop_outputs(
                        module.graph,
                        node,
                        replacement,
                        signature if is_root else None,
                    )
                    # Nodes computed from the replacement were specialized
                    # on the layout recorded for the call; refresh them
                    # from the replacement before later calls trace.
                    seeds = {
                        leaf
                        for leaf in pytree.tree_leaves(replacement)
                        if isinstance(leaf, Node)
                    }
                    new_stale, _ = _refresh_view_fakes(
                        module, seeds=seeds, replay_views=False
                    )
                    stale |= new_stale
                    _check_no_new_shape_guards(shape_env, guards_before, symbols)
                except Exception as error:
                    kernel_name = "unknown"
                    try:
                        kernel_name = _read_captured_call(node).kernel.name
                    except Exception:
                        pass
                    error.add_note(
                        f"while lowering Helion kernel {kernel_name!r} "
                        f"(node {node.name})"
                    )
                    raise
                modified = True
            if modified:
                module.graph.lint()
                module.recompile()
