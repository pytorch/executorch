# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Core AI backend for delegating an EdgeProgram to Apple's Core AI framework
# via the ``coreai-torch`` converter.

import copy
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, final, List, Optional, Tuple

from executorch.backends.apple.coreai.compiler.constants import MAIN_ENTRYPOINT
from executorch.backends.apple.coreai.compiler.enumerated_shapes import (
    apply_enumerated_shapes,
)
from executorch.backends.apple.coreai.compiler.io_compat import (
    _edge_io,
    assert_io_compatible,
    coreai_io_names,
)
from executorch.backends.apple.coreai.passes.replace_copy_ops import (
    ReplaceCopyOpsWithFunctionalPass,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects.edge._ops import EdgeOpOverload

logger = logging.getLogger(__name__)

# Alignment for the embedded asset files so the large blobs (e.g. ``main.mlirb``)
# are mmap-friendly on device.  Applied to every file flattened into the NDS.
_ASSET_ALIGNMENT = 16


class COMPILE_SPEC_KEYS(Enum):
    # Minimum OS

    # Minimum OS / deployment version.  This is a save-time property of the
    # .aimodel (save_asset(minimum_os=...)) that ALSO feeds coreai-build, so it
    # applies to every delivery path, AOT or not.  Numeric string, e.g. "27.0".
    MIN_DEPLOYMENT_VERSION = "coreai_min_deployment_version"

    # Enumerated input shapes for this delegate, derived by the partitioner from
    # ET-input enumerations and propagated to this subgraph via torch.export
    # symbols.  General (applies to every delivery mode, AOT or not).  JSON:
    #   {symbol_name: [value, ...]}   # e.g. {"s31": [4, 16, 32]}
    # Preprocess substitutes these into each user-input placeholder's symbolic
    # shape and attaches the results via AIProgram.set_static_shape_config.
    INPUT_ENUMERATIONS = "coreai_input_enumerations"

    # Ahead-of-time compilation (xcrun coreai-build).
    # Presence implies compiling the .aimodel to per-architecture .aimodelc
    # bundles at build time (requires macOS + the Metal Toolchain). The value is
    # a JSON object of build-only coreai-build options; an empty {} means all
    # defaults:
    #   {"platform": "iOS|macOS|watchOS|visionOS|tvOS",
    #    "preferred_compute": "gpu|neural-engine|none",
    #    "architectures": [...],            # empty/absent => all supported
    #    "expect_frequent_reshapes": bool}
    # (min_deployment_version is a separate general spec that also applies to
    # the portable .aimodel, so it is not part of this AOT-only blob.)
    AOT_COMPILE_CONFIG = "coreai_aot_compile_config"

    # Lower every Core AI partition of every method into one Core AI model, one
    # function per partition named <method>_<partition index>. Value: b"1".
    # Every partition must carry identical compile specs.
    MULTIFUNCTION = "coreai_multifunction"


class AssetPackaging(str, Enum):
    # Portable .aimodel embedded in the .pte via NamedDataStore.
    AIMODEL = "aimodel"
    # AOT-compiled per-arch .aimodelc bundles embedded in the .pte.
    AIMODELC = "aimodelc"


@dataclass(frozen=True)
class AOTCompileConfig:
    """Build-time ``coreai-build`` options for AOT compilation.

    Constructible in Python and (de)serializable to/from JSON so it can ride as
    the ``AOT_COMPILE_CONFIG`` compile spec. ``min_deployment_version`` is not
    part of this object; it is a general spec (it also sets the portable
    .aimodel's OS floor), so it is passed separately to the partitioner.
    """

    platform: Optional[str] = None  # iOS / macOS / watchOS / visionOS / tvOS
    preferred_compute: Optional[str] = None  # gpu / neural-engine / none
    architectures: Optional[List[str]] = None  # None / empty => all supported
    expect_frequent_reshapes: bool = False

    # Fields allowed in the JSON form (kept in sync with the dataclass fields).
    _ALLOWED = (
        "platform",
        "preferred_compute",
        "architectures",
        "expect_frequent_reshapes",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Minimal JSON-able dict (omits None/defaults)."""
        d: Dict[str, Any] = {}
        if self.platform is not None:
            d["platform"] = self.platform
        if self.preferred_compute is not None:
            d["preferred_compute"] = self.preferred_compute
        if self.architectures:
            d["architectures"] = list(self.architectures)
        if self.expect_frequent_reshapes:
            d["expect_frequent_reshapes"] = True
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOTCompileConfig":
        unexpected = set(data) - set(cls._ALLOWED)
        if unexpected:
            raise ValueError(
                f"unexpected AOTCompileConfig field(s): {sorted(unexpected)}; "
                f"allowed: {sorted(cls._ALLOWED)}"
            )
        arch = data.get("architectures")
        if arch is not None and not isinstance(arch, list):
            # A bare string would splat into one flag per character.
            raise ValueError(
                f"architectures must be a list, got {type(arch).__name__}: {arch!r}"
            )
        return cls(
            platform=data.get("platform"),
            preferred_compute=data.get("preferred_compute"),
            architectures=list(arch) if arch else None,
            expect_frequent_reshapes=bool(data.get("expect_frequent_reshapes", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "AOTCompileConfig":
        return cls.from_dict(json.loads(text))


def _get_compile_spec(
    compile_specs: List[CompileSpec], key: COMPILE_SPEC_KEYS
) -> Optional[bytes]:
    for spec in compile_specs:
        if spec.key == key.value:
            return spec.value
    return None


def _nds_key(export_hash: str, relative_path: str) -> str:
    # Byte-identical exports share embedded assets under their content hash.
    return f"coreai/{export_hash}/{relative_path}"


def _asset_metadata(staging: Path) -> Dict[str, Any]:
    """Per-file sizes and a SHA-256 digest per bundle under ``staging``.

    Each bundle digest covers a fixed prefix, the file count, then per file (in
    UTF-8 byte order) the path length, path, size and contents.
    """
    sizes: Dict[str, int] = {}
    digests: Dict[str, str] = {}
    for bundle in sorted(p for p in staging.iterdir() if p.is_dir()):
        files = sorted(
            (
                (path.relative_to(staging).as_posix(), path)
                for path in bundle.rglob("*")
                if path.is_file()
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
        digest = hashlib.sha256(b"ExecuTorch.CoreAI.raw-assets.sha256.v1\0")
        digest.update(len(files).to_bytes(8, "big"))
        for rel, path in files:
            payload = path.read_bytes()
            name = rel.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(payload).to_bytes(8, "big") + payload)
            sizes[rel] = len(payload)
        digests[bundle.name] = digest.hexdigest()
    return {"files": sizes, "bundle_digests": digests}


def _deliver(
    staging: Path,
    packaging: AssetPackaging,
    manifest_extra: Dict[str, Any],
    bindings: List[Dict[str, Any]],
) -> List[PreprocessResult]:
    """Embed a staging directory of bundles in the .pte, one manifest per binding.

    ``staging`` holds the ``.aimodel`` / ``.aimodelc`` bundles, never their
    contents, so assets land under ``coreai/<export_hash>/<bundle>/`` in the
    NamedDataStore. ``manifest_extra`` paths are staging-relative bundle names.
    Each binding (``function``, ``input_names``, ``output_names``) gets its own
    manifest; all of them reference the same embedded bundles.
    """
    metadata = _asset_metadata(staging)
    # The SDK program hash excludes metadata such as creation timestamps.
    export_hash = hashlib.sha256(
        json.dumps(metadata["bundle_digests"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_extra, **metadata}
    if packaging == AssetPackaging.AIMODEL:
        manifest["path"] = f"{export_hash}/{manifest['path']}"
    else:
        manifest["archs"] = {
            arch: f"{export_hash}/{bundle}"
            for arch, bundle in manifest["archs"].items()
        }
    store_output = _embed_dir(staging, export_hash)
    return [
        PreprocessResult(
            processed_bytes=json.dumps(
                {
                    "packaging": packaging.value,
                    "export_hash": export_hash,
                    **binding,
                    **manifest,
                }
            ).encode("utf-8"),
            data_store_output=store_output,
        )
        for binding in bindings
    ]


def _prepare_program_for_conversion(edge_program: ExportedProgram) -> ExportedProgram:
    """Rewrite an edge-dialect program into the ATen form ``coreai-torch`` expects.

    ``coreai-torch`` keys its lowering table on plain ATen overload names
    (``"addmm.default"``), but ExecuTorch hands the backend edge ops whose
    ``__name__`` is prefixed (``"aten.addmm.default"``) and whose view ops are
    functionalized ``*_copy`` variants.  So we:

    1. Remap edge ``*_copy`` ops to their functional ATen forms
       (:class:`ReplaceCopyOpsWithFunctionalPass`), then
    2. Unwrap any remaining ``EdgeOpOverload`` to its underlying ATen overload.
    """
    ep = copy.deepcopy(edge_program)
    ReplaceCopyOpsWithFunctionalPass()(ep.graph_module)
    for node in ep.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, EdgeOpOverload):
            node.target = node.target._op
    ep.graph_module.recompile()
    return ep


def _convert_to_aiprogram(
    entries: List[Tuple[ExportedProgram, str, List[str], List[str]]],
):
    """Convert ``(edge_program, entrypoint, input_names, output_names)`` entries
    into one Core AI program with one function per entry."""
    from coreai_torch import TorchConverter

    converter = TorchConverter()
    for edge_program, entrypoint, input_names, output_names in entries:
        converter.add_exported_program(
            _prepare_program_for_conversion(edge_program),
            input_names=input_names,
            output_names=output_names,
            entrypoint_name=entrypoint,
        )
    return converter.to_coreai()


def _boundary_names(edge_program: ExportedProgram) -> Tuple[List[str], List[str]]:
    mutated_inputs = edge_program.graph_signature.user_inputs_to_mutate
    if mutated_inputs:
        raise ValueError(
            "Core AI delegates cannot mutate ExecuTorch-owned user inputs: "
            f"{sorted(mutated_inputs.values())}"
        )
    edge_inputs, edge_outputs = _edge_io(edge_program)
    return (
        [f"input_{i}" for i in range(len(edge_inputs))],
        [f"output_{i}" for i in range(len(edge_outputs))],
    )


def _binding(
    program,
    edge_program: ExportedProgram,
    entrypoint: str,
    input_names: List[str],
    output_names: List[str],
) -> Dict[str, Any]:
    # Fail fast if the .aimodel boundary I/O won't match what ET feeds/reads.
    assert_io_compatible(
        program,
        edge_program,
        input_names=input_names,
        output_names=output_names,
        entrypoint=entrypoint,
    )
    bound_inputs, bound_outputs = coreai_io_names(program, entrypoint)
    return {
        "function": entrypoint,
        "input_names": bound_inputs,
        "output_names": bound_outputs,
    }


def _enumerate_binding(
    program,
    edge_program: ExportedProgram,
    binding: Dict[str, Any],
    input_names: List[str],
    compile_specs: List[CompileSpec],
) -> Dict[str, Any]:
    """Apply this partition's enumerated shapes and rewrite its binding.

    ``set_static_shape_config`` erases the bound function, so a specialized
    delegate lists every per-shape function with its input shapes, in
    ``input_names`` order, and the runtime picks one by the inputs it gets.
    """
    raw = _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.INPUT_ENUMERATIONS)
    if not raw:
        return binding
    entrypoint = binding["function"]
    shapes_config = apply_enumerated_shapes(
        program,
        edge_program,
        json.loads(raw.decode()),
        input_names=input_names,
        entrypoint=entrypoint,
    )
    if not shapes_config:
        return binding
    bound_inputs = binding["input_names"]
    functions = []
    for key, shapes in shapes_config.items():
        if set(shapes) != set(bound_inputs):
            raise ValueError(
                f"enumerated shapes for {entrypoint!r} cover inputs "
                f"{sorted(shapes)} but the delegate binds {bound_inputs}"
            )
        functions.append(
            {
                "function": f"{entrypoint}_{key}",
                "input_shapes": [list(shapes[name]) for name in bound_inputs],
            }
        )
    enumerated = {k: v for k, v in binding.items() if k != "function"}
    enumerated["enumerated_functions"] = functions
    return enumerated


def _check_unique_functions(bindings: List[Dict[str, Any]]) -> None:
    """Reject two delegates binding the same function name.

    Enumerated functions are named ``<method>_<index>_<key>``, which a method
    whose name looks like another method's index and key could reproduce.
    """
    seen = set()
    for binding in bindings:
        names = (
            [binding["function"]]
            if "function" in binding
            else [f["function"] for f in binding["enumerated_functions"]]
        )
        for name in names:
            if name in seen:
                raise ValueError(
                    f"multifunction export generates the function name {name!r} "
                    "twice; rename the method whose name collides with it"
                )
            seen.add(name)


def _spec_items(
    compile_specs: List[CompileSpec], ignore: Tuple[str, ...] = ()
) -> List[Tuple[str, bytes]]:
    return sorted(
        (spec.key, bytes(spec.value))
        for spec in compile_specs
        if spec.key not in ignore
    )


# Asset embedding helpers.
def _embed_dir(root_dir: Path, export_hash: str):
    """Flatten every file under ``root_dir`` into a NamedDataStore output.

    ``root_dir`` is the directory *containing* the ``.aimodel`` / ``.aimodelc``
    bundles, never a bundle itself, so the bundle name survives into the keys.

    Keys are ``coreai/{export_hash}/{relpath}``.
    """
    store = NamedDataStore()
    for path in sorted(root_dir.rglob("*")):
        if path.is_file():
            store.add_named_data(
                _nds_key(export_hash, path.relative_to(root_dir).as_posix()),
                path.read_bytes(),
                alignment=_ASSET_ALIGNMENT,
            )
    return store.get_named_data_store_output()


def _os_version_text(min_os) -> Optional[str]:
    """The floor actually baked into a portable asset, as ``major.minor``.

    ``save_asset`` takes an ``OSVersion``, which carries only a major version,
    so a spec of ``"27.5"`` yields a v27 asset. Deriving the manifest value
    from the OSVersion rather than the raw spec keeps the two from disagreeing.
    The AOT route reports :func:`_min_os_text` instead, since ``coreai-build``
    honours the minor version it is given.
    """
    return None if min_os is None else f"{min_os.value.lstrip('v')}.0"


# AOT (coreai-build) helpers.
def _min_os_text(compile_specs: List[CompileSpec]) -> Optional[str]:
    """The MIN_DEPLOYMENT_VERSION spec as ``major[.minor[.patch]]``, or None.

    The two consumers want different spellings of the same thing: ``coreai.
    authoring.OSVersion`` accepts only names like ``"v27"``, while
    ``coreai-build --min-deployment-version`` rejects them and wants a numeric
    version. Users may write either, so this is the one place that normalizes;
    ``_min_os_version`` renders the OSVersion form for ``save_asset`` and this
    numeric form is what reaches coreai-build and the manifest.
    """
    raw = _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION)
    if raw is None:
        return None
    text = raw.decode()
    if text.startswith("v"):
        text = text[1:]
    # Canonicalize to major.minor so "27", "v27" and "27.0" agree; two builds
    # differing only in spelling would otherwise emit different manifests.
    return f"{text}.0" if "." not in text else text


def _aot_compile_options(compile_specs: List[CompileSpec]) -> Dict[str, Any]:
    """Normalized coreai-build opts from AOT_COMPILE_CONFIG + general specs.

    ``min_deployment_version`` is pulled from its own (general) spec, since it
    also applies to the portable .aimodel.
    """
    raw = _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.AOT_COMPILE_CONFIG)
    config = AOTCompileConfig.from_json(raw.decode()) if raw else AOTCompileConfig()
    return {
        "platform": config.platform or "macOS",
        # None => let coreai-build use its own default; matches save_asset default.
        "min_deployment_version": _min_os_text(compile_specs),
        "preferred_compute": config.preferred_compute or "none",
        # empty list => compile for all supported architectures
        "architectures": list(config.architectures or []),
        "expect_frequent_reshapes": config.expect_frequent_reshapes,
    }


def _min_os_version(compile_specs: List[CompileSpec]):
    """Map the MIN_DEPLOYMENT_VERSION spec to a coreai ``OSVersion`` (or None).

    Applied to every delivery path via ``save_asset(minimum_os=...)``.  Accepts a
    numeric string ("27.0" / "27") or an ``OSVersion`` name ("v27").  Returns
    None when unset so ``save_asset`` uses its own default.
    """
    raw = _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION)
    if raw is None:
        return None
    from coreai.authoring import OSVersion

    text = raw.decode()
    name = text if text.startswith("v") else f"v{text.split('.')[0]}"
    try:
        return OSVersion(name)
    except ValueError as e:
        available = [m.value for m in OSVersion]
        raise ValueError(
            f"unsupported min_deployment_version {text!r} (maps to {name!r}); "
            f"available coreai OSVersion values: {available}"
        ) from e


def _save_asset(program, path: Path, min_os) -> None:
    """``program.save_asset`` honoring an optional ``minimum_os``."""
    if min_os is None:
        program.save_asset(path)
    else:
        program.save_asset(path, minimum_os=min_os)


def _run_coreai_build(aimodel_path: Path, out_dir: Path, opts: Dict[str, Any]) -> None:
    """Invoke ``xcrun coreai-build compile`` to produce per-arch .aimodelc."""
    cmd = [
        "xcrun",
        "coreai-build",
        "compile",
        str(aimodel_path),
        "--output",
        str(out_dir),
        "--platform",
        opts["platform"],
    ]
    if opts["min_deployment_version"]:
        cmd += ["--min-deployment-version", opts["min_deployment_version"]]
    if opts["preferred_compute"] and opts["preferred_compute"] != "none":
        cmd += ["--preferred-compute", opts["preferred_compute"]]
    for arch in opts["architectures"]:
        cmd += ["--architecture", arch]
    if opts["expect_frequent_reshapes"]:
        cmd += ["--expect-frequent-reshapes"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as e:
        raise RuntimeError(
            "`xcrun coreai-build` not found. AOT compilation requires macOS with "
            "the Metal Toolchain installed. Disable aot_compile or install the "
            "toolchain (xcodebuild -downloadComponent MetalToolchain)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"`coreai-build compile` timed out after {e.timeout}s"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"`coreai-build compile` failed (rc={result.returncode}):\n{result.stderr}"
        )


def _compiled_arch_bundles(out_dir: Path) -> List[Tuple[str, Path]]:
    """Return [(arch, bundle_dir), ...] for each ``model.<arch>.aimodelc``."""
    bundles = []
    for path in sorted(out_dir.iterdir()):
        if path.is_dir() and path.name.endswith(".aimodelc"):
            name = path.name
            # ``model.<arch>.aimodelc`` -> ``<arch>``
            arch = name[len("model.") : -len(".aimodelc")]
            bundles.append((arch, path))
    if not bundles:
        raise RuntimeError(f"coreai-build produced no .aimodelc bundles in {out_dir}")
    return bundles


def _compile_aot(program, opts: Dict[str, Any], tmp_dir: Path) -> Path:
    """Save the intermediate ``.aimodel`` and AOT-compile it to ``.aimodelc``.

    Returns the compiled output directory. The intermediate ``.aimodel``
    keeps the default OS floor; ``min_deployment_version`` targets the compiled
    ``.aimodelc`` via ``coreai-build``, so it is not double-specified.
    """
    _save_asset(program, tmp_dir / "model.aimodel", None)
    out = tmp_dir / "compiled"
    out.mkdir()
    _run_coreai_build(tmp_dir / "model.aimodel", out, opts)
    return out


@final
class CoreAIBackend(BackendDetails):
    """AOT lowering of an edge program to a Core AI asset.

    Delivery is chosen by compile specs: the portable ``.aimodel`` (default),
    or AOT-compiled per-architecture ``.aimodelc`` bundles (``aot_compile``, via
    ``xcrun coreai-build``; architecture selection is one / a list / all). The
    bundles are embedded in the ``.pte`` via NamedDataStore.

    ``processed_bytes`` contains the manifest used by the runtime delegate.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        input_names, output_names = _boundary_names(edge_program)
        program = _convert_to_aiprogram(
            [(edge_program, MAIN_ENTRYPOINT, input_names, output_names)]
        )
        # Enumerated shapes replace ``main`` with renamed clones, so bind first.
        binding = _binding(
            program, edge_program, MAIN_ENTRYPOINT, input_names, output_names
        )
        binding = _enumerate_binding(
            program, edge_program, binding, input_names, compile_specs
        )
        return CoreAIBackend._package(program, compile_specs, [binding])[0]

    @classmethod
    def preprocess_multimethod(
        cls,
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, List[PreprocessResult]]:
        partitions = [
            (method, index, program, compile_specs[method][index])
            for method, programs in edge_programs.items()
            for index, program in enumerate(programs)
        ]
        if not any(
            _get_compile_spec(specs, COMPILE_SPEC_KEYS.MULTIFUNCTION) is not None
            for _, _, _, specs in partitions
        ):
            return super().preprocess_multimethod(edge_programs, compile_specs)

        missing = sorted(
            {
                method
                for method, _, _, specs in partitions
                if _get_compile_spec(specs, COMPILE_SPEC_KEYS.MULTIFUNCTION) is None
            }
        )
        if missing:
            raise ValueError(
                "multifunction=True must be set on the CoreAIPartitioner of every "
                f"method lowered to Core AI; methods {missing} were lowered "
                "without it"
            )
        # Each partition applies its own enumerations, so methods may enumerate
        # different values; every other spec shapes the shared model.
        ignore = (COMPILE_SPEC_KEYS.INPUT_ENUMERATIONS.value,)
        reference = _spec_items(partitions[0][3], ignore)
        differing = sorted(
            {
                method
                for method, _, _, specs in partitions
                if _spec_items(specs, ignore) != reference
            }
        )
        if differing:
            raise ValueError(
                "multifunction requires identical Core AI compile specs in every "
                f"partition; methods {differing} differ from {partitions[0][0]!r}"
            )
        entries = [
            (program, f"{method}_{index}", *_boundary_names(program))
            for method, index, program, _ in partitions
        ]
        logger.info(
            "Lowering %d Core AI partitions into one model: %s",
            len(entries),
            [entrypoint for _, entrypoint, _, _ in entries],
        )
        program = _convert_to_aiprogram(entries)
        # Bind every partition before specializing any, since specializing a
        # partition erases its function.
        bindings = [_binding(program, *entry) for entry in entries]
        bindings = [
            {
                **_enumerate_binding(
                    program, edge_program, binding, input_names, specs
                ),
                "multifunction": True,
            }
            for binding, (edge_program, _, input_names, _), (_, _, _, specs) in zip(
                bindings, entries, partitions
            )
        ]
        _check_unique_functions(bindings)
        results = CoreAIBackend._package(program, partitions[0][3], bindings)

        by_method: Dict[str, List[PreprocessResult]] = {
            method: [] for method in edge_programs
        }
        for (method, _, _, _), result in zip(partitions, results):
            by_method[method].append(result)
        return by_method

    @staticmethod
    def _package(
        program, compile_specs: List[CompileSpec], bindings: List[Dict[str, Any]]
    ) -> List[PreprocessResult]:
        # min-deployment-version is a single knob applied to whichever artifact
        # ships: for aot-compiled delivery the .aimodelc (via coreai-build
        # --min-deployment-version), for portable delivery the .aimodel's floor
        # (via save_asset(minimum_os=...)). In the aot-compiled path the temp
        # .aimodel is discarded, so it keeps the default floor (no
        # double-specification).
        if _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.AOT_COMPILE_CONFIG):
            return CoreAIBackend._preprocess_aot_compiled(
                program, _aot_compile_options(compile_specs), bindings
            )
        return CoreAIBackend._preprocess_portable(
            program, _min_os_version(compile_specs), bindings
        )

    # Portable .aimodel delivery.
    @staticmethod
    def _preprocess_portable(
        program, min_os, bindings: List[Dict[str, Any]]
    ) -> List[PreprocessResult]:
        with TemporaryDirectory() as tmp:
            _save_asset(program, Path(tmp) / "model.aimodel", min_os)
            return _deliver(
                Path(tmp),
                AssetPackaging.AIMODEL,
                {
                    # relative path the runtime resolves against its base
                    "path": "model.aimodel",
                    "min_deployment_version": _os_version_text(min_os),
                },
                bindings,
            )

    # AOT-compiled .aimodelc delivery (per architecture).
    @staticmethod
    def _preprocess_aot_compiled(
        program, opts: Dict[str, Any], bindings: List[Dict[str, Any]]
    ) -> List[PreprocessResult]:
        with TemporaryDirectory() as tmp:
            out = _compile_aot(program, opts, Path(tmp))
            return _deliver(
                out,
                AssetPackaging.AIMODELC,
                {
                    "platform": opts["platform"],
                    "min_deployment_version": opts["min_deployment_version"],
                    "archs": {
                        arch: bundle.name
                        for arch, bundle in _compiled_arch_bundles(out)
                    },
                },
                bindings,
            )
