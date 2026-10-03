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


class AssetPackaging(str, Enum):
    # Portable .aimodel embedded in the .pte via NamedDataStore.
    INLINE = "inline"
    # AOT-compiled per-arch .aimodelc bundles embedded in the .pte.
    AOT_COMPILED_INLINE = "aot_compiled_inline"


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


def _nds_key(model_hash: str, relative_path: str) -> str:
    # Namespaced by content hash so identical assets dedup to one entry and the
    # hash doubles as an on-device cache key.
    return f"coreai/{model_hash}/{relative_path}"


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
    model_hash: str,
    packaging: AssetPackaging,
    manifest_extra: Dict[str, Any],
) -> PreprocessResult:
    """Embed a staging directory of bundles in the .pte with its manifest.

    ``staging`` holds the ``.aimodel`` / ``.aimodelc`` bundles, never their
    contents, so assets land under ``coreai/<hash>/<bundle>/`` in the
    NamedDataStore.
    """
    manifest = {
        "packaging": packaging.value,
        "hash": model_hash,
        **manifest_extra,
        **_asset_metadata(staging),
    }
    return _embed_dir_inline(staging, model_hash, manifest)


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
    edge_program: ExportedProgram,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
):
    from coreai_torch import TorchConverter

    aten_program = _prepare_program_for_conversion(edge_program)
    converter = TorchConverter()
    converter.add_exported_program(
        aten_program,
        input_names=input_names,
        output_names=output_names,
        entrypoint_name=MAIN_ENTRYPOINT,
    )
    return converter.to_coreai()


# Asset embedding helpers.
def _embed_dir_inline(
    root_dir: Path, model_hash: str, manifest_extra: Dict[str, Any]
) -> PreprocessResult:
    """Flatten every file under ``root_dir`` into the NamedDataStore.

    ``root_dir`` is the directory *containing* the ``.aimodel`` / ``.aimodelc``
    bundles, never a bundle itself, so the bundle name survives into the keys.

    Keys are ``coreai/{hash}/{relpath}``; ``manifest_extra`` becomes the
    manifest (packaging, files, archs, ...).
    """
    store = NamedDataStore()
    for path in sorted(root_dir.rglob("*")):
        if path.is_file():
            store.add_named_data(
                _nds_key(model_hash, path.relative_to(root_dir).as_posix()),
                path.read_bytes(),
                alignment=_ASSET_ALIGNMENT,
            )
    return PreprocessResult(
        processed_bytes=json.dumps(manifest_extra).encode("utf-8"),
        data_store_output=store.get_named_data_store_output(),
    )


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


def _save_and_hash(program, path: Path, min_os=None) -> str:
    """Save the ``.aimodel`` and return its content hash (``main.hash``)."""
    _save_asset(program, path, min_os)
    return (path / f"{MAIN_ENTRYPOINT}.hash").read_bytes().hex()


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


def _compile_aot(program, opts: Dict[str, Any], tmp_dir: Path) -> Tuple[str, Path]:
    """Save the intermediate ``.aimodel`` and AOT-compile it to ``.aimodelc``.

    Returns ``(model_hash, compiled_out_dir)``. The intermediate ``.aimodel``
    keeps the default OS floor; ``min_deployment_version`` targets the compiled
    ``.aimodelc`` via ``coreai-build``, so it is not double-specified.
    """
    model_hash = _save_and_hash(program, tmp_dir / "model.aimodel")
    out = tmp_dir / "compiled"
    out.mkdir()
    _run_coreai_build(tmp_dir / "model.aimodel", out, opts)
    return model_hash, out


@final
class CoreAIBackend(BackendDetails):
    """AOT lowering of an edge program to a Core AI asset.

    Delivery is chosen by compile specs: the portable ``.aimodel`` (default),
    or AOT-compiled per-architecture ``.aimodelc`` bundles (``aot_compile``, via
    ``xcrun coreai-build``; architecture selection is one / a list / all). The
    bundles are embedded in the ``.pte`` via NamedDataStore.

    ``processed_bytes`` is always a small JSON manifest naming what/where.
    Runtime execution is not wired up yet.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        aot_compiled = (
            _get_compile_spec(compile_specs, COMPILE_SPEC_KEYS.AOT_COMPILE_CONFIG)
            is not None
        )

        mutated_inputs = edge_program.graph_signature.user_inputs_to_mutate
        if mutated_inputs:
            raise ValueError(
                "Core AI delegates cannot mutate ExecuTorch-owned user inputs: "
                f"{sorted(mutated_inputs.values())}"
            )
        edge_inputs, edge_outputs = _edge_io(edge_program)
        input_names = [f"input_{i}" for i in range(len(edge_inputs))]
        output_names = [f"output_{i}" for i in range(len(edge_outputs))]
        program = _convert_to_aiprogram(
            edge_program, input_names=input_names, output_names=output_names
        )
        # Fail fast if the .aimodel boundary I/O won't match what ET feeds/reads.
        assert_io_compatible(
            program, edge_program, input_names=input_names, output_names=output_names
        )
        # Enumerated shapes replace ``main`` with renamed clones, so read first.
        bound_inputs, bound_outputs = coreai_io_names(program)
        raw_enum = _get_compile_spec(
            compile_specs, COMPILE_SPEC_KEYS.INPUT_ENUMERATIONS
        )
        if raw_enum:
            apply_enumerated_shapes(
                program,
                edge_program,
                json.loads(raw_enum.decode()),
                input_names=input_names,
            )
        bindings = {
            "function": MAIN_ENTRYPOINT,
            "input_names": bound_inputs,
            "output_names": bound_outputs,
        }

        # min-deployment-version is a single knob applied to whichever artifact
        # ships: for aot-compiled delivery the .aimodelc (via coreai-build
        # --min-deployment-version), for portable delivery the .aimodel's floor
        # (via save_asset(minimum_os=...)). In the aot-compiled path the temp
        # .aimodel is discarded, so it keeps the default floor (no
        # double-specification).
        if aot_compiled:
            return CoreAIBackend._preprocess_aot_compiled(
                program, _aot_compile_options(compile_specs), bindings
            )
        return CoreAIBackend._preprocess_portable(
            program, _min_os_version(compile_specs), bindings
        )

    # Portable .aimodel delivery.
    @staticmethod
    def _preprocess_portable(
        program, min_os, bindings: Dict[str, Any]
    ) -> PreprocessResult:
        with TemporaryDirectory() as tmp:
            model_hash = _save_and_hash(program, Path(tmp) / "model.aimodel", min_os)
            return _deliver(
                Path(tmp),
                model_hash,
                AssetPackaging.INLINE,
                {
                    **bindings,
                    # relative path the runtime resolves against its base
                    "path": f"{model_hash}/model.aimodel",
                    "min_deployment_version": _os_version_text(min_os),
                },
            )

    # AOT-compiled .aimodelc delivery (per architecture).
    @staticmethod
    def _preprocess_aot_compiled(
        program, opts: Dict[str, Any], bindings: Dict[str, Any]
    ) -> PreprocessResult:
        with TemporaryDirectory() as tmp:
            model_hash, out = _compile_aot(program, opts, Path(tmp))
            return _deliver(
                out,
                model_hash,
                AssetPackaging.AOT_COMPILED_INLINE,
                {
                    **bindings,
                    "platform": opts["platform"],
                    "min_deployment_version": opts["min_deployment_version"],
                    "archs": {
                        arch: f"{model_hash}/{bundle.name}"
                        for arch, bundle in _compiled_arch_bundles(out)
                    },
                },
            )
