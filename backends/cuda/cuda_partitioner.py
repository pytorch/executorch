# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List, Optional

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.cuda.cuda_backend import CudaBackend  # usort: skip
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY

# Inlined copies of the internal compile-spec key strings owned by
# ``backends/cuda/passes/weight_offload_pass.py``. We don't import from
# that module because it registers a custom op at import time
# (``@custom_op`` decorator), which would defeat the lazy-import
# pattern in ``CudaBackend.pre_aoti_transform_and_collect_named_data``.
# The drift hazard is bounded by
# ``test_partitioner_internal_keys_match_pass``, which asserts these
# values match the pass-side constants at CI time.
_WEIGHT_OFFLOAD_ENABLE_SPEC_KEY = "_weight_offload_internal_enable"
_WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY = "_weight_offload_internal_pin_fqns"
_WEIGHT_OFFLOAD_INTERNAL_KEY_PREFIX = "_weight_offload_internal_"

# Weight offload is device-0-only today: the pass hard-codes
# device_type=cuda, device_index=0 in the payload, and the runtime
# calls cudaSetDevice(0) before container/stream creation. Until
# multi-device offload lands, reject any other target_device when
# weight_offload=True so callers don't silently end up on the wrong
# GPU.
_OFFLOAD_REQUIRED_DEVICE = ("cuda", 0)


def _check_pin_fqns_input(weight_offload_pin_fqns) -> List[str]:
    """Normalize the public ``weight_offload_pin_fqns`` argument to a
    deduped first-seen-order list of strings. Rejects ``bare str``
    (the ``list("w1")`` foot-gun) and any non-list iterable (the
    public contract is ``List[str]``, and dict_keys / sets / generators
    would either lose ordering or be one-shot). The runtime payload
    parser hard-fails on empty / NUL-containing FQNs as a trust-boundary
    check; not duplicating that here."""
    if weight_offload_pin_fqns is None:
        return []
    if isinstance(weight_offload_pin_fqns, str):
        raise TypeError(
            "weight_offload_pin_fqns must be a list of strings, not a "
            "bare str; pass [...] even for a single FQN"
        )
    if not isinstance(weight_offload_pin_fqns, list):
        raise TypeError(
            f"weight_offload_pin_fqns must be a list, got "
            f"{type(weight_offload_pin_fqns).__name__}"
        )
    seen: set = set()
    out: List[str] = []
    for fqn in weight_offload_pin_fqns:
        if not isinstance(fqn, str):
            raise TypeError(
                f"weight_offload_pin_fqns elements must be strings, got "
                f"{type(fqn).__name__}: {fqn!r}"
            )
        if fqn not in seen:
            seen.add(fqn)
            out.append(fqn)
    return out


def _reject_non_default_target_device_when_offloading(
    compile_spec: List[CompileSpec],
) -> None:
    """Raise if any caller-supplied ``target_device`` compile spec is
    not ``cuda:0``. The default (key absent) is fine and gets filled in
    later by the constructor.

    Parsed inline rather than via
    ``propagate_device_pass._parse_device_spec_value`` to avoid pulling
    in the flatbuffer schema import for a one-line check.
    """
    want_type, want_index = _OFFLOAD_REQUIRED_DEVICE
    for spec in compile_spec:
        if spec.key != TARGET_DEVICE_COMPILE_SPEC_KEY:
            continue
        raw = spec.value.decode("utf-8").strip().lower()
        if ":" in raw:
            type_str, idx_str = raw.split(":", 1)
            try:
                idx = int(idx_str)
            except ValueError:
                idx = -1
        else:
            type_str, idx = raw, 0
        if type_str != want_type or idx != want_index:
            raise ValueError(
                f"CudaPartitioner: weight_offload=True currently requires "
                f"target_device='{want_type}:{want_index}'; got "
                f"{spec.value!r}. Multi-device weight offload is not "
                f"implemented yet (the payload and runtime hard-code "
                f"device 0); drop the target_device spec or set it to "
                f"'{want_type}:{want_index}'."
            )


def _validate_and_translate_weight_offload_kwargs(
    compile_spec: List[CompileSpec],
    weight_offload: bool,
    weight_offload_pin_fqns: Optional[List[str]],
) -> List[CompileSpec]:
    """Translate the public weight-offload kwargs to internal compile
    specs with strict validation. Returns the (possibly augmented)
    compile_spec list to pass to the base partitioner."""
    pin_fqns_list = _check_pin_fqns_input(weight_offload_pin_fqns)

    # Reject pin-without-enable.
    if pin_fqns_list and not weight_offload:
        raise ValueError(
            "weight_offload_pin_fqns is set but weight_offload=False; "
            "pinning requires enabling weight offload"
        )

    if weight_offload:
        _reject_non_default_target_device_when_offloading(compile_spec)

    # Strict mixed-channel rejection: when ANY public weight-offload
    # kwarg is non-default, reject ANY raw `_weight_offload_internal_*`
    # compile spec. Raw internal specs stay allowed when both public
    # kwargs are at default values (preserves the test stack).
    if weight_offload or pin_fqns_list:
        offenders = [
            spec.key
            for spec in compile_spec
            if spec.key.startswith(_WEIGHT_OFFLOAD_INTERNAL_KEY_PREFIX)
        ]
        if offenders:
            raise ValueError(
                f"CudaPartitioner: public weight-offload kwargs conflict "
                f"with raw {_WEIGHT_OFFLOAD_INTERNAL_KEY_PREFIX}* entries "
                f"in compile_spec ({sorted(set(offenders))!r}); use "
                f"exactly one channel - either the public kwargs OR the "
                f"raw compile_spec, not both"
            )

    out = list(compile_spec)
    if weight_offload:
        out.append(CompileSpec(_WEIGHT_OFFLOAD_ENABLE_SPEC_KEY, b"1"))
        if pin_fqns_list:
            out.append(
                CompileSpec(
                    _WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY,
                    b"\x00".join(f.encode("utf-8") for f in pin_fqns_list),
                )
            )
    return out


@final
@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class CudaPartitioner(AotiPartitioner):
    """
    CUDA partitioner driven by AOTInductor backend.

    This partitioner adds a target_device compile spec to enable device info
    propagation. The PropagateDevicePass will read this spec and mark delegate
    output tensors with CUDA device type, which flows through to serialization.
    """

    def __init__(
        self,
        compile_spec: List[CompileSpec],
        *,
        weight_offload: bool = False,
        weight_offload_pin_fqns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the CUDA partitioner.

        Args:
            compile_spec: List of compile specs for the backend. To specify a
                         target CUDA device, include a CompileSpec with key
                         "target_device" (e.g., value "cuda:1"). If not
                         provided, defaults to "cuda:0". NOTE: when
                         ``weight_offload=True`` only ``cuda:0`` is supported
                         today; any other ``target_device`` is rejected
                         (multi-device offload is not yet implemented).
            weight_offload: When True, opt the method into the CUDA weight-
                         offload runtime: AOTI's eager constant load is
                         skipped, the runtime installs pre-load dummies,
                         and probes serve weights through a bounded GPU
                         pool from a host mirror. The load-time budget is
                         controlled by the ``weight_offload_budget_mb``
                         runtime spec (or the default
                         ``floor + pinned_bytes`` when unset). Default
                         False. Currently device-0-only.
            weight_offload_pin_fqns: Optional list of parameter / buffer
                         FQNs to keep resident on GPU for the Session
                         lifetime (no streaming). Requires
                         ``weight_offload=True``. Duplicates are removed
                         first-seen order.
        """
        # Translate the public weight-offload kwargs to internal
        # compile specs (with validation). Extracted to a helper to
        # keep this constructor under the project's cyclomatic-
        # complexity cap.
        compile_spec = _validate_and_translate_weight_offload_kwargs(
            compile_spec, weight_offload, weight_offload_pin_fqns
        )

        # Add target_device compile spec for device propagation if not already present
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec = compile_spec + [
                CompileSpec(
                    TARGET_DEVICE_COMPILE_SPEC_KEY,
                    b"cuda:0",
                )
            ]
        super().__init__(CudaBackend.__name__, compile_spec)
