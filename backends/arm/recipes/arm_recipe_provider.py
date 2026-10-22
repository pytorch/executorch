# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.recipes.arm_recipe_types import ARM_BACKEND, ArmRecipeType
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.util._factory import create_partitioner, create_quantizer
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.pass_manager import PassType
from executorch.exir.program import EdgeProgramManager
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    RecipeType,
)


logger: logging.Logger = logging.getLogger(__name__)

# (target prefix, default MAC count). Which counts are *accepted* is Vela's to
# say, so it is asked at build time rather than restated here.
_ETHOS_U_FAMILIES: dict[ArmRecipeType, tuple[str, int]] = {
    ArmRecipeType.ETHOS_U55_INT8: ("ethos-u55", 128),
    ArmRecipeType.ETHOS_U65_INT8: ("ethos-u65", 256),
    ArmRecipeType.ETHOS_U85_INT8: ("ethos-u85", 256),
}

_ETHOS_U_KWARGS: frozenset[str] = frozenset(
    {"macs", "system_config", "memory_mode", "extra_flags", "config_ini"}
)

# Prepended to any caller-supplied Vela flags, matching `_get_compile_spec` in
# aot_arm_compiler.py.
_VELA_DEFAULT_FLAGS: tuple[str, ...] = (
    "--verbose-operators",
    "--verbose-cycle-estimate",
)


@dataclass(frozen=True)
class _TosaVersionedTarget:
    """A target with no caller-tunable options: class plus TOSA version."""

    compile_spec: Callable[[str], ArmCompileSpec]
    tosa_spec: str
    quant_mode: Optional[str]
    replace_quant_nodes: bool


# VGF keeps the quantized_decomposed QDQ ops it is given; see
# `_apply_replace_quant_nodes` in aot_arm_compiler.py.
_TOSA_VERSIONED_TARGETS: dict[ArmRecipeType, _TosaVersionedTarget] = {
    ArmRecipeType.TOSA_FP: _TosaVersionedTarget(
        TosaCompileSpec, "TOSA-1.0+FP", None, False
    ),
    ArmRecipeType.TOSA_INT8: _TosaVersionedTarget(
        TosaCompileSpec, "TOSA-1.0+INT", "INT8", True
    ),
    ArmRecipeType.TOSA_A16W8: _TosaVersionedTarget(
        TosaCompileSpec, "TOSA-1.0+INT+int16", "A16W8", True
    ),
    ArmRecipeType.VGF_FP: _TosaVersionedTarget(
        VgfCompileSpec, "TOSA-1.0+FP", None, False
    ),
    ArmRecipeType.VGF_INT8: _TosaVersionedTarget(
        VgfCompileSpec, "TOSA-1.0+INT", "INT8", False
    ),
}


def _reject_unsupported_accelerator(
    recipe_type: ArmRecipeType, family: str, target: str, macs: int
) -> None:
    """Ask Vela which accelerator configurations it accepts.

    A local copy would go stale on the next Vela bump. Without Vela there is
    nothing to check against and the compile spec still has to build, so the
    import is guarded the way `arm_vela` guards its own.

    """
    try:
        from ethosu.vela.architecture_features import Accelerator  # type: ignore
    except ImportError:
        logger.debug("ethos-u-vela is not installed; macs=%s unvalidated", macs)
        return

    supported = {accelerator.value for accelerator in Accelerator}
    if target not in supported:
        allowed = sorted(
            int(name.rsplit("-", 1)[1])
            for name in supported
            if name.startswith(f"{family}-")
        )
        raise ValueError(
            f"Recipe '{recipe_type.value}' does not support macs={macs}. "
            f"Allowed: {allowed}"
        )


def _replace_quant_nodes(
    edge_program_manager: EdgeProgramManager,
) -> list[PassType]:
    """Rewrite the QDQ ops left outside the delegate into cortex_m kernels.

    Matches `_apply_replace_quant_nodes` in aot_arm_compiler.py, which applies
    the same pass to the whole manager once the partitioner has run. Without it
    the boundary quantize/dequantize keep their `quantized_decomposed` targets,
    which have no out variants, and `to_executorch` refuses to emit them.

    """
    # Function-local: an FP recipe must not pull in the cortex_m operator
    # library, which registers its whole op set on import.
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
        ReplaceQuantNodesPass,
    )

    return [ReplaceQuantNodesPass()]


class ArmRecipeProvider(BackendRecipeProvider):
    """Builds ExportRecipes for the delegated Arm targets: Ethos-U, TOSA, VGF.

    Each recipe is built to reproduce the default
    ``backends/arm/scripts/aot_arm_compiler.py`` invocation for its target: the
    same compile spec, quantizer, pass pipeline and backend config. The CLI
    options with no recipe equivalent, debug mode and direct drive, are the
    exceptions.
    """

    @property
    def backend_name(self) -> str:
        return ARM_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(_ETHOS_U_FAMILIES) + list(_TOSA_VERSIONED_TARGETS)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if not isinstance(recipe_type, ArmRecipeType):
            return None

        if recipe_type in _ETHOS_U_FAMILIES:
            self._warn_unknown_kwargs(recipe_type, kwargs, _ETHOS_U_KWARGS)
            return self._build_recipe(
                recipe_type,
                self._ethos_u_compile_spec(recipe_type, kwargs),
                quant_mode="INT8",
                replace_quant_nodes=True,
            )

        target = _TOSA_VERSIONED_TARGETS.get(recipe_type)
        if target is None:
            return None

        self._warn_unknown_kwargs(recipe_type, kwargs, frozenset())
        return self._build_recipe(
            recipe_type,
            target.compile_spec(target.tosa_spec),
            quant_mode=target.quant_mode,
            replace_quant_nodes=target.replace_quant_nodes,
        )

    @staticmethod
    def _ethos_u_compile_spec(
        recipe_type: ArmRecipeType, kwargs: dict[str, Any]
    ) -> EthosUCompileSpec:
        family, default_macs = _ETHOS_U_FAMILIES[recipe_type]
        macs = kwargs.get("macs", default_macs)
        if not isinstance(macs, int):
            raise ValueError(f"macs must be an int, got {macs!r}")

        extra_flags = kwargs.get("extra_flags") or []
        # The list check comes first: a bare string would be iterated into one
        # flag per character, and anything not iterable would raise TypeError
        # out of `all` rather than reaching this message.
        if not isinstance(extra_flags, list) or not all(
            isinstance(flag, str) for flag in extra_flags
        ):
            raise ValueError(
                f"extra_flags must be a list of strings, got {extra_flags!r}"
            )

        target = f"{family}-{macs}"
        _reject_unsupported_accelerator(recipe_type, family, target, macs)

        return EthosUCompileSpec(
            target=target,
            system_config=kwargs.get("system_config"),
            memory_mode=kwargs.get("memory_mode"),
            extra_flags=list(_VELA_DEFAULT_FLAGS) + list(extra_flags),
            # EthosUCompileSpec owns the default.
            config_ini=kwargs.get("config_ini"),
        )

    @classmethod
    def _build_recipe(
        cls,
        recipe_type: ArmRecipeType,
        compile_spec: ArmCompileSpec,
        quant_mode: Optional[str],
        replace_quant_nodes: bool,
    ) -> ExportRecipe:
        # The partitioner snapshots the compile spec and the pipeline config is
        # materialised on first read, which the CLI gets for free by quantizing
        # before it partitions.
        compile_spec.set_pass_pipeline_config(compile_spec._get_pass_pipeline_config())

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=cls._build_quantization_recipe(
                compile_spec, quant_mode
            ),
            lowering_recipe=LoweringRecipe(
                partitioners=[create_partitioner(compile_spec)],
                # The CLI disables edge verification on every Arm path.
                edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
                edge_manager_transform_passes=(
                    [_replace_quant_nodes] if replace_quant_nodes else None
                ),
            ),
            # The Arm runtime expects the delegate payload inline rather than
            # in its own segment, as every other Arm AOT path asks for.
            executorch_backend_config=ExecutorchBackendConfig(
                extract_delegate_segments=False
            ),
        )

    @staticmethod
    def _build_quantization_recipe(
        compile_spec: ArmCompileSpec, quant_mode: Optional[str]
    ) -> Optional[QuantizationRecipe]:
        if quant_mode is None:
            return None

        if quant_mode == "INT8":
            operator_config = get_symmetric_quantization_config(is_per_channel=True)
        elif quant_mode == "A16W8":
            if not compile_spec.tosa_spec.support_extension("int16"):
                raise ValueError(
                    f"TOSA spec {compile_spec.tosa_spec} does not support int16 "
                    "(required for A16W8)"
                )
            operator_config = get_symmetric_a16w8_quantization_config(
                is_per_channel=True
            )
        else:
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")

        quantizer = create_quantizer(compile_spec)
        quantizer.set_global(operator_config)
        return QuantizationRecipe(quantizers=[quantizer])

    @staticmethod
    def _warn_unknown_kwargs(
        recipe_type: ArmRecipeType,
        kwargs: dict[str, Any],
        expected: frozenset[str],
    ) -> None:
        # Warn, as XNNPACK and QNN do: `_create_target_recipe` hands every
        # recipe in a combination the same kwargs.
        unexpected = set(kwargs.keys()) - expected
        if unexpected:
            allowed = sorted(expected) if expected else "none"
            logger.warning(
                "Arm recipe '%s' ignoring unexpected parameters: %s. Allowed: %s",
                recipe_type.value,
                sorted(unexpected),
                allowed,
            )
