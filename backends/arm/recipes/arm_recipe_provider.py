# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Optional, Sequence

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.recipes.arm_recipe_types import ARM_BACKEND, ArmRecipeType
from executorch.exir.pass_manager import PassType
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    RecipeType,
    StageType,
)


_ETHOS_U_FAMILIES: dict[ArmRecipeType, tuple[str, tuple[int, ...], int]] = {
    ArmRecipeType.ETHOS_U55_INT8: ("ethos-u55", (32, 64, 128, 256), 128),
    ArmRecipeType.ETHOS_U65_INT8: ("ethos-u65", (256, 512), 256),
    ArmRecipeType.ETHOS_U85_INT8: ("ethos-u85", (128, 256, 512, 1024, 2048), 256),
}

_ETHOS_U_KWARGS: frozenset[str] = frozenset(
    {"macs", "system_config", "memory_mode", "extra_flags", "config_ini"}
)

# Matches aot_arm_compiler.py:479-484 — bit-identical Vela invocation vs. CLI.
_VELA_DEFAULT_FLAGS: tuple[str, ...] = (
    "--verbose-operators",
    "--verbose-cycle-estimate",
)

# Pipeline used by INT8/A16W8 paths so ReplaceQuantNodesPass runs after the
# partitioner (matches aot_arm_compiler.py:200-201).
_PIPELINE_WITH_EDGE_PASSES: list[StageType] = [
    StageType.SOURCE_TRANSFORM,
    StageType.QUANTIZE,
    StageType.TORCH_EXPORT,
    StageType.TO_EDGE_TRANSFORM_AND_LOWER,
    StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
    StageType.TO_EXECUTORCH,
]


def _replace_quant_nodes_pass(_epm: Any) -> list[PassType]:
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
        ReplaceQuantNodesPass,
    )

    return [ReplaceQuantNodesPass()]


class ArmRecipeProvider(BackendRecipeProvider):
    """Note: unknown kwargs raise ``ValueError`` (vs. XNNPACK/QNN, which log a
    warning). Intentional for a new provider so typos like ``mac=128`` fail
    fast rather than silently producing a wrong-target binary."""

    @property
    def backend_name(self) -> str:
        return ARM_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(ArmRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if not isinstance(recipe_type, ArmRecipeType):
            return None

        if recipe_type in _ETHOS_U_FAMILIES:
            return self._build_ethos_u_recipe(recipe_type, kwargs)

        # Prime ethosu before importing vgf: a pre-existing circular dep
        # between tosa.backend and ethosu.backend breaks if vgf is loaded
        # first (vgf.backend → tosa.backend → _passes → ethosu.backend →
        # tosa.backend [partial]). The Arm CLI works around it by the same
        # ordering at module load (aot_arm_compiler.py:26-35).
        import executorch.backends.arm.ethosu  # noqa: F401
        from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
        from executorch.backends.arm.vgf import VgfCompileSpec

        # (compile_spec_factory, tosa_spec, quant_mode, replace_quant_nodes).
        # replace_quant_nodes is False for VGF, matching aot_arm_compiler.py:200.
        delegated: dict[
            ArmRecipeType,
            tuple[Callable[[str], ArmCompileSpec], str, Optional[str], bool],
        ] = {
            ArmRecipeType.TOSA_FP: (TosaCompileSpec, "TOSA-1.0+FP", None, False),
            ArmRecipeType.TOSA_INT8: (TosaCompileSpec, "TOSA-1.0+INT", "INT8", True),
            ArmRecipeType.TOSA_A16W8: (
                TosaCompileSpec,
                "TOSA-1.0+INT+int16",
                "A16W8",
                True,
            ),
            ArmRecipeType.VGF_FP: (VgfCompileSpec, "TOSA-1.0+FP", None, False),
            ArmRecipeType.VGF_INT8: (VgfCompileSpec, "TOSA-1.0+INT", "INT8", False),
        }
        factory, tosa_spec, quant_mode, replace_quant_nodes = delegated[recipe_type]
        return self._build_delegated_recipe(
            recipe_type, factory, tosa_spec, kwargs, quant_mode, replace_quant_nodes
        )

    def _build_ethos_u_recipe(
        self, recipe_type: ArmRecipeType, kwargs: dict[str, Any]
    ) -> ExportRecipe:
        from executorch.backends.arm.ethosu import EthosUCompileSpec
        from executorch.backends.arm.util._factory import create_partitioner

        self._validate_kwargs(recipe_type, kwargs, _ETHOS_U_KWARGS)

        family, allowed_macs, default_macs = _ETHOS_U_FAMILIES[recipe_type]
        macs = kwargs.get("macs", default_macs)
        if macs not in allowed_macs:
            raise ValueError(
                f"Recipe '{recipe_type.value}' does not support macs={macs}. "
                f"Allowed: {list(allowed_macs)}"
            )

        user_extra_flags = kwargs.get("extra_flags") or []
        compile_spec = EthosUCompileSpec(
            target=f"{family}-{macs}",
            system_config=kwargs.get("system_config"),
            memory_mode=kwargs.get("memory_mode"),
            extra_flags=list(_VELA_DEFAULT_FLAGS) + list(user_extra_flags),
            config_ini=kwargs.get("config_ini", "Arm/vela.ini"),
        )

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=self._build_quantization_recipe(compile_spec, "INT8"),
            lowering_recipe=LoweringRecipe(
                partitioners=[create_partitioner(compile_spec)],
                edge_manager_transform_passes=[_replace_quant_nodes_pass],
            ),
            pipeline_stages=_PIPELINE_WITH_EDGE_PASSES,
        )

    def _build_delegated_recipe(
        self,
        recipe_type: ArmRecipeType,
        compile_spec_factory: Callable[[str], ArmCompileSpec],
        tosa_spec: str,
        kwargs: dict[str, Any],
        quant_mode: Optional[str],
        replace_quant_nodes: bool,
    ) -> ExportRecipe:
        from executorch.backends.arm.util._factory import create_partitioner

        self._validate_kwargs(recipe_type, kwargs, frozenset())

        compile_spec = compile_spec_factory(tosa_spec)
        partitioner = create_partitioner(compile_spec)

        if replace_quant_nodes:
            lowering = LoweringRecipe(
                partitioners=[partitioner],
                edge_manager_transform_passes=[_replace_quant_nodes_pass],
            )
            pipeline = _PIPELINE_WITH_EDGE_PASSES
        else:
            lowering = LoweringRecipe(partitioners=[partitioner])
            pipeline = None

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=self._build_quantization_recipe(
                compile_spec, quant_mode
            ),
            lowering_recipe=lowering,
            pipeline_stages=pipeline,
        )

    @staticmethod
    def _build_quantization_recipe(
        compile_spec: ArmCompileSpec, quant_mode: Optional[str]
    ) -> Optional[QuantizationRecipe]:
        from executorch.backends.arm.quantizer import (
            get_symmetric_a16w8_quantization_config,
            get_symmetric_quantization_config,
        )
        from executorch.backends.arm.util._factory import create_quantizer

        if quant_mode is None:
            return None

        quantizer = create_quantizer(compile_spec)
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
        quantizer.set_global(operator_config)
        return QuantizationRecipe(quantizers=[quantizer])

    @staticmethod
    def _validate_kwargs(
        recipe_type: ArmRecipeType,
        kwargs: dict[str, Any],
        expected: frozenset[str],
    ) -> None:
        unexpected = set(kwargs.keys()) - expected
        if unexpected:
            allowed = sorted(expected) if expected else "none"
            raise ValueError(
                f"Arm recipe '{recipe_type.value}' got unexpected parameters: "
                f"{sorted(unexpected)}. Allowed: {allowed}"
            )
