# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


from typing import Any, Optional, Sequence

import coremltools as ct
import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import (
    CoreMLPartitioner,
)
from executorch.backends.apple.coreml.recipes.coreml_recipe_types import (
    COREML_BACKEND,
    CoreMLRecipeType,
)

from executorch.exir import EdgeCompileConfig
from executorch.export import (
    AOQuantizationConfig,
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    RecipeType,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig


class CoreMLRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return COREML_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(CoreMLRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create CoreML recipe with precision and compute unit combinations"""

        if recipe_type not in self.get_supported_recipes():
            return None

        if ct is None:
            raise ImportError(
                "coremltools is required for CoreML recipes. "
                "Install it with: pip install coremltools"
            )

        # Validate kwargs
        self._validate_recipe_kwargs(recipe_type, **kwargs)

        if recipe_type == CoreMLRecipeType.FP32:
            return self._build_fp_recipe(recipe_type, ct.precision.FLOAT32, **kwargs)
        elif recipe_type == CoreMLRecipeType.FP16:
            return self._build_fp_recipe(recipe_type, ct.precision.FLOAT16, **kwargs)
        elif recipe_type == CoreMLRecipeType.PT2E_INT8_STATIC:
            return self._build_pt2e_quantized_recipe(
                recipe_type, activation_dtype=torch.quint8, **kwargs
            )
        elif recipe_type == CoreMLRecipeType.PT2E_INT8_WEIGHT_ONLY:
            return self._build_pt2e_quantized_recipe(
                recipe_type, activation_dtype=torch.float32, **kwargs
            )
        elif recipe_type == CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL:
            return self._build_torchao_quantized_recipe(
                recipe_type,
                weight_dtype=torch.int4,
                is_per_channel=True,
                **kwargs,
            )
        elif recipe_type == CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP:
            group_size = kwargs.pop("group_size", 32)
            return self._build_torchao_quantized_recipe(
                recipe_type,
                weight_dtype=torch.int4,
                is_per_channel=False,
                group_size=group_size,
                **kwargs,
            )
        elif recipe_type == CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL:
            return self._build_torchao_quantized_recipe(
                recipe_type, weight_dtype=torch.int8, is_per_channel=True, **kwargs
            )
        elif recipe_type == CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP:
            group_size = kwargs.pop("group_size", 32)
            return self._build_torchao_quantized_recipe(
                recipe_type,
                weight_dtype=torch.int8,
                is_per_channel=False,
                group_size=group_size,
                **kwargs,
            )
        elif recipe_type == CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY:
            bits = kwargs.pop("bits")
            block_size = kwargs.pop("block_size")
            return self._build_codebook_quantized_recipe(
                recipe_type, bits=bits, block_size=block_size, **kwargs
            )

        return None

    def _validate_recipe_kwargs(self, recipe_type: RecipeType, **kwargs: Any) -> None:
        """Validate kwargs for each recipe type"""
        expected_keys = self._get_expected_keys(recipe_type)

        unexpected = set(kwargs.keys()) - expected_keys
        if unexpected:
            raise ValueError(
                f"Recipe '{recipe_type.value}' received unexpected parameters: {list(unexpected)}"
            )

        self._validate_base_parameters(kwargs)
        self._validate_group_size_parameter(recipe_type, kwargs)
        self._validate_codebook_parameters(recipe_type, kwargs)

    def _get_expected_keys(self, recipe_type: RecipeType) -> set:
        """Get expected parameter keys for a recipe type"""
        common_keys = {"minimum_deployment_target", "compute_unit"}

        if recipe_type in [
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,
        ]:
            return common_keys | {"group_size", "filter_fn"}
        elif recipe_type in [
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL,
        ]:
            return common_keys | {"filter_fn"}
        elif recipe_type == CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY:
            return common_keys | {"bits", "block_size", "filter_fn"}
        else:
            return common_keys

    def _validate_base_parameters(self, kwargs: Any) -> None:
        """Validate minimum_deployment_target and compute_unit parameters"""
        if "minimum_deployment_target" in kwargs:
            minimum_deployment_target = kwargs["minimum_deployment_target"]
            if not isinstance(minimum_deployment_target, ct.target):
                raise ValueError(
                    f"Parameter 'minimum_deployment_target' must be an enum of type ct.target, got {type(minimum_deployment_target)}"
                )

        if "compute_unit" in kwargs:
            compute_unit = kwargs["compute_unit"]
            if not isinstance(compute_unit, ct.ComputeUnit):
                raise ValueError(
                    f"Parameter 'compute_unit' must be an enum of type ct.ComputeUnit, got {type(compute_unit)}"
                )

    def _validate_group_size_parameter(
        self, recipe_type: RecipeType, kwargs: Any
    ) -> None:
        """Validate group_size parameter for applicable recipe types"""
        if (
            recipe_type
            in [
                CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,
                CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,
            ]
            and "group_size" in kwargs
        ):
            group_size = kwargs["group_size"]
            if not isinstance(group_size, int):
                raise ValueError(
                    f"Parameter 'group_size' must be an integer, got {type(group_size).__name__}: {group_size}"
                )
            if group_size <= 0:
                raise ValueError(
                    f"Parameter 'group_size' must be positive, got: {group_size}"
                )

    def _validate_codebook_parameters(
        self, recipe_type: RecipeType, kwargs: Any
    ) -> None:
        """Validate bits and block_size parameters for codebook recipe type"""
        if recipe_type != CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY:
            return

        # Both bits and block_size must be present
        if not ("bits" in kwargs and "block_size" in kwargs):
            raise ValueError(
                "Parameters 'bits' and 'block_size' must be present for codebook recipes"
            )

        if "bits" in kwargs:
            bits = kwargs["bits"]
            if not isinstance(bits, int):
                raise ValueError(
                    f"Parameter 'bits' must be an integer, got {type(bits).__name__}: {bits}"
                )
            if not (1 <= bits <= 8):
                raise ValueError(
                    f"Parameter 'bits' must be between 1 and 8, got: {bits}"
                )

        if "block_size" in kwargs:
            block_size = kwargs["block_size"]
            if not isinstance(block_size, list):
                raise ValueError(
                    f"Parameter 'block_size' must be a list, got {type(block_size).__name__}: {block_size}"
                )

    def _validate_and_set_deployment_target(
        self, kwargs: Any, min_target: ct.target, quantization_type: str
    ) -> None:
        """Validate or set minimum deployment target for quantization recipes"""
        minimum_deployment_target = kwargs.get("minimum_deployment_target", None)
        if minimum_deployment_target and minimum_deployment_target < min_target:
            raise ValueError(
                f"minimum_deployment_target must be {str(min_target)} or higher for {quantization_type} quantization"
            )
        else:
            # Default to the minimum target for this quantization type
            kwargs["minimum_deployment_target"] = min_target

    def _build_fp_recipe(
        self,
        recipe_type: RecipeType,
        precision: ct.precision,
        **kwargs: Any,
    ) -> ExportRecipe:
        """Build FP32/FP16 recipe"""
        lowering_recipe = self._get_coreml_lowering_recipe(
            compute_precision=precision,
            **kwargs,
        )

        return ExportRecipe(
            name=recipe_type.value,
            lowering_recipe=lowering_recipe,
        )

    def _build_pt2e_quantized_recipe(
        self,
        recipe_type: RecipeType,
        activation_dtype: torch.dtype,
        **kwargs: Any,
    ) -> ExportRecipe:
        """Build PT2E-based quantization recipe"""
        from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer

        self._validate_and_set_deployment_target(kwargs, ct.target.iOS17, "pt2e")

        # Validate activation_dtype
        assert activation_dtype in [
            torch.quint8,
            torch.float32,
        ], f"activation_dtype must be torch.quint8 or torch.float32, got {activation_dtype}"

        # Create quantization config
        config = ct.optimize.torch.quantization.LinearQuantizerConfig(
            global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
                quantization_scheme="symmetric",
                activation_dtype=activation_dtype,
                weight_dtype=torch.qint8,
                weight_per_channel=True,
            )
        )

        quantizer = CoreMLQuantizer(config)
        quantization_recipe = QuantizationRecipe(quantizers=[quantizer])

        lowering_recipe = self._get_coreml_lowering_recipe(**kwargs)

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quantization_recipe,
            lowering_recipe=lowering_recipe,
        )

    def _build_torchao_quantized_recipe(
        self,
        recipe_type: RecipeType,
        weight_dtype: torch.dtype,
        is_per_channel: bool,
        group_size: int = 32,
        **kwargs: Any,
    ) -> ExportRecipe:
        """Build TorchAO-based quantization recipe"""
        if is_per_channel:
            weight_granularity = PerAxis(axis=0)
        else:
            weight_granularity = PerGroup(group_size=group_size)

        # Use user-provided filter_fn if provided
        filter_fn = kwargs.get("filter_fn", None)
        config = AOQuantizationConfig(
            ao_base_config=IntxWeightOnlyConfig(
                weight_dtype=weight_dtype,
                granularity=weight_granularity,
            ),
            filter_fn=filter_fn,
        )

        quantization_recipe = QuantizationRecipe(
            quantizers=None,
            ao_quantization_configs=[config],
        )

        # override minimum_deployment_target to ios18 for torchao (GH issue #13122)
        self._validate_and_set_deployment_target(kwargs, ct.target.iOS18, "torchao")
        lowering_recipe = self._get_coreml_lowering_recipe(**kwargs)

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quantization_recipe,
            lowering_recipe=lowering_recipe,
        )

    def _build_codebook_quantized_recipe(
        self,
        recipe_type: RecipeType,
        bits: int,
        block_size: list,
        **kwargs: Any,
    ) -> ExportRecipe:
        """Build codebook/palettization quantization recipe"""
        from torchao.prototype.quantization.codebook_coreml import (
            CodebookWeightOnlyConfig,
        )

        self._validate_and_set_deployment_target(kwargs, ct.target.iOS18, "codebook")

        # Get the appropriate dtype (torch.uint1 through torch.uint8)
        dtype = getattr(torch, f"uint{bits}")

        # Use user-provided filter_fn or default to Linear/Embedding layers
        filter_fn = kwargs.get(
            "filter_fn",
            lambda m, fqn: (
                isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.Linear)
            ),
        )

        config = AOQuantizationConfig(
            ao_base_config=CodebookWeightOnlyConfig(
                dtype=dtype,
                block_size=block_size,
            ),
            filter_fn=filter_fn,
        )

        quantization_recipe = QuantizationRecipe(
            quantizers=None,
            ao_quantization_configs=[config],
        )

        lowering_recipe = self._get_coreml_lowering_recipe(**kwargs)

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quantization_recipe,
            lowering_recipe=lowering_recipe,
        )

    def _get_coreml_lowering_recipe(
        self,
        compute_precision: ct.precision = ct.precision.FLOAT16,
        **kwargs: Any,
    ) -> LoweringRecipe:
        """Get CoreML lowering recipe with optional precision"""
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=compute_precision,
            compute_unit=kwargs.get("compute_unit", ct.ComputeUnit.ALL),
            minimum_deployment_target=kwargs.get("minimum_deployment_target", None),
        )

        minimum_deployment_target = kwargs.get("minimum_deployment_target", None)
        take_over_mutable_buffer = True
        if minimum_deployment_target and minimum_deployment_target < ct.target.iOS18:
            take_over_mutable_buffer = False

        partitioner = CoreMLPartitioner(
            compile_specs=compile_specs,
            take_over_mutable_buffer=take_over_mutable_buffer,
        )

        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=False,
        )

        return LoweringRecipe(
            partitioners=[partitioner], edge_compile_config=edge_compile_config
        )
