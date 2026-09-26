# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import RewriteConvPass
from executorch.backends.arm._passes.arm_pass_manager import (
    _registered_pass_insertions,
    register_pass_insertions_before,
)
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.vgf import VgfPartitioner
from executorch.backends.cortex_m.quantizer_reporter import (
    QuantizerInfo,
    QuantizerReporterUser,
)
from executorch.exir.pass_base import ExportPass
from torch._ops import OpOverload
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from .passes import RewriteWarpDownsampleToTosaCustomPass

_RIFE_LIBRARY: torch.library.Library | None = None
_RIFE_FAKE_IMPLS_REGISTERED = False


def _warp_downsample_fake(
    image: torch.Tensor, flow: torch.Tensor, scale: int
) -> torch.Tensor:
    del flow
    return torch.empty(
        (
            image.shape[0],
            image.shape[1],
            image.shape[2] // scale,
            image.shape[3] // scale,
        ),
        dtype=image.dtype,
        device=image.device,
    )


def _has_warp_downsample_op(scale: int) -> bool:
    try:
        getattr(torch.ops.rife, f"warp_downsample{scale}").default
    except AttributeError:
        return False
    return True


def _register_warp_downsample_fake_impls() -> None:
    global _RIFE_FAKE_IMPLS_REGISTERED
    if _RIFE_FAKE_IMPLS_REGISTERED:
        return

    for scale in (2, 4, 8):

        def _warp_downsample_fake_impl(
            image: torch.Tensor,
            flow: torch.Tensor,
            scale: int = scale,
        ) -> torch.Tensor:
            return _warp_downsample_fake(image, flow, scale)

        try:
            torch.library.register_fake(f"rife::warp_downsample{scale}")(
                _warp_downsample_fake_impl
            )
        except RuntimeError as error:
            message = str(error)
            if "already" not in message and "CompositeImplicitAutograd" not in message:
                raise

    _RIFE_FAKE_IMPLS_REGISTERED = True


def _ensure_warp_downsample_ops_defined() -> None:
    global _RIFE_LIBRARY
    if _RIFE_LIBRARY is None:
        _RIFE_LIBRARY = torch.library.Library("rife", "FRAGMENT")
    missing_scales = [
        scale for scale in (2, 4, 8) if not _has_warp_downsample_op(scale)
    ]
    for scale in missing_scales:
        _RIFE_LIBRARY.define(
            f"warp_downsample{scale}(Tensor image, Tensor flow) -> Tensor"
        )


def _ensure_warp_downsample_ops_registered() -> None:
    _ensure_warp_downsample_ops_defined()
    _register_warp_downsample_fake_impls()


def _warp_downsample_target(scale: int) -> OpOverload:
    _ensure_warp_downsample_ops_registered()
    return getattr(torch.ops.rife, f"warp_downsample{scale}").default


def _warp_downsample_targets() -> tuple[OpOverload, ...]:
    return tuple(_warp_downsample_target(scale) for scale in (2, 4, 8))


def _warp_downsample_snorm_qspec() -> FixedQParamsQuantizationSpec:
    return FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=1.0 / 127.0,
        zero_point=0,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
    )


class _WarpDownsampleQuantizer(Quantizer, QuantizerReporterUser):
    def __init__(self) -> None:
        super().__init__()
        QuantizerReporterUser.__init__(self)
        self.targets = set(_warp_downsample_targets())
        self.snorm_qspec = _warp_downsample_snorm_qspec()

    def get_quantizer_info(self) -> QuantizerInfo:
        return QuantizerInfo(
            self.__class__.__name__,
            "rife.warp_downsample{2,4,8}",
            "rife_warp_downsample_snorm",
            "examples.arm.QAT_example.rife_vgf",
        )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for node in model.graph.nodes:
            if (
                node.op != "call_function"
                or node.target not in self.targets
                or len(node.args) != 2
            ):
                continue
            image = node.args[0]
            if not isinstance(image, torch.fx.Node):
                continue
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map={image: self.snorm_qspec},
                output_qspec=self.snorm_qspec,
                _annotated=True,
            )
            meta_custom = node.meta.get("custom", {})
            meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = ArmAnnotationInfo(
                quantized=True
            )
            node.meta["custom"] = meta_custom
            self.report_accept([node])
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        return None


def _register_pass_before(target_pass_type: type, pass_: ExportPass) -> None:
    existing_insertions = _registered_pass_insertions.get(target_pass_type)
    if existing_insertions is not None and any(
        isinstance(existing_pass, type(pass_))
        for existing_pass in existing_insertions.before_passes
    ):
        return
    register_pass_insertions_before(target_pass_type, [pass_])


def configure_rife_vgf(partitioner: VgfPartitioner | None = None) -> None:
    """Enable the Practical-RIFE warp-downsample VGF extension."""
    _ensure_warp_downsample_ops_registered()
    if partitioner is not None:
        for target in _warp_downsample_targets():
            partitioner.register_custom_partition_op(target)

    _register_pass_before(RewriteConvPass, RewriteWarpDownsampleToTosaCustomPass())


def configure_rife_vgf_quantizer(quantizer) -> None:
    """Quantize RIFE warp-downsample image input/output as int8 SNORM."""
    _ensure_warp_downsample_ops_registered()
    quantizer.add_quantizer(_WarpDownsampleQuantizer())
