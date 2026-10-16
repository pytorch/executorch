# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    HtpArch,
    LpaiHardwareVersion,
    QcomChipset,
    QnnExecuTorchBackendType,
)


@dataclass(frozen=True)
class ExecutionTarget:
    backend: QnnExecuTorchBackendType
    hardware_version: HtpArch | LpaiHardwareVersion | None


@dataclass(frozen=True)
class MetricConstraint:
    operator: Callable[[float, float], bool]
    expected: float

    def assert_value(self, actual: float, context: str) -> None:
        operator_name = {operator.ge: ">=", operator.le: "<="}.get(
            self.operator, self.operator.__name__
        )
        assert self.operator(
            actual, self.expected
        ), f"{context}: expected {operator_name} {self.expected}, got {actual}"


def minimum(value: float) -> MetricConstraint:
    return MetricConstraint(operator.ge, value)


def maximum(value: float) -> MetricConstraint:
    return MetricConstraint(operator.le, value)


@dataclass(frozen=True)
class MetricProfile:
    shared: Mapping[str, MetricConstraint]
    by_backend: Mapping[QnnExecuTorchBackendType, Mapping[str, MetricConstraint]]
    by_target: Mapping[ExecutionTarget, Mapping[str, MetricConstraint]]


def _profile(
    shared: Mapping[str, MetricConstraint],
    by_backend: (
        Mapping[QnnExecuTorchBackendType, Mapping[str, MetricConstraint]] | None
    ) = None,
    by_target: Mapping[ExecutionTarget, Mapping[str, MetricConstraint]] | None = None,
) -> MetricProfile:
    return MetricProfile(shared, by_backend or {}, by_target or {})


# TODO: extend targets for coverage
HTP_V75 = ExecutionTarget(QnnExecuTorchBackendType.kHtpBackend, HtpArch.V75)
HTP_V79 = ExecutionTarget(QnnExecuTorchBackendType.kHtpBackend, HtpArch.V79)


# TODO: use by_target for fine-grained validation
METRIC_PROFILES = {
    "mobilenet_v2": _profile({"top_1": minimum(52), "top_5": minimum(84)}),
    "mobilenet_v3": _profile({"top_1": minimum(51), "top_5": minimum(76)}),
    "inception_v3": _profile({"top_1": minimum(59), "top_5": minimum(78)}),
    "inception_v4": _profile({"top_1": minimum(64), "top_5": minimum(85)}),
    "vit": _profile({"top_1": minimum(69), "top_5": minimum(91)}),
    "edsr": _profile({"PSNR": minimum(29), "SSIM": minimum(0.94)}),
    "deeplab_v3": _profile(
        {"PA": minimum(0.88), "MPA": minimum(0.79), "MIoU": minimum(0.67)}
    ),
    "mobilebert": _profile({"cpu_htp_delta": maximum(2)}),
    "ptq_mobilebert": _profile({"cpu_htp_delta": maximum(5)}),
    "wav2letter": _profile({"wer": maximum(0.5), "cer": maximum(0.25)}),
    "albert": _profile({"accuracy": minimum(0.95)}),
    "bert": _profile({"accuracy": minimum(0.55)}),
    "conv_former": _profile({"top_1": minimum(70), "top_5": minimum(92)}),
    "convnext_small": _profile({"top_1": minimum(74), "top_5": minimum(96)}),
    "cvt": _profile({"top_1": minimum(70), "top_5": minimum(90)}),
    "deit": _profile({"top_1": minimum(76), "top_5": minimum(92)}),
    "depthanything_v2_small": _profile({"sqnr": minimum(15)}),
    "dino_v2": _profile({"top_1": minimum(62), "top_5": minimum(87)}),
    "distilbert": _profile({"accuracy": minimum(0.48)}),
    "dit": _profile({"top_1": minimum(78), "top_5": minimum(92)}),
    "efficientnet": _profile({"top_1": minimum(61), "top_5": minimum(88)}),
    "efficient_sam": _profile({"MIoU": minimum(0.95)}),
    "esrgan": _profile({"PSNR": minimum(23), "SSIM": minimum(0.85)}),
    "eurobert": _profile({"accuracy": minimum(0.54)}),
    "fastvit": _profile({"top_1": minimum(60), "top_5": minimum(77)}),
    "fbnet": _profile({"top_1": minimum(63), "top_5": minimum(88)}),
    "focalnet": _profile({"top_1": minimum(54), "top_5": minimum(80)}),
    "gmlp": _profile({"top_1": minimum(70), "top_5": minimum(88)}),
    "maxvit_t": _profile({"top_1": minimum(71), "top_5": minimum(91)}),
    "mobilevit_v2": _profile({"top_1": minimum(50), "top_5": minimum(85)}),
    "mobilevit_v1": _profile({"top_1": minimum(76), "top_5": minimum(93)}),
    "pvt": _profile({"top_1": minimum(65), "top_5": minimum(83)}),
    "regnet": _profile({"top_1": minimum(64), "top_5": minimum(86)}),
    "retinanet": _profile({"mAP": minimum(0.6)}),
    "roberta": _profile({"accuracy": minimum(0.54)}),
    "squeezenet": _profile({"top_1": minimum(27), "top_5": minimum(59)}),
    "ssd300_vgg16": _profile({"mAP": minimum(0.76)}),
    "swin_transformer": _profile({"top_1": minimum(71), "top_5": minimum(90)}),
    "swin_v2_t": _profile({"top_1": minimum(63), "top_5": minimum(91)}),
    "t5": _profile({"f1": minimum(0.72)}),
    "vit_b_16": _profile({"top_1": minimum(72), "top_5": minimum(96)}),
    "whisper": _profile({"wer": maximum(0.25)}),
    "codegen2": _profile(
        {"pte_size": maximum(1_200_000_000), "inference_speed": minimum(60)}
    ),
    "llama_stories_260k": _profile(
        {"pte_size": maximum(2_020_000), "inference_speed": minimum(1600)}
    ),
    "llama_stories_110m:int8": _profile(
        {"pte_size": maximum(135_000_000), "inference_speed": minimum(220)}
    ),
    "llama_stories_110m:fp16": _profile(
        {"pte_size": maximum(275_000_000), "inference_speed": minimum(220)}
    ),
    "attention_sink": _profile({"attention_sink_evictor_pte_size": maximum(1_700_000)}),
    "asr:granite_speech_3_3-2b": _profile(
        {
            "audio_encoder_pte_size": maximum(900_000_000),
            "tok_embedding_pte_size": maximum(240_000_000),
            "pte_size": maximum(3_000_000_000),
        },
        by_target={
            HTP_V75: {"inference_speed": minimum(5)},
            HTP_V79: {"inference_speed": minimum(8)},
        },
    ),
    "vlm:smolvlm_500m_instruct": _profile(
        {
            "vision_encoder_pte_size": maximum(110_000_000),
            "tok_embedding_pte_size": maximum(100_000_000),
            "pte_size": maximum(400_000_000),
        },
        by_target={
            HTP_V75: {"inference_speed": minimum(50)},
            HTP_V79: {"inference_speed": minimum(55)},
        },
    ),
    "vlm:internvl3_1b": _profile(
        {
            "vision_encoder_pte_size": maximum(425_000_000),
            "tok_embedding_pte_size": maximum(300_000_000),
            "pte_size": maximum(550_000_000),
        },
        by_target={
            HTP_V75: {"inference_speed": minimum(11)},
            HTP_V79: {"inference_speed": minimum(13)},
        },
    ),
}


@dataclass(frozen=True)
class LlmMetricValues:
    v75_inference_speed: float
    v79_inference_speed: float
    pte_size: float
    wiki_ppl: float | None = None
    hellaswag_acc_norm: float | None = None
    sqnr: float | None = None


_LLM_METRICS = {
    "gemma-2b": LlmMetricValues(32, 36, 2_700_000_000, wiki_ppl=17, sqnr=27),
    "gemma2-2b": LlmMetricValues(32, 36, 2_860_000_000, wiki_ppl=14, sqnr=27),
    "gemma3-1b": LlmMetricValues(70, 100, 1_200_000_000, wiki_ppl=23, sqnr=10),
    "gemma4-e2b": LlmMetricValues(20, 30, 4_500_000_000, wiki_ppl=120, sqnr=10),
    "glm-1_5b": LlmMetricValues(42, 52, 1_100_000_000, wiki_ppl=21, sqnr=14),
    "granite_3_3-2b_instruct": LlmMetricValues(
        20, 22, 1_600_000_000, hellaswag_acc_norm=0.2
    ),
    "phi_4_mini": LlmMetricValues(14, 19, 4_000_000_000, wiki_ppl=14, sqnr=20),
    "llama3_2-1b_instruct": LlmMetricValues(
        37, 45, 1_500_000_000, wiki_ppl=18, sqnr=15
    ),
    "llama3_2-3b_instruct": LlmMetricValues(
        21, 26, 2_800_000_000, wiki_ppl=11, sqnr=14
    ),
    "qwen2_5-0_5b": LlmMetricValues(115, 155, 600_000_000, wiki_ppl=15, sqnr=8),
    "qwen2_5-1_5b": LlmMetricValues(38, 47, 1_500_000_000, wiki_ppl=10, sqnr=10),
    "qwen3-0_6b": LlmMetricValues(47, 68, 700_000_000, wiki_ppl=21, sqnr=7),
    "qwen3-1_7b": LlmMetricValues(28, 34, 1_800_000_000, wiki_ppl=15, sqnr=12),
    "smollm2_135m": LlmMetricValues(214, 260, 210_000_000, wiki_ppl=23, sqnr=19),
    "smollm3-3b": LlmMetricValues(23, 28, 2_600_000_000, wiki_ppl=10, sqnr=6),
}

for _model, _metrics in _LLM_METRICS.items():
    _shared = {"pte_size": maximum(_metrics.pte_size)}
    if _metrics.wiki_ppl is not None:
        _shared["wiki_ppl"] = maximum(_metrics.wiki_ppl)
    if _metrics.hellaswag_acc_norm is not None:
        _shared["acc_norm"] = minimum(_metrics.hellaswag_acc_norm)
    if _metrics.sqnr is not None:
        _shared["sqnr"] = minimum(_metrics.sqnr)
    METRIC_PROFILES[f"llm:{_model}"] = _profile(
        _shared,
        by_target={
            HTP_V75: {"inference_speed": minimum(_metrics.v75_inference_speed)},
            HTP_V79: {"inference_speed": minimum(_metrics.v79_inference_speed)},
        },
    )

LLM_MODELS = frozenset(_LLM_METRICS)


def resolve_execution_target(qnn_config) -> ExecutionTarget:
    backend = qnn_config.backend
    if backend == QnnExecuTorchBackendType.kGpuBackend:
        return ExecutionTarget(backend, None)

    try:
        soc_info = _soc_info_table[QcomChipset[qnn_config.soc_model]]
    except KeyError as error:
        raise ValueError(f"Unknown SoC model: {qnn_config.soc_model}") from error

    if backend == QnnExecuTorchBackendType.kHtpBackend:
        if soc_info.htp_info.htp_arch == HtpArch.NONE:
            raise ValueError(f"{qnn_config.soc_model} does not support HTP")
        return ExecutionTarget(backend, soc_info.htp_info.htp_arch)
    if backend == QnnExecuTorchBackendType.kLpaiBackend:
        if (
            soc_info.lpai_info is None
            or soc_info.lpai_info.lpai_hardware_version == LpaiHardwareVersion.NONE
        ):
            raise ValueError(f"{qnn_config.soc_model} does not support LPAI")
        return ExecutionTarget(backend, soc_info.lpai_info.lpai_hardware_version)
    raise ValueError(f"Unsupported backend: {backend}")


def get_metric_constraints(workload: str, qnn_config) -> Mapping[str, MetricConstraint]:
    profile = METRIC_PROFILES[workload]
    target = resolve_execution_target(qnn_config)
    return (
        profile.shared
        | profile.by_backend.get(target.backend, {})
        | profile.by_target.get(target, {})
    )


def get_metric_constraint(
    workload: str, metric: str, qnn_config
) -> MetricConstraint | None:
    return get_metric_constraints(workload, qnn_config).get(metric)


def assert_metric(workload: str, metric: str, actual: float, qnn_config) -> None:
    constraint = get_metric_constraint(workload, metric, qnn_config)
    if constraint is None:
        return
    target = resolve_execution_target(qnn_config)
    constraint.assert_value(actual, f"{workload}.{metric} on {target}")


def assert_metrics(workload: str, result: Mapping[str, float], qnn_config) -> None:
    for metric, constraint in get_metric_constraints(workload, qnn_config).items():
        assert metric in result, f"{workload}: missing metric {metric}"
        target = resolve_execution_target(qnn_config)
        constraint.assert_value(result[metric], f"{workload}.{metric} on {target}")
