# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accuracy Evaluation Lens — auto-captures datasets and evaluates model accuracy.

This lens patches dataset loaders to transparently capture evaluation data, then
lazily configures an evaluator when the first "Exported Float" record is observed.

Patches installed on session start:
  - get_imagenet_dataset: captures (inputs, targets) for ImageNet classification
  - get_masked_language_model_dataset: captures (inputs, targets) for MLM tasks

Lazy configuration (on first "Exported Float" observe):
  - Extracts float model from ExportedProgram
  - Uses captured dataset (primary) or sample input from PipelineGraphCollectorLens (fallback)
  - Auto-detects task type, post_process, and metrics
  - Computes golden outputs for PSNR/CosineSimilarity/MSE/AbsErr

Per-sample statistics (when dataset has >1 sample):
  - Each metric emits mean (primary), min, max, and worst_idx in the digest
  - worst_idx is determined by each metric's higher_is_better direction
  - AccuracyLens._worst_indices exposes {metric_name: index} as class-level state
    so future lenses (e.g., per-layer accuracy) can read the worst input index
    during their own observe() call without re-running inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    RecordAnalysis,
    RecordDigest,
    TableBlock,
    TableRecordSpec,
    ViewList,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedOutputs:
    """Wrapper for inference results obtained externally (e.g., from device)."""

    outputs: List[torch.Tensor]

    def __post_init__(self):
        if isinstance(self.outputs, list) and self.outputs:
            if isinstance(self.outputs[0], np.ndarray):
                self.outputs = [torch.from_numpy(o) for o in self.outputs]


# ---------------------------------------------------------------------------
# Metric base class
# ---------------------------------------------------------------------------


class Metric:
    """Base class for accuracy metrics.

    Subclasses implement calculate_per_sample() which returns one scalar per
    input sample.  The base class derives the aggregate mean and provides
    worst_index() using the metric's built-in direction knowledge.

    higher_is_better controls worst-case direction:
      True  → worst = argmin  (PSNR, cosine_sim, TopK — lower means worse quality)
      False → worst = argmax  (MSE, AbsErr — higher means worse quality)
    """

    higher_is_better: bool = True

    def name(self) -> str:
        raise NotImplementedError

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        raise NotImplementedError

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        values = self.calculate_per_sample(predictions)
        return float(np.mean(values)) if values else 0.0

    def worst_index(self, per_sample: List[float]) -> int:
        if not per_sample:
            return 0
        if self.higher_is_better:
            return int(np.argmin(per_sample))
        else:
            return int(np.argmax(per_sample))


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


class TopKAccuracy(Metric):
    """Classification accuracy: fraction of samples where true label is in top-k."""

    higher_is_better = True

    def __init__(self, targets: List[Any], k: int = 1):
        self.targets = targets
        self.k = k

    def name(self) -> str:
        return f"top_{self.k}"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        values = []
        for pred, target in zip(predictions, self.targets):
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if pred.dim() == 2:
                pred = pred.squeeze(0)
            _, indices = pred.topk(self.k)
            values.append(100.0 if target.view(-1) in indices else 0.0)
        return values

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        values = self.calculate_per_sample(predictions)
        return float(np.mean(values)) if values else 0.0


class CosineSimilarity(Metric):
    """Cosine similarity between predictions and golden outputs."""

    higher_is_better = True

    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs

    def name(self) -> str:
        return "cosine_sim"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        if not self.golden or len(predictions) != len(self.golden):
            return []
        values = []
        for p, g in zip(predictions, self.golden):
            p_flat = p.flatten().float()
            g_flat = g.flatten().float()
            values.append(
                F.cosine_similarity(p_flat.unsqueeze(0), g_flat.unsqueeze(0)).item()
            )
        return values


class PSNR(Metric):
    """Peak Signal-to-Noise Ratio, capped at MAX_PSNR for UI consistency.

    Raw PSNR above MAX_PSNR (e.g., 128 dB for near-zero error) is not
    meaningfully different from perfect match, so we clamp to MAX_PSNR.
    This gives a uniform ceiling: perfect match → MAX_PSNR, real degradation
    → actual dB value below MAX_PSNR.
    """

    higher_is_better = True
    MAX_PSNR = 100.0

    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs
        self.max_val = (
            max(torch.max(g).item() for g in golden_outputs) if golden_outputs else 1.0
        )

    def name(self) -> str:
        return "psnr"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        if not self.golden or len(predictions) != len(self.golden):
            return []
        values = []
        for p, g in zip(predictions, self.golden):
            mse = F.mse_loss(p.float(), g.float())
            if mse == 0:
                values.append(self.MAX_PSNR)
            else:
                db = (
                    20
                    * torch.log10(
                        torch.tensor(self.max_val) / torch.sqrt(mse)
                    ).item()
                )
                values.append(min(db, self.MAX_PSNR))
        return values


class MSE(Metric):
    """Mean Squared Error per sample.  Lower is better (higher_is_better=False)."""

    higher_is_better = False

    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs

    def name(self) -> str:
        return "mse"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        if not self.golden or len(predictions) != len(self.golden):
            return []
        return [
            F.mse_loss(p.float(), g.float()).item()
            for p, g in zip(predictions, self.golden)
        ]


class AbsErr(Metric):
    """Mean Absolute Error per sample.  Lower is better (higher_is_better=False)."""

    higher_is_better = False

    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs

    def name(self) -> str:
        return "abs_err"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        if not self.golden or len(predictions) != len(self.golden):
            return []
        return [
            torch.mean(torch.abs(p.float() - g.float())).item()
            for p, g in zip(predictions, self.golden)
        ]


class MaskedTokenAccuracy(Metric):
    """Token-level accuracy for MLM models, filtering by ignore_index (-100)."""

    higher_is_better = True

    def __init__(self, targets: List[torch.Tensor], ignore_index: int = -100):
        self.targets = targets
        self.ignore_index = ignore_index

    def name(self) -> str:
        return "masked_token_accuracy"

    def calculate_per_sample(self, predictions: List[torch.Tensor]) -> List[float]:
        values = []
        for pred, target in zip(predictions, self.targets):
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            indices = [
                i
                for i, t in enumerate(target.view(-1))
                if t.item() != self.ignore_index
            ]
            if not indices:
                values.append(0.0)
                continue
            if pred.dim() >= 2:
                pred_tokens = pred.view(-1, pred.shape[-1]).argmax(dim=-1)
            else:
                pred_tokens = pred.view(-1)
            correct = sum(
                1
                for i in indices
                if i < len(pred_tokens)
                and pred_tokens[i].item() == target.view(-1)[i].item()
            )
            values.append((correct / len(indices)) * 100.0)
        return values

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        values = self.calculate_per_sample(predictions)
        return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(
        self,
        dataset: List[Any],
        metrics: List[Metric],
        post_process: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.post_process = post_process or (lambda x: x)

    def evaluate(self, model: Any) -> Dict[str, Any]:
        """Run inference and compute all metrics.

        For each metric, always emits the mean value under the metric name.
        When dataset has >1 sample, also emits:
          {name}_min, {name}_max  — range across samples
          {name}_worst_idx        — index of the worst-performing sample
                                    (argmin for higher_is_better, argmax otherwise)
        """
        predictions = self.run_inference(model, self.dataset)
        results: Dict[str, Any] = {"_num_samples": len(predictions)}
        for metric in self.metrics:
            name = metric.name()
            try:
                per_sample = metric.calculate_per_sample(predictions)
                if not per_sample:
                    results[name] = 0.0
                    continue
                results[name] = round(float(np.mean(per_sample)), 4)
                if len(per_sample) > 1:
                    results[f"{name}_min"] = round(float(min(per_sample)), 4)
                    results[f"{name}_max"] = round(float(max(per_sample)), 4)
                    results[f"{name}_worst_idx"] = metric.worst_index(per_sample)
            except Exception as e:
                logging.error("Metric %s failed: %s", name, e)
                results[name] = f"error: {e}"
        return results

    def run_inference(self, model: Any, dataset: List[Any]) -> List[torch.Tensor]:
        raise NotImplementedError


class StandardEvaluator(Evaluator):
    """Standard evaluator for classification and regression models."""

    def run_inference(self, model: Any, dataset: List[Any]) -> List[torch.Tensor]:
        if isinstance(model, PrecomputedOutputs):
            return model.outputs
        predictions = []
        is_ep = hasattr(model, "module") and callable(model.module)
        executable = model.module() if is_ep else model
        # torch.no_grad() matches _compute_golden_outputs — without it the
        # autograd context can cause subtle numerical differences vs golden.
        with torch.no_grad():
            for inputs in dataset:
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                raw_out = executable(*args)
                out = self.post_process(raw_out)
                if isinstance(out, torch.Tensor):
                    out = out.detach().cpu()
                predictions.append(out)
        return predictions


class MLMEvaluator(Evaluator):
    """Evaluator for masked language models with -100 masking.

    Uses self.post_process (from _auto_detect_post_process) to extract logits,
    keeping the inference path consistent with how golden outputs are computed
    in _compute_golden_outputs.  An earlier version had hardcoded
    ``out.logits if hasattr(out, "logits") else out`` which diverged from the
    golden computation and produced wrong PSNR/cosine for HuggingFace models.
    """

    def run_inference(self, model: Any, dataset: List[Any]) -> List[torch.Tensor]:
        if isinstance(model, PrecomputedOutputs):
            return model.outputs
        predictions = []
        is_ep = hasattr(model, "module") and callable(model.module)
        executable = model.module() if is_ep else model
        with torch.no_grad():
            for inputs in dataset:
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                raw_out = executable(*args)
                out = self.post_process(raw_out)
                if isinstance(out, torch.Tensor):
                    out = out.detach().cpu()
                predictions.append(out)
        return predictions


# ---------------------------------------------------------------------------
# AccuracyLens
# ---------------------------------------------------------------------------


class AccuracyLens(Lens):
    """Evaluates model accuracy at each collected pipeline stage.

    Configures itself lazily when it first observes the "Exported Float" record,
    extracting the float model from the ExportedProgram and building an evaluator
    with captured dataset + auto-detected metrics.

    Cross-lens data sharing:
      _worst_indices is a class-level dict {metric_name: worst_input_index} updated
      after every evaluate() call.  Future lenses (e.g., per-layer accuracy) can
      read it during their own observe() to focus analysis on the worst input:

        from .accuracy import AccuracyLens
        worst = AccuracyLens._worst_indices.get("psnr")  # int or None

      This follows the same pattern as PipelineGraphCollectorLens._last_export_inputs.
      Lenses run in registration order, so AccuracyLens must be registered before
      any lens that reads _worst_indices.
    """

    _installed: bool = False
    _originals: Dict[str, Any] = {}
    # Backend-specific dataset patch installers.
    _dataset_patch_installers: List[Callable] = []
    _dataset_uninstallers: List[Callable] = []

    _float_model: Any = None  # cached GraphModule from "Exported Float" ExportedProgram
    _captured_dataset: Optional[List[Any]] = None
    _captured_targets: Optional[List[Any]] = None
    _golden_outputs: Optional[List[torch.Tensor]] = None
    _post_process: Optional[Callable] = None
    _evaluator: Optional[Evaluator] = None
    _task_type: Optional[str] = None  # "classification", "mlm", or None
    _worst_indices: Dict[str, int] = {}  # {metric_name: worst_input_index}

    @classmethod
    def register_dataset_patches(
        cls, installer: Callable[["AccuracyLens"], None]
    ) -> None:
        """Register a backend-specific dataset patch installer.

        The installer receives the AccuracyLens class and should set
        cls._captured_targets and cls._task_type when dataset functions
        are called. It may also append to cls._dataset_uninstallers.
        """
        if installer not in cls._dataset_patch_installers:
            cls._dataset_patch_installers.append(installer)

    @classmethod
    def get_name(cls) -> str:
        return "accuracy"

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> None:
        if cls._installed:
            return
        for installer in cls._dataset_patch_installers:
            try:
                installer(cls)
            except Exception as exc:
                logging.warning(
                    "[AccuracyLens] Dataset patch installer failed: %s", exc
                )
        cls._installed = True

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls._uninstall_all()
        cls._clear_state()

    @classmethod
    def clear(cls) -> None:
        cls._uninstall_all()
        cls._clear_state()
        cls._dataset_patch_installers.clear()
        cls._dataset_uninstallers.clear()

    @classmethod
    def _clear_state(cls) -> None:
        cls._float_model = None
        cls._captured_dataset = None
        cls._captured_targets = None
        cls._golden_outputs = None
        cls._post_process = None
        cls._evaluator = None
        cls._task_type = None
        cls._worst_indices = {}

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:

        acc_config = context.config.get("accuracy", {})
        if not acc_config.get("enabled", True):
            return None

        record_name = context.shared_state.get("record_name", "")

        # Lazily configure evaluator on first "Exported Float" record
        if record_name == "Exported Float" and cls._evaluator is None:
            cls._configure_from_float_model(artifact)

        evaluator = acc_config.get("evaluator") or cls._evaluator

        if not evaluator:
            return None

        if not isinstance(
            artifact,
            (torch.nn.Module, torch.fx.GraphModule, torch.export.ExportedProgram),
        ):
            return None

        eval_artifact = artifact
        if record_name == "Exported Float" and cls._float_model is not None:
            eval_artifact = cls._float_model

        try:
            raw = evaluator.evaluate(eval_artifact)
            # Update class-level worst indices for cross-lens access
            cls._worst_indices = {
                k[: -len("_worst_idx")]: v
                for k, v in raw.items()
                if k.endswith("_worst_idx")
            }
            return {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in raw.items()
            }
        except Exception as e:
            logging.error("[AccuracyLens] Evaluation failed: %s", e)
            return {"error_message": str(e)}

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    @staticmethod
    def analyze(
        records: List[RecordDigest], config: Dict[str, Any]
    ) -> AnalysisResult:
        result = AnalysisResult()
        for i, record in enumerate(records):
            digest = record.data.get("accuracy")
            if digest is None:
                continue
            analysis = RecordAnalysis()
            if i > 0:
                prev = records[i - 1].data.get("accuracy", {})
                for key in digest:
                    if key.startswith("_"):
                        continue
                    if (
                        isinstance(digest.get(key), (int, float))
                        and isinstance(prev.get(key), (int, float))
                    ):
                        analysis.data[f"{key}_diff"] = round(
                            digest[key] - prev[key], 4
                        )
            result.per_record_data[record.name] = analysis
        return result

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return _AccuracyFrontend()

    # ------------------------------------------------------------------
    # Auto-configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def _auto_detect_post_process(cls, model: Any, dataset: List[Any]) -> Callable:
        try:
            sample = dataset[0]
            args = sample if isinstance(sample, (tuple, list)) else (sample,)
            with torch.no_grad():
                out = model(*args)
            if isinstance(out, torch.Tensor):
                return lambda x: x
            if hasattr(out, "logits"):
                return lambda x: x.logits
            if isinstance(out, tuple):
                return lambda x: x[0]
        except Exception as e:
            logging.debug("[AccuracyLens] post_process auto-detect failed: %s", e)
        return lambda x: x

    @classmethod
    def _compute_golden_outputs(
        cls, model: Any, dataset: List[Any], post_process: Callable
    ) -> List[torch.Tensor]:
        golden = []
        with torch.no_grad():
            for inputs in dataset:
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                out = model(*args)
                processed = post_process(out)
                if isinstance(processed, torch.Tensor):
                    processed = processed.detach().cpu()
                golden.append(processed)
        return golden

    @classmethod
    def _configure_from_float_model(cls, artifact: Any) -> None:
        """Lazily configure evaluator from the "Exported Float" ExportedProgram.

        Caches the extracted GraphModule as _float_model so that observe() can
        reuse it for the "Exported Float" evaluation instead of calling
        artifact.module() a second time (which would create a different
        GraphModule instance and risk numerical mismatch with golden outputs).
        """
        try:
            is_ep = hasattr(artifact, "module") and callable(artifact.module)
            model = artifact.module() if is_ep else artifact
            cls._float_model = model

            # Primary: captured dataset from dataset loader patches
            # Fallback: sample input captured by backend-specific patches in
            # PipelineGraphCollectorLens.
            if cls._captured_dataset is None:
                from .pipeline_graph_collector import PipelineGraphCollectorLens

                calibration_dataset = PipelineGraphCollectorLens._last_calibration_dataset
                # cls._captured_dataset might already be captured
                # from dataloader patching in _install_dataset_patches
                if calibration_dataset is not None:
                    cls._captured_dataset = calibration_dataset
                    logging.info(
                        "[AccuracyLens] Using backend-captured fallback dataset from PipelineGraphCollectorLens"
                    )

            if cls._captured_dataset is None:
                logging.debug("[AccuracyLens] No dataset available, skipping auto-config")
                return

            cls._post_process = cls._auto_detect_post_process(model, cls._captured_dataset)
            cls._golden_outputs = cls._compute_golden_outputs(
                model, cls._captured_dataset, cls._post_process
            )
            cls._evaluator = cls._build_default_evaluator()
            if cls._evaluator:
                logging.info(
                    "[AccuracyLens] Auto-configured %s evaluator with %d metrics",
                    cls._task_type,
                    len(cls._evaluator.metrics),
                )
        except Exception as e:
            logging.warning("[AccuracyLens] Auto-config from float model failed: %s", e)

    @classmethod
    def _build_default_evaluator(cls) -> Optional[Evaluator]:
        if cls._captured_dataset is None:
            logging.info("[AccuracyLens] Unable to auto build evaluator because no dataset is captured")
            return None

        dataset = cls._captured_dataset
        targets = cls._captured_targets
        golden = cls._golden_outputs
        task_type = cls._task_type

        metrics: List[Metric] = []
        if golden:
            metrics.extend([
                PSNR(golden),
                CosineSimilarity(golden),
                MSE(golden),
                AbsErr(golden),
            ])

        if task_type == "classification" and targets:
            metrics.extend([TopKAccuracy(targets, k=1), TopKAccuracy(targets, k=5)])
            return StandardEvaluator(
                dataset=dataset, metrics=metrics, post_process=cls._post_process
            )
        elif task_type == "mlm" and targets:
            metrics.append(MaskedTokenAccuracy(targets))
            return MLMEvaluator(
                dataset=dataset, metrics=metrics, post_process=cls._post_process
            )
        elif metrics:
            return StandardEvaluator(
                dataset=dataset, metrics=metrics, post_process=cls._post_process
            )
        return None

    # ------------------------------------------------------------------
    # Patches
    # ------------------------------------------------------------------

    @classmethod
    def _uninstall_all(cls) -> None:
        if not cls._installed:
            return
        for uninstaller in cls._dataset_uninstallers:
            try:
                uninstaller()
            except Exception:
                pass
        cls._dataset_uninstallers.clear()
        cls._originals.clear()
        cls._installed = False
        logging.info("[AccuracyLens] Uninstalled all patches")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


class _AccuracyFrontend(Frontend):
    def record(
        self, digest: Any, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ViewList]:
        if not digest or not isinstance(digest, dict):
            return None

        # Partition digest keys into three groups:
        #   primary  — mean metric values (no suffix)
        #   stats    — per-sample min/max (_min / _max suffix)
        #   worst    — worst-case input indices (_worst_idx suffix)
        num_samples = digest.get("_num_samples", 1)
        primary_data = {}
        stats_data = {}
        worst_data = {}
        for k, v in digest.items():
            if k.startswith("_"):
                continue  # internal keys (_num_samples, etc.)
            if k.endswith("_worst_idx"):
                worst_data[k[: -len("_worst_idx")]] = v
            elif k.endswith("_min") or k.endswith("_max"):
                stats_data[k] = v
            else:
                primary_data[k] = v

        n = f"{num_samples} sample{'s' if num_samples != 1 else ''}"
        blocks = [
            TableBlock(
                id="accuracy_table",
                title=f"Accuracy ({n})",
                record=TableRecordSpec(data=primary_data),
                order=20,
            )
        ]

        # Per-sample min/max table: only present when >1 sample was evaluated.
        if stats_data:
            blocks.append(
                TableBlock(
                    id="accuracy_stats_table",
                    title=f"Per-Sample Stats ({n})",
                    record=TableRecordSpec(data=stats_data),
                    order=21,
                )
            )

        # Worst input index table: only present when >1 sample was evaluated.
        if worst_data:
            blocks.append(
                TableBlock(
                    id="accuracy_worst_idx_table",
                    title=f"Worst Input Index ({n})",
                    record=TableRecordSpec(data=worst_data),
                    order=22,
                )
            )

        return ViewList(blocks=blocks)

    def check_index_diffs(
        self, prev_digest: Any, curr_digest: Any, analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        result = {}
        if not prev_digest or not curr_digest:
            return result
        for key in [
            "psnr", "cosine_sim", "mse", "abs_err",
            "top_1", "top_5", "masked_token_accuracy",
        ]:
            if key in prev_digest and key in curr_digest:
                prev_val = prev_digest[key]
                curr_val = curr_digest[key]
                if isinstance(prev_val, (int, float)) and isinstance(
                    curr_val, (int, float)
                ):
                    if abs(curr_val - prev_val) < 0.0001:
                        continue
                    result[key] = f"{curr_val - prev_val:+.4f}"
        return result

    def check_badges(
        self, digest: Any, analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        badges = []
        if digest and isinstance(digest, dict) and "error_message" in digest:
            badges.append({"label": "ERR", "color": "#d73a49"})
        return badges
