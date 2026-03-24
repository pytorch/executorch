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
  - Computes golden outputs for PSNR/CosineSimilarity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

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
# Data model classes
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedOutputs:
    """Wrapper for inference results obtained externally (e.g., from device)."""

    outputs: List[torch.Tensor]

    def __post_init__(self):
        if isinstance(self.outputs, list) and self.outputs:
            if isinstance(self.outputs[0], np.ndarray):
                self.outputs = [torch.from_numpy(o) for o in self.outputs]


class Metric(Protocol):
    def calculate(self, predictions: List[torch.Tensor]) -> float: ...
    def name(self) -> str: ...


class TopKAccuracy:
    def __init__(self, targets: List[Any], k: int = 1):
        self.targets = targets
        self.k = k

    def name(self) -> str:
        return f"top_{self.k}"

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        correct = 0
        total = len(predictions)
        if total == 0:
            return 0.0
        for pred, target in zip(predictions, self.targets):
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if pred.dim() == 2:
                pred = pred.squeeze(0)
            _, indices = pred.topk(self.k)
            if target.view(-1) in indices:
                correct += 1
        return (correct / total) * 100.0


class CosineSimilarity:
    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs

    def name(self) -> str:
        return "cosine_sim"

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        if not self.golden or len(predictions) != len(self.golden):
            return 0.0
        sims = []
        for p, g in zip(predictions, self.golden):
            p_flat = p.flatten().float()
            g_flat = g.flatten().float()
            sims.append(
                F.cosine_similarity(p_flat.unsqueeze(0), g_flat.unsqueeze(0)).item()
            )
        return float(np.mean(sims))


class PSNR:
    def __init__(self, golden_outputs: List[torch.Tensor]):
        self.golden = golden_outputs
        self.max_val = (
            max(torch.max(g).item() for g in golden_outputs) if golden_outputs else 1.0
        )

    def name(self) -> str:
        return "psnr"

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        if not self.golden or len(predictions) != len(self.golden):
            return 0.0
        psnrs = []
        for p, g in zip(predictions, self.golden):
            mse = F.mse_loss(p.float(), g.float())
            if mse == 0:
                psnrs.append(float("inf"))
            else:
                psnrs.append(
                    20
                    * torch.log10(
                        torch.tensor(self.max_val) / torch.sqrt(mse)
                    ).item()
                )
        valid = [x for x in psnrs if x != float("inf")]
        return float(np.mean(valid)) if valid else 100.0


class MaskedTokenAccuracy:
    """Token-level accuracy for MLM models, filtering by ignore_index (-100)."""

    def __init__(self, targets: List[torch.Tensor], ignore_index: int = -100):
        self.targets = targets
        self.ignore_index = ignore_index

    def name(self) -> str:
        return "masked_token_accuracy"

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        correct, total = 0, 0
        for pred, target in zip(predictions, self.targets):
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            indices = [
                i
                for i, t in enumerate(target.view(-1))
                if t.item() != self.ignore_index
            ]
            if not indices:
                continue
            if pred.dim() >= 2:
                pred_tokens = pred.view(-1, pred.shape[-1]).argmax(dim=-1)
            else:
                pred_tokens = pred.view(-1)
            for i in indices:
                if i < len(pred_tokens) and pred_tokens[i].item() == target.view(-1)[
                    i
                ].item():
                    correct += 1
            total += len(indices)
        return (correct / total) * 100.0 if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(
        self,
        dataset: List[Any],
        metrics: List[Any],
        post_process: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.post_process = post_process or (lambda x: x)

    def evaluate(self, model: Any) -> Dict[str, float]:
        predictions = self.run_inference(model, self.dataset)
        results = {}
        for metric in self.metrics:
            try:
                results[metric.name()] = metric.calculate(predictions)
            except Exception as e:
                logging.error("Metric %s failed: %s", metric.name(), e)
                results[metric.name()] = f"error: {e}"
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
        for inputs in dataset:
            args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
            raw_out = executable(*args)
            out = self.post_process(raw_out)
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu()
            predictions.append(out)
        return predictions


class MLMEvaluator(Evaluator):
    """Evaluator for masked language models with -100 masking."""

    def run_inference(self, model: Any, dataset: List[Any]) -> List[torch.Tensor]:
        if isinstance(model, PrecomputedOutputs):
            return model.outputs
        predictions = []
        is_ep = hasattr(model, "module") and callable(model.module)
        executable = model.module() if is_ep else model
        for inputs in dataset:
            args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
            out = executable(*args)
            logits = out.logits if hasattr(out, "logits") else out
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu()
            predictions.append(logits)
        return predictions


# ---------------------------------------------------------------------------
# AccuracyLens
# ---------------------------------------------------------------------------


class AccuracyLens(Lens):
    """Evaluates model accuracy at each collected pipeline stage.

    Configures itself lazily when it first observes the "Exported Float" record,
    extracting the float model from the ExportedProgram and building an evaluator
    with captured dataset + auto-detected metrics.
    """

    _installed: bool = False
    _originals: Dict[str, Any] = {}

    _captured_dataset: Optional[List[Any]] = None
    _captured_targets: Optional[List[Any]] = None
    _golden_outputs: Optional[List[torch.Tensor]] = None
    _post_process: Optional[Callable] = None
    _evaluator: Optional[Evaluator] = None
    _task_type: Optional[str] = None  # "classification", "mlm", or None

    @classmethod
    def get_name(cls) -> str:
        return "accuracy"

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> None:
        if cls._installed:
            return
        cls._install_dataset_patches()
        cls._installed = True

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls._uninstall_all()
        cls._clear_state()

    @classmethod
    def clear(cls) -> None:
        cls._uninstall_all()
        cls._clear_state()

    @classmethod
    def _clear_state(cls) -> None:
        cls._captured_dataset = None
        cls._captured_targets = None
        cls._golden_outputs = None
        cls._post_process = None
        cls._evaluator = None
        cls._task_type = None

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        record_name = context.shared_state.get("record_name", "")

        # Lazily configure evaluator on first "Exported Float" record
        if record_name == "Exported Float" and cls._evaluator is None:
            cls._configure_from_float_model(artifact)

        acc_config = context.config.get("accuracy", {})
        evaluator = acc_config.get("evaluator") or cls._evaluator
        if not evaluator:
            return None

        if not isinstance(
            artifact,
            (torch.nn.Module, torch.fx.GraphModule, torch.export.ExportedProgram),
        ):
            return None

        try:
            metrics = evaluator.evaluate(artifact)
            return {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
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
    def _detect_task_type(cls, targets: List[Any]) -> str:
        if not targets:
            return "unknown"
        sample = targets[0]
        if isinstance(sample, torch.Tensor):
            if sample.dim() >= 1 and (sample == -100).any():
                return "mlm"
        return "classification"

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
        """Lazily configure evaluator from the "Exported Float" ExportedProgram."""
        try:
            is_ep = hasattr(artifact, "module") and callable(artifact.module)
            model = artifact.module() if is_ep else artifact

            # Primary: captured dataset from dataset loader patches
            # Fallback: sample input from PipelineGraphCollectorLens
            dataset = cls._captured_dataset
            if dataset is None:
                from .pipeline_graph_collector import PipelineGraphCollectorLens

                sample_inputs = PipelineGraphCollectorLens._last_export_inputs
                if sample_inputs is not None:
                    dataset = [sample_inputs]
                    cls._captured_dataset = dataset
                    logging.info(
                        "[AccuracyLens] Using sample input from torch.export.export as fallback dataset"
                    )

            if dataset is None:
                logging.debug("[AccuracyLens] No dataset available, skipping auto-config")
                return

            cls._task_type = cls._detect_task_type(cls._captured_targets or [])
            cls._post_process = cls._auto_detect_post_process(model, dataset)
            cls._golden_outputs = cls._compute_golden_outputs(
                model, dataset, cls._post_process
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
            return None

        dataset = cls._captured_dataset
        targets = cls._captured_targets
        golden = cls._golden_outputs
        task_type = cls._task_type

        metrics: List[Any] = []
        if golden:
            metrics.extend([PSNR(golden), CosineSimilarity(golden)])

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
    def _install_dataset_patches(cls) -> None:
        try:
            import executorch.examples.qualcomm.utils as utils_module

            # get_imagenet_dataset
            if hasattr(utils_module, "get_imagenet_dataset"):
                original = utils_module.get_imagenet_dataset
                cls._originals["get_imagenet_dataset"] = original

                def patched_imagenet(*args, **kwargs):
                    inputs, targets = original(*args, **kwargs)
                    cls._captured_targets = targets
                    logging.info(
                        "[AccuracyLens] Captured ImageNet targets (%d samples)",
                        len(targets),
                    )
                    return inputs, targets

                utils_module.get_imagenet_dataset = patched_imagenet
                logging.info("[AccuracyLens] Installed patch: get_imagenet_dataset")

            # get_masked_language_model_dataset
            if hasattr(utils_module, "get_masked_language_model_dataset"):
                original_mlm = utils_module.get_masked_language_model_dataset
                cls._originals["get_masked_language_model_dataset"] = original_mlm

                def patched_mlm(*args, **kwargs):
                    inputs, targets = original_mlm(*args, **kwargs)
                    cls._captured_targets = targets
                    logging.info(
                        "[AccuracyLens] Captured MLM targets (%d samples)",
                        len(targets),
                    )
                    return inputs, targets

                utils_module.get_masked_language_model_dataset = patched_mlm
                logging.info(
                    "[AccuracyLens] Installed patch: get_masked_language_model_dataset"
                )
        except ImportError:
            logging.debug(
                "[AccuracyLens] qualcomm utils not available, skipping dataset patches"
            )

    @classmethod
    def _uninstall_all(cls) -> None:
        if not cls._installed:
            return
        try:
            import executorch.examples.qualcomm.utils as utils_module

            for key, original in cls._originals.items():
                if hasattr(utils_module, key):
                    setattr(utils_module, key, original)
        except ImportError:
            pass
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
        data = {}
        for k, v in digest.items():
            data[k] = v
        record_analysis = analysis.get("record", {})
        for k, v in record_analysis.items():
            if k.endswith("_diff"):
                metric = k.replace("_diff", "")
                if metric in data:
                    data[f"{metric} (diff)"] = f"{v:+.4f}" if isinstance(v, float) else v
        return ViewList(
            blocks=[
                TableBlock(
                    id="accuracy_table",
                    title="Accuracy",
                    record=TableRecordSpec(data=data),
                    order=20,
                )
            ]
        )

    def check_index_diffs(
        self, prev_digest: Any, curr_digest: Any, analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        result = {}
        if not prev_digest or not curr_digest:
            return result
        for key in ["psnr", "top_1", "top_5", "cosine_sim", "masked_token_accuracy"]:
            if key in prev_digest and key in curr_digest:
                prev_val = prev_digest[key]
                curr_val = curr_digest[key]
                if isinstance(prev_val, (int, float)) and isinstance(
                    curr_val, (int, float)
                ):
                    result[key] = f"{curr_val - prev_val:+.2f}"
        return result

    def check_badges(
        self, digest: Any, analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        badges = []
        if digest and isinstance(digest, dict) and "error_message" in digest:
            badges.append({"label": "ERR", "color": "#d73a49"})
        return badges
