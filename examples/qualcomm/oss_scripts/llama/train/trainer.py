# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.examples.qualcomm.oss_scripts.llama.train.config import TrainingArgs
from executorch.examples.qualcomm.oss_scripts.llama.train.loss import (
    CrossEntropyLoss,
    KLDivergenceLoss,
)
from executorch.examples.qualcomm.oss_scripts.llama.train.utils import (
    build_param_groups,
    get_warmup_cosine_lr,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(ABC):
    """Training loop skeleton: epoch/batch iteration, grad accumulation, clipping, optimizer step.

    Subclasses implement train_step, eval_step, and configure_optimizers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArgs,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cpu")  # TODO: validate GPU compatibility
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler = None

    @abstractmethod
    def train_step(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward + loss. Returns (loss, metrics). Base calls .backward() — do NOT call it here."""

    @abstractmethod
    def eval_step(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Validation forward for one batch. No gradients."""

    @abstractmethod
    def configure_optimizers(
        self, warmup_steps: int, total_steps: int
    ) -> Tuple[torch.optim.Optimizer, object]:
        """Build (optimizer, lr_scheduler). Called after model.to(device)."""

    def train(self) -> torch.nn.Module:
        self.model.to(self.device)

        num_batches = len(self.train_loader)
        optimizer_steps_per_epoch = math.ceil(num_batches / self.args.grad_accum_steps)
        total_steps = optimizer_steps_per_epoch * self.args.epochs
        warmup_steps = max(2, int(total_steps * self.args.warmup_ratio))
        self._optimizer, self._scheduler = self.configure_optimizers(
            warmup_steps, total_steps
        )

        if self._optimizer is None:
            logging.info(
                f"{type(self).__name__}: no trainable parameters, training loop skipped."
            )
            self.model.to("cpu")
            return self.model

        logging.info(
            f"{type(self).__name__}: device={self.device}  "
            f"train_batches={num_batches}  epochs={self.args.epochs}  "
            f"steps_per_epoch={optimizer_steps_per_epoch}  warmup={warmup_steps}"
        )

        for epoch in tqdm(range(self.args.epochs)):
            total_loss = 0.0
            t0 = time.time()

            for batch_idx, model_inputs in enumerate(self.train_loader):
                with torch.enable_grad():
                    loss, metrics = self.train_step(model_inputs)
                    (loss / self.args.grad_accum_steps).backward()

                is_last_batch = batch_idx == num_batches - 1
                if (batch_idx + 1) % self.args.grad_accum_steps == 0 or is_last_batch:
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()

                    lr = self._scheduler.get_last_lr()[0]
                    logging.info(
                        f"Epoch {epoch}, batch {batch_idx}/{num_batches}  "
                        + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                        + f"  lr={lr:.2e}"
                    )
                total_loss += loss.item()

            val_info = ""
            if self.val_loader is not None:
                val_metrics = self._run_validation()
                val_info = "  " + "  ".join(
                    f"val_{k}={v:.4f}" for k, v in val_metrics.items()
                )
            logging.info(
                f"Epoch {epoch}: avg_loss={total_loss / max(num_batches, 1):.4f}"
                f"{val_info}  elapsed={time.time() - t0:.1f}s"
            )

        self.model.to("cpu")
        return self.model

    def _to_device(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }

    @torch.no_grad()
    def _run_validation(self) -> Dict[str, float]:
        accum: Dict[str, float] = {}
        n = 0
        for model_inputs in self.val_loader:
            for k, v in self.eval_step(model_inputs).items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
        return {k: v / n for k, v in accum.items()} if n > 0 else {}


class Trainer(BaseTrainer):
    """CE-only trainer (SFT / PTQ-warmup fine-tuning).

    forward_fn and loss_fn are injected so this class is unaware of attn_masks or KV caches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArgs,
        train_loader: DataLoader,
        forward_fn: Callable,
        val_loader: Optional[DataLoader] = None,
        frozen_param_patterns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(model, args, train_loader, val_loader)
        self.forward_fn = forward_fn
        self.loss_fn = CrossEntropyLoss()
        self._frozen_param_patterns: List[str] = frozen_param_patterns or []

    def configure_optimizers(
        self, warmup_steps: int, total_steps: int
    ) -> Tuple[torch.optim.Optimizer, object]:
        self._freeze_params()
        params = build_param_groups(self.model, self.args.lr_config)
        if not params:
            logging.info(
                f"{type(self).__name__}: no trainable parameters, skipping optimizer setup."
            )
            return None, None
        optimizer = torch.optim.AdamW(
            params,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        scheduler = get_warmup_cosine_lr(
            optimizer,
            warmup_min_lr=1e-8,
            warmup_max_lr=self.args.lr,
            warmup_num_steps=max(2, warmup_steps),
            total_steps=total_steps,
        )
        logging.info(
            f"{type(self).__name__}: AdamW  lr={self.args.lr:.2e}  "
            f"weight_decay=0.01  loss={type(self.loss_fn).__name__}"
        )
        return optimizer, scheduler

    def train_step(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        model_inputs = self._to_device(model_inputs)
        gt_labels = model_inputs["labels"]
        logits = self.forward_fn(self.model, model_inputs)
        loss = self.loss_fn.compute(logits, gt_labels)
        return loss, {"ce": loss.item()}

    @torch.no_grad()
    def eval_step(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        model_inputs = self._to_device(model_inputs)
        gt_labels = model_inputs["labels"]
        logits = self.forward_fn(self.model, model_inputs)
        loss = self.loss_fn.compute(logits, gt_labels)
        return {"ce": loss.item()}

    def _freeze_params(self) -> None:
        frozen = 0
        for name, param in self.model.named_parameters():
            if any(re.search(p, name) for p in self._frozen_param_patterns):
                param.requires_grad_(False)
                frozen += 1
                logging.info(f"QAT: frozen param {name}")
        logging.info(f"QAT: frozen {frozen} params")


class KDTrainer(Trainer):
    """QAT knowledge-distillation trainer. Extends Trainer with an FP32 teacher and KD loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        teacher: torch.nn.Module,
        args: TrainingArgs,
        train_loader: DataLoader,
        forward_fn: Callable,
        val_loader: Optional[DataLoader] = None,
        frozen_param_patterns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_loader=train_loader,
            forward_fn=forward_fn,
            val_loader=val_loader,
            frozen_param_patterns=frozen_param_patterns,
        )
        self.teacher = teacher

        self._kd_loss = KLDivergenceLoss(temperature=args.temperature)
        self._alpha = args.alpha

    def configure_optimizers(
        self, warmup_steps: int, total_steps: int
    ) -> Tuple[torch.optim.Optimizer, object]:
        optimizer, scheduler = super().configure_optimizers(warmup_steps, total_steps)
        if optimizer is None:
            return None, None
        logging.info(
            f"KDTrainer: alpha={self._alpha:.3f} (KLDivergenceLoss)  "
            f"(1-alpha)={1.0 - self._alpha:.3f} (CrossEntropyLoss)"
        )
        return optimizer, scheduler

    def train(self) -> torch.nn.Module:
        self.teacher.to(self.device)
        self.teacher.eval()

        try:
            return super().train()
        finally:
            self.teacher.to("cpu")

    def train_step(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        model_inputs = self._to_device(model_inputs)
        gt_labels = model_inputs["labels"]

        with torch.no_grad():
            teacher_logits = self.forward_fn(self.teacher, model_inputs)

        student_logits = self.forward_fn(self.model, model_inputs)
        kd = self._kd_loss.compute(student_logits, teacher_logits, gt_labels)
        ce = self.loss_fn.compute(student_logits, gt_labels)
        loss = self._alpha * kd + (1.0 - self._alpha) * ce
        return loss, {"kd": kd.item(), "ce": ce.item()}

    @torch.no_grad()
    def eval_step(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        model_inputs = self._to_device(model_inputs)
        gt_labels = model_inputs["labels"]

        teacher_logits = self.forward_fn(self.teacher, model_inputs)
        student_logits = self.forward_fn(self.model, model_inputs)
        kd = self._kd_loss.compute(student_logits, teacher_logits, gt_labels)
        ce = self.loss_fn.compute(student_logits, gt_labels)
        return {"kd": kd.item(), "ce": ce.item()}
