# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from executorch.examples.qualcomm.oss_scripts.llama.inference import DecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.quantize.ptq import PTQStrategy
from executorch.examples.qualcomm.oss_scripts.llama.train.config import TrainingArgs
from executorch.examples.qualcomm.oss_scripts.llama.train.trainer import (
    KDTrainer,
    Trainer,
)
from torch.utils.data import DataLoader


def _disable_fake_quant_enable_observer(model: torch.fx.GraphModule) -> None:
    for _, submodule in model.named_modules():
        if hasattr(submodule, "disable_fake_quant"):
            submodule.disable_fake_quant()
        if hasattr(submodule, "enable_observer"):
            submodule.enable_observer()


def _enable_fake_quant_freeze_observer(model: torch.fx.GraphModule) -> None:
    for _, submodule in model.named_modules():
        if hasattr(submodule, "enable_fake_quant"):
            submodule.enable_fake_quant()
        if hasattr(submodule, "disable_observer"):
            submodule.disable_observer()


class QATStrategy(PTQStrategy):
    """Quantization-aware training strategy"""

    def __init__(
        self,
        inference: DecoderInference,
        module: torch.fx.GraphModule,
        seq_mse_candidates: int = 0,
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ):
        super().__init__(
            inference=inference,
            module=module,
            seq_mse_candidates=seq_mse_candidates,
            tok_embedding=tok_embedding,
        )

    def _calibrate(self, calib_loader: Dict[str, DataLoader]) -> None:
        _disable_fake_quant_enable_observer(self._module)
        super()._calibrate(calib_loader)
        _enable_fake_quant_freeze_observer(self._module)

    def _train(
        self,
        training_args: TrainingArgs,
        teacher: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        frozen_param_patterns: Optional[List[str]] = None,
    ) -> None:
        forward_fn = self._inference.make_forward_fn()
        if teacher is not None:
            KDTrainer(
                model=self._module,
                teacher=teacher,
                args=training_args,
                train_loader=train_loader,
                forward_fn=forward_fn,
                val_loader=val_loader,
                frozen_param_patterns=frozen_param_patterns,
            ).train()
        else:
            Trainer(
                model=self._module,
                args=training_args,
                train_loader=train_loader,
                forward_fn=forward_fn,
                val_loader=val_loader,
                frozen_param_patterns=frozen_param_patterns,
            ).train()

    def quantize(
        self,
        calib_loader: Dict[str, DataLoader],
        training_args: TrainingArgs = None,
        teacher: torch.nn.Module = None,
        train_loader: DataLoader = None,
        val_loader: Optional[DataLoader] = None,
        frozen_param_patterns: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.fx.GraphModule:
        self._calibrate(calib_loader)
        self._train(
            training_args=training_args,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            frozen_param_patterns=frozen_param_patterns,
        )
        return self._module
