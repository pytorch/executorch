# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import dataclasses
import logging
from abc import abstractmethod

import numpy as np

import torch
import torchaudio
from executorch.backends.samsung.test.utils.utils import GreedyLM


@dataclasses.dataclass
class CheckerConfig:
    checker: str
    kargs: dict


class CheckerBase:
    necessary_params = []
    default_params = {}

    def __init__(
        self,
        original_module: torch.nn.Module,
        current_module: torch.nn.Module,
        config: CheckerConfig,
    ):
        self.origin_module = original_module
        self.current_module = current_module
        self.config = config
        self.check_and_set_params()

    @abstractmethod
    def check(self, **kwargs):
        pass

    def check_and_set_params(self):
        expected_list = []
        for key in self.necessary_params:
            if key not in self.config.kargs:
                expected_list.append(key)
            else:
                setattr(self, key, self.config.kargs[key])
        assert (
            not expected_list
        ), f"More args expected for {type(self)} in config.kargs: " + ",".join(
            expected_list
        )
        for key in self.default_params:
            if key not in self.config.kargs:
                default_value = self.default_params[key]
                logging.info(
                    f"{key} not set in config.kargs for checker {type(self)}, using default value {default_value}"
                )
                setattr(self, key, default_value)
            else:
                setattr(self, key, self.config.kargs[key])


CHECKER_REGISTER = {}


def checker_register(checker_name: str):
    def _wrapper(cls):
        CHECKER_REGISTER[checker_name] = cls

    return _wrapper


def get_checker(origin_module, quantized_module, config) -> CheckerBase:
    assert config.checker in CHECKER_REGISTER, (
        f'Could not find checker "{config.checker}", registered checkers: \n\t'
        + "\n\t".join(CHECKER_REGISTER.keys())
    )
    return CHECKER_REGISTER[config.checker](origin_module, quantized_module, config)


@checker_register("classifier")
class ClassifierChecker(CheckerBase):
    necessary_params = ["dataset"]
    default_params = {
        "topktol": {
            1: 0.9,
            3: 0.95,
        },
    }

    def check(self):
        assert self.dataset
        assert min(self.topktol.keys()) > 0, "Topk number must be positive int"
        max_topk = max(self.topktol.keys())

        print("Check Quantization Classifier...")

        correct = torch.Tensor([0] * max_topk, device="cpu")
        total = 0
        for batch_data, _ in self.dataset:
            batch_size = batch_data.shape[0]
            total += batch_size
            # TODO: Use ground truth to replace fp models' result
            fp_out: torch.Tensor = self.origin_module(batch_data)
            _, fp_top1 = fp_out.topk(1, dim=-1)
            fp_top1 = fp_top1.view(1, -1)

            quant_out: torch.Tensor = self.current_module(batch_data)
            _, quant_topk = quant_out.topk(max_topk, dim=-1)
            quant_topk = quant_topk.t()
            for k_idx in range(max_topk):
                correct[k_idx:] += quant_topk[k_idx].eq(fp_top1).view(-1).sum().float()
        error_messages = []
        msg_template = "\tFailed in checking Top{}, Target: {:.2f} vs Current: {:.2f}"
        for topk_num, topk_tol in self.topktol.items():
            correct_num = correct[topk_num - 1]
            accuracy_score = correct_num / total * 100
            print(accuracy_score)
            if accuracy_score < topk_tol:

                error_messages.append(
                    msg_template.format(topk_num, topk_tol, accuracy_score)
                )
        assert not error_messages, "\n".join(["\n", *error_messages])
        print("Check Quantization Classifier Finished.")


@checker_register("super_resolution")
class SRChecker(CheckerBase):
    necessary_params = ["dataset"]
    default_params = {"threshold": 35.0}

    def check(self):
        peak = 1.0  # Images are scaled to 0-1

        def calc_unbatch_mse(x: torch.Tensor, target: torch.Tensor):
            # We calc PSNR for each single image
            num = torch.prod(torch.tensor(x.shape)[1:])
            return (x - target).pow(2).sum(dim=list(range(1, len(x.shape)))).pow(
                0.5
            ) / num

        data_num = 0
        total_psnr = 0
        for x, target in self.dataset:
            data_num += len(x)
            quant_out: torch.Tensor = self.current_module(x)
            unbatch_mse = calc_unbatch_mse(target, quant_out)
            unbatch_psnr = 10 * torch.log10(peak * peak / unbatch_mse)
            total_psnr += unbatch_psnr.sum()
        avg_psnr = total_psnr / data_num
        assert (
            avg_psnr > self.threshold
        ), "PSNR need to be larger than {:.2f}, but get {:.2f}. ".format(
            self.threshold, avg_psnr
        )
        print("Check Quantization Super Resolution Finished.")


@checker_register("segmentation")
class SegChecker(CheckerBase):
    necessary_params = ["dataset"]
    default_params = {"threshold": 0.7}

    def check(self):
        def calc_miou(target: torch.Tensor, pred: torch.Tensor, class_num=21):
            target = target.numpy().flatten()
            mask = target != 255  # Don't consider edge
            target = target[mask]
            pred = pred.numpy().flatten()[mask]
            target *= class_num
            target += pred
            # I of class a: mixmat[a, a]
            # U of class a: mixmat[a, :].sum() + mixmat[:, a].sum - mixmat[a, a]
            mixmat = np.bincount(target, minlength=class_num**2).reshape(
                (class_num, class_num)
            )
            i = mixmat.diagonal()
            return np.nanmean((i / (mixmat.sum(0) + mixmat.sum(1) - i)))

        data_num = 0
        total_miou = 0
        for x, targets in self.dataset:
            data_num += len(x)
            quant_out: torch.Tensor = self.current_module(x)["out"].argmax(1)
            total_miou += np.sum(
                [calc_miou(target, pred) for target, pred in zip(targets, quant_out)]
            )
        avg_miou_percentage = total_miou / data_num * 100
        assert (
            avg_miou_percentage > self.threshold
        ), "MIOU need to be larger than {:.2f}%, but get {:.2f}%. ".format(
            self.threshold, avg_miou_percentage
        )
        print("Check Quantization Segmentation  Finished.")


@checker_register("wave2letter")
class W2lChecker(CheckerBase):
    necessary_params = ["dataset", "labels"]
    default_params = {"threshold": 0.7}

    def check(self):
        criterion = torch.nn.CTCLoss(blank=len(self.labels) - 1, zero_infinity=True)
        data_num = 0
        lm = GreedyLM(self.labels)
        c_ldist_sum, c_ref_len_sum = 0, 0
        w_ldist_sum = 0
        test_loss_sum = 0
        for x, (targets, input_lens, output_lens) in self.dataset:
            data_num += len(x)
            quant_out: torch.Tensor = self.current_module(x)
            quant_out = quant_out.view((1, 29, quant_out.numel() // 29))
            loss = criterion(
                quant_out.permute(2, 0, 1), targets, input_lens, output_lens
            )
            test_loss_sum += loss.item()
            decoded_preds = lm.decode_ctc(quant_out)
            decoded_targets = lm.decode_ids(targets)
            decoded_targets = [t[:len] for t, len in zip(decoded_targets, output_lens)]

            for hypo, ref in zip(decoded_preds, decoded_targets):
                c_ldist_sum += torchaudio.functional.edit_distance(ref, hypo)
                c_ref_len_sum += len(ref)
                hypo_words = "".join(hypo).split()
                ref_words = "".join(ref).split()
                w_ldist_sum += torchaudio.functional.edit_distance(
                    ref_words, hypo_words
                )
        test_loss = test_loss_sum / len(self.dataset)
        assert (
            test_loss < self.threshold
        ), "CTC need to be smaller than {:.2f}%, but get {:.2f}%. ".format(
            self.threshold, test_loss
        )
        return self
