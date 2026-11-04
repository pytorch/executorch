# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import tempfile
from multiprocessing.connection import Listener
from pathlib import Path

import pytest

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    add_default_cmds,
    get_ipc,
    require_paths,
)
from executorch.backends.qualcomm.tests.rework.src.e2e_metrics import (
    assert_metric,
    assert_metrics,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require(request, artifact_names: list[str] = None):
    """Skip if any artifact is None; fail loudly if the path doesn't exist."""
    # executorch_root + artifact_dir are always required for e2e tests."""
    qnn_config = request.getfixturevalue("qnn_config")
    options = {
        n: request.config.getoption(n)
        for n in ["executorch_root", "artifact_dir"]
        + (artifact_names if artifact_names else [])
    }
    # extend this if the option needs no path validation
    keys_to_remove = ["model_name"]
    require_paths({k: v for k, v in options.items() if k not in keys_to_remove})
    return *options.values(), qnn_config


def _run(cmds, qnn_config):
    ip, port = get_ipc(qnn_config)
    p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
    with Listener((ip, port)) as listener:
        conn = listener.accept()
        p.communicate()
        return json.loads(conn.recv())


def _check(msg):
    if "Error" in msg:
        pytest.fail(msg["Error"])
    return msg


# ---------------------------------------------------------------------------
# TestExampleScript (examples/qualcomm/scripts/)
# ---------------------------------------------------------------------------


class MobileNetV2:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/mobilenet_v2.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("mobilenet_v2", msg, qnn_config)


class MobileNetV3:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/mobilenet_v3.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("mobilenet_v3", msg, qnn_config)


class InceptionV3:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/inception_v3.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("inception_v3", msg, qnn_config)


class InceptionV4:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/inception_v4.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("inception_v4", msg, qnn_config)


class Vit:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/torchvision_vit.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("vit", msg, qnn_config)


class Edsr:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/edsr.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--default_dataset",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("edsr", msg, qnn_config)


class DeepLabV3:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/deeplab_v3.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("deeplab_v3", msg, qnn_config)


class MobileBert:
    @staticmethod
    def test(request):
        root, artifact, pretrained_weight, qnn_config = _require(
            request, ["pretrained_weight"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/mobilebert_fine_tune.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--pretrained_weight",
            pretrained_weight,
            "--use_fp16",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        cpu, htp = msg["CPU"], msg["HTP"]
        for k, v in cpu.items():
            assert_metric(
                "mobilebert", "cpu_htp_delta", abs(v[0] - htp[k][0]), qnn_config
            )


class PtqMobileBert:
    @staticmethod
    def test(request):
        root, artifact, pretrained_weight, qnn_config = _require(
            request, ["pretrained_weight"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/mobilebert_fine_tune.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--pretrained_weight",
            pretrained_weight,
            "--ptq",
            "16a16w",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        cpu, htp = msg["CPU"], msg["HTP"]
        for k, v in cpu.items():
            assert_metric(
                "ptq_mobilebert", "cpu_htp_delta", abs(v[0] - htp[k][0]), qnn_config
            )


class Wav2Letter:
    @staticmethod
    def test(request):
        root, artifact, pretrained_weight, qnn_config = _require(
            request, ["pretrained_weight"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/scripts/wav2letter.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--pretrained_weight",
            pretrained_weight,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("wav2letter", msg, qnn_config)


class ExportExample:
    @staticmethod
    def test(request):
        root, artifact, model_name, qnn_config = _require(request, ["model_name"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmds = [
                "python",
                "qualcomm/scripts/export_example.py",
                "--model_name",
                model_name,
                "--output_folder",
                f"{tmp_dir}/",
                "--generate_etrecord",
            ]
            p = subprocess.Popen(
                cmds,
                stdout=subprocess.DEVNULL,
                cwd=f"{root}/examples",
            )
            p.communicate()
            assert Path(f"{tmp_dir}/{model_name}.pte").exists()


# ---------------------------------------------------------------------------
# TestExampleOssScript (examples/qualcomm/oss_scripts/)
# ---------------------------------------------------------------------------


class Albert:
    @staticmethod
    def test(request):
        root, artifact, sentence_dataset, qnn_config = _require(
            request, ["sentence_dataset"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/albert.py",
            "--dataset",
            sentence_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("albert", msg, qnn_config)


class Bert:
    @staticmethod
    def test(request):
        root, artifact, sentence_dataset, qnn_config = _require(
            request, ["sentence_dataset"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/bert.py",
            "--dataset",
            sentence_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("bert", msg, qnn_config)


class ConvFormer:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/conv_former.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("conv_former", msg, qnn_config)


class ConvNextSmall:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/convnext_small.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("convnext_small", msg, qnn_config)


class Cvt:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/cvt.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("cvt", msg, qnn_config)


class Deit:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/deit.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("deit", msg, qnn_config)


class DepthAnythingV2Small:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/depthanything_v2_small.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("depthanything_v2_small", msg, qnn_config)


class DinoV2:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/dino_v2.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("dino_v2", msg, qnn_config)


class Distilbert:
    @staticmethod
    def test(request):
        root, artifact, sentence_dataset, qnn_config = _require(
            request, ["sentence_dataset"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/distilbert.py",
            "--dataset",
            sentence_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("distilbert", msg, qnn_config)


class Dit:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/dit.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("dit", msg, qnn_config)


class EfficientNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/efficientnet.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("efficientnet", msg, qnn_config)


class EfficientSAM:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, pretrained_weight, oss_repo, qnn_config = (
            _require(request, ["image_dataset", "pretrained_weight", "oss_repo"])
        )
        if qnn_config.backend == QnnExecuTorchBackendType.kHtpBackend:
            pytest.skip("Bad accuracy, need investigation")
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/efficientSAM/efficientSAM.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--oss_repo",
            oss_repo,
            "--pretrained_weight",
            pretrained_weight,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("efficient_sam", msg, qnn_config)


class Esrgan:
    @staticmethod
    def test(request):
        root, artifact, oss_repo, qnn_config = _require(request, ["oss_repo"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/esrgan.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--default_dataset",
            "--oss_repo",
            oss_repo,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("esrgan", msg, qnn_config)


class Eurobert:
    @staticmethod
    def test(request):
        root, artifact, sentence_dataset, qnn_config = _require(
            request, ["sentence_dataset"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/eurobert.py",
            "--dataset",
            sentence_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("eurobert", msg, qnn_config)


class FastVit:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, pretrained_weight, oss_repo, qnn_config = (
            _require(request, ["image_dataset", "pretrained_weight", "oss_repo"])
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/fastvit.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--oss_repo",
            oss_repo,
            "--pretrained_weight",
            pretrained_weight,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("fastvit", msg, qnn_config)


class FbNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/fbnet.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("fbnet", msg, qnn_config)


class FocalNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/focalnet.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("focalnet", msg, qnn_config)


class GMLP:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/gMLP_image_classification.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("gmlp", msg, qnn_config)


class MaxVitT:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/maxvit_t.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("maxvit_t", msg, qnn_config)


class MobileViTV2:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/mobilevit_v2.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("mobilevit_v2", msg, qnn_config)


class MobileViTV1:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/mobilevit_v1.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("mobilevit_v1", msg, qnn_config)


class Pvt:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/pvt.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("pvt", msg, qnn_config)


class RegNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/regnet.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        for weight in ["regnet_y_400mf", "regnet_x_400mf"]:
            msg = _check(_run(cmds + ["--weights", weight], qnn_config))
            assert_metrics("regnet", msg, qnn_config)


class RetinaNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/retinanet.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--dataset",
            image_dataset,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("retinanet", msg, qnn_config)


class Roberta:
    @staticmethod
    def test(request):
        root, artifact, sentence_dataset, qnn_config = _require(
            request, ["sentence_dataset"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/roberta.py",
            "--dataset",
            sentence_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("roberta", msg, qnn_config)


class SqueezeNet:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/squeezenet.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("squeezenet", msg, qnn_config)


class Ssd300Vgg16:
    @staticmethod
    def test(request):
        root, artifact, pretrained_weight, oss_repo, qnn_config = _require(
            request, ["pretrained_weight", "oss_repo"]
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/ssd300_vgg16.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--oss_repo",
            oss_repo,
            "--pretrained_weight",
            pretrained_weight,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("ssd300_vgg16", msg, qnn_config)


class SwinTransformer:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/swin_transformer.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("swin_transformer", msg, qnn_config)


class SwinV2T:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/swin_v2_t.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("swin_v2_t", msg, qnn_config)


class T5:
    @staticmethod
    def test(request):
        root, artifact, qa_dataset, qnn_config = _require(request, ["qa_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/t5/t5.py",
            "--dataset",
            qa_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("t5", msg, qnn_config)


class VitB16:
    @staticmethod
    def test(request):
        root, artifact, image_dataset, qnn_config = _require(request, ["image_dataset"])
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/vit_b_16.py",
            "--dataset",
            image_dataset,
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("vit_b_16", msg, qnn_config)


class Whisper:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/whisper/whisper.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        assert_metrics("whisper", msg, qnn_config)
