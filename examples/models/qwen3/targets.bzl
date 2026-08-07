load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.python_library(
        name = "qwen3",
        srcs = [
            "__init__.py",
            "convert_weights.py",
        ],
        _is_external_target = True,
        base_module = "executorch.examples.models.qwen3",
        resources = {
            "config/0_6b_config.json": "config/0_6b_config.json",
            "config/1_7b_config.json": "config/1_7b_config.json",
            "config/4b_config.json": "config/4b_config.json",
            "config/qwen3_webgpu_q4gsw.yaml": "config/qwen3_webgpu_q4gsw.yaml",
            "config/qwen3_xnnpack_q8da4w.yaml": "config/qwen3_xnnpack_q8da4w.yaml",
        },
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models:model_base",
            "//executorch/examples/models/llama:llama2_model",
            "fbcode//pytorch/torchtune:lib",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.python_library(
        name = "webgpu_artifact_manifest",
        srcs = ["webgpu_artifact_manifest.py"],
        _is_external_target = True,
        base_module = "executorch.examples.models.qwen3",
        resources = {
            "manifests/qwen3_0_6b_webgpu.json": "manifests/qwen3_0_6b_webgpu.json",
        },
        typing = True,
        deps = [
            "//executorch/backends/webgpu/scripts:webgpu_artifact_manifest",
        ],
        visibility = ["PUBLIC"],
    )
