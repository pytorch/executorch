load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    fbcode_target(_kind = runtime.python_test,
        name = "test_speech_transform",
        srcs = ["test_speech_transform.py"],
        deps = [
            "//executorch/examples/models/gemma4:speech_transform",
            "fbsource//third-party/pypi/transformers:transformers",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_export_partitioners",
        srcs = ["test_export_partitioners.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:op_registry",
            "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_webgpu_rewrite_pass",
        srcs = ["test_webgpu_rewrite_pass.py"],
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:custom_ops_lib",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_export_smoke",
        srcs = ["test_export_smoke.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:text_decoder",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_gemma4_sdpa_host_contract",
        srcs = ["test_gemma4_sdpa_host_contract.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:custom_ops_lib",
            "//executorch/examples/models/gemma4:webgpu_support",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_selected_row_cross_decoder",
        srcs = ["test_selected_row_cross_decoder.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:text_decoder",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_webgpu_artifact_manifest",
        srcs = ["test_webgpu_artifact_manifest.py"],
        typing = True,
        deps = [
            "//executorch/examples/models/gemma4:webgpu_support",
        ],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_gemma4_plain_wasm_contract",
        srcs = ["test_gemma4_plain_wasm_contract.py"],
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_generate_target_prefill_oracle",
        srcs = ["test_generate_target_prefill_oracle.py"],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:target_prefill_contract",
            "//executorch/examples/models/gemma4:target_prefill_producer",
        ],
    )
