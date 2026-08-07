load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    # `mtp_export_lib` is D8-owned, so every target naming it needs D8 landed first.
    fbcode_target(_kind = runtime.python_test,
        name = "test_eagle_combined_round",
        srcs = ["test_eagle_combined_round.py"],
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:op_registry",
            "//executorch/examples/models/gemma4:mtp_export_lib",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",
        ],
    )

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
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/vulkan:op_registry",
            "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
            "//executorch/examples/models/gemma4:mtp_export_lib",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",
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

    fbcode_target(_kind = runtime.python_test,
        name = "test_export_assistant_webgpu_artifacts",
        srcs = ["test_export_assistant_webgpu_artifacts.py"],
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:mtp_export_lib",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
            "//executorch/extension/pybindings:portable_lib",
        ],
    )

    # The oracle generator ships in srcs so the suite carries its own reference.
    fbcode_target(_kind = runtime.python_test,
        name = "test_mtp_spec_oracle",
        srcs = [
            "generate_mtp_spec_oracle.py",
            "test_mtp_spec_oracle.py",
        ],
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
            "//executorch/kernels/quantized:aot_lib",
            "//pytorch/ao/torchao/csrc/cpu/shared_kernels/embedding_xbit:op_embedding_xbit_aten",
            "//pytorch/ao/torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight:op_linear_8bit_act_xbit_weight_aten",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:mtp_export_lib",
            "//executorch/examples/models/gemma4:target_prefill_producer",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
        ],
        typing = True,
    )

    fbcode_target(_kind = runtime.python_binary,
        name = "generate_mtp_spec_oracle",
        srcs = ["generate_mtp_spec_oracle.py"],
        main_function = "executorch.examples.models.gemma4.tests.generate_mtp_spec_oracle.main",
        preload_deps = [
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_py",
            "//executorch/kernels/quantized:aot_lib",
            "//pytorch/ao/torchao/csrc/cpu/shared_kernels/embedding_xbit:op_embedding_xbit_aten",
            "//pytorch/ao/torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight:op_linear_8bit_act_xbit_weight_aten",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/examples/models/gemma4:mtp_export_lib",
            "//executorch/examples/models/gemma4:target_prefill_producer",
            "//executorch/examples/models/gemma4:webgpu_support",
            "//executorch/exir:lib",
        ],
        typing = True,
    )

    fbcode_target(_kind = runtime.python_test,
        name = "test_webgpu_spec_contract",
        srcs = ["test_webgpu_spec_contract.py"],
        deps = [
            "//executorch/examples/models/gemma4:webgpu_support",
        ],
        typing = True,
    )

    # The OSS closure gate scans the whole checkout; the native CI script runs it.
    fbcode_target(_kind = runtime.python_library,
        name = "test_oss_source_closure",
        srcs = ["test_oss_source_closure.py"],
        typing = True,
    )

    fbcode_target(_kind = runtime.cxx_test,
        name = "test_gemma4_spec_runner_contract",
        srcs = ["test_gemma4_spec_runner_contract.cpp"],
        deps = [
            "//executorch/examples/models/gemma4:gemma4_spec_runner",
        ],
    )
