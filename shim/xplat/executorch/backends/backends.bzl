def get_all_cpu_backend_targets():
    """Returns a list of all CPU backend targets.

    For experimenting and testing, not for production, since it will typically
    include more than necessary for a particular product.
    """
    return [
        "//executorch/backends/xnnpack:xnnpack_backend",
        "//executorch/backends/fb/qnnpack:qnnpack_backend",
    ]

def get_all_cpu_aot_and_backend_targets():
    """Returns a list of all CPU backend targets with aot (ahead of time).

    For experimenting and testing, not for production, since it will typically
    include more than necessary for a particular product.
    """
    return [
        "//executorch/backends/xnnpack:xnnpack_preprocess",
        "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
        "//executorch/backends/fb/qnnpack:qnnpack_preprocess",
        "//executorch/backends/fb/qnnpack/partition:qnnpack_partitioner",
    ] + get_all_cpu_backend_targets()
