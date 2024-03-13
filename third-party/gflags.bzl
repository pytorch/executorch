# Copied from third-party/gflags/BUCK
load(":gflags_defs.bzl", "gflags_library", "gflags_sources")

def define_gflags():
    (exported_headers, headers, srcs) = gflags_sources(namespace = [
        "gflags",
        "google",
    ])

    gflags_library(
        name = "gflags",
        srcs = srcs,
        headers = headers,
        exported_headers = exported_headers,
        enable_static_variant = True,
        threads = True,
    )

    gflags_library(
        name = "gflags_nothreads",
        srcs = srcs,
        headers = headers,
        exported_headers = exported_headers,
        enable_static_variant = True,
        threads = False,
    )
