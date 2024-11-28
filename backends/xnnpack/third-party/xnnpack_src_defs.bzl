load(
    "//backends/xnnpack/third-party/XNNPACK:build_srcs.bzl",
    _LOGGING_SRCS = "LOGGING_SRCS",
    _OPERATOR_SRCS = "OPERATOR_SRCS",
    _SUBGRAPH_SRCS = "SUBGRAPH_SRCS",
    _TABLE_SRCS = "TABLE_SRCS",
    _XNNPACK_SRCS = "XNNPACK_SRCS",
)
load("//backends/xnnpack/third-party/XNNPACK/gen:microkernels.bzl", "prod_srcs_for_arch")
load("//third-party:glob_defs.bzl", "subdir_glob")

def define_xnnpack_build_src(xnnpack_build_src):
    return ["XNNPACK/{}".format(src) for src in xnnpack_build_src]

def prod_srcs_for_arch_wrapper(arch):
    prod_srcs = prod_srcs_for_arch(arch)
    return define_xnnpack_build_src(prod_srcs)

def get_xnnpack_headers():
    # XNNPACK Headers in the path containing xnnpack/ or configs/
    # do not contain the src/ path. However headers not in xnnpack/ or
    # configs/ are prepend with the src/ path. This function helps us
    # to correctly parse all the header files to the correct name
    src_headers = subdir_glob([
        ("XNNPACK/src", "**/*.h"),
    ])
    fixed_headers = {}
    for k, v in src_headers.items():
        new_key = k
        if not k.startswith("xnnpack") and not k.startswith("configs"):
            new_key = "src/{}".format(k)
        fixed_headers[new_key] = v
    include_headers = subdir_glob([
        ("XNNPACK/include", "*.h"),
    ])

    return fixed_headers | include_headers

OPERATOR_SRCS = define_xnnpack_build_src(_OPERATOR_SRCS)
SUBGRAPH_SRCS = define_xnnpack_build_src(_SUBGRAPH_SRCS)
TABLE_SRCS = define_xnnpack_build_src(_TABLE_SRCS)
XNNPACK_SRCS = define_xnnpack_build_src(_XNNPACK_SRCS)
LOGGING_SRCS = define_xnnpack_build_src(_LOGGING_SRCS)
