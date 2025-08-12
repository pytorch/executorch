load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
  """Defines targets that should be shared between fbcode and xplat.

  The directory containing this targets.bzl file should also contain both
  TARGETS and BUCK files that call this function.
  """

  runtime.python_test(
      name = "test_gen",
      srcs = glob(["test_*.py"]),
      package_style = "inplace",
      deps = [
          "//executorch/codegen:gen_lib",
          "fbsource//third-party/pypi/expecttest:expecttest",
      ],
      external_deps = [
          "torchgen",
      ],
  )
