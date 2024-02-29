import os

from torch.utils import cpp_extension

_HERE = os.path.abspath(__file__)
_EXECUTORCH_PATH = os.path.dirname(os.path.dirname(_HERE))


def load_inline(
    name,
    cpp_sources,
    functions=None,
    extra_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    is_python_module=True,
    with_pytorch_error_handling=True,
    keep_intermediates=True,
    use_pch=False,
):
    # Register the code into PyTorch
    aten_extra_cflags = ["-DUSE_ATEN_LIB"] + (extra_cflags if extra_cflags else [])
    extra_ldflags = [
        f"-L{_EXECUTORCH_PATH}",
        f"-Wl,-rpath,{_EXECUTORCH_PATH}",
        "-lexecutorch",
    ] + (extra_ldflags if extra_ldflags else [])
    module = cpp_extension.load_inline(
        name,
        cpp_sources,
        functions=functions,
        extra_cflags=aten_extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        is_python_module=is_python_module,
        with_pytorch_error_handling=with_pytorch_error_handling,
        keep_intermediates=keep_intermediates,
        use_pch=use_pch,
    )
    # Now register the code into ExecuTorch
    cpp_extension.load_inline(
        name,
        cpp_sources,
        functions=None,  # leave this out since we are not passing out any python module
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        is_python_module=False,  # don't register as a python module. Load shared library as a side effect.
        with_pytorch_error_handling=with_pytorch_error_handling,
        keep_intermediates=keep_intermediates,
        use_pch=use_pch,
    )
    return module
