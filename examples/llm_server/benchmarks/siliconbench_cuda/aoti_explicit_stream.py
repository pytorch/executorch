"""Call the official AOTI package API on the caller's CUDA stream.

The package's Python __call__ omits stream_handle. Pass the current non-default
stream explicitly when validating the same package under CUDA graph capture.
This only adapts the test invocation; generated kernels and packages are intact.
"""
import ctypes
import torch

_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
_new_capsule.restype = ctypes.py_object
INVOCATION_POLICY = 'official package loader.run with explicit current CUDA stream capsule'


def run_aoti_on_current_stream(model, inputs):
    stream = torch.cuda.current_stream()
    if stream.cuda_stream == 0:
        raise RuntimeError('Validation requires a non-default CUDA stream')
    # The capsule borrows the CUDA stream; torch owns its lifetime.
    capsule = _new_capsule(stream._as_parameter_, None, None)
    return model.loader.run(list(inputs), capsule)
