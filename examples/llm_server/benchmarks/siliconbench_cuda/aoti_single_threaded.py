"""Validate immutable packages through ET's official graph execution C API.

AOTInductorModelContainerRunSingleThreaded avoids the generic Python loader's
model-completion bookkeeping. Inputs/outputs use PyTorch's provided handle
conversion helpers. Generated code, cubins and quantized weights are unchanged.
"""
import ctypes
import hashlib
from pathlib import Path
import zipfile
import torch

INVOCATION_POLICY = 'official AOTInductorModelContainerRunSingleThreaded with explicit current CUDA stream'
_get_name = ctypes.pythonapi.PyCapsule_GetName
_get_name.argtypes = (ctypes.py_object,)
_get_name.restype = ctypes.c_char_p
_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
_get_pointer.restype = ctypes.c_void_p
_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
_new_capsule.restype = ctypes.py_object


class SingleThreadedPackage:
    def __init__(self, package, directory):
        self.container = ctypes.c_void_p()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.members = {}
        with zipfile.ZipFile(package) as archive:
            members = [m for m in archive.infolist() if m.filename.endswith(('.wrapper.so', '.cubin'))]
            assert sum(m.filename.endswith('.wrapper.so') for m in members) == 1
            assert any(m.filename.endswith('.cubin') for m in members)
            for member in members:
                name = Path(member.filename).name
                assert name not in self.members
                raw = archive.read(member)
                (self.directory/name).write_bytes(raw)
                self.members[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        library = next(self.directory.glob('*.wrapper.so'))
        self.library = ctypes.CDLL(str(library))
        self.create = self._function('AOTInductorModelContainerCreateWithDevice',
                                    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                     ctypes.c_char_p, ctypes.c_char_p])
        self.delete = self._function('AOTInductorModelContainerDelete', [ctypes.c_void_p])
        query_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.get_inputs = self._function('AOTInductorModelContainerGetNumInputs', query_types)
        self.get_outputs = self._function('AOTInductorModelContainerGetNumOutputs', query_types)
        self.get_constants = self._function('AOTInductorModelContainerGetNumConstants', query_types)
        self.run = self._function('AOTInductorModelContainerRunSingleThreaded',
                                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                  ctypes.c_void_p, ctypes.c_void_p])
        self._check(self.create(ctypes.byref(self.container), 1, b'cuda:0',
                                str(self.directory).encode()), 'create')
        self.ninputs = self._query(self.get_inputs)
        self.noutputs = self._query(self.get_outputs)
        assert self.ninputs == 6 and self.noutputs == 1
        assert self._query(self.get_constants) == 0, 'This validation loader only accepts explicit-weight GEMM packages'

    def _function(self, name, types):
        f = getattr(self.library, name)
        f.argtypes = types
        f.restype = ctypes.c_int
        return f

    @staticmethod
    def _check(error, action):
        if error:
            raise RuntimeError('Official AOTI C API '+action+' failed: '+str(error))

    def _query(self, fn):
        n = ctypes.c_size_t()
        self._check(fn(self.container, ctypes.byref(n)), 'metadata query')
        return n.value

    def __call__(self, *inputs):
        assert self.container.value and len(inputs) == self.ninputs
        stream = torch.cuda.current_stream()
        assert stream.cuda_stream != 0
        # The C API steals each new input Tensor handle; Python input tensors
        # retain their original ownership. PyTorch's capsules borrow handles.
        capsules = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(inputs)
        pointers = [_get_pointer(c, _get_name(c)) for c in capsules]
        assert all(pointers)
        input_handles = (ctypes.c_void_p*self.ninputs)(*pointers)
        output_handles = (ctypes.c_void_p*self.noutputs)()
        error = self.run(self.container, input_handles, self.ninputs,
                         output_handles, self.noutputs, stream._as_parameter_, None)
        outputs = [_new_capsule(p, None, None) for p in output_handles if p]
        tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(outputs)
        self._check(error, 'RunSingleThreaded')
        assert len(tensors) == self.noutputs
        return tensors

    def close(self):
        if self.container.value:
            self._check(self.delete(self.container), 'delete')
            self.container = ctypes.c_void_p()
