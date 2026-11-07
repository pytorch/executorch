Runtime Python API Reference
----------------------------------
The Python ``executorch.runtime`` module wraps the C++ ExecuTorch runtime. It can load and execute serialized ``.pte`` program files: see the `Export to ExecuTorch Tutorial <tutorials/export-to-executorch-tutorial.html>`__ for how to convert a PyTorch ``nn.Module`` to an ExecuTorch ``.pte`` program file. Execution accepts and returns ``torch.Tensor`` values, making it a quick way to validate the correctness of the program. To run a program without PyTorch installed, see `Without PyTorch`_ below.

For detailed information on how APIs evolve and the deprecation process, please refer to the `ExecuTorch API Life Cycle and Deprecation Policy <api-life-cycle.html>`__.

.. automodule:: executorch.runtime
.. autoclass:: Runtime
    :members: get, load_program

.. autoclass:: BackendRegistry
    :members: registered_backend_names, is_available, set_option, get_option

.. autoclass:: OperatorRegistry
    :members: operator_names

.. autoclass:: Program
    :members: method_names, load_method

.. autoclass:: Method
    :members: execute, set_output, metadata

.. autoclass:: ExecuTorchResult
    :members: shape, strides, dtype, device, nbytes, to_host, from_host

Without PyTorch
~~~~~~~~~~~~~~~

The runtime also works when PyTorch is not installed. Then ``Method.execute`` takes NumPy arrays as inputs and returns ``ExecuTorchResult`` values, which NumPy can read. For a model that adds its two inputs:

.. code-block:: python

    import numpy as np
    from executorch.runtime import Runtime

    method = Runtime.get().load_program("model.pte").load_method("forward")
    out = method.execute([np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)])[0]
    print(out.dtype)          # float32
    result = np.asarray(out)  # a NumPy view of the output, no copy
    print(result.tolist())    # [[2.0, 2.0], [2.0, 2.0]]

Other buffers, such as ``memoryview``, work as inputs too. NumPy has no bfloat16 type, so it reads a bfloat16 result as 16-bit integers that hold the raw bits.

For data on a GPU, use DLPack, the standard way to share tensors between Python libraries. This needs ExecuTorch built with CUDA support. Pass a CuPy array to a method that takes CUDA inputs. Read a CUDA result with ``cupy.from_dlpack(out)``, or copy it to the CPU with ``out.to_host()``. ``ExecuTorchResult.from_host()`` copies CPU data to a CUDA device.

Output types
~~~~~~~~~~~~

Importing ``executorch.runtime`` does not import PyTorch, unless the setting below asks for ``torch``. When PyTorch is installed, ``Method.execute`` returns ``torch.Tensor`` outputs, and the first call that returns one imports ``torch``. Without PyTorch, or when importing it raises an ``ImportError``, it returns ``ExecuTorchResult`` outputs. Any other error from importing PyTorch is raised by every call that returns a tensor; set the setting below to ``executorch`` to run without PyTorch.

A future release will make the outputs follow the tensor inputs of each call. A call with any ``torch.Tensor`` input will return ``torch.Tensor`` outputs. A call whose tensor inputs are all NumPy arrays, other buffers, DLPack objects, or ``ExecuTorchResult`` values will return ``ExecuTorchResult`` outputs, even when PyTorch is installed. A call with no tensor inputs keeps returning ``torch.Tensor`` outputs whenever ``torch`` can be imported. Each call whose outputs will change warns with a ``FutureWarning``.

To choose the output type now, set ``EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT`` before importing ExecuTorch. With ``executorch``, every call returns ``ExecuTorchResult`` outputs and ExecuTorch never imports PyTorch. With ``torch``, every call returns ``torch.Tensor`` outputs, and importing ExecuTorch imports PyTorch.

.. code-block:: python

    import os

    os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
    from executorch.runtime import Runtime

CUDA streams
~~~~~~~~~~~~

With ExecuTorch built with CUDA support, ``executorch.runtime.cuda`` makes the CUDA and TensorRT delegates of any number of programs run on one stream you choose, for example a CUDA green-context stream. It does not need PyTorch. The module documentation below gives an example and the rules.

.. automodule:: executorch.runtime.cuda
    :members: Stream, ExternalStream, stream
