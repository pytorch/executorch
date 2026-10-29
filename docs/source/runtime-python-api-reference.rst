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

To get ``ExecuTorchResult`` values even when PyTorch is installed, set ``EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT=executorch`` before importing ExecuTorch.

CUDA streams
~~~~~~~~~~~~

With ExecuTorch built with CUDA support, ``executorch.runtime.cuda`` makes the CUDA and TensorRT delegates of any number of programs run on one stream you choose, for example a CUDA green-context stream. It does not need PyTorch.

.. code-block:: python

    from executorch.runtime import cuda

    stream = cuda.Stream()
    with cuda.stream(stream):
        outputs = first.execute(inputs)
        second.execute(outputs)
    stream.close()

Inside a block, a program's copies between host and GPU memory also run on the stream, and a call waits only for that stream, so work queued on other streams keeps running. A call with GPU inputs or outputs, and ``to_host()``, still wait for the whole GPU. ``cuda.stream`` also takes a ``torch.cuda.Stream``, or any object with ``__cuda_stream__``. ``cuda.ExternalStream`` wraps a stream created elsewhere.

Limits for now: only streams on CUDA device 0 can be selected, the default stream (handle 0) cannot be selected, and a method that uses CUDA graphs fails inside a block. As in PyTorch, the selection belongs to the thread: a block must end on the thread that started it, which keeps the stream selected until then, so do not ``await`` or ``yield`` inside a block. Keep a stream alive while a block selects it, unless the block selects that very ``cuda.Stream`` object.

.. automodule:: executorch.runtime.cuda
    :members: Stream, ExternalStream, stream
