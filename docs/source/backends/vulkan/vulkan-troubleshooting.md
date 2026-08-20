# Troubleshooting

This page describes common issues that you may encounter when using the Vulkan
backend and how to debug and resolve them.

## Vulkan Backend Not Found

If you try to execute a .pte file that has been lowered to the Vulkan backend
and you see an error like:

```shell
E 00:00:00.366934 executorch:method.cpp:74] Backend VulkanBackend is not registered.
```

This error indicates the Vulkan backend is not registered with the runtime. This
can happen because the backend was not compiled or linked, or because the
registration code was optimized out.

First, make sure that when building ExecuTorch, cmake is configured with
`-DEXECUTORCH_BUILD_VULKAN=ON`.

Next, make sure that your application is linking the `vulkan_backend` target,
or the `executorch_backends` target.

Finally, ensure that `vulkan_backend` or `executorch_backends` is being linked
with the equivalent of `--whole-archive`.

## Slow Performance

Performance issues can be caused by a variety of factors:

* A key compute shader (most often convolution or linear) is not performing well
  on your target GPU
* Unsupported operators are causing too many graph breaks
* An existing operator is lacking support for some memory layout or storage type
  resulting in a high number of copies being inserted to ensure tensors are in
  a required representation for the next operator

If you experience poor on-device performance for a particular model, please
obtain some profiling data while running your model. The
[profiling tutorial](./tutorials/etvk-profiling-tutorial.md) can
be a good reference for how to do this.

Then, please file an issue on Github with the following details:

* The device(s) you have tested with, and which devices exhibit poor performance
  running the model
* The profiling data collected from executing the model
* The release version of ExecuTorch you are using, or the commit hash you built
  from if you built from source
* If available, an export script that can be used to export your model to aid
  in reproducing the issue
* If available, the `.pte` file you are testing with to aid in reproducing the
  issue.

We will do our best to patch performance problems in the Vulkan backend and
help you resolve your issue.
