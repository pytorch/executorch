# Partitioner API

The XNNPACK partitioner API allows for configuration of the model delegation to XNNPACK. Passing an `XnnpackPartitioner` instance with no additional parameters will run as much of the model as possible on the XNNPACK backend. This is the most common use-case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/xnnpack_partitioner.py#L31):

 - `configs`: Control which operators are delegated to XNNPACK. By default, all available operators all delegated. See [../config/\_\_init\_\_.py](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/config/__init__.py#L66) for an exhaustive list of available operator configs.
 - `config_precisions`: Filter operators by data type. By default, delegate all precisions. One or more of `ConfigPrecisionType.FP32`, `ConfigPrecisionType.STATIC_QUANT`, or `ConfigPrecisionType.DYNAMIC_QUANT`. See [ConfigPrecisionType](https://github.com/pytorch/executorch/blob/release/0.6/backends/xnnpack/partition/config/xnnpack_config.py#L24).
 - `per_op_mode`: If true, emit individual delegate calls for every operator. This is an advanced option intended to reduce memory overhead in some contexts at the cost of a small amount of runtime overhead. Defaults to false.
 - `verbose`: If true, print additional information during lowering.
