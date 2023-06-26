# Backend Dialect


## Properties

1. All operators in OpCall nodes are now backend specific ops. Backend specific ops are set of operators the that user specifies, and the target backend will need to implement for the final program to run in that backend
2. Memory formats of the tensor can be any format.

In other words, to convert Edge dialect to backend, a pass (provided by us) will use the information of which backend + edge ops present in the graph to replace them with equivalent backend operators (or subgraph). One of such pass is “quant fusion” This pass will take canonical quantization pattern, ie. “dequant - some_op - quant” into a single operator that is backend specific, i.e. quantized::some_op.


## Intent

This dialect allows introduction of operators that do not conform to the schema defined in the canonical ATen operator set. Consider to use backend operators if your use case satisfies one or more of the following criteria:

1. Your backend provides a library that optimizes a certain operator that is equivalent to a subgraph. E.g., linear_relu (equivalent to linear + relu) that can be executed faster on a certain backend.
2. There's a need to retrace the graph module after it is already lowered to a backend. When we retrace, backend operators can transform back to the original subgraph (in ATen dialect) where normal custom op doesn't take care of that.
3. Your backend specific operator doesn't have a generic CPU kernel but only a kernel for a certain backend. Using backend operator can workaround this issue by using the original subgraph as default kernel and keep the graph module runnable.

## API

We provide a decorator `bind_pattern_to_op` to allow user quickly register their backend operator. This decorator takes:
* a `torch.Library` object, it indicates which library or namespace this backend operator belongs to.
* a name or schema. If we already defined the schema of the backend operator in the `torch.Library` object, only a name is needed. Otherwise we can register the schema if a schema string is being passed in.

## Operator Set

* `executorch_prims::add.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.add
  * backend: executor
* `executorch_prims::mul.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.mul
  * backend: executor
* `executorch_prims::sub.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.sub
  * backend: executor
* `executorch_prims::floordiv.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.floordiv
  * backend: executor
* `executorch_prims::gt.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.gt
  * backend: executor
* `executorch_prims::lt.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.lt
  * backend: executor
* `executorch_prims::ge.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.ge
  * backend: executor
* `executorch_prims::le.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.le
  * backend: executor
* `executorch_prims::eq.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.eq
  * backend: executor
* `quantized_decomposed::embedding_byte(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor`
  * pattern: [source](https://fburl.com/code/05odcjq8)
  * backend: quantization
* `quantized_decomposed::add(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc`
  * pattern: [source](https://fburl.com/code/sjx01piz)
  * backend: quantization
* `quantized_decomposed::add.scalar(Tensor qa, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, ScalarType a_dtype, Scalar b, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max, ScalarType out_dtype) -> Tensor`
  * pattern: [source](https://fburl.com/code/sjx01piz)
  * backend: quantization
* `quantized_decomposed::add_relu(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc`
  * pattern: [source](https://fburl.com/code/sjx01piz)
  * backend: quantization
