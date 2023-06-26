# I/O APIs for Execution Plan

Here we present several unified APIs for execution plan I/O with different memory planning schemas.

```c++
Error set_input(const EValue& input_evalue, size_t input_idx);
```
This function sets the `input_idx`-th input of the execution plan to be `input_evalue`.

`input_idx` should be smaller than the number of plan's input. The data type of `input_evalue` should be the same as the `input_idx`-th input. If it is a tensor, dynamic shape is supported and its dtype should follow the execution plan.

Return `Error::Ok` if input setting completed successfully, otherwise return error occurs during execution.

Caution: the `execution_plan` may or may not have a buffer for the input data based on its memory plan. Users should double check the memory and take care of data lifecycle if needed.

```c++
Error set_inputs(const exec_aten::ArrayRef<EValue>& input_evalues);
```

This function sets the input of the execution plan to be `input_evalues`.

Expect the data types of EValue elements in the `input_evalues` to be aligned with the plan’s input, and the length of `input_evalues` should be the same as the number of the plan’s input.

Return `Error::Ok` if input setting completed successfully, otherwise return the error occurs during execution. Other things worth to be considered are same as above.

```c++
Error get_outputs(EValue* output_evalues, size_t length);
```

This function retrieves the plan’s output and write into the given EValue list `output_evalues` with given `length`.

Expect the data types of EValue elements in the `output_evalues` to be aligned with the plan’s input, and the `length` should be larger than or equal to the number of plan’s output.

Return `Error::Ok` if input setting completed successfully, otherwise return the error occurs during execution.

Caution: This function exposes the data pointer of inner output tensors. Uses should be careful when using it and not mutate the data in place. We are working on updating the implementation to remove such constraints.

More details can be found on [T132716305](https://www.internalfb.com/tasks/?t=132716305).
