# Profiling Models in ExecuTorch

Profiling in ExecuTorch gives users access to these runtime metrics:
- Model Load Time.
- Operator Level Execution Time.
- Delegate Execution Time.
  - If the delegate that the user is calling into has been integrated with the [Developer Tools](./delegate-debugging.md), then users will also be able to access delegated operator execution time.
- End-to-end Inference Execution Time.

One uniqe aspect of ExecuTorch Profiling is the ability to link every runtime executed operator back to the exact line of python code from which this operator originated. This capability enables users to easily identify hotspots in their model, source them back to the exact line of Python code, and optimize if chosen to.

We provide access to all the profiling data via the Python [Inspector API](./model-inspector.rst). The data mentioned above can be accessed through these interfaces, allowing users to perform any post-run analysis of their choice.

## Steps to Profile a Model in ExecuTorch

1. [Optional] Generate an [ETRecord](./etrecord.rst) while you're exporting your model. If provided this will enable users to link back profiling details to eager model source code (with stack traces and module hierarchy).
2.  Build the runtime with the pre-processor flags that enable profiling. Detailed in the [ETDump documentation](./etdump.md).
3. Run your Program on the ExecuTorch runtime and generate an [ETDump](./etdump.md).
4. Create an instance of the [Inspector API](./model-inspector.rst) by passing in the ETDump you have sourced from the runtime along with the optionally generated ETRecord from step 1.
    - Through the Inspector API, users can do a wide range of analysis varying from printing out performance details to doing more finer granular calculation on module level.


Please refer to the [Developer Tools tutorial](./tutorials/devtools-integration-tutorial.rst) for a step-by-step walkthrough of the above process on a sample model.
