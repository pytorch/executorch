PT2.0 Export Manual

# Context
At a high level, the goal of PT2 Export is to enable the execution of entire PyTorch programs by other means than the “eager” PyTorch runtime, with a representation that is amenable to meet the optimization and targeting goals of specialized use cases. Specifically, we want to enable users to convert their PyTorch models to a standardized IR, decoupled from its execution, that various domain-specific runtimes can transform and execute independently. This conversion is powered by Dynamo’s technique for sound whole-graph capture—capturing a graph without any “breaks” that would require the eager runtime to fall back to Python. The rest of this wiki documents a snapshot of PT2 Export as of early May 2023—what we consider an "MVP" release. Please note that this project is under active and heavy development: while this snapshot should give a fairly accurate picture of the final state, some some details might change in the coming weeks / months based on feedback. If you have any issues, please file an issue on Github and tag "export".


Modules and Entrypoints
Constraints API
Control Flow Operators
Custom Operators
Compiler Passes on Exported Artifact
Non-strict Mode
# Documentation
- [Overview](./overview.md)
  - [Background](./background.md)
  - [Overall Workflow](./overall_workflow.md)
  - [Soundness](./soundness.md)
  - [Errors](./errors.md)
- [Export API Reference](./export_api_reference.md)
  - [Modules and Entrypoints](./modules_and_entrypoints.md)
  - [Constraints API](./constraint_apis.md)
  - [Control Flow Operators](../ir_spec/control_flow.md)
  - [Custom Operators](./custom_operators.md)
- [Exported Programs](../ir_spec/00_exir.md#exportedprogram)
- [ExportDB](./exportdb.md)

ir_spec/control_flow
