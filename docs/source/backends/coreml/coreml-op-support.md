# Op support

The Core ML backend supports almost all PyTorch operators.

If an operator in your model is not supported by Core ML, you will see a warning about this during lowering.  If you want to guarantee that your model fully delegates to Core ML, you can set [`lower_full_graph=True`](coreml-partitioner.md) in the `CoreMLPartitioner`. When set, lowering will fail if an unsupported operator is encountered.
