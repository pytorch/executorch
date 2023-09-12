# Upcoming changes to export API in ExecuTorch

## Where are we today?

Exporting pytorch model for ExecuTorch runtime goes through multiple AoT (Ahead of Time) stages as shown in [here](../docs/website/docs/tutorials/exporting_to_executorch.md).
At high level there are 3 stages.
1. `exir.capture`: This captures model’s graph using ATen IR.
2. `to_edge`: translate ATen dialect into edge dialect with dtype specialization.
3. `to_executorch`: translate edge dialect to executorch dialect, along with running various passes, e.g. out variant, memory planning etc., to make model ready for executorch runtime.

Two important stops in model’s journey to executorch runtime are: a) quantization and b) delegation.

Entry points for quantization are between step 1 and 2. Thus quantization APIs consume ATen IR and are not edge/executorch specific.

Entry points for delegation are between step 2 and 3. Thus delegation APIs consume edge dialect IR.

## Need for the export API change.

Quantization workflow is built on top of exir.capture which is built on top of torch.export API. In order to support QAT, such exported models need to work with eager mode autograd. Current export, of step 1 above, emits ATen IR with core ATen ops. This is not autograd safe, meaning it is not safe to run such an exported model in eager mode (e.g. in python), and, expect the autograd engine to work. Thus training APIs, such as calculating loss on the output and calling `backward` on the loss, are not guaranteed to work with this IR.

It is important that quantization APIs, for QAT as well as PTQ,  work on the same IR, because a) it provides better UX to the users and b) it provides a single IR that backend specific quantizers (read more [here](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html?highlight=quantization) can target.

For this reason we aligned on two stage export, that is rooted in the idea of progressive lowering. The two stages are:
1. Export emits pre-dispatch ATen IR
2. Pre-dispatch ATen IR is lowered to core ATen IR.

Output of stage 1 is autograd safe and thus models exported at 1 can be trained via eager mode autograd engine.

## New export API.

We are rolling out changes related to new export API in three stages.

### Stage 1 (landed):

As shown in the figure below, exir.capture of the previous figure is broken down into:
* `capture_pre_autograd_graph
* `exir.capture`

![](./executorch_export_stack_at_stage_1_change.png)

Example of exporting model without quantization:
```
gm = export.capture_pre_autograd_graph(m)
ep = exir.capture(gm) # to be replaced with torch.export
```

Example of exporting model with quantization:
```
gm = torch.capture_pre_autograd_graph(m)
quantized_gm = calls_to_quantizaiton_api(gm)
ep = exir.capture(quantized_gm) # to be replaced with torch.export
```

You can see these changes [here](../examples/export/test/test_export.py) and [here](../examples/quantization/example.py) for how quantization APIs fit in.

### Stage 2 (coming soon):

We will deprecate exir.capture in favor of directly using torch.export. More updates on this will be posted soon.

### Stage 3 (timeline is to be determined):

The two APIs listed in stage 1 will be renamed to:
* `torch.export`
* `to_core_aten`

torch.export export graph with ATen IR, and full ATen opset, that is autograd safe, while to_core_aten will transform output of torch.export into core ATen IR that is NOT autograd safe.

Example of exporting model without quantization:
```
ep = torch.export(model)
ep = ep.to_core_aten()
```

Example of exporting model with quantization:
```
ep = torch.export(model)

gm = ep.module() # obtain fx.GraphModule. API name may change
quantized_gm = calls_to_quantizaiton_api(gm)
quantized_ep = torch.export(quantized_gm) # re-export for API compatibility

ep = quantized_ep.to_core_aten()
```

Timeline for this is to be determined, but this will NOT happen before PTC.

## Why this change?

There are a couple of reasons:
This change aligns well with the long term state where capture_pre_autograd_graph is replaced with torch.export to obtain autograd safe aten IR, and the current use of exir.capture (or torch.export when replaced) will be replaced with to_core_aten to obtain ATen IR with core ATen opset.

In the long term, export for quantization wont be separate. Quantization will be an optional step, like delegation, in the export journey. Thus aligning with that in the short terms helps because:
* it helps users with a correct mental model of how quantization fits in the export workflow, and
* export problems dont become quantization problems.

## Why the change now?
To minimize the migration pain later and have better alignment with the long term changes.

