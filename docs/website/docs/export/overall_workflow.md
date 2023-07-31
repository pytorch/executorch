# Overall Workflow

## Step 0: Preparation
To export a model, you need to ensure that:
- You have some example inputs you expect to work for your model.
- You are able to rewrite some of the model's code as necessary to successfully capture a single graph. See [Graph Breaks](./background.md/#graph-breaks). (NOTE: We do not have a story for how to deal with graph breaks in third-party libraries.)
- You know which shape dimensions (if any) of the model's inputs should be dynamic. See [Shapes](./background.md/#shapes).

## Step 1: Specification
Next, you express which dimensions (if any) you expect to be dynamic, and
(optionally) specify constraints on them to the best of your knowledge. (The
compiler will guide you on whether your constraints are sufficient, so if you do
not know anything, it's fine. However, the compiler will not infer which
dimensions should be dynamic.) These constraints encode conditions for soundness
of the exported program. See [Soundness](./soundness.md).

## Step 2: Trial and Error
You now call export on your model with your example inputs and constraints. See
[Export API Reference](./export_api_reference.md). At this point you may hit
various kinds of errors—typically, due to graph breaks or insufficient
constraints. See [Errors](./errors.md). We expect that the error messages should
be actionable, so you can learn how to fix them—usually, by looking at linked
examples in [ExportDB](./exportdb.md)—and fix them by rewriting code.

## Step 3. Inspection
At this point you will have an exported program, and you will be warned about
the assertions it makes on inputs: in particular, which dimensions are static
and have been specialized, and what conditions are expected on the remaining
dynamic dimensions. Make sure they make sense; otherwise you should debug them.

## Step 4: Testing
Finally, we would encourage you to try out other inputs whose shapes are
valid—i.e., they satisfy the assertions emitted by the compiler—yet different
from the example inputs you provided to export. (This only makes sense if you
had some dynamic dimensions.)* If all the inputs you try pass, great! Consider
your workflow complete.

* Otherwise, you have hit what is almost surely a over-specialization bug.
    Please file a bug on github with a pointer to your model, which example inputs
    and constraints you used for export, and which inputs it failed on. We will
    try to unblock you as best as possible.
