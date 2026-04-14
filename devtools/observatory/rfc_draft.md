# RFC for Observatory and FX Viewer


## Introduction

In this RFC I want to propose 2 components in devtool that amplifies the value of each other.


1. Observatory (`devtools/observatory`) : A unified, extensible debugging utility for Executorch. This framework is called observaotry because it aims to support debugging of intermediate graphs and other runtime attributes (Currently only aot workflow is supported, runtime analysis can be supported with inspection API and  etdump).  Backend maintainers and users can contribute backend-specific debugging script and analysis logic through observatory extension, these extensions are called "Lenses". Lenses behavior can be defined by arbitrary python code and control how the debugging information will be visualized in the final HTML report. Activation and behavior of individual lensese can be futher customized and controlled by configs in observatory context.

2. FX Viewr (`devtools/fx_viewer`):  A simple HTML viewer for visualizing fx_graph supporting both Aten dialect and Edge dialect. This visualization library consists of 4k lines of vallina JS with no dependency except from standard browser API). The library support instant rendering of more than 10k nodes, support minimap natigation , smart search bar, n-graph comparison side-by-side with view syncing (using `node.meta['from_node']`). Further more, the fx_viewer is easily customizable with python API that can be leveraged in observatory's Lense API, allowing the lense debugging logic to contribute customized coloring rule, node data, and label highlights, and even cross-graph syncing rule.

To demonstrate the value these 2 utilities together, I implemented a **zero-config  auto collection workflow of per-layer accuracy analysis**, that works on all xnnpack and qualcomm aot examples.

1. Setup : Standard executorch dev environment is enough. No special dependency required.

2. Command : Simply use observaotry.cli to invoke ordinary aot script. Use `--lense_recipe=accuracy` to enable accuracy Lenses.

```bash
 python -m executorch.backends.xnnpack.debugger.observatory \
        --output-html output.html \
        --lense_recipe=accuracy \
        {original xnnpack command and args}
```
For example

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/mv2/obs_report.html \
    --lense_recipe=accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```



