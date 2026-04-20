# LLMs Quantization Guidance

## Mixed-Precision Quantization with SQNR Analysis

When deploying LLMs at low precision (for example, `16a4w_block`), some layers can accumulate significantly higher quantization error and become accuracy bottlenecks.
mix_precision_analyzer.py is an analysis tool that helps you identify these quantization-sensitive layers and provides a directional starting point for mixed-precision tuning. It lets you selectively upgrade only the most quantization-sensitive layers to higher precision, while keeping the rest of the model at the aggressive baseline (e.g. 16a4w_block).
The tool does not aim to find a globally optimal quantization recipe, but it helps narrow the search space so you can iterate from a directional starting point rather than guessing.

### Overview

`mix_precision_analyzer.py` provides two classes and one module-level function:

- **`PerLayerSqnrAnalyzer`** — takes the FP32 `GraphModule` (before `prepare_pt2e`), the fake quant `GraphModule` (after `convert_pt2e`), and optionally the `QuantRecipe` used to produce the fake quant model. Runs both graphs on the same calibration inputs and computes per-conv2d layer SQNR by comparing intermediate outputs. Results are grouped by module path and bucketed across layer ranges.

- **`SqnrReport`** — holds the grouped SQNR results and exposes three methods:
  - `save_analysis_summary()` — writes a CSV with per-group statistics (columns: `group_name, avg_sqnr, median_sqnr, min_sqnr, max_sqnr, count`).
  - `suggest_recipe_overrides(sqnr_threshold=10.0, default_precision=use_16a4w_block, higher_precision=use_16a8w)` — flags groups whose avg, median, or min SQNR falls below `sqnr_threshold` as sensitive layer groups to builds a `QuantRecipe` that: carries over all non-sensitive strategies from the analysis recipe, and set sensitive groups to `higher_precision` per-channel. Returns a list of `QuantRecipe` objects (one per block size), or an empty list if no sensitive groups are found.

- **`save_suggest_recipes(report, suggest_recipe, output_dir=None)`** — renders the override recipes into a ready-to-use `.py` file. Each strategy is annotated with `[Original recipe]` (carried over from the analysis recipe unchanged) or `[Added by SqnrAnalyzer]` (automatically added by the analyzer).


### Step-by-Step Workflow

**1. Initial Quantization Configuration**

When starting quantization from scratch, the recommended first step is to apply an aggressive baseline precision to the model's layers. You can configure this in `examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py`.

Set your target `conv2d` layers to use **LPBQ (`16a4w_block`) with a block size of 64**.

For example, your base recipe might look like this:
```python
class Qwen3_1_7BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a4w_block,
            is_qat=False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_BLOCK,
            extra_kwargs={"block_size": (1, 64, 1, 1)},
        )
      )
```

**2. Run SQNR Evaluation**

Once your baseline recipe is set, run the main script (`llama.py`) with the `--quant_recipe_suggestion` flag. The SQNR analyzer runs automatically during calibration and writes the following files to the working directory:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
    ... \
    --quant_recipe_suggestion
```

Output files:

- `{model_name}_quantization_error.csv` — per-group SQNR statistics sorted by sensitivity (most sensitive first)
- `{model_name}_suggest_recipe.py` — ready-to-use `StaticLLMQuantRecipe` subclasses optimized to apply higher-precision quantization to the most sensitive groups.


**3. Analyze Sensitive Layers**

The analyzer automatically flags layer groups where SQNR falls below `sqnr_threshold` (default: `10.0` dB). A lower SQNR means higher quantization error and greater sensitivity.

The generated CSV is sorted by median SQNR ascending, placing the most problematic groups at the top. For example, based on the Qwen3-1.7B model:

- **`feed_forward.w2_conv`** (down-projection), **`feed_forward.w3_conv`**, and **`attention.wv_conv`** layers are consistently the most sensitive, with SQNR values below 10 dB.

*Note: `sqnr_threshold` can be adjusted via `suggest_recipe_overrides(sqnr_threshold=...)`.*

**4. Generated Recipe**

The generated `{model_name}_suggest_recipe.py` contains one class per block size candidate, e.g.:

```
QWEN3_1_7B_BlockSize16QuantRecipe
QWEN3_1_7B_BlockSize32QuantRecipe
QWEN3_1_7B_BlockSize64QuantRecipe
```

Each class extends `StaticLLMQuantRecipe` and builds a `QuantRecipe` with two types of strategies, annotated inline:

- `# [Original recipe]` — strategy was already present in the analysis-time recipe and is carried over unchanged.
- `# [Added by SqnrAnalyzer]` — strategy is new or has different precision/granularity compared to the original; added because the layer group was flagged as sensitive.


**5. Apply the Suggested Recipe**

The analyzer logs the generated classes and writes `{model_name}_suggest_recipe.py`:

```
[SqnrAnalyzer] Recipe file written to: {model_name}_suggest_recipe.py
[SqnrAnalyzer] Generated classes:
  - QWEN3_1_7B_BlockSize16QuantRecipe
  - QWEN3_1_7B_BlockSize32QuantRecipe
  - QWEN3_1_7B_BlockSize64QuantRecipe
[SqnrAnalyzer] Replace the original recipe class in your export script with one of the above and re-run calibration to evaluate accuracy.
```

Copy the classes above into `static_llm_quant_recipe.py` or replace the original recipe import in `__init__.py`:

```python
# Before (in examples/qualcomm/oss_scripts/llama/__init__.py):
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import Qwen3_1_7BQuantRecipe

# After (example with block size 64):
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import QWEN3_1_7B_BlockSize64QuantRecipe as Qwen3_1_7BQuantRecipe
```

**Iterative tuning tips:**

- Start with `BlockSize64` as a balanced starting point.
- If accuracy is still insufficient, try `BlockSize32` then `BlockSize16`.
- You can also try `annotate_kv_8bit` as a combination to balance accuracy and performance.
- Consider enabling additional PTQ (Post-Training Quantizatio) techniques in `__init__.py`, such as `seq_mse` or `r3`, to further improve baseline accuracy.
- Once satisfied, copy the final recipe into `static_llm_quant_recipe.py` as the permanent recipe for the model.

> **Note:** The primary purpose of this SQNR analysis is just to provide a guiding direction for mixed-precision quantization. While it identifies which layers are most sensitive and suggests a reasonable mixed-precision combination, it does not guarantee the best possible combination for every model. Please note that not every model will truly benefit from this mixed precision analysis, as the overall effectiveness of PTQ (Post-Training Quantization) can still be limited. The generated recipe classes are a starting point for exploration, not a final answer. You may need to experiment with different block sizes, different threshold values, or manually crafting overrides for specific layer groups to find the optimal accuracy/performance trade-off for your target model. If your requirement is to push the performance to the extreme limits, please try QAT (Quantization-Aware Training) instead.
