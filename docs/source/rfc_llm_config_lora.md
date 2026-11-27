# RFC: LoRA methods and calibration in `LlmConfig`

**Status:** RFC; implementation pending.

**Author:** Mahesh Madhavan

**Last Update:** 2026-10-01

**Prerequisite:** [Hugging Face calibration dataset support](rfc_llm_config_calibration_datasets.md).

## Summary

Let users choose LoRA adapters, quantization settings, and calibration data for
each exported method through `LlmConfig`.

A method is a named entry point for calling the exported model. It can use the
base model alone or include a LoRA adapter: a small set of trained weights that
changes the model's behavior.

## What changes

Today, `MethodConfig.lora_config` selects an adapter. This proposal adds:

- `lora_configs`: a list of adapters for each method, with optional names.
- `quantization.lora_quantize`: optional quantization and dataset settings for a
  method, selected by `method_name`.
- `HFDatasetConfig`: the same dataset config used by the calibration dataset RFC.
- `backend.qnn.lora`: QNN settings for passing adapter inputs and keeping
  quantization values fixed.

V1 allows several methods, with one lora adapter, and supports only
`freeze_all` quantization_strategy. More adapters per method and other strategies are future work.

## Proposed configuration

Changes to `extension/llm/export/config/llm_config.py`, after the prerequisite RFC:

```diff
 @dataclass
 class LoraConfig:
     # Existing fields remain unchanged.
+    name: Optional[str] = None

 @dataclass
 class MethodConfig:
     method_name: str
     lora_config: Optional[LoraConfig] = None
     export_seq_len: Optional[int] = None
+    lora_configs: List[LoraConfig] = field(default_factory=list)

+@dataclass
+class LoraQuantizationConfig:
+    method_name: str
+    lora_pt2e_quantize: Optional[Pt2eQuantize] = None
+    calibration_dataset: Optional[HFDatasetConfig] = None

 @dataclass
 class QuantizationConfig:
     # Includes calibration_dataset from the prerequisite RFC.
+    lora_quantize: List[LoraQuantizationConfig] = field(default_factory=list)

+class QNNLoraQuantizationStrategy(str, Enum):
+    freeze_all = "freeze_all"

+@dataclass
+class QNNLoraConfig:
+    weight_as_input: bool = True
+    scale_as_input: bool = True
+    quantization_strategy: QNNLoraQuantizationStrategy = (
+        QNNLoraQuantizationStrategy.freeze_all
+    )

 @dataclass
 class QNNConfig:
     # Existing fields remain unchanged.
+    lora: QNNLoraConfig = field(default_factory=QNNLoraConfig)
```

- Add `name` after the existing fields so existing calls to `LoraConfig(...)`
  still work.
- Accept the old `lora_config` field for compatibility; use `lora_configs` in new
  configs. Reject a method that sets both an adapter and a nonempty adapter list.
- If neither field supplies an adapter, the method uses only the base model.

An adapter's `adapter_checkpoint` points to its trained weights.
`adapter_config` points to a file describing its rank, scaling, and target layers.

## Choosing settings for each method

| Setting | Behavior |
|---|---|
| `LoraConfig.name` | Optional adapter name. If omitted, use `method_name` for the method's single adapter. |
| `multimethod.methods[].lora_configs` | LoRA weights and config for this method. |
| `lora_quantize[].method_name` | Names the existing method whose settings to change. That method must have an adapter. |
| `lora_pt2e_quantize` | Quantization settings for the adapter. If omitted or `None`, use base model settings. |
| `calibration_dataset` | Dataset for this method. If omitted or `None`, use base model `quantization.calibration_dataset`. |

A method without a `lora_quantize` entry uses the shared settings.

## QNN settings and `freeze_all`

These settings live under `backend.qnn.lora`:

| Setting | Proposed meaning |
|---|---|
| `weight_as_input` | If true, pass adapter weights when calling the model. If false, include them in the exported method. |
| `scale_as_input` | Pass the adapter's scaling factor as an input. This controls how much the adapter contributes to the result. |
| `quantization_strategy` | `freeze_all` keeps both base and adapter quantization values fixed during export. |



## Example

This example uses the datasets to prepare quantization values before `freeze_all`
export. It requires exporter support; model and export-shape settings are omitted.

```yaml
backend:
  qnn:
    enabled: true
    lora:
      weight_as_input: true
      scale_as_input: true
      quantization_strategy: freeze_all

quantization:
  pt2e_quantize: qnn_16a4w
  calibration_dataset:
    path: Salesforce/wikitext
    name: wikitext-2-raw-v1
    split: train
    limit: 100
    input_format: text
    input_column: text

  lora_quantize:
    - method_name: function_usecase
      lora_pt2e_quantize: qnn_16a16w
      calibration_dataset:
        path: json
        data_files: /datasets/function/calibration.jsonl
        split: train
        limit: 100
        input_format: messages
        input_column: messages
    - method_name: elementary_usecase
      lora_pt2e_quantize: qnn_16a16w
      # Inherits base calibration_dataset.

multimethod:
  methods:
    - method_name: base_usecase
    - method_name: function_usecase
      lora_configs:
        - name: function
          adapter_checkpoint: /adapters/function/adapter_model.safetensors
          adapter_config: /adapters/function/adapter_config.json
    - method_name: elementary_usecase
      lora_configs:
        - name: elementary
          adapter_checkpoint: /adapters/elementary/adapter_model.safetensors
          adapter_config: /adapters/elementary/adapter_config.json
```

| Method | Adapter | Calibration dataset |
|---|---|---|
| `base_usecase` | None | Shared WikiText dataset |
| `function_usecase` | `function` | Method's chat dataset in JSONL |
| `elementary_usecase` | `elementary` | Shared WikiText dataset |

Using the same dataset does not mean sharing collected calibration statistics or
combining adapter weights. Prepare values for each method's model and adapter.

## Checks and compatibility

- Method names must be unique. Allow at most one `lora_quantize` entry per method;
  reject entries for missing methods or methods without adapters.
- Reject multiple adapters per method and strategies other than `freeze_all` in
  v1. 
- Keep existing Python/YAML configs working. Reject combining
  `base.lora_config` with `multimethod.methods`.
- Keep `adapter_quant` behavior; reject conflicts with shared or per-method
  quantization settings.


## Future work

### Additional quantization strategies

- `freeze_base`: keep base quantization values fixed and recalibrate the adapter.
- `freeze_none`: recalibrate both base and adapter quantization values. 


### Multiple adapters in one method

The same list can describe multiple adapters in one method. Supporting their
combined execution is future work:

```yaml
multimethod:
  methods:
    - method_name: combined_usecase
      lora_configs:
        - name: function
          adapter_checkpoint: /adapters/function/adapter_model.safetensors
          adapter_config: /adapters/function/adapter_config.json
        - name: elementary
          adapter_checkpoint: /adapters/elementary/adapter_model.safetensors
          adapter_config: /adapters/elementary/adapter_config.json
```

- Both adapters would run in one method, each with a unique name.
- Settings for `combined_usecase` would select one LoRA quantization setting and
  one dataset for the whole method. 

## Decision requested

1. Use `HFDatasetConfig` for each method's calibration dataset.
2. Add adapter lists and optional names while keeping the old `lora_config` field for now.
3. Add `backend.qnn.lora` with only `freeze_all` in v1, once the exporter can
   prepare or load the required quantization values.
