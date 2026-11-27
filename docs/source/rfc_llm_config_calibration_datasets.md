# RFC: Hugging Face calibration dataset in `LlmConfig`

**Status:** RFC

**Author:** Mahesh Madhavan

**Last Update:** 2026-10-01

## Summary

Add `quantization.calibration_dataset` to `LlmConfig` for Hugging Face datasets or
local files. A shared loader returns text/chat rows with optional image or audio
values.

## Motivation

`QuantizationConfig` currently exposes a single prompt and `lm_eval` task fields:

```python
calibration_tasks: Optional[List[str]] = None
calibration_limit: Optional[int] = None
calibration_seq_length: Optional[int] = None
calibration_data: str = "Once upon a time"
```

These fields cannot describe an HF dataset or local dataset files.

## Scope

- Add a YAML- and OmegaConf-compatible dataset source using Huggingface `datasets.load_dataset()`.
- Describe text/chat, image+text, and audio+text inputs through named columns.
- Preserve existing prompt and `lm_eval` fields and behavior when the new field
  is absent.

The multimodal source contract and examples are included here.

## Proposed configuration

Add a YAML- and OmegaConf-compatible `HFDatasetConfig` to
[llm_config.py](../../extension/llm/export/config/llm_config.py), exposed as
`llm_config.quantization.calibration_dataset`.

```diff
-from typing import ClassVar, List, Optional
+from typing import Any, ClassVar, Dict, List, Optional

+class DatasetInputFormat(str, Enum):
+    text = "text"
+    messages = "messages"
+
+
+@dataclass
+class HFDatasetConfig:
+    # Forwarded to datasets.load_dataset().
+    path: str
+    name: Optional[str] = None
+    data_dir: Optional[str] = None
+    data_files: Any = None
+    split: str = "train"
+    streaming: bool = False
+    load_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
+
+    # Applied after loading.
+    limit: Optional[int] = 1
+    input_format: DatasetInputFormat = DatasetInputFormat.text
+    input_column: str = "text"
+    image_column: Optional[str] = None
+    audio_column: Optional[str] = None

 @dataclass
 class QuantizationConfig:
     calibration_tasks: Optional[List[str]] = None
     calibration_limit: Optional[int] = None
     calibration_seq_length: Optional[int] = None
     calibration_data: str = "Once upon a time"
+    calibration_dataset: Optional[HFDatasetConfig] = None
```

### Field summary

| Field | Purpose |
|---|---|
| `path` | HF dataset ID or local builder such as `json` or `parquet`. |
| `name` | Optional dataset subset or builder configuration. |
| `data_dir` / `data_files` | Local or builder-specific data locations accepted by Hugging Face. |
| `split` | One dataset split to load. |
| `streaming` | Request a Hugging Face `IterableDataset`. |
| `load_dataset_kwargs` | Additional YAML-compatible arguments to `load_dataset()`. |
| `limit` | Maximum raw rows selected; defaults to 1. |
| `input_format` | Whether the selected input is plain text or messages. |
| `input_column` | Column containing the text or messages. |
| `image_column` / `audio_column` | Optional column containing the media associated with the same row. |

All values must be YAML-compatible. A supplied `limit` must be a positive integer.
`limit: null` selects the full split for a non-streaming dataset; streaming requires
a positive limit. Skipped rows count toward the limit. The limit does
not count token sequences or batches.

## Examples

### Plain text

```yaml
quantization:
  calibration_dataset:
    path: Salesforce/wikitext
    name: wikitext-2-raw-v1
    split: train
    streaming: true
    limit: 100
    input_format: text
    input_column: text
```

Example row returned by Hugging Face:

```json
{"text": "Eight FAB subtypes were proposed in 1976 ."}
```

For local files, use `path: json` or `path: parquet` with `data_files`. For chat,
use `input_format: messages` and select the column containing the conversations.

### Image and text

```yaml
quantization:
  calibration_dataset:
    path: json
    data_files: /datasets/calibration.jsonl
    split: train
    limit: 100
    input_format: text
    input_column: prompt
    image_column: image
```

Example JSONL row:

```json
{"prompt": "What is shown?", "image": "/datasets/images/example.jpg"}
```

### Audio and messages from Parquet

```yaml
quantization:
  calibration_dataset:
    path: parquet
    data_files: /datasets/audio_calibration.parquet
    split: train
    limit: 100
    input_format: messages
    input_column: messages
    audio_column: audio
```

Example record stored in Parquet (shown as JSON for readability):

```json
{
  "messages": [
    {"role": "user", "content": "Transcribe this recording."},
    {"role": "assistant", "content": "The train leaves at six."}
  ],
  "audio": "/datasets/audio/example.wav"
}
```

This example associates one recording with the user turn. The model must
support that association, insert any required audio markers, apply chat template,
and process the recording using the model's audio requirements. The shared loader
passes the messages and audio value together without adding model-specific tokens.

The Parquet file stores the nested messages and an audio file reference directly.

### Audio and transcription directly from Hugging Face

```yaml
quantization:
  calibration_dataset:
    path: google/fleurs
    name: en_us
    split: train
    streaming: true
    limit: 100
    input_format: text
    input_column: transcription
    audio_column: audio
```

FLEURS supplies `audio` and `transcription`, without a `messages` column.
Before preprocessing, a decoded row looks like this.

```python
raw_row = {
    "transcription": "The train leaves at six.",
    "audio": {
        "path": ".../example.wav",
        "array": waveform,  # Decoded audio samples.
        "sampling_rate": 16000,
    },
}
```

An offline conversion can add messages while retaining the audio:

```python
chat_row = {
    "messages": [
        {"role": "user", "content": "Transcribe this recording."},
        {"role": "assistant", "content": raw_row["transcription"]},
    ],
    "audio": raw_row["audio"],  # Retain the audio with its sampling rate.
}
```

For the converted dataset, select `input_format: messages` and
`input_column: messages`. Audio can remain decoded or be saved as a WAV file at
its sampling rate, with the file path stored in `audio`.

## Input contract

The shared loader returns a native HF `Dataset` or `IterableDataset`, preserving
original column names:

```python
def load_hf_dataset(config: HFDatasetConfig) -> Dataset | IterableDataset:
    ...
```

Model integrations consume the rows as `Iterable[Mapping[str, Any]]`, with the
source configuration supplied separately:

- For `text`, `input_column` selects a string.
- For `messages`, pass the value unchanged to the selected model's chat-template
  adapter, which validates its supported message format. Shared code does not infer
  or rename roles.
- Media columns select values from the same row. Preserve the representation
  returned by HF.

The shared loader performs no additional media normalization or model preprocessing.

## Compatibility

- Without `calibration_dataset`, preserve existing prompt and `lm_eval` behavior.
- With it, use the dataset as the sole source; reject other explicitly supplied
  calibration sources.
- HF `limit` controls rows independently of `lm_eval` task limits.

## Implementation direction

1. Add the config and source-selection validation.
2. Add a shared LLM loader that lazily imports `datasets`, loads one split, and
   applies the row limit.
3. Perform any optional dataset normalization offline.

## Decision requested

1. Add `quantization.calibration_dataset: Optional[HFDatasetConfig]` without
   changing existing calibration fields.
2. Use native HF datasets and dict rows as the shared source interface.
