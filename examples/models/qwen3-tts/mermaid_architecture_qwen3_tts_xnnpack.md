# Qwen3-TTS XNNPACK Pipeline Architecture

Copy the code below and paste into:
- **VS Code**: Paste in any `.md` file, press `Ctrl+Shift+V` to preview
- **Mermaid Playground**: https://mermaid.live
- **GitHub**: Renders natively in `.md` files

## End-to-End Pipeline

```mermaid
flowchart TD
    subgraph Export["Export (Python)"]
        direction TB
        PyModel["Qwen3-TTS PyTorch Model"] --> ExpScript["export_unified.py"]
        ExpScript --> PTE["model.pte (2.3 GB, 8da4w)"]
    end

    subgraph Runner["C++ Runner (main_unified.cpp)"]
        direction TB
        CLI["CLI Input\n--text / --prompts_path"] --> Session["SynthesisSession\nper-session RNG + config"]
        Session --> Pipeline
    end

    subgraph Pipeline["Synthesis Pipeline (XNNPACK, CPU-only)"]
        direction TB
        Tokenize["Tokenize Text\n(HuggingFace JSON tokenizer)"] --> EncText["encode_text\ntoken_ids → embeddings ∈ ℝ¹ˣˢˣ¹⁰²⁴"]
        EncText --> Prefill["talker prefill\ntext embeddings → hidden + logits"]
        Prefill --> Loop

        subgraph Loop["⚠️ BOTTLENECK: Autoregressive Codec Loop\n~130-150ms per step on CPU\n~20s total for 11s of audio"]
            direction TB
            SampleCode0["Sample code_0\n(top-k, rep. penalty, EOS check)"]
            SampleCode0 --> Decision{{"use fused\nfast path?"}}

            Decision -- "Yes\n(contract v2, top_k=50)" --> Fused["cp_generate (fused)\n1 XNNPACK call → 15 sub-codes\ninverse-CDF top-k(50) sampling"]
            Decision -- "No\n(fallback)" --> Legacy

            subgraph Legacy["Legacy Host Loop"]
                direction TB
                CP["code_predictor"] --> CPH["cp_head"]
                CPH --> CE["codec_embed"]
                CE --> CP
            end

            Fused --> EmbSum["embedding sum → next talker input"]
            Legacy --> EmbSum
            EmbSum --> Talker["🔴 talker decode step\ndense matmul on CPU\nhidden + logits for next code_0"]
            Talker --> SampleCode0
        end

        Loop -- "EOS or limit" --> Decode["decode_audio\naudio codes → waveform (24 kHz)"]
    end

    Decode --> WAV["Output .wav file"]

    PTE -.-> Runner
    Pipeline -.-> Timing["SynthesisTiming\nprep | prefill | codegen | decode"]
```

### Current Performance

> | Metric | Legacy (host loop) | Fused cp_generate v2 |
> |--------|-------------------|----------------------|
> | Generation time | 23.9s | 19.6s |
> | Codegen | 21.1s | 17.0s |
> | Per-step cost | ~150ms | ~125ms |
> | Audio output | 11.5s | 10.8s |
>
> Fused `cp_generate` v2 collapsed 15 host round-trips into 1 graph call, achieving ~15-20% codegen speedup.

## Fused cp_generate v2 Detail

```mermaid
flowchart LR
    subgraph Inputs["Host → Graph"]
        H["talker_hidden"] 
        E0["code_0_embed"]
        T["temperature"]
        U["sample_uniforms\n(15 uniform randoms)"]
    end

    subgraph FusedGraph["cp_generate XNNPACK Graph"]
        direction TB
        Pre["CP prefill\n(hidden + code_0)"] --> G1

        subgraph G1["Group 1..15 Loop (unrolled)"]
            direction TB
            Head["LM Head\nhidden → logits"] --> TopK["top-k(50)"]
            TopK --> Softmax["softmax / temperature"]
            Softmax --> CDF["cumsum → CDF"]
            CDF --> Sample["inverse-CDF sample\nusing uniform random"]
            Sample --> Embed["codec embed lookup"]
            Embed --> CP_Step["CP transformer step"]
            CP_Step --> Head
        end
    end

    subgraph Outputs["Graph → Host"]
        Codes["sampled_subcodes\nint64 × 15"]
        ESum["embed_sum\nfloat × 1024"]
    end

    Inputs --> FusedGraph --> Outputs
```

## Warm Benchmark Session Flow

```mermaid
sequenceDiagram
    participant CLI as main_unified
    participant Runner as Qwen3TTSUnifiedRunner
    participant Session as SynthesisSession
    participant PTE as model.pte (XNNPACK)

    CLI->>Runner: construct(model_path, tokenizer_path)
    CLI->>Runner: warmup_all()
    Runner->>PTE: load + execute all 7 methods once

    loop For each prompt × repeat
        CLI->>Runner: create_synthesis_session(config)
        Runner-->>Session: new session (fresh RNG)
        CLI->>Session: synthesize(text, language)
        Session->>PTE: encode_text → talker → cp_generate loop → decode_audio
        Session-->>CLI: waveform + SynthesisTiming
        CLI->>CLI: trim silence, write WAV, log timing
    end
```

## Summary

These diagrams show the Qwen3-TTS XNNPACK pipeline at three levels:

1. **End-to-end pipeline**: text input → tokenization → 7-method model execution → WAV output, with the fused/legacy fast-path decision gate
2. **Fused cp_generate v2**: the internal XNNPACK graph that collapses 15 host round-trips into one call using inverse-CDF sampling
3. **Warm benchmark session**: how `SynthesisSession` keeps the runner warm across sequential prompts for honest latency measurement
