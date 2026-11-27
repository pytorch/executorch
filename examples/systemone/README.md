# System One: native structured decision inference

System One models make fast, structured decisions — a Choice among named
options, a Noul (yes/no), or a Score over ordered levels — from a shared text
snapshot, returning probabilities and a confidence value instead of generated
text. This directory provides a model-agnostic C++ interface for embedding
them, following
[TypeSafe's SDK operation](https://docs.typesafe.ai/sdk/python/api/clients/sync)
and [typed SDK](https://docs.typesafe.ai/sdk/python/api/types/questions).

[api.h](api.h) defines the question and answer types (`Choice`, `Noul`,
`Score`, `ChoiceAnswer`, `NoulAnswer`, `ScoreAnswer`) and the virtual
`SystemOne::system_one(state, questions)` interface in the
`executorch::systemone` namespace. It is shared across models; it has no
dependency on any one model's tokenizer, export format, or runtime.

Model-specific implementations live in their own subdirectory, each
borrowing or embedding an ExecuTorch `Module` and providing a concrete
`SystemOne` subclass:

- [kev/](kev/) — [Kev](https://huggingface.co/jaredpalmer/kev-0.8b), a System
  One model, using XNNPACK on CPU or MLX on Apple GPUs.

See a model's subdirectory for export, build, and benchmarking instructions.
