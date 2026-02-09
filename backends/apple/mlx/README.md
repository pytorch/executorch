# MLX Delegate for ExecuTorch

The MLX delegate compiles PyTorch models to run on Apple Silicon GPUs via the
[MLX](https://github.com/ml-explore/mlx) framework. It consists of:

- A **Python compilation pipeline** that converts ExportedPrograms (Edge IR) into
  a custom FlatBuffer bytecode format.
- A **C++ runtime** that loads the bytecode and executes it using MLX GPU
  primitives.

> **Adding a new op?** Jump to [How to Add a New Op](#how-to-add-a-new-op).

## Directory Layout

```
backends/apple/mlx/
├── serialization/              # Schema + code generation
│   ├── schema.fbs              # ← Source of truth (FlatBuffer schema)
│   ├── generate.py             # Code generator (schema.fbs → everything else)
│   ├── mlx_graph_schema.py     # [GENERATED] Python dataclasses for IR nodes
│   ├── mlx_graph_serialize.py  # Serialization to FlatBuffer binary
│   ├── _generated_serializers.py # [GENERATED] Per-op FlatBuffer builders
│   └── _generated/             # [GENERATED] FlatBuffer Python bindings (flatc)
├── runtime/                    # C++ runtime (loaded at inference time)
│   ├── MLXBackend.cpp          # BackendInterface (init / execute / destroy)
│   ├── MLXLoader.h/.cpp        # [GENERATED] FlatBuffer → C++ structs
│   ├── MLXExecutor.h           # ExecutionState, constant loading, helpers
│   ├── MLXInterpreter.h        # Op dispatch loop + per-op exec_* functions
│   └── schema_generated.h      # [GENERATED] FlatBuffer C++ bindings (flatc)
├── ops.py                      # Op handlers  (ATen target → MLX IR node)
├── patterns.py                 # Pattern handlers (multi-node fusions)
├── program_builder.py          # MLXProgramBuilder + REGISTRY
├── partitioner.py              # Decides which ops to delegate to MLX
├── preprocess.py               # BackendDetails.preprocess() entry point
├── custom_ops.py               # Custom torch ops (rope, etc.)
├── test/
│   ├── test_ops.py             # Op test definitions (models + configs)
│   ├── test_utils.py           # OpTestCase base class + helpers
│   ├── op_test_runner.cpp      # C++ test runner (loads .pte, runs, compares)
│   └── run_all_tests.py        # End-to-end: export → C++ run → compare
└── examples/
    ├── llama/                  # LLaMA export + run
    └── whisper/                # Whisper export + run
```

Files marked **[GENERATED]** are produced by running:

```bash
python backends/apple/mlx/serialization/generate.py
```

---

## Compilation Pipeline

The compilation pipeline converts a PyTorch model into a `.pte` file containing
the MLX delegate payload. The high-level flow:

```
torch.export()           →  ExportedProgram (ATen IR)
to_edge_transform_and_lower()  →  Edge IR + partitioning + lowering
```

Within that flow, the MLX-specific steps are:

1. **Partitioning** (`partitioner.py`) — `MLXPartitioner` walks the Edge IR
   graph and tags nodes that MLX can handle. It uses `MLXProgramBuilder` in a
   dry-run mode to determine support — so partitioning and compilation use the
   exact same logic. Unsupported ops fall back to ExecuTorch's portable
   runtime.

2. **Preprocessing** (`preprocess.py`) — For each partitioned subgraph,
   `MLXBackend.preprocess()` is called. It builds an `MLXGraph` via
   `MLXProgramBuilder`, serializes it to FlatBuffer, and returns a
   `PreprocessResult` with the binary payload and constant data.

3. **Op handling** (`ops.py`, `patterns.py`) — During the build,
   `MLXProgramBuilder` walks the FX graph node-by-node and dispatches to
   registered handlers. Single-op handlers live in `ops.py`; multi-node fused
   patterns (e.g., quantized linear, SDPA, KV cache update) live in
   `patterns.py`.

4. **Serialization** (`serialization/`) — The `MLXGraph` dataclass tree is
   serialized to a FlatBuffer binary. See [Serialization](#serialization) below.

The complete preprocessing flow:

```
ExportedProgram (subgraph)
  → MLXProgramBuilder.build()      # walks FX graph, calls op handlers
  → MLXGraph                       # Python IR (dataclasses from mlx_graph_schema.py)
  → MLXGraphSerializer.serialize() # FlatBuffer binary
  → PreprocessResult               # returned to ExecuTorch
```

---

## How to Add a New Op

This section walks through adding a new op end-to-end, using **`aten.linear`**
as an example.

### Step 1: Add the Node to `schema.fbs`

Add a new table in the "Op nodes" section and add it to the `OpNode` union:

```fbs
table LinearNode {
    x: Tid (required);
    weight: Tid (required);
    out: Tid (required);
    bias: Tid;  // optional
}
```

Then add `LinearNode` to the `union OpNode { ... }` list.

### Step 2: Run the Code Generator

```bash
python backends/apple/mlx/serialization/generate.py
```

This regenerates:

- `mlx_graph_schema.py` — adds `LinearNode` Python dataclass
- `_generated_serializers.py` — adds `_build_LinearNode` serializer
- `runtime/MLXLoader.h` — adds `LinearNode` C++ struct, `OpCode::LINEAR`, loader
- `runtime/MLXLoader.cpp` — adds FlatBuffer → `LinearNode` deserialization
- `runtime/schema_generated.h` — FlatBuffer C++ bindings

### Step 3: Add the Python Op Handler (`ops.py`)

Register a handler that converts the ATen op to your new node. Make sure to
import `LinearNode` from `mlx_graph_schema`:

```python
from executorch.backends.apple.mlx.serialization.mlx_graph_schema import LinearNode

@REGISTRY.register(target=[torch.ops.aten.linear.default])
def _linear_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 3, "aten.linear")
    require_kwargs(P.kwargs(n), set(), "aten.linear")
    x, w = args[0], args[1]
    b = args[2] if len(args) > 2 else None
    out = P.make_or_get_slot(n)
    P.emit(
        LinearNode(
            x=P.slot_to_tid(x),
            weight=P.slot_to_tid(w),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(b) if b else None,
        )
    )
    return out
```

Key APIs:
- **`P.args(n)`** — resolves FX node args to `Slot` objects (tensor/value references)
- **`P.make_or_get_slot(n)`** — allocates the output tensor slot
- **`P.slot_to_tid(slot)`** — converts a `Slot` to a `Tid` for the IR node
- **`P.emit(node)`** — appends the instruction to the graph

### Step 4: Add the C++ Op Handler (`MLXInterpreter.h`)

Add an `exec_*` function in the `ops` namespace:

```cpp
inline void exec_linear(const LinearNode& n, ExecutionState& st, StreamOrDevice s) {
    const auto& X = st.const_tensor_ref(n.x);
    auto W = transpose(st.const_tensor_ref(n.weight), {1, 0}, s);
    array Y = n.bias
        ? addmm(st.const_tensor_ref(*n.bias), X, W, 1.0f, 1.0f, s)
        : matmul(X, W, s);
    st.set_tensor(n.out, std::move(Y));
}
```

Then add the dispatch case in `Interpreter::execute_instruction()`:

```cpp
case OpCode::LINEAR:
    ops::exec_linear(std::get<LinearNode>(instr.node), st, s);
    break;
```

### Step 5: Write a Test (`test/test_ops.py`)

Each test follows a standard pattern:

1. **Define a `nn.Module`** that uses the op.
2. **Define an `OpTestCase` subclass** that specifies test configurations.
3. **Decorate with `@register_test`** to register it with the test runner.

```python
class LinearModel(nn.Module):
    def __init__(self, in_features=64, out_features=128, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

@register_test
class LinearTest(OpTestCase):
    name = "linear"
    rtol = 1e-4
    atol = 1e-4

    def __init__(self, in_features=64, out_features=128, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    @classmethod
    def get_test_configs(cls):
        return [cls(), cls(bias=False)]

    def create_model(self):
        return LinearModel(self.in_features, self.out_features, bias=self.bias)

    def create_inputs(self):
        return (torch.randn(2, 16, self.in_features),)
```

### Step 6: Run Tests

Tests are end-to-end: export `.pte` → run via C++ `op_test_runner` → compare
outputs against PyTorch reference. Since adding a new op always involves C++
changes, use `--rebuild` to recompile the runtime:

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests --rebuild linear
```

Run all tests in parallel:

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests --rebuild -j4 --clean-after
```

Other useful flags:

| Flag | Purpose |
|---|---|
| `--rebuild` | Rebuild the C++ `op_test_runner` before running |
| `-j N` / `--parallel N` | Run N tests in parallel |
| `--clean-after` | Remove generated test artifacts after running |
| `--list` | List all available test names and exit |
| `-v` / `--verbose` | Verbose output |

Test artifacts are saved to `test/op_tests/<test_name>/` (`.pte`, input/output
`.bin` files). See [`test/README.md`](test/README.md) for full details on test
architecture, prerequisites, and the `OpTestCase` API.

### Checklist

- [ ] Add `*Node` table to `schema.fbs` + add to `OpNode` union
- [ ] Run `python backends/apple/mlx/serialization/generate.py`
- [ ] Add `@REGISTRY.register` handler in `ops.py` (and import the new node class)
- [ ] Add `exec_*` function in `runtime/MLXInterpreter.h`
- [ ] Add `case OpCode::*` in `Interpreter::execute_instruction()`
- [ ] Add test model + `OpTestCase` in `test/test_ops.py`
- [ ] Run `python -m executorch.backends.apple.mlx.test.run_all_tests --rebuild <test_name>`

---

## Serialization

### Overview

The serialization system converts a Python `MLXGraph` dataclass tree into a
FlatBuffer binary that the C++ runtime can load. The source of truth is
**`schema.fbs`** — a single FlatBuffer schema file from which all code on both
sides is generated.

### Schema (`schema.fbs`)

The schema defines:

| Concept | FlatBuffer type | Purpose |
|---|---|---|
| **`Tid`** | struct | Tensor slot index (indexes into the runtime tensor array) |
| **`Vid`** | struct | Value slot index (for scalar `int32`/`float`/`bool` values) |
| **`IntOrVid`** | table | A field that is either a literal `int64` or a runtime `Vid` reference (for dynamic shapes) |
| **`FloatOrVid`** | table | Same idea for floats |
| **`TidOrVid`** | table | Either a tensor or a scalar value |
| **Op node tables** | table | One per op (e.g. `AddNode`, `SiluNode`, `ReshapeNode`). Each declares its inputs/outputs as `Tid`/`Vid` references and any scalar parameters. |
| **`OpNode`** | union | Union of all op node tables |
| **`Instruction`** | table | Wraps an `OpNode` union |
| **`MLXGraph`** | table (root) | The complete program: slot counts, instruction list, I/O maps, named slots, tensor metadata |

Key design points:

- **No embedded weights.** Constants are stored in ExecuTorch's `named_data_map`
  and loaded by name at runtime. This enables zero-copy on unified memory.
- **Tensor IDs (`Tid`) are globally ordered:** Constants → Inputs → Outputs →
  Mutable Buffers → Temps. The runtime uses this ordering for O(1) type lookup.
- **Dynamic shapes** are supported via `IntOrVid` — a shape dimension can be
  either a literal integer or a reference to a runtime value produced by
  `sym_size` / `item()` ops.

### Code Generation (`generate.py`)

`generate.py` parses `schema.fbs` and generates **all** boilerplate on both the
Python and C++ sides:

| Generated file | What it contains |
|---|---|
| `mlx_graph_schema.py` | Python `@dataclass` for every op node, `Tid`, `Vid`, `IntOrVid`, etc. |
| `_generated_serializers.py` | `GeneratedOpBuilders` mixin class with `_build_*Node` methods for every op |
| `_generated_inspector.py` | Inspector utilities for debugging `.pte` files |
| `runtime/MLXLoader.h` | C++ structs for every op node, `OpCode` enum, `NodeVariant`, `Instruction`, `MLXProgram` |
| `runtime/MLXLoader.cpp` | `load_instruction()` and `load_program()` — FlatBuffer → C++ struct conversion |
| `runtime/schema_generated.h` | Standard FlatBuffer C++ bindings (via `flatc`) |
| `_generated/` directory | Standard FlatBuffer Python bindings (via `flatc`) |

Running the generator:

```bash
python backends/apple/mlx/serialization/generate.py
```

Use `--skip-flatc` if you only changed op node definitions (not core types) and
want to skip the `flatc` invocation.

### Serialization Format

The binary payload embedded in the `.pte` file has this layout:

```
[Header: 24 bytes]
    4 bytes   padding (zeros)
    4 bytes   magic ("MLX0")
    8 bytes   data_segment_offset (uint64 LE)
    8 bytes   data_segment_size   (uint64 LE)
[FlatBuffer payload]
[Padding to 16-byte alignment]
[Data segment (currently unused — constants go via named_data_map)]
```

The `MLXGraphSerializer` class (in `mlx_graph_serialize.py`) drives
serialization. It inherits `GeneratedOpBuilders` for the per-op builders and
adds the root-table construction, I/O maps, tensor metadata, and header.

---

## Runtime

### Initialization (`init`)

When ExecuTorch loads a `.pte` with an MLX delegate blob, `MLXBackend::init()`
is called:

1. **Parse FlatBuffer** — `loader::load_program()` deserializes the binary into
   an `MLXProgram` struct (C++ mirrors of the schema).
2. **Load constants** — Iterates `named_slots`, calls
   `named_data_map->get_data(name)` for each constant tensor, wraps the buffer
   as an `mlx::core::array` (zero-copy when possible on unified memory).
3. **Initialize mutable buffers** — Creates zero-filled MLX arrays for
   persistent state (e.g., KV cache). These live across `execute()` calls.
4. **Bind execution state** — `ExecutionState::bind()` pre-computes tensor ID
   ranges for O(1) routing.

### Execution (`execute`)

Each `execute()` call:

1. **Reset** per-execution state (inputs/outputs/temps cleared; mutable buffers
   and constants are retained).
2. **Bind inputs** — Walk `input_map`, convert each ExecuTorch tensor to an
   `mlx::core::array` (zero-copy pointer wrap).
3. **Run instructions** — `Interpreter::run()` dispatches each `Instruction`
   through a `switch` on `OpCode`, calling the corresponding `exec_*` function.
4. **Evaluate** — Call `mlx::core::eval()` on output tensors to trigger
   lazy GPU computation.
5. **Copy outputs** — Convert MLX arrays back to ExecuTorch tensors via
   `memcpy`.

### Tensor ID Layout

Tensor slot IDs are assigned in a fixed order during compilation:

```
 ┌──────────┬──────────┬──────────┬────────────────┬──────────┐
 │ Constants│  Inputs  │ Outputs  │ Mutable Buffers│  Temps   │
 │  0..C-1  │  C..I-1  │  I..O-1  │   O..M-1       │  M..T-1  │
 └──────────┴──────────┴──────────┴────────────────┴──────────┘
```

The runtime stores constants and mutable buffers in separate containers
(`ConstantData`, `MutableBufferData`). Inputs, outputs, and temps share a flat
`vector<optional<Tensor>>` in `ExecutionState`.

### Key Runtime Files

| File | Role |
|---|---|
| `MLXBackend.cpp` | `init()` / `execute()` / `destroy()` — the ExecuTorch `BackendInterface` |
| `MLXLoader.h/.cpp` | [GENERATED] Deserializes FlatBuffer into `MLXProgram` (C++ structs) |
| `MLXExecutor.h` | `ExecutionState`, `ConstantData`, `MutableBufferData`, constant loading, dtype conversion |
| `MLXInterpreter.h` | The op dispatch switch + all `exec_*` implementations |
