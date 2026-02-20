# MLX Delegate Serialization

This directory contains the serialization code for the MLX delegate, which converts
Python graph representations to FlatBuffer format for execution on Apple Silicon.

## Single Source of Truth: `schema.fbs`

The FlatBuffer schema file `schema.fbs` is the **single source of truth** for all
serialization-related code. When you need to add a new op or modify existing types,
edit `schema.fbs` and regenerate all derived files.

## Code Generator

The `generate.py` script parses `schema.fbs` and generates:

| Generated File | Description |
|----------------|-------------|
| `mlx_graph_schema.py` | Python dataclasses for all schema types |
| `_generated_serializers.py` | Python FlatBuffer serialization methods |
| `_generated/` | Python FlatBuffer reader classes (via `flatc`) |
| `../runtime/MLXLoader.h` | C++ structs, OpCode enum, NodeVariant |
| `../runtime/MLXLoader.cpp` | C++ `load_instruction()` switch statement |
| `../runtime/schema_generated.h` | C++ FlatBuffer reader classes (via `flatc`) |

## Usage

### Regenerate all files

From the executorch root directory:

```bash
python backends/mlx/serialization/generate.py
```

Or with explicit flatc path:

```bash
python backends/mlx/serialization/generate.py --flatc /path/to/flatc
```

### Options

```
--flatc PATH      Path to flatc compiler (default: "flatc")
--skip-flatc      Skip running flatc (use existing FlatBuffer bindings)
--dry-run         Print what would be generated without writing files
```

## Adding a New Op

1. **Edit `schema.fbs`**: Add a new table for your op and include it in the `OpNode` union:

   ```flatbuffers
   // Add the table definition
   table MyNewNode {
       x: Tid (required);
       out: Tid (required);
       some_param: int32;
   }

   // Add to the OpNode union
   union OpNode {
       // ... existing ops ...
       MyNewNode
   }
   ```

2. **Regenerate**: Run the generator:

   ```bash
   python backends/mlx/serialization/generate.py
   ```

3. **Register the op handler** in `ops.py`:

   ```python
   from executorch.backends.mlx.serialization.mlx_graph_schema import MyNewNode

   @REGISTRY.register(torch.ops.aten.my_op.default)
   def handle_my_op(P: MLXProgramBuilder, node: Node) -> MyNewNode:
       x = P.get_tid(node.args[0])
       out = P.get_output_tid(node)
       return MyNewNode(x=x, out=out, some_param=42)
   ```

4. **Implement the C++ executor** in `MLXInterpreter.h`:

   ```cpp
   case OpCode::MY_NEW: {
       auto& op = instr.get<MyNewNode>();
       // MLX implementation
       break;
   }
   ```

## File Structure

```
serialization/
├── README.md                    # This file
├── schema.fbs                   # SOURCE OF TRUTH - FlatBuffer schema
├── generate.py                  # Code generator script
├── mlx_graph_schema.py          # [GENERATED] Python dataclasses
├── mlx_graph_serialize.py       # Main serializer (uses generated code)
├── _generated_serializers.py    # [GENERATED] Op serialization methods
└── _generated/                  # [GENERATED] FlatBuffer Python bindings
    └── mlx_delegate/
        ├── *.py                 # One file per table/enum

runtime/
├── MLXLoader.h                  # [GENERATED] C++ types and loader decls
├── MLXLoader.cpp                # [GENERATED] C++ loader implementation
├── schema_generated.h           # [GENERATED] FlatBuffer C++ bindings
├── MLXInterpreter.h             # C++ executor (manual)
├── MLXExecutor.h                # C++ executor interface (manual)
└── MLXBackend.cpp               # ExecuTorch backend integration (manual)
```

## Schema Design Notes

### Field Types

- `Tid` - Tensor slot identifier (indexes into tensor array)
- `Vid` - Value slot identifier (indexes into values array for scalars)
- `IntOrVid` - Either a literal int64 or a Vid (for dynamic shapes)
- `FloatOrVid` - Either a literal double or a Vid
- `DTypeId` - Data type enum (f16, f32, bf16, i32, etc.)

### Optional Fields

FlatBuffer fields without `(required)` are optional. In the generated Python
dataclasses, these become `Optional[T]` with default `None`.

For optional scalar fields that need a sentinel (to distinguish None from 0),
use the `= null` default:

```flatbuffers
table MyNode {
    value: float = null;  // None by default, distinguishes None from 0.0
}
```

This requires FlatBuffers 2.0+ (ExecuTorch uses 24.3.25). The generated Python
dataclass will have `value: Optional[float] = None`.

## Troubleshooting

### flatc not found

Install FlatBuffers or specify the path:

```bash
# macOS
brew install flatbuffers

# Or specify path
python generate.py --flatc /usr/local/bin/flatc
```

### Import errors after regeneration

Make sure you're running from the correct environment:

```bash
conda run -n et-mlx python backends/mlx/serialization/generate.py
```

### Generated code doesn't match schema

Delete all generated files and regenerate:

```bash
rm -rf backends/mlx/serialization/_generated
rm backends/mlx/serialization/mlx_graph_schema.py
rm backends/mlx/serialization/_generated_serializers.py
python backends/mlx/serialization/generate.py
```
