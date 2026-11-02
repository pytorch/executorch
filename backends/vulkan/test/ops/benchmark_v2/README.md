# benchmark_v2: JSON shader and operator benchmarks

`etvk_ops_bench` replays a registered ET-VK shader or operator from a JSON config.
Pass one object or an array mixing both case types. Each shader config or operator
sweep entry gets its own tensors, calibration, and measurements. Results default
to compact JSON on stdout, with argument strings and timing summaries. Runtime
failures appear as error objects and execution continues with the next config.
Use `--output_format=json_full` for full ordered JSON reports or
`--output_format=table` for a condensed table.

## Layout

Define inputs, tensor shapes, initialization, sweeps, and measurement settings in
JSON. The binary validates outputs against registered CPU references, calibrates
the repetition count, and reports GPU timestamps, wall time, and optional
throughput estimates. It executes shaders and operators linked into the ET-VK
runtime; raw shader configurations must match the selected shader's interface.

Paths below are relative to `test/ops/benchmark_v2/`:

```text
etvk_ops_bench.cpp         CLI and case execution
framework/                Parsing, graph setup, validation, timing, and reporting
  test/                   Framework and registry tests
operators/                CPU references, routes, throughput counters, and checks
  <family>/.../cases/      JSON case collections by operator and quantization scheme
examples/                 Small shader and operator configuration examples
```

Each expanded case runs separately. Shader variants can share one binding
interface through `shader_sweep`; operators can sweep argument sets, dtype and
storage variables, and execution routes.

## Running the binary

The commands below assume `etvk_ops_bench` is on `PATH` and the current directory
is `test/ops/benchmark_v2/`. A host run requires a Vulkan loader and a compute-capable
Vulkan implementation. Select a hardware GPU or configure a software implementation
such as SwiftShader in the execution environment.

```sh
etvk_ops_bench --config=examples/shaders.json
etvk_ops_bench --config=examples/linear.json --output_format=table
etvk_ops_bench --config=examples/conv2d.json --validation_only --validation_mode=first
```

For Android, stage an Android build of the executable and the JSON file with ADB.
Replace `SERIAL` with the connected device's serial and `/path/to/etvk_ops_bench`
with the built executable:

```sh
adb -s SERIAL push /path/to/etvk_ops_bench /data/local/tmp/etvk_ops_bench
adb -s SERIAL push examples/shaders.json /data/local/tmp/benchmark_config.json
adb -s SERIAL shell chmod +x /data/local/tmp/etvk_ops_bench
adb -s SERIAL shell /data/local/tmp/etvk_ops_bench --config=/data/local/tmp/benchmark_config.json
```

## Command-line options

The binary requires `--config=<JSON_FILE>`. All options follow the executable
name; tensor and benchmark settings belong in the JSON file.

| Option | Default | Behavior |
| --- | --- | --- |
| `--config=<path>` | Required | Read one case or an array of case definitions. |
| `--output_format=json\|json_full\|table` | `json` | Compact JSON, full JSON, or human-readable table. |
| `--validation_mode=all\|first\|none` | `all` | Validate every case, only the first expanded case, or no cases. |
| `--error_mode=report\|throw\|skip` | `report` | Report failures and continue, stop on the first failure, or disable validation. Unsupported cases report `SKIP`. |
| `--validate_specs` | `false` | Audit schemas and registrations without allocating case tensors or running GPU work. |
| `--validation_only` | `false` | Run correctness checks without calibration or timing passes. Requires validation enabled. |
| `--table_name_max_chars=<N>` | `32` | Limit shader/operator names in table output; `0` disables truncation. |
| `--table_inputs_max_chars=<N>` | `64` | Limit table input strings; `0` disables truncation. |
| `--helpshort` | — | Show the binary's flags and defaults. |

For a quick correctness check, pass `--validation_only --validation_mode=first`.
For timing without CPU reference work, pass `--validation_mode=none`.
Benchmark conditions such as `target_ms` and `benchmark_runs` belong in the JSON.

## Config

| Example | Cases |
| --- | --- |
| `examples/buffer_shader.json` | Raw buffer copy. |
| `examples/texture_shader.json` | Raw 3D texture addition. |
| `examples/texture_shader_lwg_sweep.json` | Texture addition with explicit LWG candidates, including `[10,10,1]`. |
| `examples/texture_shader_lwg_range.json` | Texture addition with a Cartesian LWG range. |
| `examples/shader_workgroup_sweep.json` | Two compatible 32-bit buffer-copy shaders crossed with four LWG sizes. |
| `examples/shaders.json` | Raw buffer copy and texture addition. |
| `examples/conv2d.json` | FP32 and FP16 convolution. |
| `examples/conv2d_routes.json` | FP32/FP16 direct and im2col convolution, including intermediate-storage variants. |
| `examples/q8ta_conv2d_routes.json` | Q8TA automatic, direct, and im2col convolution. |
| `examples/q8ta_conv2d.json` | Quantized int8 activations, per-channel int8 weights, and int8 output. |
| `examples/linear.json` | FP32/FP16 linear projections and feed-forward expansion, with decode and batched/prefill shapes. |
| `examples/q4gsw_linear.json` | Weight-only Q4GSW linear with FP32 and FP16 activations. |

| Field | Meaning |
| --- | --- |
| `shader` | Compiled registry name, e.g. `buffer_to_buffer_float`. |
| `operator` | Operator registry name, e.g. `aten.linear.default`. Specify exactly one of `shader`, `shader_sweep`, `operator`, or `check`. |
| `shader_sweep` | Nonempty shader-name list; every shader uses the shared bindings, initialization, parameters, and benchmark settings. |
| `name` | Optional result label; defaults to the registry name. |
| `gwg_size` | Three positive shader invocation counts or a global picker name. |
| `lwg_size` | Three positive workgroup dimensions or a local picker name. |
| `lwg_size_sweep` | Shader-only Cartesian search: `x`, `y`, and `z` ranges `[start, stop, step]`. |
| `bindings` | Shader bindings in binding order, with tensor properties declared inline. |
| `sweep_variables` | Operator variable names mapped to nonempty choice arrays; all Cartesian combinations are tested. |
| `route` | Operator-only route name or `{"name":"im2col","options":{...}}`; omitted means normal automatic routing. |
| `route_sweep` | Operator-only nonempty list of route selections, crossed with argument sets and variables. |
| `args_sweep` | Nonempty list of full operator argument arrays; each set is benchmarked separately. |
| `specialization_constants` | Optional array for IDs 3 onward; local dimensions occupy IDs 0–2. |
| `push_constants` | Optional array of typed data blocks, concatenated into at most 128 bytes. |
| `outputs` | Shader-only optional logical output names. |
| `target_ms` | Calibration target per batch, default 50. |
| `max_dispatches` | Invocation count cap, default 200 for operators and 65536 for shaders. |
| `warmup_runs` | Unmeasured batches per timing mode, default 3. |
| `benchmark_runs` | Measured batches per timing mode, default 10. |
| `check` | Registered validation-only check, e.g. capability or dynamic-resize checks. |
| `reference` | Optional registered CPU reference name; inferred for supported operators/shaders. |
| `validation_method` | Comparison function and tolerances, e.g. `{"function":"is_close","atol":0.01,"rtol":0.01}`. Optional `outputs` selects named tensors to check. |
| `show_memory_throughput` | Show registered memory-throughput estimates for each measurement; default false. |
| `show_compute_throughput` | Show registered compute-throughput estimates for each measurement; default false. |

Tensor descriptions require `sizes` in logical NCHW order. For shaders, put these
properties inside the `buffer`, `texture2d`, or `texture3d` binding object;
for operators, use the `tensor` object inside each argument set in `args_sweep`. Optional fields:

- `name`: optional tensor name; defaults to `arg_<index>`
  for operators and `binding_<index>` for shaders.
- `dtype`: `float32` (default), `float16`, `int32`, `int8`, `uint8`, or `int8x4`.
- `storage`: operator-only; `buffer` (default), `texture2d`, `texture3d`, or
  `constant`. Constant storage passes a CPU tensor reference to the operator's
  prepack route; the operator chooses its GPU storage.
  For shader bindings, the wrapper key determines storage.
- `layout`: `width` (default), `height`, or `channels` packing.
- `data`: initialization descriptor, described below; defaults to zero.
- `access`: optional operator role override, `read`, `write`, or `read_write`.
  Roles are inferred from the operator's dispatch nodes when omitted.
  Writable tensors cannot be constant.

| `data` | Initialization |
| --- | --- |
| `{"fill":0.12}` | Fill every logical element with a value representable by the dtype. |
| `{"rand_uniform_float":[0,1]}` | Uniform random values for FP32/FP16 tensors. |
| `{"rand_uniform_float":[0,1],"seed":42}` | Uniform values using an explicit uint32 seed. |
| `{"rand_uniform_int":[-8,8]}` | Uniform integers, including both bounds; also supports integer-valued FP32/FP16 inputs. Optional `seed` is supported. |
| `{"repeat":[1,2,3]}` | Repeat a fixed sequence to fill the tensor. |
| `{"abs_mod":{"bounds":[-128,127],"mod":5}}` | Generate uniform integers, then take `abs(value) % mod`. |
| `{"pad_columns":{"source":{"rand_uniform_int":[-8,8]},"valid_columns":12}}` | Generate the entire padded tensor, then zero columns after the valid prefix. |
| `{"row_sum":"weight"}` | Sum flattened source rows into int32; trailing destination rows are zero. |
| `{"column_sum":"weight"}` | Sum the source's leading dimensions into its last dimension. |
| `{"q4_group_sum":{"tensor":"weight","group_size":32}}` | Sum signed packed Q4 weights per group into group-major `[K/group_size,N]` data. |

Random data is reproducible. By default, each tensor name determines its seed,
so dtype/storage variants use the same source values and different tensors use
independent streams. FP16 conversion happens during tensor preparation. Flat
`data` arrays and a tensor-level `fill` field are also accepted; use either `data`
or `fill`. The tensor's `storage: "constant"` selects CPU prepacking inputs;
`data.fill` only controls initialization.
Example inputs use random initialization; `fill` is used only for output tensors.
`row_sum` resolves named tensor dependencies, including forward references, and
rejects missing sources, cycles, noninteger sources, and int32 overflow.

For `int8x4`, use buffer storage and a packed layout: `4w`, `4c`, `4w4c`,
`4h4w`, or `4c1w`. Supply logical int8 values; GPU prepacking packs them into
int32 storage. Reports contain tensor properties and timing data without tensor contents.

### Operator arguments

Operators use `args_sweep`, a list of ordered argument arrays; workgroup dimensions and shader bindings
are selected by their runtime implementations. Arguments support:

| JSON | Graph value |
| --- | --- |
| `{"tensor":{"name":"input","sizes":[4,32],"dtype":"float32"}}` | Inline tensor definition. Use `storage: "constant"` for a CPU tensor reference. |
| `{"tensor":{"ref":"input"}}` | Reuse an inline tensor definition. |
| `{"int":32}` | Integer. |
| `{"float":0.25}` | Floating point scalar. |
| `{"bool":false}` | Boolean. |
| `{"int_list":[1,1]}` | Integer list. |
| `{"string":"none"}` | String. |
| `null` | None, e.g. absent bias. |

Each sweep entry contains the full registry argument list, including outputs, in signature order.
Arguments are passed directly without reordering or appending output arguments.
Output tensors do not require an `access` annotation. The runner infers input,
output, and read/write roles from the dispatch nodes for reporting.

For example, each linear argument set has the order input, weight, bias, output.
To reuse an input as the output, pass `{"tensor":{"ref":"input"}}` at the output position.
References can precede definitions; names must be unique within a set and can
be reused in other sets. References cannot cross sweep entries. References accept `ref`
and optional `access`; other tensor properties belong in the definition.
Operator configs reject separate `tensors`, `args`, `inputs`, and `outputs` fields.
Weight and bias tensors usually require `storage: "constant"`. An operator
case cannot specify `gwg_size`, `lwg_size`, `bindings`, push constants,
or specialization constants.

One case can benchmark both precisions:

```json
{
  "name": "relu",
  "operator": "aten.relu.default",
  "target_ms": 10,
  "benchmark_runs": 5,
  "sweep_variables": {"v_DTYPE": ["float32", "float16"]},
  "args_sweep": [
    [
      {"tensor": {"sizes": [8], "dtype": "v_DTYPE", "data": {"rand_uniform_float": [0, 1]}}},
      {"tensor": {"sizes": [8], "dtype": "v_DTYPE", "data": {"fill": 0}}}
    ]
  ]
}
```

Case-level timing settings apply to every entry. Results follow definition and
sweep order, with separate calibration and measurements for each argument set.
Variable names start with `v_`. An entire matching string value is replaced by
the selected JSON value, including numbers, arrays, or objects. Variables can
appear in arguments or case-level settings. Choices must be concrete values;
undefined placeholders are rejected. For each argument template, variables are
expanded in lexical name order, with the last variable varying fastest.

The linear example sweeps 1, 16, and 128 input rows for 2048→2048 projections
and 2048→8192 feed-forward expansion. Each shape runs in FP32 and FP16 with
buffer and texture3d activations, for 24 cases total.
Weights use uniform `[-0.02, 0.02]` values. Its explicit `atol=0.05`, `rtol=0.01`
allows FP16 accumulation error in long reductions; tighter tolerances can be used
when studying numerical accuracy.

The convolution examples use 3x3 kernels, stride/dilation 1, and padding 1.
Q8TA weights use the flattened `[C_out, C_in * K_h * K_w]` layout. Weight sums
are derived from the random weights; the unsigned-dot path requires these sums
even when the input zero point is zero. Q4GSW weights use `[N, K/2]` uniform random
packed bytes; each byte stores two signed Q4 values with zero point 8.
Random scales remain positive and have shape `[K/group_size, N]`.

### Operator routes

Use `route_sweep` to compare execution routes with the same canonical
operator arguments and initialized data:

```json
{
  "operator": "aten.convolution.default",
  "route_sweep": [
    "auto",
    "direct",
    "im2col",
    {"name": "im2col", "options": {"intermediate_storage": "buffer"}},
    {"name": "im2col", "options": {"intermediate_storage": "texture2d"}},
    {"name": "im2col", "options": {"intermediate_storage": "texture3d"}}
  ],
  "sweep_variables": {"v_DTYPE": ["float32", "float16"]},
  "args_sweep": ["<full argument arrays, as in conv2d_routes.json>"]
}
```

For one selection, use `"route": "direct"` or an object with `name`
and optional `options`. The two fields are mutually exclusive and are rejected
for raw shaders. Variables can appear in route names and option values. Expansion
runs argument templates first, variable combinations next, and route
selections in listed order last. The combined expansion is capped at 65536 cases
per definition. Each case gets a separate graph, validation, calibration, and
measurements; route selectors are never appended to the operator argument list.

`auto` calls the ordinary production operator registry, including its performance
heuristics. Forced routes call production route entry points directly, subject
to correctness and device-capability checks. A forced route never silently falls
back to another route. Validation uses the same selection in a separate
graph and the operator's existing CPU reference.

Registered convolution routes:

| Operator | Routes and options |
| --- | --- |
| `aten.convolution.default` | `auto`, `direct`, `im2col`. Only `im2col` accepts `intermediate_storage`: `buffer`, `texture2d`, or `texture3d`; omission uses the production choice. |
| `et_vk.q8ta_conv2d.default` | `auto`, `direct` (general convolution), `im2col`. No route options. |

The floating-point forced routes currently support forward 2D convolution with
batch 1 and channels-packed texture3d input/output. Direct grouped convolution
requires group channel boundaries aligned to four; im2col requires groups 1.
The Q8TA routes use packed int8 buffers and the production streaming feasibility
checks. Route eligibility is separate from automatic performance heuristics:
for example, forcing floating-point im2col can test dilated convolutions even
when automatic routing prefers direct. Constants use `storage: "constant"`. Floating-point weight and bias constants
must share a declared dtype, whose effective GPU dtype matches the input/output
dtype; devices that promote FP16 storage to FP32 keep that promotion supported.

Unsupported forced cases report `status: "SKIP"` and `skip_reason`, with no timing
measurements. They do not fail the run and are skipped in every `error_mode`,
including `throw`. Unknown route names, malformed options, and unexpected runtime
failures remain errors. `--validation_mode=none` skips correctness validation;
it does not disable route eligibility checks. Since unsupported cases produce no
nodes from which to infer tensor roles, their table row may include all arguments.

Explicit selections add requested/resolved `route` metadata to JSON.
An `auto` result identifies the actual production route when an inspector is
registered; otherwise `resolved` is null. The table adds a `Route`
column, e.g. `auto->im2col` or `im2col[buffer]`. Shader breakdowns still show the
actual shaders, and the operator timing covers the complete execute chain.
Prepacking remains outside benchmark timing.

### Registering routes

`framework/RouteRegistry.h` registers a builder function directly under
an operator/route pair, plus an optional inspector for normal automatic routing:

```cpp
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(aten.my_op.default, direct, build_my_direct);
  REGISTER_ROUTE_INSPECTOR(aten.my_op.default, inspect_my_route);
}
```

Builders live under `operators/<category>/`. Their signature is
`Json(ComputeGraph&, const std::vector<ValueRef>&, const Json& options)`.
They receive the canonical argument list and return resolved metadata
`{"name":"direct","options":{}}`. Reject unknown options with
`std::invalid_argument`; throw `UnsupportedRoute` for expected shape,
storage, or capability restrictions before adding nodes. Preflight entry points
that can fall back internally so a forced result always describes the requested
route. The registry reserves `auto` for ordinary operator dispatch.

An inspector receives the graph, canonical arguments, and the first/end execute
node indices for that operator. It returns resolved metadata or null when the
route cannot be identified. Route names are scoped to each operator; duplicate
operator/route registrations and null functions are rejected. Multiple pairs
can register the same builder function. The route library uses
whole-archive linking to retain static registrations.

### Workgroup pickers

Shader workgroup settings accept vectors or picker names, independently:

```json
{
  "gwg_size": "default_pick_gwg",
  "lwg_size": "default_pick_lwg"
}
```

Use `shader_sweep` instead of `shader` to test variants with the same binding
interface and tensor representation:

```json
{
  "shader_sweep": ["buffer_to_buffer_float", "buffer_to_buffer_int32"],
  "gwg_size": "default_pick_gwg",
  "lwg_size_sweep": {"x": [32, 128, 32]}
}
```

Shader names run in listed order, with every workgroup candidate tested before
moving to the next shader. Each combination gets a fresh graph and its own
`sweep_index`, calibration, validation, and measurements. The combined Cartesian
sweep is capped at 65536 cases. `shader_sweep` also combines with an explicit LWG
list; omitting a workgroup sweep tests each shader once with the selected LWG.
Each shader is checked against the bindings when executed; incompatible variants
report case errors using the configured `error_mode`.

The combined example keeps FP32 buffer inputs unchanged. Its int32 copy variant
loads and stores the same 32-bit words, preserving the floating-point bits.
The shared `reference: "copy"` checks identical logical outputs for both shaders.

The global picker is `default_pick_gwg`. It uses the first tensor binding whose
`access` is `write` or `read_write`: logical element count for buffers, texture
extents for textures. Local pickers are `default_pick_lwg`, `pick_required_lwg`,
`pick_xy_square_lwg`, `pick_xz_square_lwg`, and `pick_120shape_lwg`.
`pick_required_lwg` needs a global grid with a wrapping hint, such as the default
buffer grid. Unsupported names produce a case error.

For a shader, `lwg_size` also accepts a nonempty list of vectors:

```json
"lwg_size": [[8, 8, 1], [10, 10, 1]]
```

Alternatively, omit `lwg_size` and specify ranges:

```json
"lwg_size_sweep": {"x": [4, 12, 4], "y": [4, 12, 4], "z": [1, 1, 1]}
```

Ranges include the stop when reached by the step. Omitted axes default to 1.
The Cartesian order is x outermost, then y, then z; each candidate has its own
`sweep_index`, graph, calibration, validation, and measurements. A definition
can expand to at most 65536 cases. Invalid device configurations report case
errors; they are not silently filtered. `lwg_size` and `lwg_size_sweep` are
mutually exclusive.

Explicit local dimensions must be positive and fit device limits. For a
linear buffer grid, an explicit or picked local size also sets the wrapping
geometry. Explicit global vectors are rounded up to local workgroup multiples,
with no output tile scaling. Results report resolved `gwg_size` and `lwg_size`
vectors and, when used, `gwg_picker` and `lwg_picker` names.
Shaders using generic linear buffer indexing require `lwg_size.z=1`; larger z
dimensions need a shader that explicitly handles those invocations.

### Raw shader bindings

Binding forms:

```json
[
  {"buffer": {"name": "output", "sizes": [8], "access": "write"}},
  {"buffer": {"name": "input", "sizes": [8], "dtype": "float32", "data": {"rand_uniform_float": [-1, 1]}}},
  {"meta_ubo": "output", "field": "meta"},
  {"uniform": [
    {"dtype": "int32", "data": [8, 0, 0, 0]},
    {"dtype": "float32", "data": [0.5, 0, 0, 0]}
  ]}
]
```

Tensor bindings use a `buffer`, `texture2d`, or `texture3d` wrapper. Put tensor
properties and `access` inside it, with `sizes` defining a new tensor. Omit
`storage` because the wrapper determines it. `name` is optional and defaults to
`binding_<index>`; names must be unique. `meta_ubo`, specialization layout
constants, and `outputs` refer to these names. References can precede definitions.
Use `{"buffer":{"ref":"input","access":"read"}}` to bind the same buffer again
without allocating another resource. A reference's wrapper type must match its
tensor. Shader configs do not accept a separate `tensors` object.

`access` defaults to `read`; `read_write` is also supported. `meta_ubo` fields are
`meta` (default; `BufferMetadata` or `TextureMetadata`), `sizes`, `strides`,
`numel`, and `logical_limits`. Bindings must match the shader signature.

UBOs and push constants use arrays of typed blocks. Blocks are concatenated
without implicit padding: include the shader's alignment/padding explicitly.
Supported block dtypes: `float32`, `int32`, `uint32`, `int8`, and `uint8`.
Total byte size must be a multiple of four.

Specialization constants use `{"dtype":"int32","value":8}` or
`{"layout":"input"}` for a tensor's hashed layout. Supported scalar dtypes are
`int32` (default), `uint32`, `float32`, and `bool`.

The binary executes shaders and operators already linked into the runtime.
It does not compile GLSL or load external SPIR-V. Raw shader cases require
the selected shader's tensor layouts and parameters. Operator cases invoke
the operator's prepack and dispatch routes.

## Output and timing

The default output is compact JSON. Use `--output_format=table` for a condensed
table:

```sh
etvk_ops_bench --config=examples/shaders.json --output_format=table
```

Columns show shader/operator name, resolved GWG/LWG, inputs, and mean/p90 GPU
timestamp latency in microseconds. Tensor prefixes indicate storage: `b` buffer,
`t2` texture2d, `t3` texture3d, and `c` CPU constant. Non-constant tensors append
their layout name after the dtype, e.g. `b[1,1,1,8]f32.width` or
`b[60,64,40,51]i8x4.4w4c`. Constants omit the layout suffix because the operator
determines their prepacked GPU layout, e.g. `c[128,576]i8`.
Shader inputs include `read` and `read_write` tensor bindings in binding order;
operator inputs follow `args` order and include constant tensors and `read_write`
arguments. Scalar, list, string, and null operator arguments are included in
signature order. Raw shader uniform and push constant values are included.
Floating-point arguments use three significant figures, with scientific notation
when needed. This also applies to floating-point list elements in `args_str`.
Output-only
tensors and UBOs are omitted. Dtype suffixes reflect actual precision. Operator
rows use `-` for workgroup sizes and report latency for the full execute chain.
Unsupported timestamps show `N/A`; failed cases include their error message.
Operator labels append the zero-based sweep index, e.g. `aten.linear.default [1]`.
Separators follow the header and separate cases; operator shader breakdowns stay
in the same group. Names default to a 32-character limit and Inputs to 64;
long values end in `...`. Configure `--table_name_max_chars` and
`--table_inputs_max_chars`; `0` disables the corresponding limit. JSON strings
remain complete.

Use `--output_format=json_full` for full reports. Shader fields follow this order:

Workgroup dimension values and tensor size arrays are printed on one line.
Other report containers retain their indentation.

1. `case_name`, `shader_name`.
2. `gwg_size`, `lwg_size`: `{"method": "explicit", "value": [8, 1, 1]}` or
   the picker name as `method`, with resolved dimensions as `value`.
3. `bindings`: named tensor/UBO descriptions with physical `binding_index`,
   type, sizes, dtype, layout, and access. Shared bindings include `ref`.
4. `specialization_constants`: actual values keyed by constant ID (3 onward;
   local dimensions are already shown in `lwg_size`). Layout-derived constants
   also identify their tensor. Push constants follow when configured.
5. Run info: device, target duration, repetition cap/count, warmup and benchmark
   runs, and timestamp availability/precision.
6. `gpu_timestamp_shader_us`: statistics with `min`, `max`, `mean`, `p50`,
   `p90`, and `p95`.
7. `gpu_timestamp_batch_us`, `wall_time_shader_us`, `wall_time_batch_us`:
   arrays with one value per measured batch, in execution order.

Tensor values are excluded. Dtype metadata reports the actual dtype;
`requested_dtype` appears when promotion occurs, such as FP16 to FP32 on
SwiftShader.

### JSON summaries (`json`)

The default `--output_format=json` keeps the case grouping and field names of
full JSON, with these substitutions:

- Operator `args` becomes `args_str`, using the full `Inputs` notation, including
  non-tensor arguments, storage, dtype, and non-constant layouts. Output-only
  tensors are excluded. Table width limits do not truncate this string.
- `gpu_timestamp_operator_us`, `gpu_timestamp_batch_us`, and wall timing arrays
  become objects with `min`, `median`, and `max`. Median is the middle sorted
  sample, averaging the two middle samples for even counts. Empty timing arrays
  become `null`.
- Each `*_observed_gpu_clock_freq_MHz` array becomes `{min, max}` over that
  pass's start/end readings. Unavailable readings are ignored; a pass without
  usable readings becomes `null`. Zero remains a valid observation.

Shader timestamp statistics, shader breakdowns, raw shader bindings, validation,
and other metadata retain their full JSON representation.

```json
{
  "operator_name": "aten.linear.default",
  "args_str": "b[4,32]f32.width, c[16,32]f32, c[16]f32",
  "operator_invocations_per_batch": 10,
  "gpu_timestamp_operator_us": {"min": 2, "median": 4.5, "max": 9},
  "gpu_timestamp_batch_us": {"min": 20, "median": 45, "max": 90},
  "gpu_batch_observed_gpu_clock_freq_MHz": {"min": 800, "max": 950}
}
```

### Batches and calibration

A batch is one recorded command buffer containing repeated executions of the
case. For a raw shader, each repetition is one shader dispatch; for an operator,
each repetition is the full execute chain, which may dispatch several shaders.
The repetition count is reported as `dispatches_per_batch` or
`operator_invocations_per_batch`, respectively.

Calibration always warms and probes query-free batches, increasing repetitions
until the measured wall time reaches roughly `target_ms` or `max_dispatches`.
Calibration probes are excluded from reported latency samples. Commands are
recorded once per calibration step, then reused for warmup and measurement.
Every expanded sweep combination is calibrated and measured separately.
Operator graphs always invoke resize functions before benchmarking, including
shader and workgroup selection even when input shapes have not changed.

### Measurement passes

The benchmark runs the enabled passes below in the listed order, using separate
submissions.

| Pass | Measurement boundary | Reported latency |
| --- | --- | --- |
| Wall time | Host monotonic clock from command-buffer submission through fence completion, without timestamp queries. Includes host submission overhead, queue/fence waiting, and GPU execution. | `wall_time_batch_us` contains batch totals. Dividing each total by the repetition count gives `wall_time_shader_us` or `wall_time_operator_us`. |
| Whole-batch GPU timestamps | One timestamp pair brackets the entire repeated dispatch chain, including GPU dependency barriers and inter-dispatch gaps. | `gpu_timestamp_batch_us` describes batch totals. Dividing each total by the repetition count gives `gpu_timestamp_shader_us` or `gpu_timestamp_operator_us`. These are two views of the same capture. |
| Per-dispatch GPU timestamps | Operator-only profiling pass with a timestamp pair around each executed shader, including its associated barriers. | `shader_breakdown[].gpu_timestamp_shader_us` describes each dispatch position, averaged across that profiling batch's operator repetitions. |

Pass selection depends on case type and timestamp support:

| Case | GPU timestamps available | Enabled passes |
| --- | --- | --- |
| Raw shader | Yes | Wall time, whole-batch GPU timestamps. |
| Operator | Yes | Wall time, whole-batch GPU timestamps, per-dispatch GPU timestamps. |
| Raw shader or operator | No | Wall time only. GPU batch/operator timings are `null` in `json` and empty arrays in `json_full`; raw shader statistics are `null`, and shader breakdowns are omitted. |

Each enabled pass performs `warmup_runs` unmeasured replays (default 3), followed
by `benchmark_runs` measured replays (default 10; many examples specify 5).
The wall and whole-batch GPU passes use the calibrated repetition count. The
profiling pass may reduce that count to fit its timestamp query pool; its divisor
is `shader_profile_invocations_per_batch`.

### Android GPU clock samples

Android builds query the GPU clock immediately before submission and after fence
completion for every measured replay. Reads are outside the latency intervals;
warmups, calibration, and validation do not produce clock samples. JSON reports
include separate clock series for each pass. `json` summarizes each series as
`{min, max}`; `json_full` preserves arrays of start/end pairs in MHz:

```json
{
  "gpu_batch_observed_gpu_clock_freq_MHz": [
    [903, 903],
    [903, 950]
  ],
  "wall_time_batch_observed_gpu_clock_freq_MHz": [
    [903, 903],
    [903, 903]
  ]
}
```

| Clock field | Measurement pass | Associated timing |
| --- | --- | --- |
| `gpu_batch_observed_gpu_clock_freq_MHz` | Whole-batch GPU timestamps | `gpu_timestamp_batch_us` and operator batch averages in `gpu_timestamp_operator_us`. |
| `wall_time_batch_observed_gpu_clock_freq_MHz` | Wall time | `wall_time_batch_us` and `wall_time_shader_us`/`wall_time_operator_us`. |
| `shader_profile_batch_observed_gpu_clock_freq_MHz` | Operator shader profiling | Runs summarized by `shader_breakdown[].gpu_timestamp_shader_us`. |

In `json_full`, each enabled pass contributes `benchmark_runs` pairs in run
order. For GPU-batch and wall-time arrays, entry `i` corresponds to timing sample
`i` from that pass.
The GPU-batch and wall-time clock arrays immediately follow their corresponding
`_batch_us` fields in JSON output. Array indexing applies to `json_full`; compact
`json` reports extrema over each pass's observations.
Shader-profile entry `i` describes profiling batch `i`, whose dispatch latencies
feed the shader-breakdown statistics. The GPU batch array is empty without GPU
timestamp support; the shader profiling array appears only when operator
profiling runs.

An unreadable clock sample is `null`; a driver-reported zero is retained. These
are driver-reported snapshots, not average clocks during GPU execution. Host
builds omit clock fields.

When clock observations are present, tables add `Min GPU MHz` and `Max GPU MHz`
after the latency columns. A main case row reports the minimum and maximum
across all start/end observations from its wall, GPU batch, and shader profiling
runs. Indented shader rows share the range from the operator's profiling runs.
Missing, nonnumeric, or nonfinite readings are ignored; rows without usable
observations show `N/A`. Driver-reported zero values remain valid observations.

The sampler discovers readable KGSL/devfreq, Pixel Mali, and Samsung SGPU sysfs
sources and converts their driver-specific Hz, kHz, or MHz units to MHz. Clock
access is best effort and does not require root or change GPU clock settings.

### Samples and percentiles

Each measured batch contributes one sample to each reported latency series.
For whole-batch GPU timing, the sample is the batch timestamp interval. For
amortized shader/operator timing, it is that interval divided by the repetition
count. For a shader breakdown entry, it is the average interval for that dispatch
position across the profiling batch's repetitions. Repeated shader names at
different dispatch positions remain separate series.

In `json_full`, `gpu_timestamp_batch_us`, `gpu_timestamp_operator_us`, and
wall-time fields contain arrays with one value per measured batch, in execution
order. Compact `json` summarizes these arrays as `min`, `median`, and `max`.
Operator GPU values are batch totals divided by
`operator_invocations_per_batch`; for example, in `json_full`:

```json
{
  "operator_invocations_per_batch": 10,
  "gpu_timestamp_operator_us": [3, 1, 2],
  "gpu_timestamp_batch_us": [30, 10, 20]
}
```

`gpu_timestamp_shader_us`, both for raw shaders and shader breakdown entries,
reports `min`, `max`, `mean`, `p50`, `p90`, and `p95`. These shader statistics
currently use one batch-average sample per benchmark run. The table derives
mean/p90 from operator sample arrays and reads shader statistics for raw shader
and indented profiling rows.

Percentiles use linear interpolation. For `R` sorted samples `x[0]` through
`x[R-1]` and percentile fraction `f`, the position is `f * (R - 1)`; the result
interpolates between the adjacent samples at that position. With five sorted
samples `x1 <= x2 <= x3 <= x4 <= x5`:

| Statistic | Result |
| --- | --- |
| p50 | `x3` |
| p90 | `0.4 * x4 + 0.6 * x5` |
| p95 | `0.2 * x4 + 0.8 * x5` |

With `benchmark_runs: 5`, even a batch containing hundreds of dispatches yields
only five batch-average samples. High percentiles then largely interpolate
toward the largest observation and provide weak estimates of the tail.
Increasing `benchmark_runs` to 50 or 100 improves sampling, but these remain
percentiles of batch-average latency; averaging within each batch smooths
variation among individual executions.

### Measurement boundaries and shader breakdowns

Wall timing samples cover query-free submission through fence completion. GPU
timestamps bracket the repeated dispatch chain, including dependency barriers. Input
preparation, uploads, pipeline creation, command recording, query retrieval, and
readback are outside these intervals. Unsupported timestamps produce `null` GPU
batch/operator timings in `json`, empty arrays in `json_full`, and `null` raw
shader statistics. Conversion preserves fractional `timestampPeriod` and handles
timestamp counter width.

Operator reports use `operator_name`, `sweep_index`, and the selected inputs:
`args_str` in `json`, or ordered `args` with inline tensor metadata and access
roles in `json_full`. They also include
the runtime-selected `execute_nodes`. Per-operator fields are
`gpu_timestamp_operator_us` and `wall_time_operator_us`; they include the full
execute chain (e.g. im2col, transpose, GEMM), with weight prepacking outside timing.
`operator_invocations_per_batch` is the divisor for both wall and batch GPU
measurements. Nodes may self-gate and issue
no dispatch. Repeats reuse initialized storage, so in-place operators accumulate
updates.
When variables are present, reports include their selected `sweep_variables`
values and the original `args_sweep_index`. `sweep_index` identifies the expanded
run within that definition.

When GPU timestamps are available, operator reports also include `shader_breakdown`.
Each entry describes an actual dispatch in execution order: `dispatch_index`,
runtime-selected `shader_name`, resolved GWG/LWG, and `gpu_timestamp_shader_us`
statistics. Repeated shader names remain separate entries; skipped nodes are
omitted. Table output shows these shaders indented below their operator.

The breakdown uses a separate profiling pass with the calibrated invocation count,
capped by timestamp query-pool capacity. `shader_profile_invocations_per_batch`
reports that count. Warmup and benchmark run counts are the same as the aggregate
timing passes. Each shader sample averages that dispatch position over the batch
and includes its dependency barrier.
Per-dispatch timestamp overhead and inter-dispatch gaps mean the sum need not
match the separately measured operator latency. The passes can also run under
different GPU clock/cache conditions, and their repetition counts may differ.
For small workloads, per-dispatch timestamps can substantially increase measured
latency. Compare the batch GPU timestamps and query-free wall measurements
when evaluating that overhead.
Prepacking remains outside timing.
Breakdowns are omitted when timestamps are unavailable, including on this
repository's SwiftShader build.

## Throughput

Enable either or both metrics in a case definition:

```json
{
  "show_memory_throughput": true,
  "show_compute_throughput": true
}
```

The report's `throughput` object contains `memory_bytes` and/or `compute_flops`
per invocation. `gpu_timestamp` and `gpu_timestamp_batch` contain rate statistics;
`wall_time` and `wall_time_batch` contain one rate per measured batch. Memory
rates are available as `memory_gib_per_s` (GiB/s, 2^30 bytes/s) and
`memory_mib_per_s` (MiB/s, 2^20 bytes/s); compute rates use `compute_gflops`
(10^9 FLOPs/s). Throughput is computed from each sample before aggregation, so
its percentiles rank rates, with higher values indicating faster execution.
Batch rates account for all repeated invocations and match the invocation rates.

Table output appends `GPU GiB/s` and/or `GPU GFLOPS`, showing mean GPU rates.
Shader breakdown entries include their own GPU throughput when a counter is
registered for the executed shader. No parent operator work is assigned to a
child shader. Unregistered counters produce null rates, table `N/A`, and a
diagnostic; a known zero-work counter produces zero. Unavailable GPU timestamps
produce null GPU rates while wall-time rates remain available.

Memory counters estimate logical tensor traffic: each distinct read tensor once,
each write tensor once, and read/write tensors twice. Operator counters include
logical input/weight/bias/output tensors; child shader counters estimate their bound tensor footprints, including packed
weights and intermediates. Tiled/streaming dispatches may touch only part of those
footprints. The model excludes
UBOs, staging, tensor padding, prepacking, and additional/cache-dependent traffic.
GEMM/convolution counters use two operations per multiply-accumulate, plus
optional bias additions. Quantized operators report arithmetic-equivalent work,
excluding weight dequantization and activation requantization. Q4 linear
counters can use the model `3*M*N*K + M*N`; FPA counters use the two-operation
multiply-accumulate model. The registered counter determines the model. Custom
counters can implement another work model.

Counters register beside operator implementations using
`framework/ThroughputRegistry.h`, with independent memory/compute associations:

```cpp
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(aten.my_op.default, my_byte_count);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.my_op.default, my_flop_count);
  REGISTER_SHADER_MEMORY_TRAFFIC(my_shader, my_byte_count);
  REGISTER_SHADER_COMPUTE_FLOPS(my_shader, my_flop_count);
}
```

Shader counters can register a case-sensitive prefix. For example:

```cpp
REGISTER_SHADER_MEMORY_TRAFFIC(conv2d, throughput::tensor_traffic);
```

This covers every shader whose name starts with `conv2d`, including dtype,
storage, and tile variants. Lookup checks exact registrations first, then uses
the longest matching prefix. `REGISTER_SHADER_MEMORY` and
`REGISTER_SHADER_COMPUTE` register exact names for individual overrides;
operator associations remain exact. Memory and compute prefixes are independent,
and empty/duplicate prefix registrations and null callables are rejected. Each
registration stores the function pointer directly; one callable can serve
multiple operators, shaders, or prefixes.

Prefixes group variants with the same work model. For example, `linear_vec_bias`
selects the bias-aware compute counter over `linear_vec`, and
`q4gsw_linear_gemm__tin__` selects its binding-aware memory counter over
`q4gsw_linear_gemm__`. Prefix association happens in the benchmark's throughput
registry when the actual dispatched shader name is reported.

A counter returns a finite, nonnegative `double` from a `const Json&` containing
resolved case metadata. Operators receive `operator_name` and ordered `args`
with tensor descriptions and scalar values. Shaders receive `shader_name`,
resolved GWG/LWG, and named `bindings` with sizes, actual dtype, storage, and
access; shader breakdown counters receive metadata from that dispatch's tensors.
Registration is performed before benchmarking. Duplicate keys in each
association category are rejected. Functions are passed directly to registration
macros and receive resolved case metadata; no separate function-name registry
is required.

## Correctness validation

Validation runs before timing by default. It constructs a separate graph with
input/output staging and executes the target shader or operator once. Benchmark
graphs have no execute-stage input/output transfers. CPU reference work and
readback are outside all benchmark intervals.
Tensor data is generated once per expanded case and shared by the validation
and benchmark graphs. Each graph uploads its inputs during setup; benchmarking
does not regenerate input data. When validation is skipped, data is still
generated once to initialize the benchmark graph.
Validation GPU execution latency is not reported. The separate
`validation.reference_wall_time_us` field times CPU reference computation,
including input decoding and precision conversion; it excludes output readback
and comparison and is not a shader/operator benchmark latency.

Select correctness checks with `--validation_mode`:

| Value | Behavior |
| --- | --- |
| `all` (default) | Validate every expanded case when supported. |
| `first` | Validate only the first expanded case in the entire input file, for smoke testing. |
| `none` | Skip correctness validation and CPU reference computation. |

`first` selects the first case after argument, shader, workgroup, and route sweep
expansion. If that case is unsupported, it reports `SKIP`; later cases are not
validated. All cases still run their benchmark passes when supported. Results
include a JSON `validation` object only for selected checks; the table includes
a `Validation` column when any result has validation information.

`--error_mode=report` (default) reports `PASS` or `FAIL` for selected validation
checks and continues benchmarking. Any validation or execution failure produces
exit status 1.
`--error_mode=throw` stops at the first validation or execution failure with
exit status 1. `--error_mode=skip` disables validation, overriding
`--validation_mode`; prefer `--validation_mode=none`.
Malformed configs and flags produce exit status 2.

Example reference names include `copy`, `add`, `linear`, `conv2d`, `q8ta_conv2d`,
and `q4gsw_linear`. They are inferred for the example shaders/operators;
`reference` can select one explicitly for a compatible custom shader. Unknown
references report validation failure. New CPU references belong under
`operators/<category>/` and register beside their implementations.

### Registering references

`framework/ReferenceRegistry.h` provides two registration levels:

```cpp
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(my_reference, my_reference_impl);
  REGISTER_OPERATOR_REFERENCE(aten.my_op.default, my_reference);
  REGISTER_SHADER_REFERENCE(my_shader, my_reference);
}
```

The first macro registers a callable under a unique name. The remaining macros
associate an operator or shader name with that reference name. Multiple names
may share a reference; operator and shader associations have separate namespaces.
Registration blocks run at static initialization, following the ET-VK operator
registration pattern. Associations may be declared before the function's
registration in another translation unit; lookup verifies that the function
exists. Duplicate names or associations are errors.
Each associated function must understand that case's operator arguments or
shader bindings. Registration is performed before benchmarking starts.

A reference callable has the signature
`reference::Tensor(const Json&, const reference::TensorMap&, size_t threads)`.
Tensor inputs have already been decoded and rounded to actual GPU precision.
The returned tensor is shape-checked and rounded to output precision by the
framework. A JSON `reference` override resolves through the same function registry.
The reference library uses whole-archive linking to retain registration
initializers from every reference source file.

Linear references use a tiled, vectorizable CPU GEMM with up to eight threads.
Convolution references parallelize output work. Validation does not skip cases
based on tensor size. Inputs are rounded to their actual GPU precision,
including FP16 constants, before computing reference results.

The supported comparison is `is_close`: every element must satisfy
`abs(actual - expected) <= atol + rtol * abs(expected)`. Nonfinite values,
shape/count mismatches, and missing/unexpected outputs fail validation.
Defaults are FP32 `atol=rtol=1e-4`, FP16 `atol=rtol=1e-2`, and exact integer
comparison. `validation_method` overrides either tolerance; tolerances must be
finite and nonnegative. FP16 accumulation may require larger tolerances for
long reductions. Reports include per-output mismatch counts and error maxima,
and `reference_wall_time_us`, without tensor values.

## Operator source organization

Operator code follows family and quantization-scheme folders, using the production
runtime organization as a guide:

```text
operators/
  convolution/
    Conv2dReference.{h,cpp}
    Conv2dRoutes.{h,cpp}
    Conv2dThroughput.cpp
    cases/
    q8ta/
      Q8taConv2dReference.{h,cpp}
      Q8taConv2dRoutes.{h,cpp}
      Q8taConv2dThroughput.cpp
      cases/
  gemm/
    linear/
      LinearReference.{h,cpp}
      LinearThroughput.cpp
      cases/
      qw/q4gsw/
        Q4gswLinearReference.{h,cpp}
        Q4gswLinearThroughput.cpp
        cases/
```

Scheme-specific arithmetic, argument validation, and registrations live with the
scheme. Shared convolution geometry/traversal and route guards live in the
convolution parent; shared GEMM projection and work-count helpers live in the
GEMM parent. Add other quantized implementations under their family’s `q8ta/` or
`qw/<scheme>/` folder.

## Framework components

Shared headers and implementations live in `framework/`: `Config` validates JSON,
`Sweep` expands templates, Cartesian variables, and route selections, `DataInitializer` generates
typed input data, and `TensorUtils` handles tensor setup and dtype metadata.
`Timestamps` owns GPU queries, `Timing` handles calibration and measurement,
`WorkgroupUtils` resolves workgroup vectors and named pickers, and `Report`
formats ordered case metadata and timing statistics.
`RouteRegistry` resolves named operator routes and optional automatic
route inspectors. `ShaderBenchmark` and `OperatorBenchmark` assemble and run their respective cases.
Component tests live under `framework/test/`, including registry, reference,
route, reporting, and validation tests.
`etvk_ops_bench.cpp` handles the CLI and dispatches configs to the runners.

## Framework tests

Tests under `framework/test/` cover schemas and sweep expansion, data
initialization, reference and throughput registrations, route lookup, validation,
and report formatting. Registry and reporting tests run without GPU execution.
Operator route tests exercise shader chains, route options, automatic route
inspection, and unsupported-case guards; these require a Vulkan implementation
such as SwiftShader or a hardware GPU.

## Case collections

JSON collections live under `operators/<family>/.../cases/*.json`, alongside the
references and registrations for that operator family or quantization scheme.
Pass any collection to `--config`. Each definition and its expanded sweeps run
in file order.

Operator cases call registered production operators directly. Q8TA cases use
packed `int8x4` activations and outputs; weight-only quantized operators use their
native weight, scale, and activation arguments. Quantize and dequantize are
separate operator cases. Quantized comparisons use quantized units with zero
relative tolerance. Use `route` or `route_sweep` to select execution routes.

## Schema audits and validation-only runs

`--validate_specs` parses all expanded cases and checks operator, route, reference,
and custom-check registrations without allocating case tensors or executing
operators. It emits a JSON audit summary and exits nonzero on errors. The binary
attempts texture creation without screening dimensions against driver-reported limits,
which can understate usable dimensions. Ordinary execution still validates
outputs according to `--validation_mode`. Use
`--validation_only` to execute GPU correctness checks without calibration or
benchmark passes; it requires `--validation_mode=all` or `first` and
`--error_mode=report` or `throw`. With `first`, remaining cases report `SKIP`
without allocating tensors or executing benchmark passes.

Device support is determined while constructing and executing each case. The
binary catches ET-VK `ShaderNotSupportedError` and unsupported-route exceptions
and reports `SKIP` with a reason, then continues to the next case. This applies
to validation and benchmarking, including `--error_mode=throw`; that option
still stops on validation failures and other errors. JSON `requirements` gates
are not supported. `--validate_specs` checks schemas and registrations, rather
than executing cases to establish device support.

Validation-only `check` cases return PASS, FAIL, or SKIP with diagnostic details;
they have no latency measurements. Unselected checks report `SKIP`, including
all checks under `--validation_mode=none` or `--error_mode=skip`.

To preserve a case's declared validation scope when an operator also updates
auxiliary state, select outputs inside the comparison settings:

```json
"validation_method": {
  "function": "is_close",
  "atol": 0.001,
  "rtol": 0.001,
  "outputs": ["output"]
}
```

The default checks all inferred writable tensors. An explicit list must be
nonempty, unique, and name writable GPU tensors from the case. Each selected
tensor must be produced by the reference and the GPU. Unselected auxiliary
tensors remain in the operator arguments and execute normally.
