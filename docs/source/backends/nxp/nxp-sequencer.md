# NXP eIQ Neutron Sequencer Mode

The Neutron sequencer mode can improve inference speed on `Neutron-C` targets, such as `i.MX RT700`, at the cost of
a slightly larger model. In this mode, the Neutron Compiler generates bytecode that is consumed directly by a hardware
component of the Neutron unit called the sequencer, instead of generating microcode that is processed by the Neutron
Driver at runtime.

The bytecode is a stream of encoded instructions that programs the Neutron hardware registers. Part of the microcode
processing overhead is thus moved from runtime to compile time. Multiple consecutive kernels are grouped into
sequences and executed atomically, which reduces the runtime dispatch overhead.

> **Note:** The sequencer mode is available only for `Neutron-C` targets. It cannot be used for `Neutron-S` targets
> (with subsystem).

## Enable the sequencer mode

Set the `use_sequencer` option when generating the Neutron compile spec:

```python
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec

compile_spec = generate_neutron_compile_spec("imxrt700", use_sequencer=True)
```

When exporting with the Neutron recipes, set `use_sequencer=True` in `NeutronRecipeConfig`.

The `aot_neutron_compile.py` example enables the mode with the `--use_sequencer` flag:

```bash
python -m examples.nxp.aot_neutron_compile \
    --quantize \
    --delegate \
    --use_sequencer \
    --model_name cifar10
```

For the full list of compile spec options, see {doc}`nxp-partitioner`.

## Trade-offs

- The sequencer mode is not guaranteed to be faster for every model. The benefit depends on the model and the target,
  and the inference can even be slower, so benchmark the model in both modes before choosing.
- The total size of the microcode is larger than in the standard mode, because the kernel internal loops are unrolled
  in the bytecode.
- When combined with `fetch_constants_to_sram`, each sequence contains only a single kernel. Dynamic weight fetching
  requires kernel-level synchronization, which prevents the kernel grouping. The sequencer bytecode is copied from
  FLASH to SRAM once before the execution, as the hardware can execute bytecode only from SRAM.

## Cycle estimation and profiling

The cycle estimate reported by the Neutron Compiler (`dump-statistics`) consists of the host-side overhead and the NPU
execution. With the sequencer enabled, the host-side part is suppressed considerably, so the estimate is more reliable
and closer to the hardware.

In the statistics table, the individual kernel rows are replaced by grouped `Sequence_N (N kernels)` rows. When
`fetch_constants_to_sram` is also used, an additional `Bytecode fetch` row describes the one-shot copy of the bytecode
to SRAM.

The compiler option `dump-bytecode` prints the generated sequencer bytecode to the console. It can be passed through
the `extra_flags` compile spec option.

For the profiling support, see {doc}`nxp-profiling`.
