# Memory Overflow Debugging

## Diagnosing link-time overflow

When `west build` fails with linker errors like:

```
ld.bfd: region `FLASH' overflowed by 3476864 bytes
ld.bfd: region `RAM' overflowed by 3128920 bytes
```

### Step 1: Identify what's overflowing

Read the error to determine which region and by how much:

| Region | Contains | Typical cause of overflow |
|--------|----------|--------------------------|
| FLASH | .text, .rodata, vector table | Model blob in wrong section, too many portable ops |
| RAM | .data, .bss (pools, runtime copy of model) | Pools too large, model SRAM copy |

### Step 2: Check for orphan sections

```
warning: orphan section `network_model_sec' from `app/libapp.a(main.cpp.obj)'
```

This means the model blob section has no matching linker rule. It gets dumped into
FLASH by default. Fix: add a DTS memory region + linker snippet to route it
to DDR or other external memory. See `board_bringup.md` for the pattern.

### Step 3: Check the memory map

On a successful build, the output shows:

```
Memory region         Used Size  Region Size  %age Used
           FLASH:      459668 B       512 KB     87.67%
             RAM:     1963160 B         2 MB     93.61%
       MODEL_DDR:     3541440 B        16 MB     21.11%
```

Calculate headroom: `Region Size - Used Size`. If any region is over 95%, it's fragile.

## Common overflow patterns and fixes

### Pattern 1: Model blob overflows FLASH

**Symptom:** FLASH overflows by roughly the .pte file size.

**Cause:** The model blob (in `network_model_sec`) lands in FLASH because there's
no linker rule routing it elsewhere.

**Fix:** Add a DDR/external memory region via DTS overlay and a linker snippet.
Also enable `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y` in the board conf to
skip the SRAM copy (the Kconfig sets the `ET_ARM_MODEL_PTE_DMA_ACCESSIBLE`
compile-time macro that main.cpp checks).

### Pattern 2: Model SRAM copy overflows RAM

**Symptom:** RAM overflows by roughly the .pte file size.

**Cause:** Without `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y`, main.cpp creates
`model_pte_runtime[sizeof(model_pte)]` — a writable copy of the entire model in SRAM.

**Fix:** If the model is in DMA-accessible memory (DDR, MRAM), set
`CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y` in the board conf. This defines
`ET_ARM_MODEL_PTE_DMA_ACCESSIBLE` at compile time, telling main.cpp to use
the model blob in-place.

### Pattern 3: Pools overflow RAM

**Symptom:** RAM overflows but model is already in DDR.

**Cause:** Combined pool sizes exceed available SRAM.

**Fix:** Either reduce pool sizes (if the model allows) or redirect `zephyr,sram`
to a larger memory region. Check `board_bringup.md` for the memory budget formula.

### Pattern 4: Both FLASH and RAM overflow

**Symptom:** Both regions overflow simultaneously.

**Cause:** Model blob in FLASH + model SRAM copy in RAM. Double penalty.

**Fix:** Apply fixes for patterns 1 and 2 together: DDR placement + DMA_ACCESSIBLE.

## Diagnosing runtime allocation failures

When the build links but execution fails with:

```
Memory allocation failed: 602112B requested (adjusted for alignment), 33136B available
```

### Reading the error

The error tells you exactly:
- **How much was requested** (602112 B)
- **How much was available** (33136 B)
- **Which allocator** (from the call stack: `method_allocator` or `temp_allocator`)

### Method allocator failures

The method pool holds planned buffers (model-determined, fixed) plus input tensors.
Look at the log lines before the error:

```
Setting up planned buffer 0, size 752640.
Method allocator pool size: 786432 bytes.
```

Here: 752640 used for the planned buffer, leaving 33792 free. The input tensor
needs 602112, which doesn't fit.

**Fix:** `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` >= planned_buffer + input_tensor_size + alignment padding.

### Temp allocator failures

The temp pool holds delegate scratch buffers. For Ethos-U:

```
Failed to allocate scratch buffer of 1509792 bytes from temp_allocator
```

**Fix:** `CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE` >= requested size + alignment padding.

### When pools don't fit in SRAM

If total required pools exceed available SRAM:

1. **Use a board with more memory** (e.g., Corstone-320 has 4 MiB ISRAM vs Corstone-300's 2 MiB)
2. **Redirect `zephyr,sram` to a larger region** (e.g., ISRAM instead of SRAM — verify flash/sram don't conflict)
3. **Place pools in DDR** via custom linker sections (slower but no size limit)
4. **Use a smaller model** that needs less scratch space

## Useful commands

### Inspect ELF sections

```bash
arm-zephyr-eabi-objdump -h build/zephyr/zephyr.elf
```

Shows all section names, VMAs (virtual memory addresses), and sizes. Use this to
verify the model section landed in the expected memory region.

### Inspect linker memory regions

```bash
grep "MEMORY" build/zephyr/linker_zephyr_pre0.cmd
```

Shows all memory regions with origins and sizes as the linker sees them.

### Check generated linker snippet

```bash
cat build/model_section.ld
```

Verify the section name matches (no leading dot) and the target region is correct.

### Check DTS overlay was applied

```bash
grep "model_ddr\|zephyr,sram\|zephyr,flash" build/zephyr/zephyr.dts
```

Verify your overlay nodes appear in the final merged DTS.
