# Adding a New Board

## What you need

1. A Zephyr BSP for the board (upstream or custom)
2. The board's memory map (from datasheet or DTS)
3. A .pte model file to embed

## Files to create

For a sample called `my-sample` targeting board `my_board`:

```
zephyr/samples/my-sample/
├── boards/
│   ├── my_board.overlay    # DTS: memory regions, chosen nodes
│   └── my_board.conf       # Kconfig: drivers, pool sizes
├── CMakeLists.txt           # Build: model embedding, ops, linking
├── Kconfig                  # Sample-level config options
├── prj.conf                 # Default config (all boards)
└── src/main.cpp             # Application code
```

If adding a board to an **existing** sample (e.g., `mv2-ethosu`), you only need the `boards/` files.

## Step 1: DTS overlay

The overlay configures memory regions for your board. Key decisions:

### Which `chosen` nodes to set

```dts
/ {
    chosen {
        zephyr,sram = &my_sram;   /* where .data/.bss land (allocator pools) */
        /* zephyr,flash is usually set by the board DTS already */
    };
};
```

**Rule:** `zephyr,sram` must point to the largest contiguous RAM region that fits
the allocator pools. If the default is too small, override it.

**Warning:** If `zephyr,flash` and `zephyr,sram` point to the same physical memory,
Zephyr's non-XIP linker places .text and .data sequentially. This works but means
code and data share the region's capacity.

### Adding a DDR / external memory region

If the model blob is too large for on-chip memory, place it in external memory:

```dts
/ {
    model_ddr: memory@70000000 {
        compatible = "zephyr,memory-region", "mmio-sram";
        reg = <0x70000000 DT_SIZE_M(16)>;
        zephyr,memory-region = "MODEL_DDR";
    };
};
```

Then create a linker snippet (`model_section.ld.in`) to route the model section there.
See `zephyr/samples/mv2-ethosu/model_section.ld.in` for the template.

The CMakeLists.txt detects the DTS node and generates the linker snippet:

```cmake
dt_nodelabel(model_ddr_path NODELABEL "model_ddr")
if(model_ddr_path)
  configure_file(model_section.ld.in ${CMAKE_CURRENT_BINARY_DIR}/model_section.ld @ONLY)
  zephyr_linker_sources(SECTIONS ${CMAKE_CURRENT_BINARY_DIR}/model_section.ld)
endif()
```

### DMA accessibility check

If using an NPU (Ethos-U), the model data and scratch buffers must be in
DMA-accessible memory. Check the board's TRM (Technical Reference Manual) for
which memory regions the NPU's DMA engine can reach.

Common patterns:
- **FVP boards**: ISRAM and DDR are DMA-accessible; DTCM is not
- **Alif Ensemble**: MRAM and SRAM are DMA-accessible
- **General rule**: tightly coupled memories (TCM) are usually NOT DMA-accessible

## Step 2: Kconfig conf

```kconfig
# Enable NPU driver (if applicable)
CONFIG_ETHOS_U=y

# Skip SRAM model copy if model is in DMA-accessible memory.
# This Kconfig symbol is defined per-sample (e.g., mv2-ethosu/Kconfig),
# not globally. If your sample doesn't define it, add the Kconfig entry
# or pass -DET_ARM_MODEL_PTE_DMA_ACCESSIBLE via CMake directly.
CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y

# Pool sizes — adjust based on model requirements
# Run a build first, then tune from runtime error messages
CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE=1572864
CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE=1572864
```

### Pool sizing strategy

1. Start with the sample's defaults (check `prj.conf` and board `.conf` — varies by sample)
2. Build and run — if allocation fails, the error tells you exactly what was requested
3. **Method pool**: must hold largest planned buffer + all input tensors
4. **Temp pool**: must hold delegate scratch buffer (varies by model and backend)
5. Total pools must fit in the `zephyr,sram` region minus ~112 KiB overhead (stack, heap, .data/.bss)

### Memory budget formula

```
available_for_pools = zephyr_sram_size - code_if_shared - stack - heap - bss_overhead
```

Where:
- `code_if_shared`: 0 if flash and sram are separate regions; .text size if they share a region
- `stack`: `CONFIG_MAIN_STACK_SIZE` (default 16 KiB)
- `heap`: `CONFIG_HEAP_MEM_POOL_SIZE` (default 64 KiB)
- `bss_overhead`: ~30 KiB for ET runtime + Zephyr kernel

## Step 3: Build and verify

```bash
west build -b my_board modules/lib/executorch/zephyr/samples/my-sample -- \
    -DET_PTE_FILE_PATH=model.pte
```

Check the memory map in the build output:

```
Memory region         Used Size  Region Size  %age Used
           FLASH:      459668 B       512 KB     87.67%
             RAM:     1963160 B         2 MB     93.61%
       MODEL_DDR:     3541440 B        16 MB     21.11%
```

**Green flags:** all regions under 95%, model in the expected region.
**Red flags:** any region near 100%, orphan section warnings, overflow errors.

## Reference: existing board configs

| Board | Overlay | Conf | Notes |
|-------|---------|------|-------|
| Corstone-300 | `mps3_corstone300_fvp.overlay` | `mps3_corstone300_fvp.conf` | ISRAM for sram, DDR for model, reduced pools |
| Corstone-320 | `mps4_corstone320_fvp.overlay` | `mps4_corstone320_fvp.conf` | Shared ISRAM for flash+sram, DDR for model |
| Alif E8 | (no overlay needed) | (no conf needed) | MRAM holds model in-place, defaults work |

Use these as templates — copy the closest match and adapt the memory addresses and sizes.
