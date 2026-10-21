# QNN FullyConnected + LPBQ bias-drop repro

Debug material for D125031333 / T291919349.

QNN HTP sometimes drops the int32 bias of a `FullyConnected` whose weight uses
`QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION`, at graph prepare. The likely
cause: `Qnn_BlockwiseExpansion_t` has no length field, and for FullyConnected
QNN sizes `scaleOffsets` as `in_features` entries and `blocksScale8` as
`in_features * numBlocksPerAxis` bytes. The documented sizes use `axisSize`,
which is `dims[0]` = `out_features`. So QNN reads past both arrays.

`repro_guard.c` calls the QNN C API directly, with no ExecuTorch code. It builds
one graph:
- input `x`: `[8, 512]` uint16;
- `FullyConnected` with a `[32, 512]` int8/int4 weight, `axis = 0`, 16 blocks of
  32 and 4-bit block scales;
- an int32 bias;
- output: `[8, 32]` uint16.

It prepares the graph, then searches the context binary for the first two bias
values to tell whether the bias survived.

- `blob0.bin`: weight data, `32 x 512` int8, values in `[-7, 7]`.
- `blob1.bin`: bias data, 32 int32.

## Build

Use QNN headers and the x86 `libQnnHtp.so` from the same SDK. Seen on 2.37, 2.45,
2.48 and 2.51.

```
SDK=/path/to/qairt/2.48.x
g++ -std=gnu++17 -I $SDK/include/QNN -x c++ repro_guard.c -ldl -o repro_guard
```

## Run

```
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang
LIB=$SDK/lib/x86_64-linux-clang/libQnnHtp.so

# 1. scaleOffsets sized as documented (32 entries) and ending at a PROT_NONE
#    guard page: any read past entry 32 faults.
GUARD_SO=32 ./repro_guard $LIB 1    # Segmentation fault

# 2. Same, padded with zeros to in_features (512 entries): runs, bias kept.
GUARD_SO=512 ./repro_guard $LIB 1   # bias present in 1/1 context binaries

# 3. Plain heap arrays (what a normal client does): intermittent bias drop,
#    roughly 3-10% of fresh processes depending on heap layout.
for i in $(seq 300); do HEAP=1 ./repro_guard $LIB 1 | tail -1; done | sort | uniq -c
#    e.g. 34 "bias present in 0/1", 266 "bias present in 1/1" (2.48)

# 4. Fill the over-read region with random bits: the bias is dropped most of the time.
GUARD_SO=512 PAD=4 ./repro_guard $LIB 1   # bias present in 0/1 context binaries
```

`GUARD_BS=N` does the same for `blocksScale8`, with N bytes, where 512 is the
documented size. The smallest sizes that run without a fault are
`GUARD_SO=512` and `GUARD_BS=8192`. With `NBLK=8` (block 64), they are 512
and 4096. In other words, the count follows `in_features`, not
`out_features * numBlocksPerAxis`.

### Observed

| QNN | `HEAP=1`, bias dropped |
|---|---|
| 2.37 | 93 / 2000 |
| 2.45 | 119 / 2000 |
| 2.48 | 118 / 2000 |
| 2.51 | 50 / 1000 |

- The same graph with an `AXIS_SCALE_OFFSET` (per-channel) weight drops the bias
  0 times.
- Under valgrind memcheck, case 3 reports 510 invalid reads in
  `tensorCreateGraphTensor(weight)`, starting at the end of the 256 B
  `scaleOffsets` and 512 B `blocksScale8` allocations.
- Compiling on the device (aarch64 `libQnnHtp.so` 2.48, soc 95) reproduces it:
  15 / 300 with heap arrays, a SIGSEGV every run with the guard page.

## Other knobs

| Env | Default | Meaning |
|---|---|---|
| `SOC` / `ARCH` | 75 / 79 | HTP SoC model and arch for `deviceCreate` |
| `NBLK` | 16 | blocks per channel along `in_features` |
| `PAD` | 0 | `GUARD_SO` fill for entries past 32: 0 zero, 1 copy, 2 `0xff`, 4 random |
| `PADBS` | 1 | the same, for `GUARD_BS` |
| `REC_DIR` | `.` | directory containing `blob0.bin` / `blob1.bin` |
| `VTCM_MB` | 8 | graph VTCM size |

The second argument is the number of compiles to run in one process. A third
argument, a path prefix, dumps each context binary.
