# ESP-NN runner

ESP-IDF application that runs a test bundle on an Espressif board, checks every
model against the host (bit-exact for int8 graphs, bit-exact or SQNR for
ExecuTorch programs with CPU operators) and times it.

Bundles are created by `../aot_espnn_compiler.py` (one model) or
`../scripts/make_test_bundles.py` (the backend's operator test models):

| Bundle | Contents | Needs |
|---|---|---|
| `.xtb` (XTB1) | XTS3 int8 graphs run by the standalone `espnn_runtime` interpreter | nothing else |
| `.xtp` (XTP1) | ExecuTorch `.pte` programs, ESP-NN delegate plus portable CPU kernels | ExecuTorch cross-compiled for the chip, `CONFIG_ESPNN_EXECUTORCH_DELEGATE=y` |

## Build and run

From an activated ESP-IDF v5.1+ shell:

```bash
cd examples/espressif/espnn_runner
python ../scripts/make_test_bundles.py                      # bundles/op_suite.xtb, bundles/pte_suite.xtp

# Standalone int8 graphs (any ESP32 chip)
idf.py set-target esp32s3                                   # or esp32
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults" menuconfig  # ESP-NN runner -> bundle path
idf.py -p /dev/ttyACM0 flash monitor

# ExecuTorch programs (needs ../scripts/build_executorch_esp.sh esp32s3 first)
idf.py -B build-et -DSDKCONFIG=build-et/sdkconfig \
    -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.executorch" \
    -DESPNN_EXECUTORCH_BUILD_DIR=$EXECUTORCH_ROOT/cmake-out-esp-esp32s3 build flash monitor
```

`../scripts/run_bundle.py BUNDLE --port PORT` does the same (target detection,
build, flash, capture) and prints a summary.

Output, one line per model:

```
CHIP target=esp32s3 cores=2 revision=2 cpu_mhz=240 flash_mb=16 psram_kb=8192 ...
SUITE CASE 1/11 name=conv-act-pool-linear ops=9 ... exact=3 ... ansi=1 us=2444 min_us=... status=PASS
ET CASE 5/7 name=tiny-transformer ... delegate_us=2288 cpu_us=6040 sqnr_db=41.6 ... status=PASS
ESPNN RUNNER: PASS
```

`sdkconfig.defaults.esp32s3` assumes 16 MB flash and octal PSRAM (for example
ESP32-S3-DevKitC-1 N16R8). For other modules change the flash size, partition
table and PSRAM mode. Boards without PSRAM still boot (`CONFIG_SPIRAM_IGNORE_NOTFOUND`)
and run models whose activations fit in internal RAM.
