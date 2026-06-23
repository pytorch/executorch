# XNNPACK Model Tests

## Optional Pytest Perf Stage

Model tests can opt in to an optional latency check after correctness
comparison:

```python
.run_method_and_compare_outputs(xnnpack_perf=True)
```

The perf stage is skipped by default. Enable it only for runs where local
latency comparison is useful.

### Commands

Run the normal correctness test with no perf stage:

```bash
python3 -m pytest -q \
backends/xnnpack/test/models/resnet.py::TestResNet18::test_fp32_resnet18
```

Run correctness plus perf comparison against the recorded result:

```bash
EXECUTORCH_XNNPACK_PYTEST_PERF=1 \
python3 -m pytest -q \
backends/xnnpack/test/models/resnet.py::TestResNet18::test_fp32_resnet18
```

Record or update the local perf result:

```bash
EXECUTORCH_XNNPACK_PYTEST_PERF=1 \
EXECUTORCH_XNNPACK_PYTEST_PERF_UPDATE=1 \
python3 -m pytest -q \
backends/xnnpack/test/models/resnet.py::TestResNet18::test_fp32_resnet18
```

Use more samples for a less noisy result:

```bash
EXECUTORCH_XNNPACK_PYTEST_PERF=1 \
EXECUTORCH_XNNPACK_PYTEST_PERF_RUNS=50 \
EXECUTORCH_XNNPACK_PYTEST_PERF_WARMUP_RUNS=10 \
python3 -m pytest -q \
backends/xnnpack/test/models/resnet.py::TestResNet18::test_fp32_resnet18
```

### Environment Variables

| Variable                                       | Default | Meaning                                          |
| ---------------------------------------------- | ------: | ------------------------------------------------ |
| `EXECUTORCH_XNNPACK_PYTEST_PERF`               |     off | Enables the optional perf stage.                 |
| `EXECUTORCH_XNNPACK_PYTEST_PERF_UPDATE`        |     off | Records the current result instead of comparing. |
| `EXECUTORCH_XNNPACK_PYTEST_PERF_RUNS`          |    `10` | Number of timed runs.                            |
| `EXECUTORCH_XNNPACK_PYTEST_PERF_WARMUP_RUNS`   |     `2` | Number of warmup runs.                           |
| `EXECUTORCH_XNNPACK_PYTEST_PERF_THRESHOLD_PCT` |  `10.0` | Allowed slowdown before comparison fails.        |

Boolean variables accept values such as `1`, `true`, `yes`, and `on`.

The perf stage forces the ExecuTorch threadpool to one thread while timing,
then restores the previous thread count after the run.

### Results

Perf results are keyed by the pytest test name and host/runtime identity. For
example, the ResNet test writes:

```text
backends/xnnpack/test/models/resnet_pytest_perf_results.json
backends/xnnpack/test/models/resnet_pytest_perf_results.md
```

The JSON file is the complete machine-readable record used for comparisons.
It includes samples, host details, runtime details, linked-symbol hints, and
the recorded latency metrics.

The Markdown file is a compact human-readable summary for review. It includes
the host, processor, expected hardware path, thread count, mean latency,
median latency, coefficient of variation, and recorded time.

`Expected HW path` is deduced from host ML features and linked runtime symbols.
It is not proof of the exact runtime microkernel selected for every operator.
