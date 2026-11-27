# NXP Neutron backend tests

## Selecting the Neutron target

One pytest run tests exactly one Neutron target:

```bash
pytest -c /dev/null --strict-markers backends/nxp/tests/ --neutron-target=imx95
```

The target is taken from, in this order:

1. the `--neutron-target=<target>` option,
2. the `NEUTRON_TARGET` env variable (an empty value counts as unset),
3. the default `imxrt700`.

An unknown target is an error, it never falls back to the default.

### Targets and architecture markers

Every target belongs to an architecture, and every architecture has a pytest marker:

| Target     | Marker             | Architecture            |
|------------|--------------------|-------------------------|
| `imxrt700` | `neutron_c`        | Neutron-C (MCU)         |
| `imx943`   | `neutron_s_single` | single-core Neutron-S   |
| `imx95`    | `neutron_s_multi`  | multi-core Neutron-S    |

The mapping lives in `TARGET_TO_MARKER` in `target_selection.py`. Adding a target is one new entry there. A new
architecture also needs a marker description in `ARCHITECTURE_MARKER_DESCRIPTIONS`.

## Marker policy

- Markers are inclusive: they list the architectures a test is eligible for. They can be stacked, and module level
  `pytestmark = [...]` is allowed.
- Tests not eligible for the selected target are **deselected**, not skipped.
- **Interim rule:** a test without an architecture marker runs only on the default target `imxrt700`. This covers
  both Neutron-C specific and target-agnostic tests for now. Their classification is EIEX-1251.
- A test with any architecture marker (`neutron_c`, `neutron_s_single`, `neutron_s_multi`) must take the `target`
  fixture. Only unmarked tests may omit it. Otherwise collection fails, on every run including the default one,
  so that a marked test can never silently run with the default target.
- Do not select targets or architectures with `-m` (e.g. `-m neutron_c` or `-m imx95`). This is an error. Use
  `--neutron-target` instead. Other markers in `-m` work as usual.
- Run with `--strict-markers` (as `run_unittests.sh` does) so that a typo in a marker name fails.

## Enabling a test for more targets

Existing tests stay unmarked and keep using the `imxrt700` default of the helpers. To enable a test for more
architectures:

1. mark it with every architecture it is eligible for,
2. add the `target` fixture argument (the name of the selected target, e.g. `"imx95"`),
3. pass it to every helper that takes a target (`to_quantized_edge_program`, `to_quantized_executorch_program`,
   `lower_run_compare`, `NeutronTargetSpec`, ...).

```python
@pytest.mark.neutron_c
@pytest.mark.neutron_s_single
@pytest.mark.neutron_s_multi
def test_my_operator(target: str):
    edge_program = to_quantized_edge_program(model, input_shape, target=target)
```

`generic_tests/test_neutron_s.py::test_to_quantized_edge_program__selected_target` is a working example.

`unittest.TestCase` tests cannot take pytest fixtures. Convert them to plain pytest tests before enabling them for
other targets.

## IDE runs

When running tests from an IDE for a non-default target, set `NEUTRON_TARGET=<target>` in the environment of the run
configuration, or add `--neutron-target=<target>` to its pytest arguments.

## Tests of the selection logic

`generic_tests/test_target_selection.py` checks the selection logic itself in separate pytest sessions (default,
env variable, option precedence, invalid values, eligibility, deselection, the `-m` and `target` fixture guards,
and `--strict-markers`).
