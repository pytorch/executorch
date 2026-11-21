# Muse Glimmer MLX DFlash tests

The native suite loads serialized PTEs through `MuseGlimmerMLXExecutor::create`.
Controlled fixtures use small hidden dimensions and a 200,096-token vocabulary;
no model download or tokenizer is required. The export script generates a synthetic
GGUF in its output directory with Q4_K/Q6_K weights and the production DFlash2
block size, selector rank, top-k, and 202,048-token vocabulary. Layer count,
hidden width, and context are reduced; it does not measure model quality or
acceptance rate.

After `make muse-glimmer-batching-mlx`:

```bash
python -m examples.models.muse_glimmer.tests.export_mlx_dflash_fixture /tmp/muse-glimmer-mlx-fixtures
cmake -S examples/models/muse-glimmer -B cmake-out/examples/models/muse-glimmer-batching \
  -DMUSE_GLIMMER_BUILD_BATCHING_TESTS=ON \
  -DMUSE_GLIMMER_MLX_TEST_FIXTURES=/tmp/muse-glimmer-mlx-fixtures
cmake --build cmake-out/examples/models/muse-glimmer-batching \
  --target muse_glimmer_mlx_dflash_test sampling_test
ctest --test-dir cmake-out/examples/models/muse-glimmer-batching \
  -R '^muse_glimmer_sampling$' --output-on-failure --no-tests=error
ctest --test-dir cmake-out/examples/models/muse-glimmer-batching \
  -R '^muse_glimmer_mlx_dflash$' --output-on-failure --no-tests=error
```

Random and quantized PTE fixtures compare draft numerics against eager references.
Controlled rejection and cache-lifecycle checks assert expected tokens and
acceptance counts; continuation checks compare with target-only execution,
including legacy artifacts. Seeded continuations are compared across
prompt chunking and mixed prefill/decode batches for both draft formats.
Routing-sensitive fixtures also compare per-round tokens and acceptance counts
between solo and packed execution with unequal prompts and reordered owners.
Input-sensitive continuations check shared-payload slices against target-only decoding.
A vendored stateless official reference separately checks DFlash2 model semantics,
including sliding-window boundaries. Native sampling checks target-only,
original DFlash, and DFlash2 at temperature 0.8 with top-k/top-p filtering;
a separate temperature control detects ignored settings. The MLX CI job runs
these fixtures; backend cache and primitive suites cover window semantics and TF32.
