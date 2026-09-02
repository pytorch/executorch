# XNNPACK Pytest Perf Results

Generated from `resnet_pytest_perf_results.json`.

| Test                                                               | Host                 | Processor    | ML features                             | Expected HW path | Threads | Mean ms | Median ms |  CV % | Recorded             |
| ------------------------------------------------------------------ | -------------------- | ------------ | --------------------------------------- | ---------------- | ------: | ------: | --------: | ----: | -------------------- |
| xnnpack.models.resnet.testresnet18.test_fp32_resnet18.68745d2d8d5d | Darwin arm64 Mac15,6 | Apple M3 Pro | fp, asimd, bf16, asimddp, fphp, i8mm    | kleidiai+neon    |       1 |  41.686 |    41.679 | 0.247 | 2026-06-25 09:24 UTC |
| xnnpack.models.resnet.testresnet18.test_fp32_resnet18.68745d2d8d5d | Linux aarch64        | aarch64      | fp, asimd, asimddp, asimdhp, bf16, fphp | kleidiai+neon    |       1 |  41.900 |    41.906 | 1.108 | 2026-06-26 13:41 UTC |
