# Arm CI

`arm.yml` runs the shared `_arm-trunk.yml` suite on `ciflow/trunk-arm/*` tags.
`trunk.yml` calls the same suite of Arm, Zephyr, and Cortex-M tests.
Both workflows use the name `trunk` so their job names match for Dr. CI;
the narrow workflow's run title is `Arm trunk (<ref>)`. Extracting the suite
adds `Arm suite /` to earlier job names, so those older runs will not match
automatically.

The Labeler workflow selects the CI label from the complete PR diff, including
deleted files and both paths of a rename. It retries changing snapshots before
falling back, and retries transient API errors twice.

| Change | Label |
| --- | --- |
| All files within `backends/arm/` and `examples/arm/` | `ciflow/trunk-arm` |
| Arm files plus any outside path | `ciflow/trunk` |
| Incomplete metadata with observed Arm paths or `ciflow/trunk-arm` | `ciflow/trunk` |
| Incomplete metadata without Arm evidence or either trunk label | Fail selection without changing labels; re-run or request full CI |
| PR head lacks either Arm workflow file | `ciflow/trunk` |
| No Arm files, including an empty diff | No additional CI label |

An existing `ciflow/trunk` label always wins. The workflow never removes it,
preserving manual full-run requests. If a mixed PR becomes Arm-only, a
maintainer can remove that label; the next push recomputes the selection.
`ciflow/trunk-arm` is managed automatically and is removed when the PR no
longer has Arm paths. Generated docs and CI workflow files remain outside the
two-directory rule. Other module and Docker labels are updated even if Arm
selection fails.

The existing PyTorch bot handles label-to-tag creation and fork approval.
Labeling and tag updates are independent. During a scope transition, both the
narrow run and the full trunk run can execute the entire Arm suite to completion.

Older PR heads fall back to full CI until they incorporate both `arm.yml`
and `_arm-trunk.yml`. After updating such a branch, remove its full-run
label to opt into narrow CI on the next push.
