# Arm CI

`arm.yml` runs the shared `_arm-trunk.yml` suite on `ciflow/trunk-arm/*` tags.
`trunk.yml` calls the same suite, preserving its 33 Arm, Zephyr, and Cortex-M
test entries. Ordinary PR CI and full trunk runs keep their existing coverage.

The Labeler workflow selects the CI label from the complete PR diff, including
deleted files and both paths of a rename:

| Change | Label |
| --- | --- |
| All files within `backends/arm/` and `examples/arm/` | `ciflow/trunk-arm` |
| Arm files plus any outside path | `ciflow/trunk` |
| Unavailable or incomplete file metadata | `ciflow/trunk` |
| PR head lacks either Arm workflow file | `ciflow/trunk` |
| No Arm files | No additional CI label |

An existing `ciflow/trunk` label always wins. The workflow never removes it,
preserving manual full-run requests. If a mixed PR becomes Arm-only, a
maintainer can remove that label; the next push recomputes the selection.
Obsolete `ciflow/trunk-arm` labels are removed.

The existing PyTorch bot handles label-to-tag creation and fork approval.
Labeling and tag updates are independent, so a scope transition can briefly
start both runs. This change requires no test-infra deployment.

Older PR heads fall back to full CI until they incorporate both `arm.yml`
and `_arm-trunk.yml`. After updating such a branch, remove its full-run
label to opt into narrow CI on the next push.
