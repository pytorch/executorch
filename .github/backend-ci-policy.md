# Backend CI coverage

QNN, Arm (including VGF), XNNPACK and Vulkan use full matrices for relevant
backend and shared export/runtime/test/build/dependency changes. Unrelated PRs
and unsampled main pushes use the reduced matrices. The existing CI run
decision supplies the main sample; its rules are not modified.

| Event | Matrix |
| --- | --- |
| Relevant PR or main-push change | Full |
| Unrelated PR | Reduced |
| Unrelated sampled main push | Full |
| Unrelated unsampled main push | Reduced |
| Schedule, existing release/tag/manual run | Full, as before |
| Unknown/unavailable change information | Full |

Git diffs include deleted and renamed paths and do not have the API's 300-file
limit. Callers without a reduced matrix continue to use their full flow list.
This change does not alter release publication or viable/strict rules.
