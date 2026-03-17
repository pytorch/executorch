# Questions / Assumptions Log

This file records open design questions and assumptions without blocking implementation.

## Open Questions
1. Should new observatory become default import path in examples immediately, or remain opt-in while old `debugging_utils` stays primary?
2. For GraphView compare, should `max_parallel > 2` be hard-rejected in v1 minimal or accepted and clipped to 2 in UI?
3. For ETRecord auto-collect naming, should method labels include source (`Exported`, `Edge`, `Extra`) or keep a flat naming convention for continuity with old reports?

## Current Assumptions
1. New infra remains opt-in to avoid regression risk; old infra remains untouched.
2. Compare defaults use RFC compact profile and allow only 2 panes in first pass.
3. Auto-collected ETRecord records are prefixed to keep source traceable and review-friendly.
4. Minimal migration includes only metadata + stack trace lenses plus GraphLens built-in.

## Follow-up (non-blocking)
- If reviewers request strict old naming compatibility, rename collect labels in one small follow-up commit.
- If reviewers want default adoption, switch imports in selected examples in a dedicated commit.
