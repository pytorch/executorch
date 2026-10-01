# PR Review Checklist

This checklist covers areas that CI cannot check. Skip items related to linting, formatting, type checking, and import ordering.

## Infrastructure

### GitHub Actions Storage

For every PR, determine whether it creates or expands GitHub Actions artifacts
or caches beyond what downstream jobs, releases, debugging, or users need.
Required, bounded storage is not a finding; unnecessary storage is.

- [ ] **Artifact contents** - Check changes to `actions/upload-artifact`, reusable workflow upload inputs, paths, globs, include/exclude filters, compression, and generated artifacts such as models, checkpoints, test outputs, and build outputs
- [ ] **Consumers and necessity** - Every stored artifact or cache has a concrete consumer. Flag uploads retained only as an intermediate side effect, debug output with no stated use, and files included because an upload path is broader than its consumer needs
- [ ] **Fan-out and duplication** - Check schedules, retries, triggers, and matrix dimensions that multiply uploads; flag identical or overlapping payloads stored by multiple jobs, shards, or upload steps
- [ ] **Retention and cache churn** - Artifact retention is no longer than its consumer requires, and cache keys do not create avoidable generations that are never restored
- [ ] **Indirect producers** - Check whether changed code makes directories or archives captured by an unchanged artifact upload larger or more numerous
- [ ] **Quantify the footprint** - When practical, estimate bytes per upload multiplied by job or matrix fan-out, run frequency, and retention. State assumptions rather than inventing precision
- [ ] **Treat waste as bugs** - Confirmed unnecessary GitHub Actions storage is a must-fix Infrastructure finding. Cite the responsible changed line, explain why the data is unnecessary and estimate its scale, then require the author to remove it, narrow the stored contents, deduplicate it, or shorten retention. It blocks approval and requires **Request Changes**. If storage is necessary and bounded, do not mention this check in the review
