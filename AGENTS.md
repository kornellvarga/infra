# infra agent rules

## Durable truth

- `kornellvarga/infra` is the canonical source/history truth.
- Valeria's `/home/kornel/repos/infra` checkout is a synchronized canonical local clone, not a scratch workspace.

## Workspace isolation

- Mutable agent and experiment work must run in isolated workspaces/worktrees derived from a declared source revision.
- Do not edit, reset, merge, rebase, or otherwise mutate the canonical Valeria checkout during an experiment.
- Preserve unrelated or unpublished work; fail closed rather than normalizing a dirty checkout destructively.

## Experiment provenance

Every accepted run should identify its source revision, configuration, model/runtime identity, hardware target, timestamps, exit state, and compact result summary.

## Storage boundary

Do not commit model weights, large datasets, caches, compiled model blobs, or large raw traces. Store reproducible definitions and compact accepted results in Git; keep large artifacts external and reference them by path/hash/manifest where useful.

## Portability

Experiment semantics must not depend on Valeria-specific absolute model/data paths or GPU numbers. Resolve those through node-local configuration so the same experiment can later be repeated on consumer hardware.
