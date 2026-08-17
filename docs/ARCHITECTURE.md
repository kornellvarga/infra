# Infra execution architecture

## Durable truth

`kornellvarga/infra` is the only durable source/history truth for this research project.

## Initial execution node

Valeria is the only execution node for the first phase. Its canonical checkout is:

`/home/kornel/repos/infra`

The canonical checkout is for synchronization and source inspection. Autonomous coding jobs and experiments must not mutate it directly.

## Workspace isolation

Every mutable run must use an isolated task/experiment workspace derived from the exact Git revision declared by the run. A run records at least:

- experiment/task id
- source revision
- model/runtime configuration
- hardware target
- start/end timestamps
- exit state
- compact metrics/result summary
- paths or hashes for external large artifacts when applicable

## Git vs external storage

Git stores source, experiment definitions, tests, compact summaries, and provenance. Model weights, datasets, caches, compiled model blobs, and large raw traces remain on the execution node or external artifact storage.

## Portability rule

Experiment definitions must not hard-code Valeria-specific GPU identities or absolute data/model paths. Node-local configuration resolves those resources so the same experiment can later run on consumer hardware without changing experiment semantics.

## Control boundary

Remote orchestration selects fixed-purpose project actions or declared experiment profiles. It must not accept arbitrary shell commands, arbitrary repository URLs, or arbitrary filesystem paths from remote requests.
