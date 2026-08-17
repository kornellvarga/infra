# infra

Research project for exploring how far a coding agent can approach frontier-class software-engineering capability using locally executed models, tools, memory, verification, and inference-time search.

## Architecture

- **GitHub (`kornellvarga/infra`)** is the canonical source/history truth.
- **Valeria** is the initial execution and experiment machine.
- `/home/kornel/repos/infra` is Valeria's canonical local clone; experiments and autonomous coding work must run in isolated workspaces derived from it.
- Large model weights, datasets, caches, and raw experiment artifacts stay outside Git. Git stores code, experiment definitions, compact result summaries, and provenance.

## Initial goals

1. Build a reproducible local coding-agent experiment harness.
2. Establish baseline tasks and measurements before training or fine-tuning anything.
3. Compare raw model capability against increasingly capable harnesses: tools, verification, retries, memory, and reviewer/critic passes.
4. Keep experiment definitions hardware-portable so the same run can later be repeated on consumer hardware.

See `docs/ARCHITECTURE.md` for the execution contract.