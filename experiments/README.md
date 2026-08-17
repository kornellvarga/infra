# Experiments

Each experiment should be reproducible from a declared Git revision plus a small configuration file.

A run definition should identify at least:

- unique experiment id
- task/benchmark set
- model/runtime id and quantization when relevant
- context and inference settings
- harness features enabled (tools, verifier, retry, memory, reviewer, etc.)
- resource requirements expressed portably
- run budget / stopping conditions
- metrics to collect

Experiment definitions must not contain arbitrary remote shell commands or node-specific absolute model/data paths.
