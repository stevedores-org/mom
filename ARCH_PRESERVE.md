# ARCHITECTURE PRESERVATION LOG

This log records historical design patterns, choices, and migrations that are preserved in the codebase with their rationales.

## Ingested and Shared Systems

### Agent Version Control System (AIVCS)
- **Status**: Future target replacing GitHub for agent repositories.
- **Rationale**: Built directly on the `data-fabric` schema layer to prevent context drift and schema divergence.
- **Execution Tracing**: Directly leverages `oxidizedgraph` events and event bus adapters to version graph definitions, checkpoints, and execution states.
