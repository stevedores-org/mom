# MOM gRPC Integration Guide

This guide describes how to integrate with the MOM gRPC service (on port `50051`) for memory store operations. gRPC is the recommended protocol for agent operations requiring low latency and high throughput.

## Schema & Proto Definition

The official protobuf schema is defined in [protos/memory.proto](../protos/memory.proto).

### Core Service Methods

- **`Write(MemoryItem) -> MemoryId`**
  Writes a single memory item to the store.
- **`Get(MemoryId) -> MemoryItem`**
  Retrieves a single memory item by ID (enforcing tenant-aware isolation if a scope is provided).
- **`Query(QueryRequest) -> Stream<ScoredMemoryItem>`**
  Queries memories with lexical and metadata filters.
- **`Delete(MemoryId) -> Empty`**
  Deletes a memory item.
- **`Recall(QueryRequest) -> Stream<ScoredMemoryItem>`**
  Recall context using hybrid/semantic search.
- **`BulkWrite(stream MemoryItem) -> Stream<MemoryId>`**
  Bidirectional streaming to write a batch of memory items.
- **`BulkGet(stream MemoryId) -> Stream<MemoryItem>`**
  Bidirectional streaming to retrieve a batch of memory items.

---

## Configuration

The gRPC server runs alongside the Axum HTTP server and is exposed on port `50051`.
Ensure the environment variables are set correctly (e.g., `MOM_DB_PATH` for SurrealDB connection).

### gRPC Reflection

gRPC Reflection is enabled on port `50051` to allow visual inspection and debugging using tools like `grpcurl` or `Postman`.

Example listing services via `grpcurl`:
```bash
grpcurl -plaintext localhost:50051 list
# Output:
# grpc.reflection.v1alpha.ServerReflection
# memory.MemoryStoreService
```

Example querying via `grpcurl`:
```bash
grpcurl -plaintext -d '{"id": "doc-123", "scope": {"tenant_id": "example-tenant"}}' \
  localhost:50051 memory.MemoryStoreService/Get
```

---

## Client Examples

### Rust Client (Tonic)

To run the Rust example client:
```bash
cargo run --example grpc_client --manifest-path crates/mom-grpc/Cargo.toml
```

See the source file at [crates/mom-grpc/examples/grpc_client.rs](../crates/mom-grpc/examples/grpc_client.rs).

### Go Client (gRPC)

See the source file template at [examples/go_client/client.go](../examples/go_client/client.go).

---

## Benchmarks

A performance benchmark script is included to compare gRPC vs HTTP write latency and throughput.

To run the benchmark:
1. Ensure the MOM server is running (`cargo run -p mom-service`).
2. Run the benchmark tool:
   ```bash
   cargo run --example grpc_benchmark --manifest-path crates/mom-grpc/Cargo.toml
   ```

Typical benchmark comparison output shows gRPC with up to **2-3x throughput** and significantly **lower latency** than HTTP due to HTTP header overhead and connection reusing:

```text
=== Write Performance Comparison (100 runs) ===
Protocol | Avg Latency | p95 Latency | Throughput (req/sec)
---------|-------------|-------------|---------------------
gRPC     |       1.24ms |       2.10ms |          804.2 rps
HTTP     |       4.15ms |       6.80ms |          240.5 rps
===================================================
```
