# Memory Graph Relationships (US-11)

MOM supports explicit semantic edges between memories for traversal, causality chains, and conflict detection.

## Relationship types

| Type | Semantics | Example |
|---|---|---|
| `causal` | A caused B | deployment error → outage |
| `derived_from` | B was derived from A | summary derived from events |
| `contradicts` | A and B conflict | two facts with incompatible values |
| `same_as` | A and B are equivalent | duplicate canonical fact |
| `references` | A references B | decision references prior analysis |

Each link stores:

- `weight` (0..1) — edge strength for ranking traversals
- `confidence` (0..1) — confidence in the relationship claim

## HTTP API

All endpoints require `tenant_id` as a query parameter (see [TENANT_ISOLATION.md](./TENANT_ISOLATION.md)).

### Create link

```http
POST /v1/links?tenant_id=acme
Content-Type: application/json

{
  "src_memory_id": "event-a",
  "dst_memory_id": "event-b",
  "rel": "causal",
  "weight": 0.9,
  "confidence": 0.95
}
```

Both memories must exist in the tenant scope.

### Update link metadata

```http
PATCH /v1/links/{link_id}?tenant_id=acme
Content-Type: application/json

{ "weight": 0.8, "confidence": 0.9 }
```

### Delete link

```http
DELETE /v1/links/{link_id}?tenant_id=acme
```

### Traverse graph

Breadth-first traversal from a starting memory, optionally filtered by relationship type:

```http
GET /v1/links/traverse?tenant_id=acme&from=event-a&rel=causal&depth=5
```

Returns ordered steps with depth and the link used to reach each memory.

### Conflict detection

Find `contradicts` edges tenant-wide or for a specific memory:

```http
GET /v1/links/conflicts?tenant_id=acme
GET /v1/links/conflicts?tenant_id=acme&memory_id=fact-1
```

## Storage

Links are stored in SurrealDB table `memory_links` with tenant-scoped indexes on source, destination, and relationship type. Traversal runs in-process via BFS with a configurable depth cap (max 32).

## Future work

- LLM-driven relationship discovery
- SurrealDB native graph `RELATE` edges for index-free traversals at scale
