# MOM - Design Specification

Event-sourced memory kernel + retrieval engine for autonomous agents.

## Core Concepts

### Memory Scopes (where a memory "lives")

1. **Workspace** (team/product)
2. **Project** (repo/task)
3. **Agent** (one agent's private state)
4. **Conversation/Run** (single execution trace)

Each record includes `(tenant_id, workspace_id, project_id, agent_id, run_id)` where many fields can be null depending on scope.

### Memory Types

- **Event**: Raw facts ("user said X", "tool returned Y", "error Z")
- **Episode**: A clustered slice of events (task chunk)
- **Summary**: Condensed representation of episode/run
- **Long-term fact**: Durable structured claim (with provenance + confidence)
- **Preference/Policy**: "How to behave" items

### Trust & Provenance (Mandatory)

Every record has:

```rust
pub struct MemoryItem {
  pub id: String,
  pub scope: ScopeKey,          // tenant/workspace/project/agent/run
  pub kind: MemoryKind,         // Event | Episode | Summary | Fact | Preference
  pub time: DateTime<Utc>,
  pub content: Content,         // Text + optional JSON payload
  pub tags: Vec<String>,
  pub links: Vec<MemoryLink>,   // edges to other memories
  pub importance: f32,          // 0..1
  pub confidence: f32,          // 0..1
  pub ttl: Option<Duration>,
  pub source: Source,           // user/tool/model/system
  pub integrity_hash: [u8; 32], // SHA256
}
```

## Storage Architecture

### Pluggable Backend Trait

```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
  async fn put(&self, item: &MemoryItem) -> Result<()>;
  async fn batch_put(&self, items: &[MemoryItem]) -> Result<()>;
  async fn get(&self, id: MemoryId) -> Result<Option<MemoryItem>>;
  async fn query(&self, q: Query) -> Result<Vec<Scored<MemoryItem>>>;
  async fn delete(&self, id: MemoryId) -> Result<()>;
}
```

Implementations:
- **SQLite** (default, local/dev)
- **Postgres** (scale-out + pgvector)
- Custom (S3 + metadata DB, etc.)

### Indexing Layers (Composable)

1. **Vector index** (semantic): Embeddings of content
2. **BM25 / full-text** (lexical): Keyword matching
3. **Graph edges**: Causal, derived_from, contradicts, same_as

**Ranking**:
```
score = w_v*vector + w_l*lexical + w_t*time_decay + w_i*importance + w_s*source_trust
```

## Retrieval API: Context Packs

Return tailored context (no manual stitching):

```rust
pub struct ContextPack {
  pub highlights: Vec<MemoryItem>,   // Top K recent
  pub summaries: Vec<MemoryItem>,    // Mid/long
  pub facts: Vec<MemoryItem>,        // Durable
  pub citations: Vec<Citation>,      // Provenance
}
```

Agents call:
- `recall(query, scope, budget_tokens, filters)`
- `recall_for_tool(tool_name, scope, schema_hash)`
- `recall_for_planning(goal, constraints)`

## HTTP Endpoints (axum)

```
POST   /v1/memory                 # Write event/summary/fact
POST   /v1/recall                 # Hybrid query → ContextPack
POST   /v1/consolidate            # Run summarization
GET    /v1/memory/{id}            # Get specific item
POST   /v1/links                  # Create graph edges
POST   /v1/policy                 # Set retention/privacy
```

### Payload Compression & Decompression

All HTTP endpoints support transparent payload compression and decompression:
- **Request Decompression**: Request bodies compressed with `gzip` or `zstd` are transparently decompressed when clients provide the `Content-Encoding` header (e.g. `Content-Encoding: gzip`).
- **Response Compression**: Response payloads are compressed with `gzip` or `zstd` when clients send the corresponding `Accept-Encoding` header (e.g. `Accept-Encoding: gzip, zstd`). Uncompressed clients receive plain JSON as normal.


## Hierarchical Consolidation Pipeline

1. **Capture**: Append raw events
2. **Chunk + Episode builder**: Group events by run + time gaps
3. **Summarizer**: Produce summaries when thresholds hit
4. **Fact distiller**: Extract durable facts with provenance
5. **Garbage collection**: TTL expiry + forgetting policies

Key: Never throw away raw events by default; stop retrieving them (cold storage).

## SQLite Schema (Starter)

```sql
CREATE TABLE memory_items (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  workspace_id TEXT,
  project_id TEXT,
  agent_id TEXT,
  run_id TEXT,
  kind TEXT,           -- event|episode|summary|fact|preference
  time DATETIME DEFAULT CURRENT_TIMESTAMP,
  content_text TEXT,
  content_json JSON,
  tags JSON,
  links JSON,
  importance REAL,
  confidence REAL,
  source TEXT,         -- user|tool|model|system
  integrity_hash TEXT,
  expires_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memory_embeddings (
  item_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  dims INTEGER,
  vector BLOB NOT NULL,
  FOREIGN KEY(item_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

CREATE TABLE memory_tags (
  item_id TEXT,
  tag TEXT,
  FOREIGN KEY(item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
  PRIMARY KEY (item_id, tag)
);

CREATE TABLE memory_links (
  src_id TEXT,
  dst_id TEXT,
  rel TEXT,  -- causal|derived_from|contradicts|same_as
  weight REAL,
  FOREIGN KEY(src_id) REFERENCES memory_items(id) ON DELETE CASCADE,
  FOREIGN KEY(dst_id) REFERENCES memory_items(id) ON DELETE CASCADE,
  PRIMARY KEY (src_id, dst_id, rel)
);

CREATE TABLE memory_audit (
  id INTEGER PRIMARY KEY,
  item_id TEXT,
  action TEXT,         -- create|update|delete
  actor TEXT,
  time DATETIME DEFAULT CURRENT_TIMESTAMP,
  details_json JSON,
  FOREIGN KEY(item_id) REFERENCES memory_items(id) ON DELETE SET NULL
);
```

## Security & Multi-Tenancy

- **Tenant isolation**: Mandatory filter at query layer
- **Field-level encryption**: Optional (encrypt content at rest)
- **PII redaction**: Policy-driven hook on write
- **Signed integrity chain**: Detect tampering

## Phase 1 MVP

1. SQLite store + migrations
2. Axum HTTP API: `POST /v1/memory`, `POST /v1/recall`
3. Hybrid recall: Lexical FTS + pluggable embeddings
4. Episode summarization: Manual trigger
5. TypeScript/Bun client wrapper

## Phase 2

- Vector embedding support (OpenAI, local, etc.)
- BM25 full-text indexing
- Episode building automation
- Fact distillation
- Postgres backend
- ACL support

## Phase 3

- Graph edges + queries
- TTL + garbage collection
- MCP server bridge
- gRPC (tonic) service
- Production observability

## Building Blocks

### Use Existing Baselines

- **Mem0**: Importance scoring, budgeting, retrieval
- **MemoryOS**: Hierarchical storage, consolidation
- **LangChain**: Agent integration patterns

### TypeScript/Bun Integration

Optional SDK:

```typescript
import { MomClient } from '@lornu/mom';

const mom = new MomClient({ baseUrl: 'http://localhost:8080' });

await mom.write({
  kind: 'event',
  content: 'User asked...',
  tags: ['deployment'],
  scope: { agentId: 'agent-1' }
});

const context = await mom.recall({
  query: 'deployment patterns',
  budgetTokens: 4096
});
```

### MCP Server (Tool Bridge)

```
mom_write(scope, kind, content, tags)
mom_recall(query, scope, budget)
mom_search(text, scope)
mom_consolidate(scope)
```

---

**Service Choice**: axum (HTTP) for MVP, tonic (gRPC) in Phase 3.
