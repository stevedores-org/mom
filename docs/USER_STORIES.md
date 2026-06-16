# MOM User Stories

Event-sourced memory kernel for autonomous agents. Complete list of features organized by phase and priority.

**Total: 23 user stories** across Phase 1 (MVP), Phase 2 (Enhancement), and Phase 3 (Advanced).

---

## Phase 1: MVP (Core Functionality)

Essential memory operations for agents.

### US-1: Write Event Memory
- **Goal**: Agents write event memories (observations, decisions)
- **Endpoint**: `POST /v1/memory`
- **Status**: 🟢 Ready to implement
- **Complexity**: Low
- **Issue**: [#4](https://github.com/lornu-ai/mom/issues/4)

### US-2: Retrieve Specific Memory
- **Goal**: Get a known memory by ID
- **Endpoint**: `GET /v1/memory/:id`
- **Status**: 🟢 Ready to implement
- **Complexity**: Low
- **Depends on**: US-1
- **Issue**: [#5](https://github.com/lornu-ai/mom/issues/5)

### US-3: List & Filter Memories
- **Goal**: Browse memory history with filters (scope, kind, tags, time)
- **Endpoint**: `GET /v1/memory`
- **Status**: 🟢 Ready to implement
- **Complexity**: Medium
- **Depends on**: US-1
- **Issue**: [#6](https://github.com/lornu-ai/mom/issues/6)

### US-4: Recall Memories (Lexical Search)
- **Goal**: Search memories by text query
- **Endpoint**: `POST /v1/recall`
- **Status**: 🟢 Ready to implement
- **Complexity**: Medium
- **Depends on**: US-1
- **Scoring**: Importance + recency + text match
- **Issue**: [#7](https://github.com/lornu-ai/mom/issues/7)

### US-5: Delete Memory
- **Goal**: Remove a memory by ID
- **Endpoint**: `DELETE /v1/memory/:id`
- **Status**: 🟢 Ready to implement
- **Complexity**: Low
- **Depends on**: US-1
- **Issue**: [#8](https://github.com/lornu-ai/mom/issues/8)

### US-6: Agent-Scoped Memories
- **Goal**: Isolate memories to specific agents
- **Scoping**: Tenant > Workspace > Project > Agent > Run
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Security**: Critical
- **Issue**: [#9](https://github.com/lornu-ai/mom/issues/9)

### US-7: Multi-Tenant Isolation
- **Goal**: Enforce strict tenant separation
- **Database**: Constraints at SurrealDB layer
- **Status**: ✅ Implemented (#11 closed 2026-06-16)
- **Complexity**: Medium
- **Security**: Critical
- **Issue**: [#11](https://github.com/lornu-ai/mom/issues/11)

### US-13: TypeScript/Bun Client SDK
- **Goal**: Simple SDK for TS/Node.js integration
- **Package**: `@lornu/mom`
- **Transport**: Standard `fetch` API
- **Status**: 🟢 Ready to implement
- **Complexity**: Low
- **Issue**: [#18](https://github.com/lornu-ai/mom/issues/18)

### US-14: Context Packs for Agents
- **Goal**: Return structured memory bundles with budgeting
- **Endpoint**: `POST /v1/context-pack`
- **Returns**: highlights, summaries, facts, citations
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Issue**: [#19](https://github.com/lornu-ai/mom/issues/19)

---

## Phase 2: Enhancement (Advanced Features)

Features that build on MVP foundation.

### US-8: Vector Embeddings & Semantic Search
- **Goal**: Store embeddings and search by semantic similarity
- **Trait**: `Embedder` (pluggable providers)
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Blocks**: Hybrid search, vector recall
- **Issue**: [#13](https://github.com/lornu-ai/mom/issues/13)

### US-9: Memory Consolidation (Summarization)
- **Goal**: Auto-summarize old events into durable summaries
- **Endpoint**: `POST /v1/consolidate`
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Future**: LLM-powered summarization
- **Issue**: [#14](https://github.com/lornu-ai/mom/issues/14)

### US-10: Durable Facts & Preferences
- **Goal**: Store learned facts with provenance and confidence
- **Kind**: `Fact` and `Preference` memory types
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Issue**: [#15](https://github.com/lornu-ai/mom/issues/15)

### US-11: Memory Graph Relationships
- **Goal**: Link memories with semantic relationships
- **Types**: causal, derived_from, contradicts, same_as, references
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Issue**: [#16](https://github.com/lornu-ai/mom/issues/16)

### US-12: Retention Policies & TTL
- **Goal**: Auto-expire or soft-delete old memories
- **Endpoint**: `POST /v1/policy`
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Job**: Background cleanup/archival
- **Issue**: [#17](https://github.com/lornu-ai/mom/issues/17)

### US-15: MCP Server Bridge
- **Goal**: Expose MOM as Model Context Protocol tools
- **Tools**: mom_write, mom_recall, mom_search, mom_consolidate
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Issue**: [#20](https://github.com/lornu-ai/mom/issues/20)

### US-16: Observability & Metrics
- **Goal**: Monitor performance and debug issues
- **Export**: OpenTelemetry metrics
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Issue**: [#21](https://github.com/lornu-ai/mom/issues/21)

### US-17: Authentication & Authorization
- **Goal**: Secure with JWT and RBAC
- **Transport**: Bearer tokens, role-based access control
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Security**: Critical
- **Issue**: [#22](https://github.com/lornu-ai/mom/issues/22)

### US-18: Docker & Deployment
- **Goal**: Deploy in containers, Kubernetes
- **Artifacts**: Dockerfile, docker-compose.yml, Helm charts
- **Status**: 🟡 Design phase
- **Complexity**: Low-Medium
- **Issue**: [#23](https://github.com/lornu-ai/mom/issues/23)

### US-19: Batch Operations & Optimization
- **Goal**: Efficient bulk write, delete, query
- **Endpoints**: `/memory/batch`, `/memory/batch/delete`
- **Status**: 🟡 Design phase
- **Complexity**: Medium
- **Performance**: 1000 items < 1 sec
- **Issue**: [#24](https://github.com/lornu-ai/mom/issues/24)

### US-21: data-fabric Integration
- **Goal**: Import agent decisions and modifications from data-fabric
- **Direction**: data-fabric → MOM
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Issue**: [#26](https://github.com/lornu-ai/mom/issues/26)

### US-22: oxidizedgraph Integration
- **Goal**: Track workflow execution and decisions
- **Direction**: oxidizedgraph → MOM
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Issue**: [#27](https://github.com/lornu-ai/mom/issues/27)

### US-23: oxidizedRAG Integration
- **Goal**: Store and query code analysis memories
- **Direction**: oxidizedRAG → MOM
- **Status**: 🟡 Design phase
- **Complexity**: High
- **Issue**: [#28](https://github.com/lornu-ai/mom/issues/28)

---

## Phase 3: Advanced (Future Enhancements)

High-performance and sophisticated features.

### US-20: gRPC Service
- **Goal**: High-performance binary protocol for agents
- **Port**: 50051
- **Status**: 🔴 Future
- **Complexity**: High
- **Alongside**: Axum HTTP service
- **Issue**: [#25](https://github.com/lornu-ai/mom/issues/25)

---

## Implementation Roadmap

### Phase 1 (Weeks 1-4)
1. ✅ **Infrastructure**: Multi-crate Rust workspace, SurrealDB store, Axum service
2. ⏳ **Core Write Path**: US-1 (Write Event Memory)
3. ⏳ **Core Read Path**: US-2, US-3, US-4, US-5 (CRUD operations)
4. ⏳ **Scoping**: US-6, US-7 (Agent + tenant isolation)
5. ⏳ **Client**: US-13 (TypeScript/Bun SDK)

### Phase 2 (Weeks 5-12)
1. ⏳ **Vector Search**: US-8 (Embeddings + semantic search)
2. ⏳ **Consolidation**: US-9 (Summarization pipeline)
3. ⏳ **Advanced Types**: US-10 (Facts, preferences), US-11 (Graph relationships)
4. ⏳ **Management**: US-12 (Retention policies), US-16 (Metrics)
5. ⏳ **Security**: US-17 (Auth/RBAC), US-18 (Docker)
6. ⏳ **Integration**: US-21, US-22, US-23 (Ecosystem integration)

### Phase 3 (Weeks 13+)
1. ⏳ **Performance**: US-19 (Batch ops), US-20 (gRPC)
2. ⏳ **Advanced**: MCP bridge (US-15), LLM integration

---

## Metrics & KPIs

### Phase 1 Success Criteria
- ✅ All CRUD operations working
- ✅ Agent + tenant isolation enforced
- ✅ TypeScript client functional
- ✅ < 100ms p95 latency for single item ops

### Phase 2 Success Criteria
- ⏳ Vector search working
- ⏳ < 500ms p95 for complex queries
- ⏳ 10K+ items searchable in < 1 sec
- ⏳ Ecosystem integrations stable

### Phase 3 Success Criteria
- ⏳ gRPC service < 50ms p95
- ⏳ Batch operations 1000 items < 1 sec
- ⏳ MCP tools production-ready

---

## Dependencies Matrix

```
US-1 (Write)
  ├─ US-2 (Get)
  ├─ US-3 (List)
  ├─ US-4 (Recall)
  ├─ US-5 (Delete)
  ├─ US-6 (Agent Scoping)
  ├─ US-7 (Tenant Isolation)
  └─ US-14 (Context Packs)

US-8 (Vector Search)
  ├─ US-4 (needs recall foundation)
  └─ Hybrid Recall

US-9 (Consolidation)
  ├─ US-1 (write summaries)
  └─ US-4 (query for consolidation)

US-21, US-22, US-23 (Integrations)
  └─ US-1, US-4 (read/write infrastructure)
```

---

## References

- **Repository**: https://github.com/lornu-ai/mom
- **Issues**: [All Issues](https://github.com/lornu-ai/mom/issues)
- **Architecture**: [docs/DESIGN.md](./DESIGN.md)
- **Status**: [Updated 2026-03-06]
