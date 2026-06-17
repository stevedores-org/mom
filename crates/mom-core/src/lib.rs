//! MOM Core - Stable kernel API for event-sourced memory
//!
//! This is the minimal "MOM contract" - everything depends on it.

pub mod context_pack;
pub mod facts;
pub mod task;
pub use context_pack::{
    build_context_pack, content_embed_text, content_preview, Citation, ContextPack,
    ContextPackRequest, DEFAULT_BUDGET_TOKENS, MAX_EMBED_TEXT_CHARS, TOKENS_PER_ITEM,
};
pub use facts::{
    read_provenance_ids, read_superseded_by, read_version, record_semantic_conflict,
    write_provenance_ids, write_superseded_by, write_version, FactPayload, FactValidationError,
    PreferencePayload, PreferenceValidationError, META_FACT, META_PREFERENCE, META_PROVENANCE_IDS,
    META_SEMANTIC_CONFLICTS, META_SUPERSEDED_BY, META_VERSION,
};
pub use task::{
    task_tag, CheckpointRecord, TaskParseError, TaskRecord, TaskStatus, META_TASK_ID,
    TAG_TASK_PREFIX,
};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryId(pub String);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    #[serde(alias = "Event")]
    Event,
    #[serde(alias = "Summary")]
    Summary,
    #[serde(alias = "Fact")]
    Fact,
    #[serde(alias = "Preference")]
    Preference,
    /// Agent-task tracking: an item describing work to do or in progress.
    /// Status, scratchpad, and dependency edges live in `Content::Json` / `meta`.
    #[serde(alias = "Task")]
    Task,
    /// Durable-execution checkpoint: a serialized snapshot of an agent's
    /// state taken at a pause point, suitable for resume on the same or
    /// a different worker. References the originating `Task` via `meta`.
    #[serde(alias = "Checkpoint")]
    Checkpoint,
}

/// `Display` mirrors the serde lowercase encoding so the textual form is
/// the same everywhere: serde, store, HTTP query parameters.
impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Infallible: every variant has a lowercase serde encoding.
        let s = serde_plain::to_string(self).expect("MemoryKind serializes as a plain string");
        f.write_str(&s)
    }
}

/// Parses the same lowercase tokens produced by [`Display`] / serde. Used
/// by HTTP filter parsing and by the SurrealDB store on read.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("unknown memory kind: {0}")]
pub struct ParseMemoryKindError(pub String);

impl FromStr for MemoryKind {
    type Err = ParseMemoryKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_plain::from_str(s).map_err(|_| ParseMemoryKindError(s.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeKey {
    pub tenant_id: String,
    pub workspace_id: Option<String>,
    pub project_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Json(serde_json::Value),
    TextJson {
        text: String,
        json: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: MemoryId,
    pub scope: ScopeKey,
    pub kind: MemoryKind,
    pub created_at_ms: i64,
    pub content: Content,
    pub tags: Vec<String>,

    // ranking knobs
    pub importance: f32, // 0..1
    pub confidence: f32, // 0..1

    // provenance / safety
    pub source: String, // "user" | "tool" | "agent" | "system"
    pub ttl_ms: Option<i64>,
    pub meta: BTreeMap<String, serde_json::Value>,

    // semantic search (Phase 2)
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub scope: ScopeKey,
    pub text: String,
    pub kinds: Option<Vec<MemoryKind>>,
    pub tags_any: Option<Vec<String>>,
    pub limit: usize,

    // optional: time bounds (ms since epoch)
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,

    // optional: cursor for pagination
    pub cursor: Option<String>,
}

impl Query {
    pub fn encode_cursor(created_at_ms: i64, id: &str) -> String {
        use base64::{prelude::BASE64_STANDARD, Engine};
        let raw = format!("{}:{}", created_at_ms, id);
        BASE64_STANDARD.encode(raw)
    }

    pub fn decode_cursor(cursor: &str) -> Option<(i64, String)> {
        use base64::{prelude::BASE64_STANDARD, Engine};
        let decoded_bytes = BASE64_STANDARD.decode(cursor).ok()?;
        let decoded_str = String::from_utf8(decoded_bytes).ok()?;
        let mut parts = decoded_str.splitn(2, ':');
        let created_at_ms = parts.next()?.parse::<i64>().ok()?;
        let id = parts.next()?.to_string();
        Some((created_at_ms, id))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scored<T> {
    pub score: f32,
    pub item: T,
}

/// Core storage trait - implement this for new backends
#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    async fn put(&self, item: MemoryItem) -> anyhow::Result<()>;
    async fn get(&self, id: &MemoryId) -> anyhow::Result<Option<MemoryItem>>;
    async fn query(&self, q: Query) -> anyhow::Result<Vec<Scored<MemoryItem>>>;
    async fn delete(&self, id: &MemoryId) -> anyhow::Result<()>;

    /// Tenant-aware get: retrieves an item only if it belongs to the specified scope
    /// Returns None if item doesn't exist or doesn't belong to the tenant
    /// SECURITY: This method enforces multi-tenant isolation
    async fn get_scoped(
        &self,
        id: &MemoryId,
        scope: &ScopeKey,
    ) -> anyhow::Result<Option<MemoryItem>> {
        // Default implementation: get item and verify tenant match
        if let Some(item) = self.get(id).await? {
            if item.scope.tenant_id == scope.tenant_id {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    /// Tenant-aware delete: deletes an item only if it belongs to the specified scope
    /// Returns Ok(()) whether item exists or not (idempotent)
    /// SECURITY: This method enforces multi-tenant isolation
    async fn delete_scoped(&self, id: &MemoryId, scope: &ScopeKey) -> anyhow::Result<()> {
        // Default implementation: get item and verify tenant match before delete
        if let Some(item) = self.get(id).await? {
            if item.scope.tenant_id == scope.tenant_id {
                self.delete(id).await?;
            }
        }
        Ok(())
    }

    /// Vector-based semantic search (Phase 2)
    async fn vector_recall(
        &self,
        _query_embedding: &[f32],
        _scope: &ScopeKey,
        _limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        // Default implementation returns empty - implementations can override
        Ok(Vec::new())
    }

    /// Hybrid recall combining lexical + semantic search with RRF fusion (Phase 2)
    async fn hybrid_recall(
        &self,
        _q: Query,
        _query_embedding: &[f32],
        _limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        // Default implementation returns empty - implementations can override
        Ok(Vec::new())
    }

    /// Batch write: writes multiple items, returning the assigned ids in input
    /// order. Best-effort (non-atomic) by default — a mid-batch failure leaves
    /// a partial result. Backends that support transactions (e.g. SurrealDB)
    /// can override for true atomicity (tracked in #68).
    ///
    /// US-19a (#63).
    async fn write_batch(
        &self,
        items: Vec<MemoryItem>,
        _atomic: bool,
    ) -> anyhow::Result<Vec<MemoryId>> {
        let mut ids = Vec::with_capacity(items.len());
        for mut item in items {
            if item.id.0.is_empty() {
                item.id = MemoryId(uuid::Uuid::new_v4().to_string());
            }
            let id = item.id.clone();
            self.put(item).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Batch delete: deletes multiple ids in input order. Idempotent —
    /// missing ids are not an error (mirrors single-item `delete`).
    /// Best-effort (non-atomic) by default; backends with transactions
    /// can override (tracked in #68).
    ///
    /// US-19b (#64).
    async fn delete_batch(&self, ids: Vec<MemoryId>, _atomic: bool) -> anyhow::Result<()> {
        for id in ids {
            self.delete(&id).await?;
        }
        Ok(())
    }

    /// Batch query: runs N independent queries and returns results aligned
    /// by input index. Default impl is sequential; backends with cheap
    /// concurrency can override (e.g. `futures::join_all`) for parallelism.
    /// First error short-circuits the batch and is returned.
    ///
    /// US-19c (#65).
    async fn query_batch(
        &self,
        queries: Vec<Query>,
    ) -> anyhow::Result<Vec<Vec<Scored<MemoryItem>>>> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.query(q).await?);
        }
        Ok(results)
    }
}

/// Optional: embedder for semantic search (plug in later)
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, input: &str) -> anyhow::Result<Vec<f32>>;
    fn dims(&self) -> usize;
    fn model_id(&self) -> &str;
}

impl MemoryItem {
    pub fn new(
        id: MemoryId,
        scope: ScopeKey,
        kind: MemoryKind,
        content: Content,
        source: String,
    ) -> Self {
        Self {
            id,
            scope,
            kind,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            content,
            tags: Vec::new(),
            importance: 0.5,
            confidence: 1.0,
            source,
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_item_new() {
        let item = MemoryItem::new(
            MemoryId("test-1".to_string()),
            ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            MemoryKind::Event,
            Content::Text("Hello world".to_string()),
            "user".to_string(),
        );

        assert_eq!(item.id.0, "test-1");
        assert_eq!(item.kind, MemoryKind::Event);
        assert_eq!(item.importance, 0.5);
    }

    #[test]
    fn test_memory_kind_task_serializes_lowercase() {
        let json = serde_json::to_string(&MemoryKind::Task).unwrap();
        assert_eq!(json, "\"task\"");
    }

    #[test]
    fn test_memory_kind_checkpoint_serializes_lowercase() {
        let json = serde_json::to_string(&MemoryKind::Checkpoint).unwrap();
        assert_eq!(json, "\"checkpoint\"");
    }

    #[test]
    fn test_memory_kind_task_deserializes_from_lowercase() {
        let kind: MemoryKind = serde_json::from_str("\"task\"").unwrap();
        assert_eq!(kind, MemoryKind::Task);
    }

    #[test]
    fn test_memory_kind_checkpoint_deserializes_from_lowercase() {
        let kind: MemoryKind = serde_json::from_str("\"checkpoint\"").unwrap();
        assert_eq!(kind, MemoryKind::Checkpoint);
    }

    #[test]
    fn test_memory_kind_round_trip_all_variants() {
        for kind in [
            MemoryKind::Event,
            MemoryKind::Summary,
            MemoryKind::Fact,
            MemoryKind::Preference,
            MemoryKind::Task,
            MemoryKind::Checkpoint,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let parsed: MemoryKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, parsed, "round-trip failed for {:?}", kind);
        }
    }

    #[test]
    fn test_task_item_carries_status_via_json_content() {
        let item = MemoryItem::new(
            MemoryId("task-1".to_string()),
            ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            MemoryKind::Task,
            Content::Json(serde_json::json!({
                "status": "in_progress",
                "depends_on": ["task-0"],
            })),
            "agent".to_string(),
        );

        assert_eq!(item.kind, MemoryKind::Task);
        match &item.content {
            Content::Json(v) => {
                assert_eq!(v["status"], "in_progress");
                assert_eq!(v["depends_on"][0], "task-0");
            }
            _ => panic!("expected Content::Json"),
        }
    }

    #[test]
    fn test_checkpoint_item_references_originating_task_via_meta() {
        let mut item = MemoryItem::new(
            MemoryId("ckpt-1".to_string()),
            ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            MemoryKind::Checkpoint,
            Content::Json(serde_json::json!({"step": 4, "scratchpad": {}})),
            "agent".to_string(),
        );
        item.meta
            .insert("task_id".to_string(), serde_json::json!("task-1"));

        assert_eq!(item.kind, MemoryKind::Checkpoint);
        assert_eq!(item.meta.get("task_id").unwrap(), "task-1");
    }

    #[test]
    fn test_deserialize_kind() {
        let k: MemoryKind = serde_json::from_str("\"Fact\"").unwrap();
        assert_eq!(k, MemoryKind::Fact);
    }
}
