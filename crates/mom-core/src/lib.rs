//! MOM Core - Stable kernel API for event-sourced memory
//!
//! This is the minimal "MOM contract" - everything depends on it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryId(pub String);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Event,
    Summary,
    Fact,
    Preference,
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
}
