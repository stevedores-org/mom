//! MOM SurrealDB Store - Multi-model persistence layer
//!
//! Leverages SurrealDB's document model, relationships, and queries
//! for efficient memory storage and hybrid retrieval.

use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey, Scored};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use surrealdb::engine::local::{Db, Mem};
use surrealdb::Surreal;
use tracing::{debug, error};

pub mod hybrid;

pub use hybrid::{HybridConfig, RankedResult};

#[allow(dead_code)]
pub struct SurrealDBStore {
    db: Arc<Surreal<Db>>,
    namespace: String,
    database: String,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let magnitude = norm_a.sqrt() * norm_b.sqrt();
    if magnitude == 0.0 {
        0.0
    } else {
        dot_product / magnitude
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StoredItem {
    id: String,
    tenant_id: String,
    workspace_id: Option<String>,
    project_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,

    kind: String,
    created_at_ms: i64,

    content_text: Option<String>,
    content_json: Option<serde_json::Value>,

    importance: f32,
    confidence: f32,
    source: String,
    ttl_ms: Option<i64>,
    meta: serde_json::Value,

    tags: Vec<String>,

    // Phase 2: Vector embeddings
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
}

impl SurrealDBStore {
    pub async fn new(_db_path: &str) -> anyhow::Result<Self> {
        // For in-memory backend, create new Surreal instance
        // Note: Initialize with Mem endpoint, returns Surreal<Db> connection
        let db: Surreal<Db> = Surreal::new::<Mem>(()).await?;
        db.use_ns("mom").use_db("main").await?;

        Self::init_schema(&db).await?;

        Ok(Self {
            db: Arc::new(db),
            namespace: "mom".to_string(),
            database: "main".to_string(),
        })
    }

    async fn init_schema(db: &Surreal<Db>) -> anyhow::Result<()> {
        // Create table for memory items
        db.query(
            r#"
            DEFINE TABLE memory_items SCHEMAFULL PERMISSIONS
              FOR select WHERE tenant_id = $scope_tenant_id;
            DEFINE FIELD id ON TABLE memory_items TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD tenant_id ON TABLE memory_items TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD workspace_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD project_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD agent_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD run_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD kind ON TABLE memory_items TYPE string ASSERT $value IN ['Event', 'Summary', 'Fact', 'Preference'];
            DEFINE FIELD created_at_ms ON TABLE memory_items TYPE number;
            DEFINE FIELD content_text ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD content_json ON TABLE memory_items TYPE option<object>;
            DEFINE FIELD importance ON TABLE memory_items TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD confidence ON TABLE memory_items TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD source ON TABLE memory_items TYPE string;
            DEFINE FIELD ttl_ms ON TABLE memory_items TYPE option<number>;
            DEFINE FIELD meta ON TABLE memory_items TYPE object;
            DEFINE FIELD tags ON TABLE memory_items TYPE array<string>;
            DEFINE FIELD embedding ON TABLE memory_items TYPE option<array<float>>;
            DEFINE FIELD embedding_model ON TABLE memory_items TYPE option<string>;

            DEFINE INDEX idx_tenant_time ON TABLE memory_items COLUMNS tenant_id, created_at_ms;
            DEFINE INDEX idx_scope ON TABLE memory_items COLUMNS tenant_id, workspace_id, project_id, agent_id, run_id;
            DEFINE INDEX idx_embedding ON TABLE memory_items COLUMNS embedding;
            "#
        )
        .await?;

        debug!("SurrealDB schema initialized");
        Ok(())
    }

    fn kind_to_str(k: MemoryKind) -> &'static str {
        match k {
            MemoryKind::Event => "Event",
            MemoryKind::Summary => "Summary",
            MemoryKind::Fact => "Fact",
            MemoryKind::Preference => "Preference",
        }
    }

    fn str_to_kind(s: &str) -> Option<MemoryKind> {
        match s {
            "Event" => Some(MemoryKind::Event),
            "Summary" => Some(MemoryKind::Summary),
            "Fact" => Some(MemoryKind::Fact),
            "Preference" => Some(MemoryKind::Preference),
            _ => None,
        }
    }

    // Fails open (returns 0) if the system clock is before UNIX_EPOCH so a
    // broken clock cannot mass-expire stored items; the error is logged so the
    // condition surfaces in metrics rather than silently corrupting TTL state.
    fn current_time_ms() -> i64 {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_millis() as i64,
            Err(err) => {
                error!(
                    error = %err,
                    "system clock is before UNIX_EPOCH; treating now_ms as 0 so TTL filtering fails open"
                );
                0
            }
        }
    }

    fn is_expired(created_at_ms: i64, ttl_ms: Option<i64>, now_ms: i64) -> bool {
        ttl_ms
            .and_then(|ttl| created_at_ms.checked_add(ttl))
            .is_some_and(|expires_at| expires_at <= now_ms)
    }
}

#[async_trait::async_trait]
impl mom_core::MemoryStore for SurrealDBStore {
    async fn put(&self, item: MemoryItem) -> anyhow::Result<()> {
        let (content_text, content_json) = match &item.content {
            Content::Text(t) => (Some(t.clone()), None),
            Content::Json(v) => (None, Some(v.clone())),
            Content::TextJson { text, json } => (Some(text.clone()), Some(json.clone())),
        };

        let stored = StoredItem {
            id: item.id.0.clone(),
            tenant_id: item.scope.tenant_id.clone(),
            workspace_id: item.scope.workspace_id.clone(),
            project_id: item.scope.project_id.clone(),
            agent_id: item.scope.agent_id.clone(),
            run_id: item.scope.run_id.clone(),
            kind: Self::kind_to_str(item.kind).to_string(),
            created_at_ms: item.created_at_ms,
            content_text,
            content_json,
            importance: item.importance,
            confidence: item.confidence,
            source: item.source.clone(),
            ttl_ms: item.ttl_ms,
            meta: serde_json::to_value(&item.meta)?,
            tags: item.tags.clone(),
            embedding: item.embedding.clone(),
            embedding_model: item.embedding_model.clone(),
        };

        let item_id = item.id.0.clone();
        let _: Vec<StoredItem> = self
            .db
            .query("UPSERT type::thing('memory_items', $id) MERGE $data")
            .bind(("id", item_id.clone()))
            .bind(("data", stored))
            .await?
            .take(0)?;

        debug!("Stored memory item: {}", item_id);
        Ok(())
    }

    async fn get(&self, id: &MemoryId) -> anyhow::Result<Option<MemoryItem>> {
        let results: Vec<StoredItem> = self
            .db
            .query("SELECT * FROM type::thing('memory_items', $id)")
            .bind(("id", id.0.clone()))
            .await?
            .take(0)?;

        Ok(results.into_iter().next().map(|s| {
            let content = match (s.content_text, s.content_json) {
                (Some(text), None) => Content::Text(text),
                (None, Some(json)) => Content::Json(json),
                (Some(text), Some(json)) => Content::TextJson { text, json },
                _ => Content::Text(String::new()),
            };

            let kind = Self::str_to_kind(&s.kind).unwrap_or(MemoryKind::Event);

            MemoryItem {
                id: MemoryId(s.id),
                scope: mom_core::ScopeKey {
                    tenant_id: s.tenant_id,
                    workspace_id: s.workspace_id,
                    project_id: s.project_id,
                    agent_id: s.agent_id,
                    run_id: s.run_id,
                },
                kind,
                created_at_ms: s.created_at_ms,
                content,
                tags: s.tags,
                importance: s.importance,
                confidence: s.confidence,
                source: s.source,
                ttl_ms: s.ttl_ms,
                meta: serde_json::from_value(s.meta).unwrap_or_default(),
                embedding: s.embedding,
                embedding_model: s.embedding_model,
            }
        }))
    }

    async fn get_scoped(
        &self,
        id: &MemoryId,
        scope: &ScopeKey,
    ) -> anyhow::Result<Option<MemoryItem>> {
        // SECURITY: Query with tenant_id filter to enforce multi-tenant isolation at DB level
        let results: Vec<StoredItem> = self
            .db
            .query("SELECT * FROM memory_items WHERE id = $id AND tenant_id = $tenant")
            .bind(("id", id.0.clone()))
            .bind(("tenant", scope.tenant_id.clone()))
            .await?
            .take(0)?;

        Ok(results.into_iter().next().and_then(|s| {
            if Self::is_expired(s.created_at_ms, s.ttl_ms, Self::current_time_ms()) {
                return None;
            }

            let content = match (s.content_text, s.content_json) {
                (Some(text), None) => Content::Text(text),
                (None, Some(json)) => Content::Json(json),
                (Some(text), Some(json)) => Content::TextJson { text, json },
                _ => Content::Text(String::new()),
            };

            let kind = Self::str_to_kind(&s.kind).unwrap_or(MemoryKind::Event);

            Some(MemoryItem {
                id: MemoryId(s.id),
                scope: mom_core::ScopeKey {
                    tenant_id: s.tenant_id,
                    workspace_id: s.workspace_id,
                    project_id: s.project_id,
                    agent_id: s.agent_id,
                    run_id: s.run_id,
                },
                kind,
                created_at_ms: s.created_at_ms,
                content,
                tags: s.tags,
                importance: s.importance,
                confidence: s.confidence,
                source: s.source,
                ttl_ms: s.ttl_ms,
                meta: serde_json::from_value(s.meta).unwrap_or_default(),
                embedding: s.embedding,
                embedding_model: s.embedding_model,
            })
        }))
    }

    async fn query(&self, q: Query) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        // Build SurrealQL query with tenant filter + optional refinements.
        // Clauses are appended conditionally; parameters are bound below.
        let mut query_str = String::from("SELECT * FROM memory_items WHERE tenant_id = $tenant");

        if q.scope.workspace_id.is_some() {
            query_str.push_str(" AND workspace_id = $workspace");
        }
        if q.scope.project_id.is_some() {
            query_str.push_str(" AND project_id = $project");
        }
        if q.scope.agent_id.is_some() {
            query_str.push_str(" AND agent_id = $agent");
        }
        if q.kinds.is_some() {
            query_str.push_str(" AND kind IN $kinds");
        }
        if q.since_ms.is_some() {
            query_str.push_str(" AND created_at_ms >= $since");
        }
        if q.until_ms.is_some() {
            query_str.push_str(" AND created_at_ms <= $until");
        }
        if !q.text.is_empty() {
            query_str.push_str(" AND (content_text CONTAINS $text OR tags CONTAINS [$text])");
        }

        // Sort by importance + recency. Fetch extra rows before the Rust-side
        // TTL filter so expired high-rank items do not starve fresh results.
        let fetch_limit = q.limit.saturating_mul(4).max(q.limit).max(1);
        query_str.push_str(" ORDER BY importance DESC, created_at_ms DESC LIMIT $limit");

        let mut builder = self
            .db
            .query(query_str)
            .bind(("tenant", q.scope.tenant_id.clone()))
            .bind(("limit", fetch_limit as i64));
        if let Some(ref ws) = q.scope.workspace_id {
            builder = builder.bind(("workspace", ws.clone()));
        }
        if let Some(ref proj) = q.scope.project_id {
            builder = builder.bind(("project", proj.clone()));
        }
        if let Some(ref agent) = q.scope.agent_id {
            builder = builder.bind(("agent", agent.clone()));
        }
        if let Some(ref kinds) = q.kinds {
            let kind_strs: Vec<String> = kinds
                .iter()
                .map(|k| Self::kind_to_str(*k).to_string())
                .collect();
            builder = builder.bind(("kinds", kind_strs));
        }
        if let Some(since) = q.since_ms {
            builder = builder.bind(("since", since));
        }
        if let Some(until) = q.until_ms {
            builder = builder.bind(("until", until));
        }
        if !q.text.is_empty() {
            builder = builder.bind(("text", q.text.clone()));
        }

        let results: Vec<StoredItem> = builder.await?.take(0)?;
        let now_ms = Self::current_time_ms();

        let mut scored = Vec::with_capacity(results.len());
        for (idx, item) in results
            .into_iter()
            .filter(|item| !Self::is_expired(item.created_at_ms, item.ttl_ms, now_ms))
            .take(q.limit)
            .enumerate()
        {
            // Simple scoring: importance + recency bonus
            let recency_bonus = (1.0 - (idx as f32 / q.limit as f32).min(1.0)) * 0.2;
            let score = (item.importance + recency_bonus).min(1.0);

            let content = match (item.content_text, item.content_json) {
                (Some(text), None) => Content::Text(text),
                (None, Some(json)) => Content::Json(json),
                (Some(text), Some(json)) => Content::TextJson { text, json },
                _ => Content::Text(String::new()),
            };

            let kind = Self::str_to_kind(&item.kind).unwrap_or(MemoryKind::Event);

            scored.push(Scored {
                score,
                item: MemoryItem {
                    id: MemoryId(item.id),
                    scope: mom_core::ScopeKey {
                        tenant_id: item.tenant_id,
                        workspace_id: item.workspace_id,
                        project_id: item.project_id,
                        agent_id: item.agent_id,
                        run_id: item.run_id,
                    },
                    kind,
                    created_at_ms: item.created_at_ms,
                    content,
                    tags: item.tags,
                    importance: item.importance,
                    confidence: item.confidence,
                    source: item.source,
                    ttl_ms: item.ttl_ms,
                    meta: serde_json::from_value(item.meta).unwrap_or_default(),
                    embedding: item.embedding,
                    embedding_model: item.embedding_model,
                },
            });
        }

        debug!("Query found {} results", scored.len());
        Ok(scored)
    }

    async fn delete(&self, id: &MemoryId) -> anyhow::Result<()> {
        let _: Vec<StoredItem> = self
            .db
            .query("DELETE type::thing('memory_items', $id)")
            .bind(("id", id.0.clone()))
            .await?
            .take(0)?;
        debug!("Deleted memory item: {}", id.0);
        Ok(())
    }

    async fn delete_scoped(&self, id: &MemoryId, scope: &ScopeKey) -> anyhow::Result<()> {
        // SECURITY: Delete with tenant_id filter to enforce multi-tenant isolation at DB level
        // This ensures we can only delete items that belong to the calling tenant
        let _: Vec<StoredItem> = self
            .db
            .query("DELETE memory_items WHERE id = $id AND tenant_id = $tenant")
            .bind(("id", id.0.clone()))
            .bind(("tenant", scope.tenant_id.clone()))
            .await?
            .take(0)?;
        debug!(
            "Deleted memory item scoped to tenant: {} (id: {})",
            scope.tenant_id, id.0
        );
        Ok(())
    }

    /// Vector-based semantic search (Phase 2d)
    async fn vector_recall(
        &self,
        query_embedding: &[f32],
        scope: &ScopeKey,
        limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        let results = semantic_recall(&self.db, scope, query_embedding, limit).await?;

        let mut scored = Vec::with_capacity(results.len());
        for (id, score) in results {
            let memory_id = MemoryId(id);
            if let Some(item) = self.get(&memory_id).await? {
                scored.push(Scored { score, item });
            }
        }

        Ok(scored)
    }

    /// Hybrid recall: lexical + semantic with RRF fusion (Phase 2d - Issue #12)
    async fn hybrid_recall(
        &self,
        q: Query,
        query_embedding: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        let config = HybridConfig::default();
        hybrid_recall_impl(self, &q.scope, &q.text, query_embedding, limit, &config).await
    }
}

/// Helper: Lexical search using content text (Phase 2d)
async fn lexical_recall(
    db: &Surreal<Db>,
    scope: &ScopeKey,
    query_text: &str,
    limit: usize,
) -> anyhow::Result<Vec<(String, f32)>> {
    // Build SurrealQL query for full-text search; clauses appended conditionally
    // and bound below to keep user-supplied values out of the query string.
    let mut query_str =
        String::from("SELECT id, importance FROM memory_items WHERE tenant_id = $tenant");

    if scope.workspace_id.is_some() {
        query_str.push_str(" AND workspace_id = $workspace");
    }
    if scope.project_id.is_some() {
        query_str.push_str(" AND project_id = $project");
    }
    if scope.agent_id.is_some() {
        query_str.push_str(" AND agent_id = $agent");
    }
    if !query_text.is_empty() {
        query_str.push_str(" AND (content_text CONTAINS $text OR tags CONTAINS [$text])");
    }
    query_str.push_str(" ORDER BY importance DESC, created_at_ms DESC LIMIT $limit");

    let mut builder = db
        .query(query_str)
        .bind(("tenant", scope.tenant_id.clone()))
        .bind(("limit", limit as i64));
    if let Some(ref ws) = scope.workspace_id {
        builder = builder.bind(("workspace", ws.clone()));
    }
    if let Some(ref proj) = scope.project_id {
        builder = builder.bind(("project", proj.clone()));
    }
    if let Some(ref agent) = scope.agent_id {
        builder = builder.bind(("agent", agent.clone()));
    }
    if !query_text.is_empty() {
        builder = builder.bind(("text", query_text.to_string()));
    }

    let results: Vec<StoredItem> = builder.await?.take(0)?;

    let scored: Vec<(String, f32)> = results
        .into_iter()
        .map(|item| (item.id, item.importance))
        .collect();

    debug!(
        "Lexical recall found {} results for query '{}'",
        scored.len(),
        query_text
    );
    Ok(scored)
}

/// Helper: Semantic search using embeddings (Phase 2d)
async fn semantic_recall(
    db: &Surreal<Db>,
    scope: &ScopeKey,
    query_embedding: &[f32],
    limit: usize,
) -> anyhow::Result<Vec<(String, f32)>> {
    // Vector similarity search - fetch all items with embeddings and compute cosine similarity.
    // Clauses appended conditionally and bound below.
    let mut query_str = String::from(
        "SELECT id, embedding FROM memory_items WHERE tenant_id = $tenant AND embedding IS NOT NULL",
    );

    if scope.workspace_id.is_some() {
        query_str.push_str(" AND workspace_id = $workspace");
    }
    if scope.project_id.is_some() {
        query_str.push_str(" AND project_id = $project");
    }
    if scope.agent_id.is_some() {
        query_str.push_str(" AND agent_id = $agent");
    }

    // Order by created_at_ms for stable ordering before similarity computation
    query_str.push_str(" ORDER BY created_at_ms DESC LIMIT 1000");

    let mut builder = db
        .query(query_str)
        .bind(("tenant", scope.tenant_id.clone()));
    if let Some(ref ws) = scope.workspace_id {
        builder = builder.bind(("workspace", ws.clone()));
    }
    if let Some(ref proj) = scope.project_id {
        builder = builder.bind(("project", proj.clone()));
    }
    if let Some(ref agent) = scope.agent_id {
        builder = builder.bind(("agent", agent.clone()));
    }

    let results: Vec<StoredItem> = builder.await?.take(0)?;

    // Compute cosine similarity for each item
    let mut scored: Vec<(String, f32)> = results
        .into_iter()
        .filter_map(|item| {
            item.embedding.as_ref().map(|embedding| {
                let similarity = cosine_similarity(query_embedding, embedding);
                (item.id, similarity)
            })
        })
        .collect();

    // Sort by similarity and truncate
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    debug!(
        "Semantic recall found {} results with embedding",
        scored.len()
    );
    Ok(scored)
}

/// Helper: Hybrid recall with RRF fusion (Phase 2d - Issue #12)
async fn hybrid_recall_impl(
    store: &SurrealDBStore,
    scope: &ScopeKey,
    query_text: &str,
    query_embedding: &[f32],
    limit: usize,
    config: &hybrid::HybridConfig,
) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
    // Run lexical and semantic searches in parallel
    let (lexical_results, semantic_results) = tokio::join!(
        lexical_recall(&store.db, scope, query_text, limit),
        semantic_recall(&store.db, scope, query_embedding, limit),
    );

    let lexical = lexical_results?;
    let semantic = semantic_results?;

    // Merge using RRF
    let merged_ids =
        hybrid::merge_results_with_rrf(lexical.clone(), semantic.clone(), config, limit);

    // Fetch full items and rebuild Scored results
    let mut scored = Vec::with_capacity(merged_ids.len());
    for (id, rrf_score) in merged_ids.iter() {
        let memory_id = MemoryId(id.clone());
        if let Some(item) = store.get_scoped(&memory_id, scope).await? {
            // Re-score: use RRF score from fusion
            scored.push(Scored {
                score: *rrf_score,
                item,
            });
        }
    }

    debug!(
        "Hybrid recall found {} results (lexical={}, semantic={}, merged={})",
        scored.len(),
        lexical.len(),
        semantic.len(),
        merged_ids.len()
    );
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttl_expiry_helper_marks_expired_items() {
        assert!(SurrealDBStore::is_expired(1_000, Some(500), 1_500));
        assert!(SurrealDBStore::is_expired(1_000, Some(500), 1_501));
    }

    #[test]
    fn ttl_expiry_helper_keeps_fresh_or_unbounded_items() {
        assert!(!SurrealDBStore::is_expired(1_000, Some(500), 1_499));
        assert!(!SurrealDBStore::is_expired(1_000, None, 10_000));
    }

    // NOTE: Cross-tenant integration tests against the live SurrealDB store
    // were drafted but block on a pre-existing surrealdb 2.x schema mismatch
    // discovered during this PR — `DEFINE FIELD id … TYPE string ASSERT
    // string::len($value) > 0` fights surrealdb 2's record-id semantics, and
    // `option<…>` schema fields reject the JSON `null` that
    // `serde_json::to_string(&stored)` produces for `None`. Both need to be
    // addressed (schema rework + `#[serde(skip_serializing_if)]` on
    // `StoredItem`) before the store is testable end-to-end. The two unit
    // tests above still cover the TTL helper; the scope-isolation HTTP-layer
    // properties remain covered by the type-level tests in mom-service.
}
