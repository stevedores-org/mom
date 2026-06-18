//! MOM SurrealDB Store - Multi-model persistence layer
//!
//! Leverages SurrealDB's document model, relationships, and queries
//! for efficient memory storage and hybrid retrieval.

use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey, Scored};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;
use surrealdb::engine::local::{Db, Mem};
use surrealdb::RecordId;
use surrealdb::Surreal;
use tracing::{debug, info};

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

/// Row shape returned by SurrealDB 2 queries (`id` is a record Thing).
#[derive(Debug, Deserialize)]
struct StoredItemFromDb {
    id: RecordId,
    tenant_id: String,
    #[serde(default)]
    workspace_id: Option<String>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    kind: String,
    created_at_ms: i64,
    #[serde(default)]
    content_text: Option<String>,
    #[serde(default)]
    content_json: Option<serde_json::Value>,
    importance: f32,
    confidence: f32,
    source: String,
    #[serde(default)]
    ttl_ms: Option<i64>,
    meta: serde_json::Value,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    embedding_model: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct StoredItem {
    id: String,
    tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,

    kind: String,
    created_at_ms: i64,

    #[serde(skip_serializing_if = "Option::is_none")]
    content_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_json: Option<serde_json::Value>,

    importance: f32,
    confidence: f32,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
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
            -- US-7 AC-3 deferred to US-17 (auth context). The previous
            -- PERMISSIONS clause referenced `$scope_tenant_id`, a param
            -- that was never bound (the `Surreal<Db>` connection is a
            -- single shared `Arc` across all requests, with no per-request
            -- session). It silently fired against NONE on every query,
            -- which made the table effectively unfiltered at the DB layer
            -- while telegraphing a false sense of security. Tenant
            -- isolation is enforced today by the app layer (every read
            -- and write goes through a `WHERE tenant_id = '...'` filter
            -- built in Rust). Real RLS lands once authenticated identity
            -- exists per US-17.
            DEFINE TABLE memory_items SCHEMAFULL;
            DEFINE FIELD id ON TABLE memory_items TYPE string;
            DEFINE FIELD tenant_id ON TABLE memory_items TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD workspace_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD project_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD agent_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD run_id ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD kind ON TABLE memory_items TYPE string ASSERT $value IN ['event', 'summary', 'fact', 'preference', 'task', 'checkpoint'];
            DEFINE FIELD created_at_ms ON TABLE memory_items TYPE number;
            DEFINE FIELD content_text ON TABLE memory_items TYPE option<string>;
            DEFINE FIELD content_json ON TABLE memory_items TYPE option<object>;
            DEFINE FIELD importance ON TABLE memory_items TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD confidence ON TABLE memory_items TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD source ON TABLE memory_items TYPE string;
            DEFINE FIELD ttl_ms ON TABLE memory_items TYPE option<number>;
            DEFINE FIELD meta ON TABLE memory_items FLEXIBLE TYPE object;
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

    /// Single source of truth for the `MemoryKind` <-> string encoding,
    /// reusing the lowercase serde representation via `Display` / `FromStr`
    /// on `MemoryKind`. Previously this had a parallel titlecase mapping,
    /// which silently diverged from the serde encoding used by every other
    /// caller — the new shape makes that impossible.
    fn kind_to_str(k: MemoryKind) -> String {
        k.to_string()
    }

    fn str_to_kind(s: &str) -> Option<MemoryKind> {
        MemoryKind::from_str(s).ok()
    }

    /// Escape single quotes in SQL string values to prevent injection
    /// Replaces ' with '' (SQL standard escape)
    fn escape_sql_string(s: &str) -> String {
        s.replace('\'', "''")
    }

    /// SurrealDB 2 record reference safe for IDs containing hyphens.
    fn record_ref(id: &str) -> String {
        format!(
            "type::thing('memory_items', '{}')",
            Self::escape_sql_string(id)
        )
    }

    fn record_id_to_string(id: &RecordId) -> String {
        String::try_from(id.key().clone()).unwrap_or_else(|_| format!("{id}"))
    }

    fn from_db_row(row: StoredItemFromDb) -> StoredItem {
        StoredItem {
            id: Self::record_id_to_string(&row.id),
            tenant_id: row.tenant_id,
            workspace_id: row.workspace_id,
            project_id: row.project_id,
            agent_id: row.agent_id,
            run_id: row.run_id,
            kind: row.kind,
            created_at_ms: row.created_at_ms,
            content_text: row.content_text,
            content_json: row.content_json,
            importance: row.importance,
            confidence: row.confidence,
            source: row.source,
            ttl_ms: row.ttl_ms,
            meta: row.meta,
            tags: row.tags,
            embedding: row.embedding,
            embedding_model: row.embedding_model,
        }
    }

    fn stored_item_from_memory_item(item: &MemoryItem) -> anyhow::Result<StoredItem> {
        let (content_text, content_json) = match &item.content {
            Content::Text(t) => (Some(t.clone()), None),
            Content::Json(v) => (None, Some(v.clone())),
            Content::TextJson { text, json } => (Some(text.clone()), Some(json.clone())),
        };

        Ok(StoredItem {
            id: item.id.0.clone(),
            tenant_id: item.scope.tenant_id.clone(),
            workspace_id: item.scope.workspace_id.clone(),
            project_id: item.scope.project_id.clone(),
            agent_id: item.scope.agent_id.clone(),
            run_id: item.scope.run_id.clone(),
            kind: Self::kind_to_str(item.kind),
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
        })
    }

    fn build_parameterized_query(q: &Query, order_by_clause: &str, include_limit: bool) -> String {
        let mut query_str = "SELECT * FROM memory_items WHERE tenant_id = $tenant_id".to_string();

        if q.scope.workspace_id.is_some() {
            query_str.push_str(" AND workspace_id = $workspace_id");
        }
        if q.scope.project_id.is_some() {
            query_str.push_str(" AND project_id = $project_id");
        }
        if q.scope.agent_id.is_some() {
            query_str.push_str(" AND agent_id = $agent_id");
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

        if let Some(ref cursor_str) = q.cursor {
            if let Some((cursor_time, cursor_id)) = Query::decode_cursor(cursor_str) {
                let safe_cursor_id = Self::escape_sql_string(&cursor_id);
                query_str.push_str(&format!(
                    " AND (created_at_ms < {} OR (created_at_ms = {} AND id < type::thing('memory_items', '{}')))",
                    cursor_time, cursor_time, safe_cursor_id
                ));
            }
        }

        if !q.text.is_empty() {
            query_str.push_str(" AND (content_text CONTAINS $text OR tags CONTAINS [$text])");
        }

        query_str.push_str(&format!(" {}", order_by_clause));

        if include_limit {
            query_str.push_str(" LIMIT $limit");
        }

        query_str
    }

    fn bind_query_params<'a>(
        mut query: surrealdb::method::Query<'a, Db>,
        q: &Query,
    ) -> surrealdb::method::Query<'a, Db> {
        query = query.bind(("tenant_id", q.scope.tenant_id.clone()));

        if let Some(ref ws) = q.scope.workspace_id {
            query = query.bind(("workspace_id", ws.clone()));
        }
        if let Some(ref proj) = q.scope.project_id {
            query = query.bind(("project_id", proj.clone()));
        }
        if let Some(ref agent) = q.scope.agent_id {
            query = query.bind(("agent_id", agent.clone()));
        }

        if let Some(kinds) = &q.kinds {
            let kind_strs: Vec<_> = kinds.iter().map(|k| Self::kind_to_str(*k)).collect();
            query = query.bind(("kinds", kind_strs));
        }

        if let Some(since) = q.since_ms {
            query = query.bind(("since", since));
        }
        if let Some(until) = q.until_ms {
            query = query.bind(("until", until));
        }

        if !q.text.is_empty() {
            query = query.bind(("text", q.text.clone()));
        }

        query
    }

    /// US-10: find every active (not yet superseded) Fact in the given
    /// scope whose `meta.fact.subject` and `meta.fact.predicate` match the
    /// caller-supplied triple key. Used by the put-Fact path to detect
    /// contradictions before commit.
    ///
    /// Note: matches the existing `query` method's scope-filter shape
    /// (workspace_id / project_id / agent_id). `run_id` is intentionally
    /// NOT filtered here so facts learned in one run are visible to
    /// supersession checks in sibling runs of the same agent — facts are
    /// agent-scoped knowledge by design.
    pub async fn find_active_facts_with_key(
        &self,
        scope: &ScopeKey,
        subject: &str,
        predicate: &str,
    ) -> anyhow::Result<Vec<MemoryItem>> {
        let mut query_str =
            "SELECT * FROM memory_items WHERE tenant_id = $tenant_id AND kind = 'fact' \
                             AND meta.fact.subject = $subject AND meta.fact.predicate = $predicate \
                             AND (meta.superseded_by IS NONE OR meta.superseded_by IS NULL)"
                .to_string();

        if scope.workspace_id.is_some() {
            query_str.push_str(" AND workspace_id = $workspace_id");
        }
        if scope.project_id.is_some() {
            query_str.push_str(" AND project_id = $project_id");
        }
        if scope.agent_id.is_some() {
            query_str.push_str(" AND agent_id = $agent_id");
        }

        let mut query = self
            .db
            .query(&query_str)
            .bind(("tenant_id", scope.tenant_id.clone()))
            .bind(("subject", subject.to_string()))
            .bind(("predicate", predicate.to_string()));

        if let Some(ref ws) = scope.workspace_id {
            query = query.bind(("workspace_id", ws.clone()));
        }
        if let Some(ref proj) = scope.project_id {
            query = query.bind(("project_id", proj.clone()));
        }
        if let Some(ref agent) = scope.agent_id {
            query = query.bind(("agent_id", agent.clone()));
        }

        let rows: Vec<StoredItemFromDb> = query.await?.take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();
        Ok(results.into_iter().map(stored_item_to_memory).collect())
    }

    /// US-10 Phase 2: find active Facts in scope whose embedding cosine
    /// similarity to `query_embedding` is at or above `threshold`. Used by
    /// the semantic-conflict advisory pass on Fact write: a Fact whose
    /// embedding is "close enough" to an existing Fact's, but whose triple
    /// key didn't match, may still be in conflict and is worth flagging
    /// even though we don't auto-supersede.
    ///
    /// `exclude_id` skips the new item itself when this is called after the
    /// row has already been written (it isn't on the put_memory path, but
    /// is needed by future re-conflict-detection sweeps over the corpus).
    ///
    /// Returns items sorted by descending similarity, truncated to `max`.
    pub async fn find_semantic_fact_conflicts(
        &self,
        scope: &ScopeKey,
        query_embedding: &[f32],
        exclude_id: Option<&MemoryId>,
        threshold: f32,
        max: usize,
    ) -> anyhow::Result<Vec<(MemoryItem, f32)>> {
        if query_embedding.is_empty() || max == 0 {
            return Ok(Vec::new());
        }
        let safe_tenant = Self::escape_sql_string(&scope.tenant_id);
        let mut query_str = format!(
            "SELECT * FROM memory_items WHERE tenant_id = '{}' AND kind = 'fact' \
             AND embedding IS NOT NULL \
             AND (meta.superseded_by IS NONE OR meta.superseded_by IS NULL)",
            safe_tenant
        );
        if let Some(ref ws) = scope.workspace_id {
            let safe_ws = Self::escape_sql_string(ws);
            query_str.push_str(&format!(" AND workspace_id = '{}'", safe_ws));
        }
        if let Some(ref proj) = scope.project_id {
            let safe_proj = Self::escape_sql_string(proj);
            query_str.push_str(&format!(" AND project_id = '{}'", safe_proj));
        }
        if let Some(ref agent) = scope.agent_id {
            let safe_agent = Self::escape_sql_string(agent);
            query_str.push_str(&format!(" AND agent_id = '{}'", safe_agent));
        }

        let rows: Vec<StoredItemFromDb> = self.db.query(&query_str).await?.take(0)?;
        let mut scored: Vec<(MemoryItem, f32)> = rows
            .into_iter()
            .map(Self::from_db_row)
            .filter_map(|s| {
                let embedding = s.embedding.clone()?;
                let sim = cosine_similarity(query_embedding, &embedding);
                if sim < threshold {
                    return None;
                }
                let mem = stored_item_to_memory(s);
                if let Some(id) = exclude_id {
                    if mem.id == *id {
                        return None;
                    }
                }
                Some((mem, sim))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max);
        Ok(scored)
    }
}

fn stored_item_to_memory(s: StoredItem) -> MemoryItem {
    let content = match (s.content_text, s.content_json) {
        (Some(text), None) => Content::Text(text),
        (None, Some(json)) => Content::Json(json),
        (Some(text), Some(json)) => Content::TextJson { text, json },
        _ => Content::Text(String::new()),
    };

    let kind = SurrealDBStore::str_to_kind(&s.kind).unwrap_or(MemoryKind::Event);

    MemoryItem {
        id: MemoryId(s.id),
        scope: ScopeKey {
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
}

#[async_trait::async_trait]
impl mom_core::MemoryStore for SurrealDBStore {
    async fn put(&self, item: MemoryItem) -> anyhow::Result<()> {
        let stored = Self::stored_item_from_memory_item(&item)?;

        // Upsert using MERGE statement
        let query = format!(
            "UPSERT {} MERGE {}",
            Self::record_ref(&item.id.0),
            serde_json::to_string(&stored)?
        );

        let _: Vec<StoredItemFromDb> = self.db.query(&query).await?.take(0)?;

        // US-7 AC-5: audit log for every memory write.
        info!(
            target: "mom.audit",
            op = "put",
            tenant_id = %item.scope.tenant_id,
            item_id = %item.id.0,
            outcome = "ok",
            "memory write"
        );
        debug!("Stored memory item: {}", item.id.0);
        Ok(())
    }

    /// **SECURITY**: This unscoped `get` reads by id ONLY — no tenant
    /// filter. Callers MUST verify tenant ownership before exposing the
    /// returned item to a caller; HTTP handlers should always go through
    /// [`MemoryStore::get_scoped`] instead.
    ///
    /// Kept as a trait method because the default `get_scoped` impl in
    /// `mom-core` uses it as its primitive (fetch-then-tenant-check). The
    /// SurrealDBStore override of `get_scoped` does its own scoped SELECT
    /// and does NOT call back here. Internal callers in this crate
    /// (vector_recall, hybrid_recall_impl) hydrate via `get_scoped` for
    /// belt-and-suspenders tenant isolation.
    async fn get(&self, id: &MemoryId) -> anyhow::Result<Option<MemoryItem>> {
        let query = format!("SELECT * FROM {}", Self::record_ref(&id.0));
        let rows: Vec<StoredItemFromDb> = self.db.query(&query).await?.take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();

        Ok(results.into_iter().next().map(stored_item_to_memory))
    }

    async fn get_scoped(
        &self,
        id: &MemoryId,
        scope: &ScopeKey,
    ) -> anyhow::Result<Option<MemoryItem>> {
        let query = format!(
            "SELECT * FROM {} WHERE tenant_id = $tenant_id",
            Self::record_ref(&id.0)
        );
        let rows: Vec<StoredItemFromDb> = self
            .db
            .query(&query)
            .bind(("tenant_id", scope.tenant_id.clone()))
            .await?
            .take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();

        let item = results.into_iter().next().map(stored_item_to_memory);
        // US-7 AC-5: audit log for every memory read.
        info!(
            target: "mom.audit",
            op = "get_scoped",
            tenant_id = %scope.tenant_id,
            item_id = %id.0,
            outcome = if item.is_some() { "ok" } else { "miss" },
            "memory read"
        );
        Ok(item)
    }

    async fn query(&self, q: Query) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        let sort_clause = if !q.text.is_empty() && q.cursor.is_none() {
            "ORDER BY importance DESC, created_at_ms DESC, id DESC"
        } else {
            "ORDER BY created_at_ms DESC, id DESC"
        };
        let query_str = Self::build_parameterized_query(&q, sort_clause, true);

        let mut query = self.db.query(&query_str);
        query = Self::bind_query_params(query, &q);
        query = query.bind(("limit", q.limit));

        let rows: Vec<StoredItemFromDb> = query.await?.take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();

        let mut scored = Vec::with_capacity(results.len());
        for (idx, item) in results.into_iter().enumerate() {
            // Simple scoring: importance + recency bonus
            let recency_bonus = (1.0 - (idx as f32 / q.limit as f32).min(1.0)) * 0.2;
            let score = (item.importance + recency_bonus).min(1.0);

            scored.push(Scored {
                score,
                item: stored_item_to_memory(item),
            });
        }

        debug!("Query found {} results", scored.len());
        // US-7 AC-5: audit log for every memory query.
        info!(
            target: "mom.audit",
            op = "query",
            tenant_id = %q.scope.tenant_id,
            outcome = "ok",
            result_count = scored.len(),
            "memory query"
        );
        Ok(scored)
    }

    /// **SECURITY**: This unscoped `delete` removes by id ONLY — no tenant
    /// filter. Callers MUST verify tenant ownership first. HTTP handlers
    /// should always use [`MemoryStore::delete_scoped`].
    ///
    /// Kept as a trait method because the default `delete_scoped` impl in
    /// `mom-core` uses it as its primitive (fetch-then-tenant-check). The
    /// SurrealDBStore override of `delete_scoped` does its own scoped
    /// DELETE and does NOT call back here.
    async fn delete(&self, id: &MemoryId) -> anyhow::Result<()> {
        let query = format!("DELETE {}", Self::record_ref(&id.0));
        let _: Vec<StoredItemFromDb> = self.db.query(&query).await?.take(0)?;
        debug!("Deleted memory item: {}", id.0);
        Ok(())
    }

    async fn delete_scoped(&self, id: &MemoryId, scope: &ScopeKey) -> anyhow::Result<()> {
        // SECURITY: Delete with tenant_id filter to enforce multi-tenant isolation at DB level
        // This ensures we can only delete items that belong to the calling tenant
        let query = format!(
            "DELETE {} WHERE tenant_id = $tenant_id",
            Self::record_ref(&id.0)
        );
        let _: Vec<StoredItemFromDb> = self
            .db
            .query(&query)
            .bind(("tenant_id", scope.tenant_id.clone()))
            .await?
            .take(0)?;
        // US-7 AC-5: audit log for every scoped delete (idempotent — we
        // don't know whether anything actually matched without a SELECT,
        // which would double the round-trip; the audit line records the
        // attempt).
        info!(
            target: "mom.audit",
            op = "delete_scoped",
            tenant_id = %scope.tenant_id,
            item_id = %id.0,
            outcome = "ok",
            "memory delete"
        );
        debug!(
            "Deleted memory item scoped to tenant: {} (id: {})",
            scope.tenant_id, id.0
        );
        Ok(())
    }

    async fn write_batch(
        &self,
        items: Vec<MemoryItem>,
        atomic: bool,
    ) -> anyhow::Result<Vec<MemoryId>> {
        if atomic {
            let mut query_str = String::new();
            query_str.push_str("BEGIN TRANSACTION;\n");
            let mut ids = Vec::with_capacity(items.len());
            for mut item in items {
                if item.id.0.is_empty() {
                    item.id = MemoryId(uuid::Uuid::new_v4().to_string());
                }
                let id = item.id.clone();

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
                    kind: Self::kind_to_str(item.kind),
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

                query_str.push_str(&format!(
                    "UPSERT {} MERGE {};\n",
                    Self::record_ref(&item.id.0),
                    serde_json::to_string(&stored)?
                ));
                ids.push(id);
            }
            query_str.push_str("COMMIT TRANSACTION;\n");

            let response = self.db.query(&query_str).await?;
            response.check()?;
            Ok(ids)
        } else {
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
    }

    async fn delete_batch(&self, ids: Vec<MemoryId>, atomic: bool) -> anyhow::Result<()> {
        if atomic {
            let mut query_str = String::new();
            query_str.push_str("BEGIN TRANSACTION;\n");
            for id in &ids {
                query_str.push_str(&format!("DELETE {};\n", Self::record_ref(&id.0)));
            }
            query_str.push_str("COMMIT TRANSACTION;\n");
            let response = self.db.query(&query_str).await?;
            response.check()?;
            Ok(())
        } else {
            for id in ids {
                self.delete(&id).await?;
            }
            Ok(())
        }
    }

    async fn delete_batch_scoped(
        &self,
        ids: Vec<MemoryId>,
        scope: &ScopeKey,
        atomic: bool,
    ) -> anyhow::Result<()> {
        if atomic {
            let mut query_str = String::new();
            query_str.push_str("BEGIN TRANSACTION;\n");
            for id in &ids {
                query_str.push_str(&format!(
                    "DELETE {} WHERE tenant_id = $tenant_id;\n",
                    Self::record_ref(&id.0)
                ));
            }
            query_str.push_str("COMMIT TRANSACTION;\n");
            let response = self
                .db
                .query(&query_str)
                .bind(("tenant_id", scope.tenant_id.clone()))
                .await?;
            response.check()?;

            for id in &ids {
                // US-7 AC-5: audit log for every scoped delete (idempotent)
                info!(
                    target: "mom.audit",
                    op = "delete_scoped",
                    tenant_id = %scope.tenant_id,
                    item_id = %id.0,
                    outcome = "ok",
                    "memory delete"
                );
            }
            Ok(())
        } else {
            for id in ids {
                self.delete_scoped(&id, scope).await?;
            }
            Ok(())
        }
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
            // US-7 AC-4: hydrate via the scoped path so a future change to
            // `semantic_recall` that drops the tenant filter (or a future
            // bug that returns ids outside the scope) cannot leak rows.
            // The underlying `semantic_recall` already filters by tenant +
            // sub-scope; this is the belt-and-suspenders guard.
            if let Some(item) = self.get_scoped(&memory_id, scope).await? {
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

    async fn query_batch(
        &self,
        queries: Vec<Query>,
    ) -> anyhow::Result<Vec<Vec<Scored<MemoryItem>>>> {
        let futures = queries.into_iter().map(|q| self.query(q));
        let results = futures::future::join_all(futures).await;

        let mut final_results = Vec::with_capacity(results.len());
        for res in results {
            final_results.push(res?);
        }
        Ok(final_results)
    }
}

impl SurrealDBStore {
    pub async fn query_items(&self, q: Query) -> anyhow::Result<Vec<MemoryItem>> {
        let query_str = Self::build_parameterized_query(&q, "ORDER BY created_at_ms ASC, id ASC", false);

        let mut query = self.db.query(&query_str);
        query = Self::bind_query_params(query, &q);

        let rows: Vec<StoredItemFromDb> = query.await?.take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();
        Ok(results.into_iter().map(stored_item_to_memory).collect())
    }

    pub async fn put_and_delete_atomic(
        &self,
        item: MemoryItem,
        delete_ids: &[MemoryId],
    ) -> anyhow::Result<()> {
        if delete_ids.is_empty() {
            return self.put(item).await;
        }

        let stored = Self::stored_item_from_memory_item(&item)?;
        let _ = self.db.query("BEGIN TRANSACTION;").await?;

        let tx_result: anyhow::Result<()> = async {
            let upsert_query = format!(
                "UPSERT {} MERGE {}",
                Self::record_ref(&item.id.0),
                serde_json::to_string(&stored)?
            );
            let _: Vec<StoredItemFromDb> = self.db.query(&upsert_query).await?.take(0)?;

            for id in delete_ids {
                let delete_query = format!("DELETE {}", Self::record_ref(&id.0));
                let _: Vec<StoredItemFromDb> = self.db.query(&delete_query).await?.take(0)?;
            }

            Ok(())
        }
        .await;

        match tx_result {
            Ok(()) => {
                let _ = self.db.query("COMMIT TRANSACTION;").await?;
                Ok(())
            }
            Err(err) => {
                let _ = self.db.query("ROLLBACK TRANSACTION;").await;
                Err(err)
            }
        }
    }
}
async fn lexical_recall(
    db: &Surreal<Db>,
    scope: &ScopeKey,
    query_text: &str,
    limit: usize,
) -> anyhow::Result<Vec<(String, f32)>> {
    // Build SurrealQL query for full-text search
    let mut query_str =
        "SELECT id, importance FROM memory_items WHERE tenant_id = $tenant_id".to_string();

    // Apply scope refinements
    if scope.workspace_id.is_some() {
        query_str.push_str(" AND workspace_id = $workspace_id");
    }
    if scope.project_id.is_some() {
        query_str.push_str(" AND project_id = $project_id");
    }
    if scope.agent_id.is_some() {
        query_str.push_str(" AND agent_id = $agent_id");
    }

    // Text match: search in content_text or tags
    if !query_text.is_empty() {
        query_str.push_str(" AND (content_text CONTAINS $text OR tags CONTAINS [$text])");
    }

    // Sort by importance, limit results
    query_str.push_str(" ORDER BY importance DESC, created_at_ms DESC LIMIT $limit");

    let mut query = db
        .query(&query_str)
        .bind(("tenant_id", scope.tenant_id.clone()))
        .bind(("limit", limit));

    if let Some(ref ws) = scope.workspace_id {
        query = query.bind(("workspace_id", ws.clone()));
    }
    if let Some(ref proj) = scope.project_id {
        query = query.bind(("project_id", proj.clone()));
    }
    if let Some(ref agent) = scope.agent_id {
        query = query.bind(("agent_id", agent.clone()));
    }

    if !query_text.is_empty() {
        query = query.bind(("text", query_text.to_string()));
    }

    let rows: Vec<StoredItemFromDb> = query.await?.take(0)?;
    let results: Vec<StoredItem> = rows.into_iter().map(SurrealDBStore::from_db_row).collect();

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
    // Vector similarity search - fetch all items with embeddings and compute cosine similarity
    let mut query_str = "SELECT id, embedding FROM memory_items WHERE tenant_id = $tenant_id AND embedding IS NOT NULL".to_string();

    // Apply scope refinements
    if scope.workspace_id.is_some() {
        query_str.push_str(" AND workspace_id = $workspace_id");
    }
    if scope.project_id.is_some() {
        query_str.push_str(" AND project_id = $project_id");
    }
    if scope.agent_id.is_some() {
        query_str.push_str(" AND agent_id = $agent_id");
    }

    // Order by created_at_ms for stable ordering before similarity computation
    query_str.push_str(" ORDER BY created_at_ms DESC LIMIT 1000");

    let mut query = db
        .query(&query_str)
        .bind(("tenant_id", scope.tenant_id.clone()));

    if let Some(ref ws) = scope.workspace_id {
        query = query.bind(("workspace_id", ws.clone()));
    }
    if let Some(ref proj) = scope.project_id {
        query = query.bind(("project_id", proj.clone()));
    }
    if let Some(ref agent) = scope.agent_id {
        query = query.bind(("agent_id", agent.clone()));
    }

    #[derive(Debug, Deserialize)]
    struct IdEmbeddingRow {
        id: RecordId,
        embedding: Option<Vec<f32>>,
    }

    let rows: Vec<IdEmbeddingRow> = query.await?.take(0)?;

    // Compute cosine similarity for each item
    let mut scored: Vec<(String, f32)> = rows
        .into_iter()
        .filter_map(|row| {
            let embedding = row.embedding?;
            let id = SurrealDBStore::record_id_to_string(&row.id);
            let similarity = cosine_similarity(query_embedding, &embedding);
            Some((id, similarity))
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

    // Fetch full items and rebuild Scored results.
    // US-7 AC-4: hydrate via `get_scoped` so a missing tenant filter in
    // the upstream RRF pipeline can't leak items across tenants. The
    // lexical / semantic helpers already filter by scope; this is the
    // belt-and-suspenders guard at the hydration step.
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
mod store_tests {
    use super::*;
    use mom_core::MemoryKind;
    use std::collections::BTreeMap;

    fn sample_item(id: &str) -> MemoryItem {
        MemoryItem {
            id: MemoryId(id.to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: Some("agent-1".to_string()),
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 1_700_000_000_000,
            content: Content::Text("hello surrealdb 2".to_string()),
            tags: vec!["test".to_string()],
            importance: 0.8,
            confidence: 0.9,
            source: "agent".to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        }
    }

    #[tokio::test]
    async fn put_get_roundtrip_surrealdb2() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let item = sample_item("roundtrip-1");
        store.put(item.clone()).await.unwrap();

        let fetched = store
            .get(&MemoryId("roundtrip-1".to_string()))
            .await
            .unwrap()
            .expect("item should exist");
        assert_eq!(fetched.id.0, "roundtrip-1");
        assert_eq!(fetched.scope.tenant_id, "acme");
        match fetched.content {
            Content::Text(t) => assert_eq!(t, "hello surrealdb 2"),
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn put_task_kind_surrealdb2() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let mut item = sample_item("task-1");
        item.kind = MemoryKind::Task;
        item.content = Content::Json(serde_json::json!({"status": "pending"}));
        store.put(item).await.unwrap();

        let fetched = store
            .get(&MemoryId("task-1".to_string()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.kind, MemoryKind::Task);
    }

    #[tokio::test]
    async fn delete_batch_surrealdb2() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let item1 = sample_item("del-1");
        let item2 = sample_item("del-2");
        store.put(item1).await.unwrap();
        store.put(item2).await.unwrap();

        assert!(store
            .get(&MemoryId("del-1".to_string()))
            .await
            .unwrap()
            .is_some());
        assert!(store
            .get(&MemoryId("del-2".to_string()))
            .await
            .unwrap()
            .is_some());

        let scope = ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: Some("agent-1".to_string()),
            run_id: None,
        };

        store
            .delete_batch_scoped(
                vec![MemoryId("del-1".to_string()), MemoryId("del-2".to_string())],
                &scope,
                false,
            )
            .await
            .unwrap();

        assert!(store
            .get(&MemoryId("del-1".to_string()))
            .await
            .unwrap()
            .is_none());
        assert!(store
            .get(&MemoryId("del-2".to_string()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn write_batch_surrealdb2() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let item1 = sample_item("batch-1");
        let item2 = sample_item("batch-2");
        let items = vec![item1, item2];

        let ids = store.write_batch(items, false).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].0, "batch-1");
        assert_eq!(ids[1].0, "batch-2");

        let fetched1 = store
            .get(&MemoryId("batch-1".to_string()))
            .await
            .unwrap()
            .expect("item 1 should exist");
        let fetched2 = store
            .get(&MemoryId("batch-2".to_string()))
            .await
            .unwrap()
            .expect("item 2 should exist");
        assert_eq!(fetched1.id.0, "batch-1");
        assert_eq!(fetched2.id.0, "batch-2");
    }

    /// US-19b (#64): batch delete via trait default impl loops `delete`.
    /// Verify N writes then a batch delete removes all rows; an unknown
    /// id in the batch is not an error (idempotent, mirroring `delete`).
    #[tokio::test]
    async fn delete_batch_default_impl_removes_all_and_is_idempotent() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let items: Vec<MemoryItem> = (0..3).map(|i| sample_item(&format!("del-{i}"))).collect();
        for it in &items {
            store.put(it.clone()).await.unwrap();
        }

        let mut to_delete: Vec<MemoryId> = items.iter().map(|it| it.id.clone()).collect();
        to_delete.push(MemoryId("does-not-exist".into()));

        store.delete_batch(to_delete, false).await.unwrap();

        for it in &items {
            let fetched = store.get(&it.id).await.unwrap();
            assert!(fetched.is_none(), "{} should be gone", it.id.0);
        }
    }

    #[tokio::test]
    async fn test_write_batch_atomic_rollback() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let item1 = sample_item("atomic-good");

        // item2 will violate schema constraints: importance must be <= 1.0, let's set to 2.0
        let mut item2 = sample_item("atomic-bad");
        item2.importance = 2.0;

        let items = vec![item1, item2];
        let res = store.write_batch(items, true).await;

        assert!(res.is_err());

        let fetched = store
            .get(&MemoryId("atomic-good".to_string()))
            .await
            .unwrap();
        assert!(
            fetched.is_none(),
            "Good item should not be persisted on transaction failure"
        );
    }

    /// US-19c (#65): batch query default impl runs each query in sequence
    /// and aligns results by input index. Seed two tenants with one item
    /// each, batch-query for both, assert per-tenant scope isolation in
    /// the aligned result slots.
    #[tokio::test]
    async fn query_batch_default_impl_returns_aligned_per_scope_results() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();

        // Tenant A item
        let mut a = sample_item("ta-1");
        a.scope.tenant_id = "tenant-a".into();
        store.put(a).await.unwrap();

        // Tenant B item
        let mut b = sample_item("tb-1");
        b.scope.tenant_id = "tenant-b".into();
        store.put(b).await.unwrap();

        let q_a = mom_core::Query {
            scope: ScopeKey {
                tenant_id: "tenant-a".into(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            text: String::new(),
            kinds: None,
            tags_any: None,
            limit: 10,
            since_ms: None,
            until_ms: None,
            cursor: None,
        };
        let q_b = mom_core::Query {
            scope: ScopeKey {
                tenant_id: "tenant-b".into(),
                ..q_a.scope.clone()
            },
            ..q_a.clone()
        };

        let results = store.query_batch(vec![q_a, q_b]).await.unwrap();
        assert_eq!(results.len(), 2, "two queries → two result slots");

        let ids_a: Vec<&str> = results[0].iter().map(|s| s.item.id.0.as_str()).collect();
        let ids_b: Vec<&str> = results[1].iter().map(|s| s.item.id.0.as_str()).collect();

        assert!(ids_a.contains(&"ta-1"), "slot 0 has tenant-a item");
        assert!(!ids_a.contains(&"tb-1"), "slot 0 must not leak tenant-b");
        assert!(ids_b.contains(&"tb-1"), "slot 1 has tenant-b item");
        assert!(!ids_b.contains(&"ta-1"), "slot 1 must not leak tenant-a");
    }

    #[tokio::test]
    async fn test_sql_injection_get_scoped() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();

        let mut item1 = sample_item("inj-1");
        item1.scope.tenant_id = "acme".into();
        store.put(item1).await.unwrap();

        let mut item2 = sample_item("inj-2");
        item2.scope.tenant_id = "other".into();
        store.put(item2).await.unwrap();

        // Let's test a few payloads to see if we can bypass the tenant check
        let payloads = vec![
            "acme\\' OR tenant_id = 'other".to_string(),
            "acme\\' OR tenant_id != '".to_string(),
            "acme\\' OR 1=1 --".to_string(),
            "acme\\' OR true //".to_string(),
        ];

        for payload in payloads {
            let scope = ScopeKey {
                tenant_id: payload.clone(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            };

            let res = store
                .get_scoped(&MemoryId("inj-2".to_string()), &scope)
                .await;
            if let Ok(Some(item)) = res {
                panic!(
                    "SQL Injection Succeeded with payload: {}! Returned item: {:?}",
                    payload, item
                );
            }
        }
    }
}
