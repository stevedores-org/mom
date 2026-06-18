//! MOM SurrealDB Store - Multi-model persistence layer
//!
//! Leverages SurrealDB's document model, relationships, and queries
//! for efficient memory storage and hybrid retrieval.
//!
//! # Tenant isolation
//!
//! Tenant and sub-scope isolation is enforced **at the Rust layer** by the
//! `WHERE tenant_id = $tenant AND <sub-scope clauses>` filters that every
//! scoped accessor (`get_scoped`, `query`, `delete_scoped`) builds before
//! sending to SurrealDB. This is the sole isolation mechanism today.
//!
//! Earlier versions of the schema carried SCHEMAFULL PERMISSIONS clauses
//! intended as defense-in-depth. They were dropped in the resolution of
//! [#36] after an empirical spike showed they did not enforce: SurrealDB
//! table PERMISSIONS only apply to record-level users authenticated via
//! `DEFINE ACCESS`. System users (root, OWNER, EDITOR, VIEWER — the only
//! options when the application owns the connection) bypass them entirely.
//! Leaving the clauses in the schema invited a false sense of security.
//!
//! Re-introducing real DB-level enforcement is tracked separately as an
//! architectural follow-up: per-tenant sessions backed by `DEFINE ACCESS`
//! and a record user per tenant. Until that lands, **the WHERE-clause
//! filter and its integration test suite (`mod tests` in this file) are
//! the contract** that callers and reviewers must hold.
//!
//! Bare `MemoryStore::get(&MemoryId)` and `MemoryStore::delete(&MemoryId)`
//! deliberately have no tenant argument; they are tenant-unsafe primitives
//! that the default `get_scoped` / `delete_scoped` implementations in
//! `mom-core` compose with `scope_matches`. Direct callers of the bare
//! variants bypass tenant isolation and should be limited to admin /
//! migration / introspection code paths.
//!
//! ## Layer diagram
//!
//! ```text
//!  caller (multi-tenant service)
//!    │
//!    │  ScopeKey { tenant_id, workspace_id, project_id, agent_id, run_id }
//!    ▼
//!  MemoryStore::{get_scoped, query, delete_scoped}
//!    │  builds:  WHERE tenant_id = $tenant
//!    │       AND workspace_id = $workspace      (when set)
//!    │       AND project_id   = $project        (when set)
//!    │       AND agent_id     = $agent          (when set)
//!    │       AND run_id       = $run            (when set)
//!    ▼
//!  SurrealDB (Mem / kv-tikv / kv-rocksdb)
//!    └─ SCHEMAFULL definitions only; no PERMISSIONS until #40 lands.
//! ```
//!
//! ## Adding a new tenant-scoped table
//!
//! When a future PR adds another table that holds tenant data, follow
//! this checklist so isolation doesn't silently regress:
//!
//! 1. **Schema.** The table is `DEFINE TABLE foo SCHEMAFULL;` (no
//!    PERMISSIONS clause until [#40]). Include a `tenant_id` column
//!    with `TYPE string ASSERT string::len($value) > 0;` and a
//!    `DEFINE INDEX` keyed on `tenant_id` plus any sub-scope fields
//!    you filter on.
//! 2. **Rust accessors.** Every read path filters on
//!    `tenant_id = $tenant` *and* every sub-scope field the caller has
//!    set, mirroring the conditional-clause pattern in `query` /
//!    `get_scoped` / `delete_scoped`. Every write path stores the
//!    caller's `ScopeKey` verbatim.
//!
//!    Look-alike helpers worth reusing: `append_scope_where_clauses` /
//!    `bind_scope_filters` in this file already handle the
//!    workspace/project/agent/run conditional pattern; route through
//!    them rather than re-implementing.
//! 3. **Validation.** Reject empty `tenant_id` at the API boundary via
//!    `mom_core::require_tenant_id` and reject empty query scope via
//!    `mom_core::require_query_scope`.
//! 4. **Tests.** Add the four test shapes from `mod tests` against the
//!    new table:
//!    - cross-tenant query returns empty,
//!    - sub-scope query / point-lookup / delete each refuse a
//!      mismatched scope on every sub-scope field they support.
//! 5. **Bare `get` / `delete`.** If the new table needs raw-id
//!    accessors, mirror the tenant-unsafe doc warning on the impls
//!    so callers don't reach for them by default.
//!
//! See also: #34 (Rust-layer WHERE-clause coverage merged), #36 (why
//! PERMISSIONS were dropped), #40 (the architecture work that would
//! re-introduce DB-level enforcement).

use mom_core::{
    require_query_scope, require_tenant_id, Content, MemoryId, MemoryItem, MemoryKind, MemoryStore,
    Query, ScopeKey, Scored,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use surrealdb::engine::local::{Db, Mem};
use surrealdb::RecordId;
use surrealdb::Surreal;
use tracing::{debug, error, info};

pub mod hybrid;
pub mod links;

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

#[derive(Debug, Deserialize)]
struct IdImportanceRow {
    memory_id: String,
    importance: f32,
}

#[derive(Debug, Deserialize)]
struct EmbeddingRow {
    memory_id: String,
    embedding: Vec<f32>,
}

impl SurrealDBStore {
    fn escape_sql_string(s: &str) -> String {
        s.replace('\\', "\\\\").replace('\'', "\\'")
    }

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
        // Tenant isolation is enforced by the Rust-layer WHERE clauses on
        // every scoped accessor; see the crate-level docstring and #36 for
        // why we no longer ship inert SCHEMAFULL PERMISSIONS clauses here.
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

            DEFINE TABLE memory_links SCHEMAFULL;
            DEFINE FIELD link_id ON TABLE memory_links TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD tenant_id ON TABLE memory_links TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD src_memory_id ON TABLE memory_links TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD dst_memory_id ON TABLE memory_links TYPE string ASSERT string::len($value) > 0;
            DEFINE FIELD rel ON TABLE memory_links TYPE string ASSERT $value IN ['causal', 'derived_from', 'contradicts', 'same_as', 'references'];
            DEFINE FIELD weight ON TABLE memory_links TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD confidence ON TABLE memory_links TYPE number ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD created_at_ms ON TABLE memory_links TYPE number;
            DEFINE INDEX idx_links_tenant_src ON TABLE memory_links COLUMNS tenant_id, src_memory_id;
            DEFINE INDEX idx_links_tenant_dst ON TABLE memory_links COLUMNS tenant_id, dst_memory_id;
            DEFINE INDEX idx_links_rel ON TABLE memory_links COLUMNS tenant_id, rel;
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
        if item.scope.tenant_id.trim().is_empty() {
            return Err(anyhow::anyhow!("tenant_id is required"));
        }
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
}

/// Appends an `AND <field> = $<param>` clause to `query_str` for each
/// sub-scope field the caller has set. Used by every read / write
/// method that filters by scope so a missing field can never silently
/// widen the result set.
///
/// **Canonical order: `workspace_id → project_id → agent_id → run_id`.**
/// Reordering this function without reordering `bind_scope_filters` will
/// produce a parameter-name mismatch the SurrealDB driver surfaces at
/// query time. The
/// `append_scope_where_clauses_emits_all_set_fields_in_canonical_order`
/// test guards against accidental re-shuffles.
///
/// Symmetric with `bind_scope_filters` below; keep the two in lockstep.
/// The same inline pattern lives in [`MemoryStore::get_scoped`] /
/// [`MemoryStore::delete_scoped`] (post-#23) and should be DRYed up
/// onto these helpers as a follow-up.
fn append_scope_where_clauses(query_str: &mut String, scope: &ScopeKey) {
    if scope.workspace_id.is_some() {
        query_str.push_str(" AND workspace_id = $workspace");
    }
    if scope.project_id.is_some() {
        query_str.push_str(" AND project_id = $project");
    }
    if scope.agent_id.is_some() {
        query_str.push_str(" AND agent_id = $agent");
    }
    if scope.run_id.is_some() {
        query_str.push_str(" AND run_id = $run");
    }
}

/// Binds each sub-scope parameter that the caller has set. Pair with
/// `append_scope_where_clauses` above — order MUST match.
fn bind_scope_filters<'a>(
    mut builder: surrealdb::method::Query<'a, Db>,
    scope: &ScopeKey,
) -> surrealdb::method::Query<'a, Db> {
    if let Some(ref ws) = scope.workspace_id {
        builder = builder.bind(("workspace", ws.clone()));
    }
    if let Some(ref proj) = scope.project_id {
        builder = builder.bind(("project", proj.clone()));
    }
    if let Some(ref agent) = scope.agent_id {
        builder = builder.bind(("agent", agent.clone()));
    }
    if let Some(ref run) = scope.run_id {
        builder = builder.bind(("run", run.clone()));
    }
    builder
}

impl SurrealDBStore {

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
        if q.scope.run_id.is_some() {
            query_str.push_str(" AND run_id = $run_id");
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
        if let Some(ref run) = q.scope.run_id {
            query = query.bind(("run_id", run.clone()));
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
        require_query_scope(scope)?;
        let mut query_str = format!(
            "SELECT * FROM {} WHERE tenant_id = $tenant_id",
            Self::record_ref(&id.0)
        );
        if scope.workspace_id.is_some() {
            query_str.push_str(" AND workspace_id = $workspace_id");
        }
        if scope.project_id.is_some() {
            query_str.push_str(" AND project_id = $project_id");
        }
        if scope.agent_id.is_some() {
            query_str.push_str(" AND agent_id = $agent_id");
        }
        if scope.run_id.is_some() {
            query_str.push_str(" AND run_id = $run_id");
        }

        let mut query = self
            .db
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
        if let Some(ref run) = scope.run_id {
            query = query.bind(("run_id", run.clone()));
        }

        let rows: Vec<StoredItemFromDb> = query.await?.take(0)?;
        let results: Vec<StoredItem> = rows.into_iter().map(Self::from_db_row).collect();

        let item = results.into_iter().next().and_then(|s| {
            if Self::is_expired(s.created_at_ms, s.ttl_ms, Self::current_time_ms()) {
                return None;
            }
            Some(stored_item_to_memory(s))
        });

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
        require_query_scope(scope)?;
        let mut query_str = format!(
            "DELETE {} WHERE tenant_id = $tenant_id",
            Self::record_ref(&id.0)
        );
        if scope.workspace_id.is_some() {
            query_str.push_str(" AND workspace_id = $workspace_id");
        }
        if scope.project_id.is_some() {
            query_str.push_str(" AND project_id = $project_id");
        }
        if scope.agent_id.is_some() {
            query_str.push_str(" AND agent_id = $agent_id");
        }
        if scope.run_id.is_some() {
            query_str.push_str(" AND run_id = $run_id");
        }

        let mut query = self
            .db
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
        if let Some(ref run) = scope.run_id {
            query = query.bind(("run_id", run.clone()));
        }

        let _: Vec<StoredItemFromDb> = query.await?.take(0)?;

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
            "Deleted memory item scoped to tenant {} workspace {:?} (id: {})",
            scope.tenant_id, scope.workspace_id, id.0
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
        require_query_scope(scope)?;
        let results = semantic_recall(&self.db, scope, query_embedding, limit).await?;
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<String> = results
            .iter()
            .map(|(id, _)| {
                format!(
                    "type::thing('memory_items', '{}')",
                    Self::escape_sql_string(id)
                )
            })
            .collect();
        let ids_clause = ids.join(", ");
        let query = format!(
            "SELECT * FROM memory_items WHERE id IN [{}] AND tenant_id = $tenant_id",
            ids_clause
        );
        let rows: Vec<StoredItemFromDb> = self
            .db
            .query(&query)
            .bind(("tenant_id", scope.tenant_id.clone()))
            .await?
            .take(0)?;

        let mut items_map: std::collections::HashMap<String, MemoryItem> = rows
            .into_iter()
            .map(Self::from_db_row)
            .map(stored_item_to_memory)
            .map(|item| (item.id.0.clone(), item))
            .collect();

        let mut scored = Vec::with_capacity(results.len());
        for (id, score) in results {
            if let Some(item) = items_map.remove(&id) {
                // US-7 AC-5: audit log for every memory read.
                info!(
                    target: "mom.audit",
                    op = "get_scoped",
                    tenant_id = %scope.tenant_id,
                    item_id = %id,
                    outcome = "ok",
                    "memory read"
                );
                scored.push(Scored { score, item });
            } else {
                // US-7 AC-5: audit log for every memory read.
                info!(
                    target: "mom.audit",
                    op = "get_scoped",
                    tenant_id = %scope.tenant_id,
                    item_id = %id,
                    outcome = "miss",
                    "memory read"
                );
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
        require_query_scope(&q.scope)?;
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
        let query_str =
            Self::build_parameterized_query(&q, "ORDER BY created_at_ms ASC, id ASC", false);

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
mod tests {
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

    fn sample_item_custom(tenant_id: &str, memory_id: &str, text: &str) -> MemoryItem {
        MemoryItem {
            id: MemoryId(memory_id.to_string()),
            scope: scope(tenant_id, None, None, None, None),
            kind: MemoryKind::Event,
            created_at_ms: 1_700_000_000_000,
            content: Content::Text(text.to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        }
    }

    fn scoped_item(scope: ScopeKey, memory_id: &str, text: &str) -> MemoryItem {
        MemoryItem {
            id: MemoryId(memory_id.to_string()),
            scope,
            kind: MemoryKind::Event,
            created_at_ms: 1_700_000_000_000,
            content: Content::Text(text.to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        }
    }

    fn unscored_query(scope: ScopeKey) -> Query {
        Query {
            scope,
            text: String::new(),
            kinds: None,
            tags_any: None,
            limit: 10,
            since_ms: None,
            until_ms: None,
            cursor: None,
        }
    }

    fn scope(
        tenant: &str,
        workspace: Option<&str>,
        project: Option<&str>,
        agent: Option<&str>,
        run: Option<&str>,
    ) -> ScopeKey {
        ScopeKey {
            tenant_id: tenant.to_string(),
            workspace_id: workspace.map(String::from),
            project_id: project.map(String::from),
            agent_id: agent.map(String::from),
            run_id: run.map(String::from),
        }
    }

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

    #[tokio::test]
    #[ignore]
    async fn spike_table_permissions_inert_for_system_users() {
        let db: Surreal<Db> = Surreal::new::<Mem>(()).await.expect("db");
        db.use_ns("spike").use_db("main").await.expect("use ns");
        db.query(
            r#"
            DEFINE TABLE items SCHEMAFULL PERMISSIONS
              FOR select WHERE tenant_id = $scope_tenant_id
              FOR create WHERE tenant_id = $scope_tenant_id;
            DEFINE FIELD memory_id ON TABLE items TYPE string;
            DEFINE FIELD tenant_id ON TABLE items TYPE string;
            "#,
        )
        .await
        .expect("schema");
        db.query("CREATE items SET memory_id='a', tenant_id='tenant-a'")
            .await
            .expect("create a");
        db.query("CREATE items SET memory_id='b', tenant_id='tenant-b'")
            .await
            .expect("create b");

        #[derive(serde::Deserialize, Debug)]
        #[allow(dead_code)]
        struct Row {
            memory_id: String,
            tenant_id: String,
        }

        let r: Vec<Row> = db
            .query("SELECT memory_id, tenant_id FROM items")
            .await
            .expect("select1")
            .take(0)
            .expect("take1");
        println!("CASE 1 (no var, root session): {} rows -> {:?}", r.len(), r);

        db.set("scope_tenant_id", "tenant-a")
            .await
            .expect("set var");
        let r: Vec<Row> = db
            .query("SELECT memory_id, tenant_id FROM items")
            .await
            .expect("select2")
            .take(0)
            .expect("take2");
        println!(
            "CASE 2 (db.set tenant-a, root session): {} rows -> {:?}",
            r.len(),
            r
        );

        let r: Vec<Row> = db
            .query("SELECT memory_id, tenant_id FROM items")
            .bind(("scope_tenant_id", "tenant-b"))
            .await
            .expect("select3")
            .take(0)
            .expect("take3");
        println!(
            "CASE 3 (bind tenant-b, root session): {} rows -> {:?}",
            r.len(),
            r
        );

        let define_user = db
            .query("DEFINE USER tenant_a ON DATABASE PASSWORD 'pw' ROLES VIEWER")
            .await;
        println!("CASE 4 DEFINE USER result: {:?}", define_user.is_ok());
        if define_user.is_ok() {
            let signin = db
                .signin(surrealdb::opt::auth::Database {
                    namespace: "spike",
                    database: "main",
                    username: "tenant_a",
                    password: "pw",
                })
                .await;
            println!("CASE 4 signin result: {:?}", signin.is_ok());
            if signin.is_ok() {
                db.set("scope_tenant_id", "tenant-a")
                    .await
                    .expect("set var case 4");
                let r: Vec<Row> = db
                    .query("SELECT memory_id, tenant_id FROM items")
                    .await
                    .expect("select4")
                    .take(0)
                    .expect("take4");
                println!(
                    "CASE 4 (DEFINE USER + signin + var=tenant-a): {} rows",
                    r.len()
                );
            }
        }
    }

    #[tokio::test]
    async fn cross_tenant_query_returns_empty() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        store
            .put(sample_item_custom("tenant-a", "mem-a", "tenant-a secret"))
            .await
            .expect("put tenant-a");

        let results = store
            .query(Query {
                scope: scope("tenant-b", None, None, None, None),
                text: String::new(),
                kinds: None,
                tags_any: None,
                limit: 10,
                since_ms: None,
                until_ms: None,
                cursor: None,
            })
            .await
            .expect("query tenant-b");

        assert!(
            results.is_empty(),
            "tenant-b must not see tenant-a memories"
        );
    }

    #[tokio::test]
    async fn cross_tenant_scoped_get_returns_none() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        store
            .put(sample_item_custom("tenant-a", "mem-shared-id", "secret"))
            .await
            .expect("put tenant-a");

        let cross = store
            .get_scoped(
                &MemoryId("mem-shared-id".into()),
                &scope("tenant-b", None, None, None, None),
            )
            .await
            .expect("scoped get tenant-b");

        assert!(cross.is_none(), "cross-tenant scoped get must miss");
    }

    #[tokio::test]
    async fn put_rejects_blank_tenant_id() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let mut item = sample_item_custom("tenant-a", "mem-x", "data");
        item.scope.tenant_id = "  ".into();
        let err = store.put(item).await.expect_err("blank tenant rejected");
        assert!(err.to_string().contains("tenant_id is required"));
    }

    #[tokio::test]
    async fn query_isolates_by_workspace_id_in_same_tenant() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let ws_a = scope("tenant-a", Some("ws-alpha"), None, None, None);
        let ws_b = scope("tenant-a", Some("ws-bravo"), None, None, None);
        store
            .put(scoped_item(ws_a.clone(), "mem-alpha", "alpha secret"))
            .await
            .expect("put ws-alpha");
        store
            .put(scoped_item(ws_b.clone(), "mem-bravo", "bravo secret"))
            .await
            .expect("put ws-bravo");

        let alpha_results = store
            .query(unscored_query(ws_a))
            .await
            .expect("query alpha");
        let bravo_results = store
            .query(unscored_query(ws_b))
            .await
            .expect("query bravo");

        let alpha_ids: Vec<_> = alpha_results.iter().map(|s| s.item.id.0.clone()).collect();
        let bravo_ids: Vec<_> = bravo_results.iter().map(|s| s.item.id.0.clone()).collect();
        assert_eq!(alpha_ids, vec!["mem-alpha".to_string()]);
        assert_eq!(bravo_ids, vec!["mem-bravo".to_string()]);
    }

    #[tokio::test]
    async fn query_isolates_by_run_id_in_same_agent() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let run_1 = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-1"),
        );
        let run_2 = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-2"),
        );
        store
            .put(scoped_item(run_1.clone(), "mem-run-1", "first run"))
            .await
            .expect("put run-1");
        store
            .put(scoped_item(run_2.clone(), "mem-run-2", "second run"))
            .await
            .expect("put run-2");

        let run_1_results = store
            .query(unscored_query(run_1))
            .await
            .expect("query run-1");
        let run_2_results = store
            .query(unscored_query(run_2))
            .await
            .expect("query run-2");

        assert_eq!(
            run_1_results
                .iter()
                .map(|s| s.item.id.0.as_str())
                .collect::<Vec<_>>(),
            vec!["mem-run-1"]
        );
        assert_eq!(
            run_2_results
                .iter()
                .map(|s| s.item.id.0.as_str())
                .collect::<Vec<_>>(),
            vec!["mem-run-2"]
        );
    }

    #[tokio::test]
    async fn get_scoped_misses_when_any_subscope_differs() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let stored_scope = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-1"),
        );
        store
            .put(scoped_item(stored_scope.clone(), "mem-shared", "secret"))
            .await
            .expect("put stored");

        let hit = store
            .get_scoped(&MemoryId("mem-shared".into()), &stored_scope)
            .await
            .expect("exact scoped get");
        assert!(hit.is_some(), "exact-scope point-lookup must succeed");

        let id = MemoryId("mem-shared".into());
        for (label, mismatched) in [
            (
                "workspace_id",
                scope(
                    "tenant-a",
                    Some("ws-2"),
                    Some("proj-1"),
                    Some("agent-1"),
                    Some("run-1"),
                ),
            ),
            (
                "project_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-2"),
                    Some("agent-1"),
                    Some("run-1"),
                ),
            ),
            (
                "agent_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-1"),
                    Some("agent-2"),
                    Some("run-1"),
                ),
            ),
            (
                "run_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-1"),
                    Some("agent-1"),
                    Some("run-2"),
                ),
            ),
        ] {
            let miss = store
                .get_scoped(&id, &mismatched)
                .await
                .expect("scoped get does not error");
            assert!(
                miss.is_none(),
                "scoped get must miss when {label} differs but the id matches"
            );
        }
    }

    #[tokio::test]
    async fn delete_scoped_refuses_to_delete_other_subscope() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let target_scope = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-1"),
        );
        let id = MemoryId("mem-keepme".into());
        store
            .put(scoped_item(target_scope.clone(), &id.0, "preserve"))
            .await
            .expect("put target");

        for (label, mismatched) in [
            (
                "workspace_id",
                scope(
                    "tenant-a",
                    Some("ws-2"),
                    Some("proj-1"),
                    Some("agent-1"),
                    Some("run-1"),
                ),
            ),
            (
                "project_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-2"),
                    Some("agent-1"),
                    Some("run-1"),
                ),
            ),
            (
                "agent_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-1"),
                    Some("agent-2"),
                    Some("run-1"),
                ),
            ),
            (
                "run_id",
                scope(
                    "tenant-a",
                    Some("ws-1"),
                    Some("proj-1"),
                    Some("agent-1"),
                    Some("run-2"),
                ),
            ),
        ] {
            store
                .delete_scoped(&id, &mismatched)
                .await
                .expect("delete_scoped does not error on wrong sub-scope");
            let still_there = store
                .get_scoped(&id, &target_scope)
                .await
                .expect("post-delete scoped get");
            assert!(
                still_there.is_some(),
                "delete_scoped must not remove the item when {label} differs"
            );
        }

        store
            .delete_scoped(&id, &target_scope)
            .await
            .expect("delete_scoped exact");
        let after_exact = store
            .get_scoped(&id, &target_scope)
            .await
            .expect("post-exact-delete scoped get");
        assert!(
            after_exact.is_none(),
            "delete_scoped with matching sub-scope must remove the item"
        );
    }

    #[test]
    fn append_scope_where_clauses_includes_run_id_when_set() {
        let mut sql = String::new();
        append_scope_where_clauses(&mut sql, &scope("acme", None, None, None, Some("run-1")));
        assert!(
            sql.contains("AND run_id = $run"),
            "expected `run_id` clause when scope.run_id = Some; got `{sql}`"
        );
    }

    #[test]
    fn append_scope_where_clauses_emits_no_run_id_when_unset() {
        let mut sql = String::new();
        append_scope_where_clauses(&mut sql, &scope("acme", Some("w1"), None, None, None));
        assert!(
            !sql.contains("run_id"),
            "run_id should be unconstrained when scope.run_id = None; got `{sql}`"
        );
    }

    #[test]
    fn append_scope_where_clauses_emits_all_set_fields_in_canonical_order() {
        let mut sql = String::new();
        append_scope_where_clauses(
            &mut sql,
            &scope("acme", Some("w1"), Some("p1"), Some("a1"), Some("r1")),
        );
        let ws = sql.find("workspace_id").expect("workspace clause");
        let proj = sql.find("project_id").expect("project clause");
        let agent = sql.find("agent_id").expect("agent clause");
        let run = sql.find("run_id").expect("run clause");
        assert!(
            ws < proj && proj < agent && agent < run,
            "clauses out of canonical order: `{sql}`"
        );
    }

    #[test]
    fn append_scope_where_clauses_emits_nothing_when_all_unset() {
        let mut sql = String::new();
        append_scope_where_clauses(&mut sql, &scope("acme", None, None, None, None));
        assert!(
            sql.is_empty(),
            "expected empty clause when only tenant_id is set; got `{sql}`"
        );
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

    #[tokio::test]
    async fn query_batch_default_impl_returns_aligned_per_scope_results() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();

        let mut a = sample_item("ta-1");
        a.scope.tenant_id = "tenant-a".into();
        store.put(a).await.unwrap();

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
