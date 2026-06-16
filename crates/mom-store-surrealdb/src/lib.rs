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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use surrealdb::engine::local::{Db, Mem};
use surrealdb::Surreal;
use tracing::{debug, error};

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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StoredItem {
    memory_id: String,
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
            DEFINE TABLE memory_items SCHEMAFULL;
            DEFINE FIELD memory_id ON TABLE memory_items TYPE string ASSERT string::len($value) > 0;
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
///
/// The `'a` lifetime is the lifetime of the borrow the caller holds on
/// the SurrealDB connection; it threads through the returned `Query`
/// builder so the caller can keep chaining more `.bind(...)` calls
/// before awaiting. Typical usage:
///
/// ```ignore
/// let mut builder = self.db.query(query_str).bind(("tenant", t));
/// builder = bind_scope_filters(builder, &q.scope);
/// // ...more binds, then:
/// let results: Vec<StoredItem> = builder.await?.take(0)?;
/// ```
///
/// The signature is currently coupled to `Db` (the local in-memory
/// SurrealDB engine). If the store ever needs a non-`Db` backend
/// (e.g. `Ws` for remote / clustered SurrealDB) this is the one place
/// the engine type leaks through the helper API.
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

#[async_trait::async_trait]
impl mom_core::MemoryStore for SurrealDBStore {
    async fn put(&self, item: MemoryItem) -> anyhow::Result<()> {
        require_tenant_id(&item.scope.tenant_id)?;

        let (content_text, content_json) = match &item.content {
            Content::Text(t) => (Some(t.clone()), None),
            Content::Json(v) => (None, Some(v.clone())),
            Content::TextJson { text, json } => (Some(text.clone()), Some(json.clone())),
        };

        let stored = StoredItem {
            memory_id: item.id.0.clone(),
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

    /// **Tenant-unsafe.** Fetches by raw record id with no tenant or
    /// sub-scope filtering. Intended only for admin/migration paths or as
    /// the building block the default `MemoryStore::get_scoped` composes
    /// with `scope_matches`. Application code should call `get_scoped`
    /// instead — see the crate-level docstring on tenant isolation.
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
                id: MemoryId(s.memory_id),
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
        require_query_scope(scope)?;
        // SECURITY: filter on tenant_id AND every sub-scope field the
        // caller has set. Without the sub-scope filters two callers in
        // the same tenant but different workspaces could resolve each
        // other's items via point-lookup. Semantics match those used by
        // `MemoryStore::query` and the in-trait `scope_matches` helper.
        let mut query_str = String::from(
            "SELECT * FROM memory_items WHERE memory_id = $id AND tenant_id = $tenant",
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
        if scope.run_id.is_some() {
            query_str.push_str(" AND run_id = $run");
        }
        let mut builder = self
            .db
            .query(&query_str)
            .bind(("id", id.0.clone()))
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
        if let Some(ref run) = scope.run_id {
            builder = builder.bind(("run", run.clone()));
        }
        let results: Vec<StoredItem> = builder.await?.take(0)?;

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
                id: MemoryId(s.memory_id),
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
        require_query_scope(&q.scope)?;
        // Build SurrealQL query with tenant filter + optional refinements.
        // Clauses are appended conditionally; parameters are bound below.
        let mut query_str = String::from("SELECT * FROM memory_items WHERE tenant_id = $tenant");

        append_scope_where_clauses(&mut query_str, &q.scope);

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
        builder = bind_scope_filters(builder, &q.scope);
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
                    id: MemoryId(item.memory_id),
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

    /// **Tenant-unsafe.** Deletes by raw record id with no tenant or
    /// sub-scope filtering. Intended only for admin/migration paths or as
    /// the building block the default `MemoryStore::delete_scoped`
    /// composes with `scope_matches`. Application code should call
    /// `delete_scoped` instead — see the crate-level docstring.
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
        require_query_scope(scope)?;
        // SECURITY: filter on tenant_id AND every sub-scope field the
        // caller has set. Same rationale + semantics as `get_scoped`
        // above — without the sub-scope clauses a delete in the same
        // tenant but different workspace would silently wipe another
        // workspace's item.
        let mut query_str =
            String::from("DELETE memory_items WHERE memory_id = $id AND tenant_id = $tenant");
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
        let mut builder = self
            .db
            .query(&query_str)
            .bind(("id", id.0.clone()))
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
        if let Some(ref run) = scope.run_id {
            builder = builder.bind(("run", run.clone()));
        }
        let _: Vec<StoredItem> = builder.await?.take(0)?;
        debug!(
            "Deleted memory item scoped to tenant {} workspace {:?} (id: {})",
            scope.tenant_id, scope.workspace_id, id.0
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
        require_query_scope(scope)?;
        let results = semantic_recall(&self.db, scope, query_embedding, limit).await?;

        let mut scored = Vec::with_capacity(results.len());
        for (id, score) in results {
            let memory_id = MemoryId(id);
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
        require_query_scope(&q.scope)?;
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
        String::from("SELECT memory_id, importance FROM memory_items WHERE tenant_id = $tenant");

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

    let results: Vec<IdImportanceRow> = builder.await?.take(0)?;

    let scored: Vec<(String, f32)> = results
        .into_iter()
        .map(|item| (item.memory_id, item.importance))
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
        "SELECT memory_id, embedding FROM memory_items WHERE tenant_id = $tenant AND embedding IS NOT NULL",
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

    let results: Vec<EmbeddingRow> = builder.await?.take(0)?;

    // Compute cosine similarity for each item
    let mut scored: Vec<(String, f32)> = results
        .into_iter()
        .map(|item| {
            let similarity = cosine_similarity(query_embedding, &item.embedding);
            (item.memory_id, similarity)
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

    // Spike for #36 (kept as a regression guard): empirically demonstrates
    // that SurrealDB table PERMISSIONS clauses gated on a session variable
    // are not enforced for system users (root, OWNER, EDITOR, VIEWER) on
    // the Mem engine, regardless of whether the variable is set via
    // db.set or per-query bind. This is why the schema no longer ships
    // PERMISSIONS clauses on memory_items / memory_links — re-introducing
    // them would only enforce under DEFINE ACCESS / record-user signin,
    // which is tracked as a separate architectural follow-up. Run with
    // `cargo test -p mom-store-surrealdb spike_ -- --nocapture --ignored`
    // to re-verify if/when SurrealDB changes this semantic.
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
        #[allow(dead_code)] // fields read by the Debug print only.
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

        // CASE 4: DEFINE USER at DB level with VIEWER role (least privileged).
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

    // Cross-tenant and sub-scope integration tests below run against the live
    // in-memory SurrealDB store. They cover both tenant_id isolation and the
    // four sub-scope dimensions (workspace, project, agent, run) end-to-end
    // for `get_scoped`, `query`, and `delete_scoped`. Closes #27.

    fn sample_item(tenant_id: &str, memory_id: &str, text: &str) -> MemoryItem {
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

    #[tokio::test]
    async fn cross_tenant_query_returns_empty() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        store
            .put(sample_item("tenant-a", "mem-a", "tenant-a secret"))
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
            .put(sample_item("tenant-a", "mem-shared-id", "secret"))
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
        let mut item = sample_item("tenant-a", "mem-x", "data");
        item.scope.tenant_id = "  ".into();
        let err = store.put(item).await.expect_err("blank tenant rejected");
        assert!(err.to_string().contains("tenant_id is required"));
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
        }
    }

    // Same-tenant / different-workspace items must not see each other through
    // `query`. Regression guard for #3 item #2 — pre-fix, `get_scoped` /
    // `delete_scoped` ignored sub-scope filters at the SQL layer.
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

    // Regression guard for the #24 SQL gap: prior to the fix, `query` dropped
    // the `run_id` filter so two callers in the same agent but different runs
    // could see each other's items.
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

    // Point-lookup must refuse to return an item that exists in the tenant
    // but under a different sub-scope. Covers every sub-scope field — workspace,
    // project, agent, run — in one pass so a regression in any of the four
    // SQL conditional clauses surfaces immediately.
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

        // Sanity: exact scope hits.
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

    // delete_scoped must not remove an item whose sub-scope differs from the
    // caller's, even when the tenant and memory_id match. Regression guard for
    // the same SQL gap covered by #23 / #24, exercised against delete instead
    // of get / query so all three scoped operations are end-to-end verified.
    #[tokio::test]
    async fn delete_scoped_refuses_to_delete_other_subscope() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        let target_scope = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-keep"),
        );
        let other_scope = scope(
            "tenant-a",
            Some("ws-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("run-other"),
        );
        store
            .put(scoped_item(target_scope.clone(), "mem-keepme", "preserve"))
            .await
            .expect("put target");

        // Delete attempt against the other sub-scope is a no-op.
        store
            .delete_scoped(&MemoryId("mem-keepme".into()), &other_scope)
            .await
            .expect("delete_scoped does not error on wrong sub-scope");

        let after = store
            .get_scoped(&MemoryId("mem-keepme".into()), &target_scope)
            .await
            .expect("post-delete scoped get");
        assert!(
            after.is_some(),
            "delete_scoped must not remove the item when the caller's sub-scope differs"
        );

        // Delete against the matching sub-scope does remove it.
        store
            .delete_scoped(&MemoryId("mem-keepme".into()), &target_scope)
            .await
            .expect("delete_scoped exact");
        let after_exact = store
            .get_scoped(&MemoryId("mem-keepme".into()), &target_scope)
            .await
            .expect("post-exact-delete scoped get");
        assert!(
            after_exact.is_none(),
            "delete_scoped with matching sub-scope must remove the item"
        );
    }

    // The tests below exercise the SQL-clause builder directly so the
    // scope-filter coverage doesn't have to wait on the schema rework.

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
    fn append_scope_where_clauses_includes_run_id_when_set() {
        // Regression guard for the gap previously present in `MemoryStore::query`:
        // workspace_id / project_id / agent_id were filtered but `run_id` was
        // silently dropped, so two callers in the same agent but different runs
        // saw each other's items.
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
        // Order matters for SurrealQL planner determinism. Mirror the order
        // the bind helper uses so a mismatch surfaces as a missing $param,
        // not a silently-wrong query.
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
}
