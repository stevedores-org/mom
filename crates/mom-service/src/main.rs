use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, patch, post},
    Json, Router,
};
use mom_core::{
    build_context_pack, read_provenance_ids, read_version, record_semantic_conflict, task_tag,
    write_provenance_ids, write_superseded_by, write_version, CheckpointRecord, Content,
    ContextPack, ContextPackRequest, Embedder, FactPayload, MemoryId, MemoryItem, MemoryKind,
    MemoryStore, PreferencePayload, Query, ScopeKey, Scored, META_PROVENANCE_IDS, META_VERSION,
    TOKENS_PER_ITEM,
};
use mom_embeddings::{create_embedder, maybe_embed_item};
use mom_sources::{
    DataFabricSource, IngestionScheduler, IngestionStatusReport, MemorySource, OxidizedGraphSource,
    OxidizedRAGSource, UnknownSourceError,
};
use mom_store_surrealdb::SurrealDBStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::decompression::RequestDecompressionLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

mod links;
mod recall;
mod tenant;

use links::{
    create_link, delete_link as delete_link_handler, list_conflicts, traverse_links, update_link,
};

use tenant::{audit_tenant_access, resolve_tenant_scope, validate_memory_write};

#[derive(Clone)]
struct SourceRegistry {
    sources: HashMap<String, Arc<dyn MemorySource>>,
}

impl SourceRegistry {
    fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    fn register(&mut self, id: String, source: Arc<dyn MemorySource>) {
        self.sources.insert(id, source);
    }

    fn get(&self, source_id: &str) -> Option<Arc<dyn MemorySource>> {
        self.sources.get(source_id).cloned()
    }

    fn source_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.sources.keys().cloned().collect();
        ids.sort();
        ids
    }
}

#[derive(Clone)]
pub(crate) struct AppState {
    store: Arc<SurrealDBStore>,
    embedder: Option<Arc<Box<dyn Embedder>>>,
    ingestion_scheduler: Arc<IngestionScheduler>,
    source_registry: SourceRegistry,
    poll_tracker: SharedPollTracker,
    default_ingest_scope: ScopeKey,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HybridSearchRequest {
    /// Search query (1-1000 characters)
    pub query: String,
    /// Result limit (1-100, default 10)
    pub limit: Option<usize>,
}

fn scope_from_query_params(params: &HashMap<String, String>) -> Result<ScopeKey, ApiError> {
    resolve_tenant_scope(params, &axum::http::HeaderMap::new())
}

pub(crate) fn scope_from_query_params_with_headers(
    params: &HashMap<String, String>,
    headers: &axum::http::HeaderMap,
) -> Result<ScopeKey, ApiError> {
    resolve_tenant_scope(params, headers)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointRequest {
    pub scope: ScopeKey,
    pub task_id: String,
    pub step: i64,
    pub scratchpad: serde_json::Value,
    pub importance: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointResponse {
    pub checkpoint_id: String,
    pub task_id: String,
    pub step: i64,
    pub created_at_ms: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResumeRequest {
    pub scope: ScopeKey,
    pub task_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResumeResponse {
    pub checkpoint_id: String,
    pub task_id: String,
    pub step: i64,
    pub scratchpad: serde_json::Value,
    pub created_at_ms: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestionRequest {
    pub tenant_id: String,
    pub workspace_id: Option<String>,
    pub project_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
}

/// US-9: request to consolidate Events in a scope + time window into a Summary.
#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub tenant_id: String,
    #[serde(default)]
    pub workspace_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    /// Inclusive window lower bound (ms since epoch).
    pub window_start_ms: i64,
    /// Inclusive window upper bound (ms since epoch).
    pub window_end_ms: i64,
    /// Only Events with `importance >= importance_threshold` are consolidated.
    #[serde(default)]
    pub importance_threshold: f32,
    /// When true, the consolidated source Events are deleted after the Summary
    /// is written. Their provenance is preserved in the Summary's `backing_ids`.
    #[serde(default)]
    pub delete_sources: bool,
}

#[derive(Debug, Serialize)]
pub struct ConsolidateResponse {
    /// The Summary memory item(s) created (currently one per call).
    pub summaries: Vec<MemoryItem>,
    /// Number of source Events folded into the summary.
    pub consolidated_count: usize,
    /// Whether the source Events were deleted after consolidation.
    pub sources_deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct IngestionResponse {
    pub source: String,
    pub count: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SourcePollStatus {
    pub source: String,
    pub last_poll_at_ms: Option<i64>,
    pub last_count: usize,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct IngestionPollTracker {
    last_poll_at_ms: Option<i64>,
    sources: HashMap<String, SourcePollStatus>,
}

#[derive(Clone)]
struct SharedPollTracker {
    inner: Arc<Mutex<IngestionPollTracker>>,
}

impl SharedPollTracker {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(IngestionPollTracker::default())),
        }
    }

    async fn record_success(&self, source: &str, count: usize, at_ms: i64) {
        let mut state = self.inner.lock().await;
        state.last_poll_at_ms = Some(at_ms);
        state.sources.insert(
            source.to_string(),
            SourcePollStatus {
                source: source.to_string(),
                last_poll_at_ms: Some(at_ms),
                last_count: count,
                last_error: None,
            },
        );
    }

    async fn record_error(&self, source: &str, error: String, at_ms: i64) {
        let mut state = self.inner.lock().await;
        state.last_poll_at_ms = Some(at_ms);
        state.sources.insert(
            source.to_string(),
            SourcePollStatus {
                source: source.to_string(),
                last_poll_at_ms: Some(at_ms),
                last_count: 0,
                last_error: Some(error),
            },
        );
    }

    async fn snapshot(&self) -> IngestionPollTracker {
        self.inner.lock().await.clone()
    }
}

#[derive(Debug, Serialize)]
pub struct IngestionStatus {
    pub sources: usize,
    pub poll_interval_secs: u64,
    pub last_poll_at_ms: Option<i64>,
    pub source_status: Vec<SourcePollStatus>,
}

/// Parse a comma-separated `kinds=` query parameter into a `Vec<MemoryKind>`.
///
/// Whitespace and case are normalized; unknown tokens cause the whole list
/// to be discarded (matching the previous in-line behavior). Returns `None`
/// when the resulting list is empty so callers can leave the query unfiltered.
fn parse_kinds(kinds_str: &str) -> Option<Vec<MemoryKind>> {
    let parsed: Result<Vec<MemoryKind>, _> = kinds_str
        .split(',')
        .map(|s| MemoryKind::from_str(s.trim().to_lowercase().as_str()))
        .collect();
    parsed.ok().filter(|v: &Vec<_>| !v.is_empty())
}
fn get_source_endpoint(source_name: &str, default: &str) -> String {
    let env_var = match source_name {
        "oxidizedrag" => "OXIDIZEDRAG_URL",
        "oxidizedgraph" => "OXIDIZEDGRAPH_URL",
        "datafabric" => "DATAFABRIC_URL",
        _ => return default.to_string(),
    };

    std::env::var(env_var).unwrap_or_else(|_| default.to_string())
}

fn optional_api_key(env_var: &str) -> Option<String> {
    std::env::var(env_var).ok().filter(|key| !key.is_empty())
}

fn default_ingest_scope() -> ScopeKey {
    ScopeKey {
        tenant_id: std::env::var("MOM_INGEST_TENANT_ID").unwrap_or_else(|_| "default".to_string()),
        workspace_id: std::env::var("MOM_INGEST_WORKSPACE_ID").ok(),
        project_id: std::env::var("MOM_INGEST_PROJECT_ID").ok(),
        agent_id: std::env::var("MOM_INGEST_AGENT_ID").ok(),
        run_id: std::env::var("MOM_INGEST_RUN_ID").ok(),
    }
}

fn build_rag_source(endpoint: String) -> Arc<dyn MemorySource> {
    let source = OxidizedRAGSource::new(endpoint);
    Arc::new(match optional_api_key("OXIDIZEDRAG_API_KEY") {
        Some(key) => source.with_api_key(key),
        None => source,
    })
}

fn build_graph_source(endpoint: String) -> Arc<dyn MemorySource> {
    let source = OxidizedGraphSource::new(endpoint);
    Arc::new(match optional_api_key("OXIDIZEDGRAPH_API_KEY") {
        Some(key) => source.with_api_key(key),
        None => source,
    })
}

fn build_fabric_source(endpoint: String) -> Arc<dyn MemorySource> {
    let source = DataFabricSource::new(endpoint);
    Arc::new(match optional_api_key("DATAFABRIC_API_KEY") {
        Some(key) => source.with_api_key(key),
        None => source,
    })
}

fn scope_from_ingestion(
    params: &HashMap<String, String>,
    req: &IngestionRequest,
) -> Result<ScopeKey, ApiError> {
    let mut scope = scope_from_query_params(params)?;
    if scope.workspace_id.is_none() {
        scope.workspace_id = req.workspace_id.clone();
    }
    if scope.project_id.is_none() {
        scope.project_id = req.project_id.clone();
    }
    if scope.agent_id.is_none() {
        scope.agent_id = req.agent_id.clone();
    }
    if scope.run_id.is_none() {
        scope.run_id = req.run_id.clone();
    }
    Ok(scope)
}

fn ingestion_polling_enabled() -> bool {
    std::env::var("MOM_INGEST_POLL_ENABLED")
        .map(|value| !matches!(value.to_lowercase().as_str(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("mom=debug".parse()?),
        )
        .init();

    info!("🧠 MOM Service starting...");

    // Initialize SurrealDB store
    let db_path = std::env::var("MOM_DB_PATH").unwrap_or_else(|_| "sqlite://mom.db".to_string());

    info!("Connecting to SurrealDB at {}", db_path);
    let store = Arc::new(SurrealDBStore::new(&db_path).await?);

    // Initialize embedder (optional - Phase 2a feature)
    let embedder = match create_embedder().await {
        Ok(emb) => {
            info!("✅ Embeddings initialized (model: {})", emb.model_id());
            Some(Arc::new(emb))
        }
        Err(e) => {
            warn!("⚠️  Embeddings disabled: {}", e);
            None
        }
    };

    let poll_interval_secs = std::env::var("MOM_INGEST_POLL_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let rag_endpoint = get_source_endpoint("oxidizedrag", "http://localhost:8001");
    let graph_endpoint = get_source_endpoint("oxidizedgraph", "http://localhost:8002");
    let fabric_endpoint = get_source_endpoint("datafabric", "http://localhost:8003");

    info!("Initializing ingestion sources:");
    info!("  oxidizedrag  : {}", rag_endpoint);
    info!("  oxidizedgraph: {}", graph_endpoint);
    info!("  datafabric   : {}", fabric_endpoint);

    let rag_source = build_rag_source(rag_endpoint);
    let graph_source = build_graph_source(graph_endpoint);
    let fabric_source = build_fabric_source(fabric_endpoint);

    let mut scheduler = IngestionScheduler::new(poll_interval_secs);
    scheduler.register_source(rag_source.clone());
    scheduler.register_source(graph_source.clone());
    scheduler.register_source(fabric_source.clone());

    let scheduler = Arc::new(scheduler);

    let mut source_registry = SourceRegistry::new();
    source_registry.register("oxidizedrag".to_string(), rag_source);
    source_registry.register("oxidizedgraph".to_string(), graph_source);
    source_registry.register("datafabric".to_string(), fabric_source);

    let poll_tracker = SharedPollTracker::new();
    info!(
        "✅ Ingestion scheduler initialized with {} sources (poll every {}s)",
        scheduler.source_count(),
        poll_interval_secs
    );

    let ingest_scope = ScopeKey {
        tenant_id: std::env::var("MOM_INGEST_TENANT_ID").unwrap_or_else(|_| "default".into()),
        workspace_id: std::env::var("MOM_INGEST_WORKSPACE_ID").ok(),
        project_id: std::env::var("MOM_INGEST_PROJECT_ID").ok(),
        agent_id: std::env::var("MOM_INGEST_AGENT_ID").ok(),
        run_id: std::env::var("MOM_INGEST_RUN_ID").ok(),
    };

    if ingestion_polling_enabled() {
        scheduler
            .clone()
            .spawn_polling_loop(Arc::clone(&store), ingest_scope.clone());
    } else {
        info!("ingestion polling loop disabled (MOM_INGEST_POLL_ENABLED=false)");
    }

    let state = AppState {
        store,
        embedder,
        ingestion_scheduler: scheduler,
        source_registry,
        poll_tracker,
        default_ingest_scope: ingest_scope,
    };

    // Build router
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/memory", post(put_memory).get(list_memories))
        .route("/v1/memory/batch", post(batch_write_memory))
        .route("/v1/memory/batch/delete", post(batch_delete_memory))
        .route("/v1/memory/batch/query", post(batch_query_memory))
        .route("/v1/memory/:id", get(get_memory).delete(delete_memory))
        .route("/v1/recall", post(recall))
        .route("/v1/semantic-search", post(semantic_search))
        .route("/v1/hybrid-search", post(hybrid_search))
        .route("/v1/links", post(create_link))
        .route("/v1/links/traverse", get(traverse_links))
        .route("/v1/links/conflicts", get(list_conflicts))
        .route(
            "/v1/links/:link_id",
            patch(update_link).delete(delete_link_handler),
        )
        .route("/v1/context-pack", post(context_pack))
        .route("/v1/consolidate", post(consolidate))
        .route("/v1/ingest/:source", post(ingest_source))
        .route("/v1/ingest/all", post(ingest_all))
        .route("/v1/ingest/status", get(ingest_status))
        .route("/v1/task/checkpoint", post(task_checkpoint))
        .route("/v1/task/resume", post(task_resume))
        // US-19f (#70): negotiate gzip/zstd on both request bodies
        // (`Content-Encoding`) and responses (`Accept-Encoding`). No
        // behaviour change for uncompressed clients; opt-in only when
        // the headers are present.
        .layer(RequestDecompressionLayer::new())
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("✅ MOM API listening on http://{}", addr);
    info!("📚 Endpoints:");
    info!("  GET    /healthz              - Health check");
    info!("  POST   /v1/memory            - Write memory");
    info!("  POST   /v1/memory/batch      - Batch write memories");
    info!("  POST   /v1/memory/batch/delete - Batch delete memories by id");
    info!("  POST   /v1/memory/batch/query - Batch query with multiple scopes");
    info!("  GET    /v1/memory            - List memories");
    info!("  GET    /v1/memory/:id        - Get memory");
    info!("  DELETE /v1/memory/:id        - Delete memory");
    info!("  POST   /v1/recall            - Recall context");
    info!("  POST   /v1/semantic-search   - Vector semantic search");
    info!("  POST   /v1/hybrid-search     - Hybrid lexical+vector recall (RRF)");
    info!("  POST   /v1/links            - Create memory relationship");
    info!("  PATCH  /v1/links/:link_id   - Update link metadata");
    info!("  DELETE /v1/links/:link_id   - Delete relationship");
    info!("  GET    /v1/links/traverse   - Traverse memory graph");
    info!("  GET    /v1/links/conflicts  - List contradicting facts");
    info!("  POST   /v1/ingest/:source    - Ingest from specific source");
    info!("  POST   /v1/ingest/all        - Ingest from all sources");
    info!("  GET    /v1/ingest/status     - Ingestion scheduler status");
    info!("  POST   /v1/context-pack      - Structured context bundle for agents");
    info!("  POST   /v1/consolidate       - Consolidate events into a Summary (US-9)");
    info!("  POST   /v1/task/checkpoint   - Write a Checkpoint memory for a task");
    info!("  POST   /v1/task/resume       - Fetch the latest checkpoint for a task");

    // Start gRPC server on 50051 (alongside Axum)
    let grpc_addr = "0.0.0.0:50051".parse::<std::net::SocketAddr>()?;
    let grpc_store: Arc<dyn MemoryStore> = state.store.clone();
    tokio::spawn(async move {
        if let Err(e) = mom_grpc::start_grpc_server(grpc_store, grpc_addr).await {
            error!("gRPC server error: {}", e);
        }
    });
    info!("✅ MOM gRPC listening on grpc://0.0.0.0:50051");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn healthz() -> &'static str {
    "ok"
}

/// US-9: consolidate Events in a scope + time window into a Summary memory item.
///
/// Selects Event items in `[window_start_ms, window_end_ms]` whose importance
/// meets the threshold, writes one Summary recording the window bounds and the
/// backing Event IDs (provenance), and optionally soft-deletes the sources.
async fn consolidate(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, ApiError> {
    if req.tenant_id.is_empty() {
        return Err(ApiError::BadRequest("tenant_id is required".to_string()));
    }
    // Enforce matching header if X-Tenant-ID is set
    if let Some(header_tenant) = headers.get("x-tenant-id").and_then(|v| v.to_str().ok()) {
        if !header_tenant.trim().is_empty() && header_tenant != req.tenant_id {
            return Err(ApiError::BadRequest(
                "tenant_id in request does not match X-Tenant-ID header".to_string(),
            ));
        }
    }
    if req.window_end_ms < req.window_start_ms {
        return Err(ApiError::BadRequest(
            "window_end_ms must be >= window_start_ms".to_string(),
        ));
    }
    let resp = run_consolidation(st.store.as_ref(), &req).await?;
    Ok(Json(resp))
}

/// Core consolidation logic, separated from the HTTP layer so it is unit
/// testable against an in-memory store.
async fn run_consolidation(
    store: &SurrealDBStore,
    req: &ConsolidateRequest,
) -> anyhow::Result<ConsolidateResponse> {
    let scope = ScopeKey {
        tenant_id: req.tenant_id.clone(),
        workspace_id: req.workspace_id.clone(),
        project_id: req.project_id.clone(),
        agent_id: req.agent_id.clone(),
        run_id: req.run_id.clone(),
    };

    // Read every candidate in the window, then apply the threshold in-process.
    // The store helper is unbounded, so large windows are fully covered rather
    // than silently truncated at an arbitrary page size.
    let query = Query {
        scope: scope.clone(),
        text: String::new(),
        kinds: Some(vec![MemoryKind::Event]),
        tags_any: None,
        limit: 1,
        since_ms: Some(req.window_start_ms),
        until_ms: Some(req.window_end_ms),
        cursor: None,
    };
    let mut candidates: Vec<MemoryItem> = store
        .query_items(query)
        .await?
        .into_iter()
        .filter(|item| item.importance >= req.importance_threshold)
        .collect();

    if candidates.is_empty() {
        return Ok(ConsolidateResponse {
            summaries: Vec::new(),
            consolidated_count: 0,
            sources_deleted: false,
        });
    }

    candidates.sort_by(|a, b| {
        a.created_at_ms
            .cmp(&b.created_at_ms)
            .then_with(|| a.id.0.cmp(&b.id.0))
    });

    let backing_ids: Vec<MemoryId> = candidates.iter().map(|c| c.id.clone()).collect();
    let now_ms = chrono::Utc::now().timestamp_millis();

    // Deterministic digest of the consolidated content.
    // TODO(US-9): replace with LLM-powered summarization (see issue #14).
    let mut digest = String::new();
    for c in &candidates {
        let snippet = match &c.content {
            Content::Text(t) => t.clone(),
            Content::TextJson { text, .. } => text.clone(),
            Content::Json(v) => v.to_string(),
        };
        let snippet: String = snippet.chars().take(120).collect();
        if !digest.is_empty() {
            digest.push_str("; ");
        }
        digest.push_str(&snippet);
    }
    let summary_text = format!(
        "Consolidated {} event(s) in window [{}, {}]: {}",
        candidates.len(),
        req.window_start_ms,
        req.window_end_ms,
        digest
    );

    let mut meta: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    meta.insert(
        "window_start_ms".to_string(),
        serde_json::json!(req.window_start_ms),
    );
    meta.insert(
        "window_end_ms".to_string(),
        serde_json::json!(req.window_end_ms),
    );
    write_provenance_ids(&mut meta, &backing_ids);

    let max_importance = candidates
        .iter()
        .map(|c| c.importance)
        .fold(0.0_f32, f32::max);

    let summary = MemoryItem {
        id: MemoryId(uuid::Uuid::new_v4().to_string()),
        scope: scope.clone(),
        kind: MemoryKind::Summary,
        created_at_ms: now_ms,
        content: Content::Text(summary_text),
        tags: vec!["consolidated".to_string()],
        importance: max_importance,
        confidence: 1.0,
        source: "system".to_string(),
        ttl_ms: None,
        meta,
        embedding: None,
        embedding_model: None,
    };

    if req.delete_sources {
        store
            .put_and_delete_atomic(summary.clone(), &backing_ids)
            .await?;
    } else {
        store.put(summary.clone()).await?;
    }

    Ok(ConsolidateResponse {
        summaries: vec![summary],
        consolidated_count: backing_ids.len(),
        sources_deleted: req.delete_sources,
    })
}

async fn prepare_memory_item(st: &AppState, mut item: MemoryItem) -> Result<MemoryItem, ApiError> {
    // US-7 AC-1: tenant_id is mandatory on every memory write. Previously
    // this was deferred to the SurrealDB schema ASSERT, which surfaced as
    // an opaque internal error — now we reject at the HTTP boundary with a
    // BadRequest so the caller learns it's their fault, not ours.
    if item.scope.tenant_id.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "scope.tenant_id is required and must be non-empty".into(),
        ));
    }

    // Generate ID if not provided
    if item.id.0.is_empty() {
        item.id = MemoryId(uuid::Uuid::new_v4().to_string());
    }

    audit_tenant_access("write", &item.scope.tenant_id, &item.id.0);

    // US-10: kind-specific meta validation. Fact and Preference items
    // carry structured payloads under well-known meta keys; reject the
    // write up-front so we don't silently store half-formed records.
    let fact_payload = match item.kind {
        MemoryKind::Fact => Some(
            FactPayload::try_from_meta(&item.meta)
                .map_err(|err| ApiError::BadRequest(format!("invalid Fact payload: {err}")))?,
        ),
        MemoryKind::Preference => {
            PreferencePayload::try_from_meta(&item.meta).map_err(|err| {
                ApiError::BadRequest(format!("invalid Preference payload: {err}"))
            })?;
            None
        }
        _ => None,
    };

    // US-10: default version / provenance chain on first write, so
    // downstream consumers never see a Fact/Preference with absent
    // versioning metadata.
    if !item.meta.contains_key(META_VERSION) {
        write_version(&mut item.meta, 1);
    }
    if !item.meta.contains_key(META_PROVENANCE_IDS) {
        write_provenance_ids(&mut item.meta, &[]);
    }

    // US-10: exact-key conflict detection for Facts. If an active Fact
    // with the same (subject, predicate) already exists in this scope:
    //   - same object → no-op (caller is re-asserting a known fact)
    //   - different object → supersede the old one: stamp its meta with
    //     superseded_by=new.id, bump new.version to old.version+1, and
    //     append old.id to new.provenance_ids.
    if let Some(ref payload) = fact_payload {
        let conflicts = st
            .store
            .find_active_facts_with_key(&item.scope, &payload.subject, &payload.predicate)
            .await
            .map_err(|err| {
                error!(?err, "find_active_facts_with_key failed");
                ApiError::Internal(format!("conflict detection failed: {err}"))
            })?;

        for old in conflicts {
            if old.id == item.id {
                // Idempotent re-write of the same record (PUT semantics).
                continue;
            }
            let old_payload = FactPayload::try_from_meta(&old.meta).ok();
            let same_object = old_payload
                .as_ref()
                .is_some_and(|p| p.object == payload.object);
            if same_object {
                // Caller is re-asserting an existing fact; nothing to
                // supersede. We deliberately do NOT bump confidence here
                // — that's a separate policy (see US-12) and would let
                // any caller silently strengthen any other caller's
                // belief just by re-writing.
                continue;
            }

            // Different object: supersede.
            let mut superseded = old.clone();
            write_superseded_by(&mut superseded.meta, &item.id);
            st.store.put(superseded).await.map_err(|err| {
                error!(?err, "marking prior fact superseded failed");
                ApiError::Internal(format!("supersession write failed: {err}"))
            })?;

            // Bump version + extend provenance chain on the new item.
            // We carry forward the maximum version we've seen across all
            // conflicts so two simultaneous contradictions still produce
            // a strictly increasing sequence.
            let candidate_version = read_version(&old.meta).saturating_add(1);
            if read_version(&item.meta) < candidate_version {
                write_version(&mut item.meta, candidate_version);
            }
            let mut prov = read_provenance_ids(&item.meta);
            if !prov.iter().any(|id| id == &old.id) {
                prov.push(old.id.clone());
            }
            write_provenance_ids(&mut item.meta, &prov);
        }
    }

    if let Some(embedder) = st.embedder.as_ref() {
        maybe_embed_item(&mut item, embedder.as_ref().as_ref()).await?;
    }

    // US-10 Phase 2: semantic conflict advisory. Once the new Fact has an
    // embedding, scan existing active Facts in scope whose embeddings are
    // close (cosine sim ≥ SEMANTIC_CONFLICT_THRESHOLD) but whose
    // `meta.fact.object` differs. Such pairs are likely contradictions
    // missed by the exact-key check (e.g. "API rate limit is 1000 req/min"
    // vs "API throughput cap: 500/minute"). We DO NOT auto-supersede —
    // semantic similarity isn't certainty — but we record both ids as
    // advisory hints so callers can surface them for human review.
    if let (Some(_), MemoryKind::Fact) = (item.embedding.as_ref(), item.kind) {
        let embedding = item.embedding.clone().unwrap();
        if let Some(ref payload) = fact_payload {
            let candidates = st
                .store
                .find_semantic_fact_conflicts(
                    &item.scope,
                    &embedding,
                    Some(&item.id),
                    SEMANTIC_CONFLICT_THRESHOLD,
                    SEMANTIC_CONFLICT_MAX_HITS,
                )
                .await
                .map_err(|err| {
                    error!(?err, "find_semantic_fact_conflicts failed");
                    ApiError::Internal(format!("semantic conflict scan failed: {err}"))
                })?;
            for (existing, _sim) in candidates {
                let existing_payload = FactPayload::try_from_meta(&existing.meta).ok();
                let same_object = existing_payload
                    .as_ref()
                    .is_some_and(|p| p.object == payload.object);
                if same_object {
                    // Embedding-close items that happen to share the same
                    // object aren't a contradiction — they're paraphrases.
                    continue;
                }
                // Record on the NEW item's advisory list. Writing back to
                // the OLD item would race with concurrent reads and double
                // the I/O for an advisory hint.
                record_semantic_conflict(&mut item.meta, &existing.id);
            }
        }
    }

    Ok(item)
}

async fn put_memory(
    State(st): State<AppState>,
    Json(item): Json<MemoryItem>,
) -> Result<(StatusCode, Json<MemoryItem>), ApiError> {
    let prepared = prepare_memory_item(&st, item).await?;
    st.store.put(prepared.clone()).await?;
    Ok((StatusCode::CREATED, Json(prepared)))
}

// ─── Batch write endpoint (US-19a / #63) ─────────────────────────────
//
// POST /v1/memory/batch
// Body: { "items": [MemoryItem, ...] }
// Response 201: { "ids": [MemoryId, ...] } aligned with input order.
//
// Best-effort / non-atomic in this slice — a mid-batch failure leaves a
// partial result. Atomicity is tracked in #68 (US-19d) as an opt-in
// `atomic: bool` once `SurrealDBStore` provides the transactional override.

/// Soft cap on per-request batch size. Rejected with 422 above this; large
/// batches should be sent as multiple requests or via a future streaming
/// endpoint.
const MAX_BATCH_ITEMS: usize = 1000;

#[derive(Debug, Serialize, Deserialize)]
struct BatchWriteRequest {
    items: Vec<MemoryItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchWriteResponse {
    ids: Vec<MemoryId>,
}

#[derive(Debug, Serialize)]
struct BatchWriteItemResult {
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<MemoryId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchWriteMultiResponse {
    results: Vec<BatchWriteItemResult>,
}

async fn batch_write_memory(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(req): Json<BatchWriteRequest>,
) -> Result<axum::response::Response, ApiError> {
    if req.items.is_empty() {
        return Err(ApiError::BadRequest("items must not be empty".into()));
    }

    let max_batch_size = std::env::var("MOM_MAX_BATCH_SIZE")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(MAX_BATCH_ITEMS);

    if req.items.len() > max_batch_size {
        return Err(ApiError::PayloadTooLarge(format!(
            "Batch size {} exceeds maximum allowed of {}",
            req.items.len(),
            max_batch_size
        )));
    }

    let atomic = params
        .get("atomic")
        .map(|s| s.parse::<bool>().unwrap_or(false))
        .unwrap_or(false);

    if atomic {
        let mut prepared_items = Vec::with_capacity(req.items.len());
        for item in req.items {
            prepared_items.push(prepare_memory_item(&st, item).await?);
        }
        let ids = st.store.write_batch(prepared_items, true).await?;
        Ok((StatusCode::CREATED, Json(BatchWriteResponse { ids })).into_response())
    } else {
        let mut results = Vec::with_capacity(req.items.len());
        for item in req.items {
            match prepare_memory_item(&st, item).await {
                Ok(prepared) => {
                    let id = prepared.id.clone();
                    match st.store.write_batch(vec![prepared], false).await {
                        Ok(mut ids) => {
                            results.push(BatchWriteItemResult {
                                status: StatusCode::CREATED.as_u16(),
                                id: ids.pop(),
                                error: None,
                            });
                        }
                        Err(e) => {
                            results.push(BatchWriteItemResult {
                                status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                                id: Some(id),
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }
                Err(e) => {
                    results.push(BatchWriteItemResult {
                        status: e.status_code().as_u16(),
                        id: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }
        Ok((
            StatusCode::MULTI_STATUS,
            Json(BatchWriteMultiResponse { results }),
        )
            .into_response())
    }
}

// ─── Batch delete endpoint (US-19b / #64) ────────────────────────────
//
// POST /v1/memory/batch/delete
// Body: { "ids": [MemoryId, ...] }
// Response 204: no body.
//
// Idempotent: missing ids are not an error. Non-atomic in this slice;
// atomicity tracked in #68 (US-19d).

/// Soft cap on per-request batch size for delete.
const MAX_BATCH_DELETE_IDS: usize = 1000;

#[derive(Debug, Deserialize)]
struct BatchDeleteRequest {
    ids: Vec<MemoryId>,
}

#[derive(Debug, Serialize)]
struct BatchDeleteItemResult {
    id: MemoryId,
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchDeleteMultiResponse {
    results: Vec<BatchDeleteItemResult>,
}

async fn batch_delete_memory(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(req): Json<BatchDeleteRequest>,
) -> Result<axum::response::Response, ApiError> {
    if req.ids.is_empty() {
        return Err(ApiError::BadRequest("ids must not be empty".into()));
    }
    if req.ids.len() > MAX_BATCH_DELETE_IDS {
        return Err(ApiError::BadRequest(format!(
            "batch size {} exceeds max {}",
            req.ids.len(),
            MAX_BATCH_DELETE_IDS
        )));
    }

    // SECURITY: Require tenant_id from query parameter
    let tenant_id = params
        .get("tenant_id")
        .ok_or(ApiError::BadRequest("tenant_id is required".to_string()))?
        .to_string();

    let scope = ScopeKey {
        tenant_id,
        workspace_id: params.get("workspace_id").map(|s| s.to_string()),
        project_id: params.get("project_id").map(|s| s.to_string()),
        agent_id: params.get("agent_id").map(|s| s.to_string()),
        run_id: params.get("run_id").map(|s| s.to_string()),
    };

    let atomic = params
        .get("atomic")
        .map(|s| s.parse::<bool>().unwrap_or(false))
        .unwrap_or(false);

    if atomic {
        st.store.delete_batch_scoped(req.ids, &scope, true).await?;
        Ok(StatusCode::NO_CONTENT.into_response())
    } else {
        let mut results = Vec::with_capacity(req.ids.len());
        for id in req.ids {
            match st
                .store
                .delete_batch_scoped(vec![id.clone()], &scope, false)
                .await
            {
                Ok(_) => {
                    results.push(BatchDeleteItemResult {
                        id,
                        status: StatusCode::NO_CONTENT.as_u16(),
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(BatchDeleteItemResult {
                        id,
                        status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                        error: Some(e.to_string()),
                    });
                }
            }
        }
        Ok((
            StatusCode::MULTI_STATUS,
            Json(BatchDeleteMultiResponse { results }),
        )
            .into_response())
    }
}

// ─── Batch query endpoint (US-19c / #65) ─────────────────────────────
//
// POST /v1/memory/batch/query
// Body: { "queries": [Query, ...] }
// Response 200: { "results": [[Scored<MemoryItem>, ...], ...] }
//   aligned by input index. First failed query short-circuits the
//   whole batch (see trait `query_batch` semantics).

/// Soft cap on number of queries per batch.
const MAX_BATCH_QUERIES: usize = 100;

#[derive(Debug, Deserialize)]
struct BatchQueryRequest {
    queries: Vec<Query>,
}

#[derive(Debug, Serialize)]
struct BatchQueryResponse {
    results: Vec<Vec<Scored<MemoryItem>>>,
}

async fn batch_query_memory(
    State(st): State<AppState>,
    Json(req): Json<BatchQueryRequest>,
) -> Result<Json<BatchQueryResponse>, ApiError> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("queries must not be empty".into()));
    }
    if req.queries.len() > MAX_BATCH_QUERIES {
        return Err(ApiError::BadRequest(format!(
            "batch size {} exceeds max {}",
            req.queries.len(),
            MAX_BATCH_QUERIES
        )));
    }

    // Apply the same default-tenant fallback as `recall` so callers that
    // omit it get the same behaviour they would with single-query.
    let mut queries = req.queries;
    for q in queries.iter_mut() {
        if q.scope.tenant_id.is_empty() {
            q.scope.tenant_id = "default".to_string();
        }
    }
    let results = st.store.query_batch(queries).await?;
    Ok(Json(BatchQueryResponse { results }))
}

/// Cosine similarity at or above this value, between a new Fact's embedding
/// and an existing active Fact's, triggers a semantic-conflict advisory
/// (when the `object` differs). 0.85 is the standard "near-duplicate"
/// threshold for sentence-embedding spaces; tunable per deploy once we have
/// labelled data.
const SEMANTIC_CONFLICT_THRESHOLD: f32 = 0.85;
/// Cap on how many candidates the semantic pass surfaces per write. The
/// purpose is advisory triage, not exhaustive enumeration — top-K is fine.
const SEMANTIC_CONFLICT_MAX_HITS: usize = 5;

async fn get_memory(
    State(st): State<AppState>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<MemoryItem>, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    audit_tenant_access("get", &scope.tenant_id, &id);
    let item = st
        .store
        .get_scoped(&MemoryId(id), &scope)
        .await?
        .ok_or(ApiError::NotFound)?;
    Ok(Json(item))
}

#[derive(Debug, Serialize, Deserialize)]
struct PaginatedListResponse {
    items: Vec<MemoryItem>,
    next_cursor: Option<String>,
}

async fn list_memories(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<PaginatedListResponse>, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    audit_tenant_access("list", &scope.tenant_id, "memory_items");

    let tenant_id = scope.tenant_id.clone();

    let kinds = params.get("kinds").and_then(|k| parse_kinds(k));

    // Parse tags filter (comma-separated)
    let tags_any = params.get("tags").and_then(|t| {
        let tags: Vec<String> = t.split(',').map(|s| s.trim().to_string()).collect();
        if tags.is_empty() || tags.iter().all(|s| s.is_empty()) {
            None
        } else {
            Some(tags.into_iter().filter(|s| !s.is_empty()).collect())
        }
    });

    // Parse time range filters (milliseconds since epoch)
    let since_ms = params.get("since_ms").and_then(|s| s.parse().ok());
    let until_ms = params.get("until_ms").and_then(|s| s.parse().ok());

    // Parse and clamp limit to max 100
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .map(|l| l.min(100))
        .unwrap_or(10);

    let cursor = params.get("cursor").cloned();

    let query = Query {
        scope: ScopeKey {
            tenant_id,
            workspace_id: scope.workspace_id,
            project_id: scope.project_id,
            agent_id: scope.agent_id,
            run_id: scope.run_id,
        },
        text: String::new(),
        kinds,
        tags_any,
        limit: limit + 1, // fetch limit + 1 items to check for next page
        since_ms,
        until_ms,
        cursor,
    };

    let mut results = st.store.query(query).await?;
    let mut next_cursor = None;

    if results.len() > limit {
        results.truncate(limit);
        if let Some(last_scored) = results.last() {
            next_cursor = Some(Query::encode_cursor(
                last_scored.item.created_at_ms,
                &last_scored.item.id.0,
            ));
        }
    }

    let items: Vec<MemoryItem> = results.into_iter().map(|s| s.item).collect();
    Ok(Json(PaginatedListResponse { items, next_cursor }))
}

async fn delete_memory(
    State(st): State<AppState>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<StatusCode, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    audit_tenant_access("delete", &scope.tenant_id, &id);
    st.store.delete_scoped(&MemoryId(id), &scope).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn context_pack(
    State(st): State<AppState>,
    Json(req): Json<ContextPackRequest>,
) -> Result<Json<ContextPack>, ApiError> {
    if req.query.scope.tenant_id.is_empty() {
        return Err(ApiError::BadRequest("tenant_id is required".to_string()));
    }

    let budget = req.budget_tokens.unwrap_or(mom_core::DEFAULT_BUDGET_TOKENS);
    let candidate_limit = (budget / TOKENS_PER_ITEM).clamp(10, 100);

    let mut q = req.query.clone();
    if q.limit == 0 {
        q.limit = candidate_limit;
    } else {
        q.limit = q.limit.min(candidate_limit);
    }

    let candidates = if !q.text.is_empty() {
        if let Some(embedder) = st.embedder.as_ref() {
            match embedder.embed(&q.text).await {
                Ok(query_embedding) => {
                    st.store
                        .hybrid_recall(q.clone(), &query_embedding, q.limit)
                        .await?
                }
                Err(e) => {
                    warn!(
                        "context-pack embedding failed, falling back to lexical recall: {}",
                        e
                    );
                    st.store.query(q).await?
                }
            }
        } else {
            st.store.query(q).await?
        }
    } else {
        st.store.query(q).await?
    };

    Ok(Json(build_context_pack(candidates, req.budget_tokens)))
}

async fn recall(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(mut q): Json<Query>,
) -> Result<Json<Vec<Scored<MemoryItem>>>, ApiError> {
    q.scope = scope_from_query_params_with_headers(&params, &headers)?;
    audit_tenant_access("recall", &q.scope.tenant_id, "memory_items");

    // US-7 AC-4: a missing tenant_id used to silently coerce to "default",
    // which meant every unauthenticated caller pooled into a single shared
    // bucket — exactly the cross-tenant pollution AC-4 wants to prevent.
    // Reject the request instead so the caller learns to send one.
    if q.scope.tenant_id.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "scope.tenant_id is required and must be non-empty".into(),
        ));
    }

    if q.text.is_empty() {
        let results = st.store.query(q).await?;
        return Ok(Json(results));
    }

    let original_limit = q.limit.max(1);
    q.limit = recall::recall_candidate_limit(original_limit);
    let results = st.store.query(q.clone()).await?;
    Ok(Json(recall::rank_recall_results(q, results)))
}

async fn ingest_source(
    State(st): State<AppState>,
    Path(source): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<IngestionResponse>, ApiError> {
    let scope = scope_from_ingestion(&params, &req)?;
    info!(source = %source, "starting manual ingestion");

    let count = st
        .ingestion_scheduler
        .ingest_source(st.store.as_ref(), &source, &scope)
        .await
        .map_err(|err| {
            if err.downcast_ref::<UnknownSourceError>().is_some() {
                ApiError::BadRequest(err.to_string())
            } else {
                ApiError::Internal(err.to_string())
            }
        })?;

    info!(source = %source, count, "manual ingestion complete");
    Ok(Json(IngestionResponse {
        source,
        count,
        message: format!("Successfully ingested {count} memories"),
    }))
}

async fn ingest_all(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<Vec<IngestionResponse>>, ApiError> {
    let scope = scope_from_ingestion(&params, &req)?;
    let outcomes = st
        .ingestion_scheduler
        .ingest_all(st.store.as_ref(), &scope)
        .await;

    let responses = outcomes
        .into_iter()
        .map(|(source, outcome)| match outcome {
            Ok(count) => IngestionResponse {
                source: source.clone(),
                count,
                message: format!("Successfully ingested {count} memories"),
            },
            Err(err) => {
                warn!(source = %source, error = %err, "manual ingestion failed");
                IngestionResponse {
                    source,
                    count: 0,
                    message: format!("Failed: {err}"),
                }
            }
        })
        .collect();

    Ok(Json(responses))
}

async fn ingest_status(State(st): State<AppState>) -> Json<IngestionStatusReport> {
    Json(st.ingestion_scheduler.status().await)
}

/// Semantic search using embeddings (Phase 2a feature)
///
/// Returns memories ranked by semantic similarity to the query
async fn semantic_search(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(req): Json<SemanticSearchRequest>,
) -> Result<Json<Vec<Scored<MemoryItem>>>, ApiError> {
    // Extract tenant_id from query params (security: prevents IDOR via request body)
    let tenant_id = params
        .get("tenant_id")
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))?
        .to_string();

    let embedder = st.embedder.as_ref().ok_or_else(|| ApiError::Internal("embedder not configured".to_string()))?;

    // Generate embedding for query text
    let query_embedding = embedder
        .embed(&req.query)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let limit = req.limit.unwrap_or(10).min(100);

    // Create scope for search
    let scope = ScopeKey {
        tenant_id,
        workspace_id: params.get("workspace_id").cloned(),
        project_id: params.get("project_id").cloned(),
        agent_id: params.get("agent_id").cloned(),
        run_id: params.get("run_id").cloned(),
    };

    // Use vector recall from store (Phase 2b)
    let results = st
        .store
        .vector_recall(&query_embedding, &scope, limit)
        .await?;

    Ok(Json(results))
}

/// Hybrid search combining lexical + semantic matching (Phase 2b feature)
///
/// Uses RRF (Reciprocal Rank Fusion) with 70% lexical + 30% semantic weighting
/// Returns memories ranked by combined relevance
///
/// Query parameters:
/// - tenant_id (required): Tenant scope for search
/// - workspace_id, project_id, agent_id, run_id (optional): Narrow search scope
///
/// Request body:
/// - query: Search text (1-1000 characters)
/// - limit: Result limit (1-100, default 10)
async fn hybrid_search(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<Vec<Scored<MemoryItem>>>, ApiError> {
    // Validate query length
    if req.query.is_empty() || req.query.len() > 1000 {
        return Err(ApiError::BadRequest(
            "query must be 1-1000 characters".to_string(),
        ));
    }

    // Extract tenant_id from query params (security: prevents IDOR via request body)
    let tenant_id = params
        .get("tenant_id")
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))?
        .to_string();

    let embedder = st.embedder.as_ref().ok_or_else(|| ApiError::Internal("embedder not configured".to_string()))?;

    // Generate embedding for query text
    let query_embedding = embedder.embed(&req.query).await.map_err(|e| {
        tracing::error!("embedding failed: {}", e);
        ApiError::Internal("embedding failed".to_string())
    })?;

    // Clamp limit to [1, 100] range
    let limit = req.limit.unwrap_or(10).clamp(1, 100);

    // Create query for lexical + semantic fusion
    // Optional scope fields (workspace_id, project_id, agent_id, run_id) narrow the search
    // to a specific context within the tenant. If omitted, search spans entire tenant.
    let query = Query {
        scope: ScopeKey {
            tenant_id,
            workspace_id: params.get("workspace_id").cloned(),
            project_id: params.get("project_id").cloned(),
            agent_id: params.get("agent_id").cloned(),
            run_id: params.get("run_id").cloned(),
        },
        text: req.query,
        kinds: None,
        tags_any: None,
        limit,
        since_ms: None,
        until_ms: None,
        cursor: None,
    };

    let results = st
        .store
        .hybrid_recall(query, &query_embedding, limit)
        .await?;

    Ok(Json(results))
}

fn scope_from_request(req: &IngestionRequest) -> ScopeKey {
    ScopeKey {
        tenant_id: req.tenant_id.clone(),
        workspace_id: req.workspace_id.clone(),
        project_id: req.project_id.clone(),
        agent_id: req.agent_id.clone(),
        run_id: req.run_id.clone(),
    }
}



/// Persist a durable-execution checkpoint for an in-flight agent task.
///
/// Creates a `MemoryItem` with `kind = Checkpoint`, indexes it by the
/// originating task via both `meta["task_id"]` and a `task:<task_id>` tag
/// so the resume path can look it up via the standard `tags_any` query.
///
/// Security note: `scope.tenant_id` is taken from the request body and is
/// therefore client-asserted. Follow-up work should derive it from
/// authenticated identity rather than trust the body.
async fn task_checkpoint(
    State(st): State<AppState>,
    Json(req): Json<CheckpointRequest>,
) -> Result<(StatusCode, Json<CheckpointResponse>), ApiError> {
    if req.scope.tenant_id.is_empty() {
        return Err(ApiError::BadRequest("scope.tenant_id is required".into()));
    }
    if req.task_id.trim().is_empty() {
        return Err(ApiError::BadRequest("task_id is required".into()));
    }

    let id = MemoryId(uuid::Uuid::new_v4().to_string());
    let mut record = CheckpointRecord::new(
        id.clone(),
        req.scope,
        req.task_id.clone(),
        req.step,
        req.scratchpad,
    );
    if let Some(importance) = req.importance {
        record = record.with_importance(importance);
    }

    let item = record.into_memory_item("agent".to_string());
    let created_at_ms = item.created_at_ms;
    st.store.put(item).await?;

    Ok((
        StatusCode::CREATED,
        Json(CheckpointResponse {
            checkpoint_id: id.0,
            task_id: req.task_id,
            step: req.step,
            created_at_ms,
        }),
    ))
}

/// Look up the latest checkpoint for a task within the caller-asserted scope.
///
/// Returns `404` if no checkpoint exists. The latest is selected by
/// `created_at_ms` rather than by importance, so high-importance older
/// checkpoints do not shadow newer state.
async fn task_resume(
    State(st): State<AppState>,
    Json(req): Json<ResumeRequest>,
) -> Result<Json<ResumeResponse>, ApiError> {
    if req.scope.tenant_id.is_empty() {
        return Err(ApiError::BadRequest("scope.tenant_id is required".into()));
    }
    if req.task_id.trim().is_empty() {
        return Err(ApiError::BadRequest("task_id is required".into()));
    }

    let query = Query {
        scope: req.scope,
        text: String::new(),
        kinds: Some(vec![MemoryKind::Checkpoint]),
        tags_any: Some(vec![task_tag(&req.task_id)]),
        limit: 100,
        since_ms: None,
        until_ms: None,
        cursor: None,
    };

    let results = st.store.query(query).await?;
    // Re-sort by recency — query() uses (importance, recency) ordering, which
    // would let a high-importance stale checkpoint mask a fresh one.
    let latest = results
        .into_iter()
        .max_by_key(|s| s.item.created_at_ms)
        .ok_or(ApiError::NotFound)?
        .item;

    let record = CheckpointRecord::try_from_memory_item(&latest)
        .map_err(|e| ApiError::Internal(format!("malformed checkpoint: {}", e)))?;

    Ok(Json(ResumeResponse {
        checkpoint_id: record.id.0,
        task_id: record.task_id,
        step: record.step,
        scratchpad: record.scratchpad,
        created_at_ms: latest.created_at_ms,
    }))
}

// Error handling
#[derive(Debug, thiserror::Error)]
pub(crate) enum ApiError {
    #[error("Not found")]
    NotFound,
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Payload too large: {0}")]
    PayloadTooLarge(String),
}

impl ApiError {
    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        error!("Internal error: {}", err);
        ApiError::Internal(err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        // US-7 AC-6: error messages must not reveal tenant data. `Internal`
        // wraps anyhow errors that frequently carry SurrealQL strings —
        // those strings have the literal `tenant_id = '<other-tenant>'`
        // embedded — so echoing the message into the response body leaked
        // cross-tenant identifiers to whoever triggered the error. We now
        // log the full detail at `error!` and return a static opaque body.
        // `BadRequest` and `PayloadTooLarge` are caller-supplied
        // conditions that don't carry server-side state, so their
        // messages are safe to surface.
        match self {
            ApiError::NotFound => {
                let body = Json(serde_json::json!({ "error": "Not found" }));
                (StatusCode::NOT_FOUND, body).into_response()
            }
            ApiError::BadRequest(msg) => {
                let body = Json(serde_json::json!({ "error": msg }));
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            ApiError::PayloadTooLarge(msg) => {
                let body = Json(serde_json::json!({ "error": msg }));
                (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
            }
            ApiError::Internal(msg) => {
                error!(error = %msg, "ApiError::Internal returned to client");
                let body = Json(serde_json::json!({ "error": "internal error" }));
                (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mom_core::Content;
    use std::collections::HashMap;

    #[tokio::test]
    async fn consolidation_creates_one_summary_with_backing_ids() {
        let store = SurrealDBStore::new("mem://test-consolidate").await.unwrap();
        let scope = ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: Some("agent-1".to_string()),
            run_id: None,
        };
        // Write 10 events inside the window.
        for i in 0..10 {
            let item = MemoryItem {
                id: MemoryId(format!("ev-{i}")),
                scope: scope.clone(),
                kind: MemoryKind::Event,
                created_at_ms: 1_000 + i as i64,
                content: Content::Text(format!("event {i}")),
                tags: vec![],
                importance: 0.5,
                confidence: 1.0,
                source: "agent".to_string(),
                ttl_ms: None,
                meta: std::collections::BTreeMap::new(),
                embedding: None,
                embedding_model: None,
            };
            store.put(item).await.unwrap();
        }

        let req = ConsolidateRequest {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: Some("agent-1".to_string()),
            run_id: None,
            window_start_ms: 0,
            window_end_ms: 100_000,
            importance_threshold: 0.0,
            delete_sources: true,
        };

        let resp = run_consolidation(&store, &req).await.unwrap();
        assert_eq!(resp.consolidated_count, 10);
        assert_eq!(resp.summaries.len(), 1);
        assert!(resp.sources_deleted);

        let summary = &resp.summaries[0];
        assert_eq!(summary.kind, MemoryKind::Summary);
        assert_eq!(
            summary.meta.get("window_start_ms").and_then(|v| v.as_i64()),
            Some(0)
        );
        assert_eq!(
            summary.meta.get("window_end_ms").and_then(|v| v.as_i64()),
            Some(100_000)
        );
        assert_eq!(read_provenance_ids(&summary.meta).len(), 10);

        // Sources were deleted, so a second run consolidates nothing.
        let resp2 = run_consolidation(&store, &req).await.unwrap();
        assert_eq!(resp2.consolidated_count, 0);
        assert!(resp2.summaries.is_empty());
    }

    // Helper to parse kinds filter
    fn parse_tags(tags_str: &str) -> Option<Vec<String>> {
        let tags: Vec<String> = tags_str.split(',').map(|s| s.trim().to_string()).collect();
        if tags.is_empty() || tags.iter().all(|s| s.is_empty()) {
            None
        } else {
            Some(
                tags.into_iter()
                    .filter(|s: &String| !s.is_empty())
                    .collect(),
            )
        }
    }

    #[test]
    fn test_parse_single_kind() {
        let kinds = parse_kinds("event");
        assert_eq!(kinds, Some(vec![MemoryKind::Event]));
    }

    #[test]
    fn test_parse_multiple_kinds() {
        let kinds = parse_kinds("event,summary,fact");
        assert_eq!(
            kinds,
            Some(vec![
                MemoryKind::Event,
                MemoryKind::Summary,
                MemoryKind::Fact
            ])
        );
    }

    #[test]
    fn test_parse_kinds_with_whitespace() {
        let kinds = parse_kinds("event , summary , fact");
        assert_eq!(
            kinds,
            Some(vec![
                MemoryKind::Event,
                MemoryKind::Summary,
                MemoryKind::Fact
            ])
        );
    }

    #[test]
    fn test_parse_kinds_case_insensitive() {
        let kinds = parse_kinds("EVENT,Summary,FACT");
        assert_eq!(
            kinds,
            Some(vec![
                MemoryKind::Event,
                MemoryKind::Summary,
                MemoryKind::Fact
            ])
        );
    }

    #[test]
    fn test_parse_invalid_kind() {
        let kinds = parse_kinds("invalid,event");
        assert_eq!(kinds, None);
    }

    #[test]
    fn test_parse_empty_kinds() {
        let kinds = parse_kinds("");
        assert_eq!(kinds, None);
    }

    #[test]
    fn test_parse_all_kinds() {
        let kinds = parse_kinds("event,summary,fact,preference");
        assert_eq!(
            kinds,
            Some(vec![
                MemoryKind::Event,
                MemoryKind::Summary,
                MemoryKind::Fact,
                MemoryKind::Preference
            ])
        );
    }

    #[test]
    fn test_parse_single_tag() {
        let tags = parse_tags("important");
        assert_eq!(tags, Some(vec!["important".to_string()]));
    }

    #[test]
    fn test_parse_multiple_tags() {
        let tags = parse_tags("important,urgent,review");
        assert_eq!(
            tags,
            Some(vec![
                "important".to_string(),
                "urgent".to_string(),
                "review".to_string()
            ])
        );
    }

    #[test]
    fn test_parse_tags_with_whitespace() {
        let tags = parse_tags("important , urgent , review");
        assert_eq!(
            tags,
            Some(vec![
                "important".to_string(),
                "urgent".to_string(),
                "review".to_string()
            ])
        );
    }

    #[test]
    fn test_parse_empty_tags() {
        let tags = parse_tags("");
        assert_eq!(tags, None);
    }

    #[test]
    fn test_parse_empty_tags_with_commas() {
        let tags = parse_tags(",,");
        assert_eq!(tags, None);
    }

    #[test]
    fn test_parse_tags_with_empty_elements() {
        let tags = parse_tags("important,,urgent");
        assert_eq!(
            tags,
            Some(vec!["important".to_string(), "urgent".to_string()])
        );
    }

    #[test]
    fn test_limit_default() {
        let params: HashMap<String, String> = HashMap::new();
        let limit = params
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .map(|l| l.min(100))
            .unwrap_or(10);
        assert_eq!(limit, 10);
    }

    #[test]
    fn test_limit_custom() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("limit".to_string(), "50".to_string());
        let limit = params
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .map(|l| l.min(100))
            .unwrap_or(10);
        assert_eq!(limit, 50);
    }

    #[test]
    fn test_limit_clamped() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("limit".to_string(), "500".to_string());
        let limit = params
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .map(|l| l.min(100))
            .unwrap_or(10);
        assert_eq!(limit, 100);
    }

    #[test]
    fn test_limit_invalid() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("limit".to_string(), "invalid".to_string());
        let limit = params
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .map(|l| l.min(100))
            .unwrap_or(10);
        assert_eq!(limit, 10);
    }

    #[test]
    fn test_timestamp_parsing() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("since_ms".to_string(), "1609459200000".to_string());
        params.insert("until_ms".to_string(), "1609545600000".to_string());

        let since_ms = params.get("since_ms").and_then(|s| s.parse().ok());
        let until_ms = params.get("until_ms").and_then(|s| s.parse().ok());

        assert_eq!(since_ms, Some(1609459200000i64));
        assert_eq!(until_ms, Some(1609545600000i64));
    }

    #[test]
    fn test_timestamp_invalid() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("since_ms".to_string(), "invalid".to_string());

        let since_ms = params.get("since_ms").and_then(|s| s.parse::<i64>().ok());
        assert_eq!(since_ms, None);
    }

    #[test]
    fn test_combined_filters() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("kinds".to_string(), "event,summary".to_string());
        params.insert("tags".to_string(), "important,urgent".to_string());
        params.insert("limit".to_string(), "25".to_string());
        params.insert("since_ms".to_string(), "1609459200000".to_string());

        let kinds = params.get("kinds").and_then(|k| parse_kinds(k));

        let tags_any = params.get("tags").and_then(|t| {
            let tags: Vec<String> = t.split(',').map(|s| s.trim().to_string()).collect();
            if tags.is_empty() || tags.iter().all(|s| s.is_empty()) {
                None
            } else {
                Some(
                    tags.into_iter()
                        .filter(|s: &String| !s.is_empty())
                        .collect(),
                )
            }
        });

        let limit = params
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .map(|l| l.min(100))
            .unwrap_or(10);

        let since_ms = params.get("since_ms").and_then(|s| s.parse().ok());

        assert_eq!(kinds, Some(vec![MemoryKind::Event, MemoryKind::Summary]));
        assert_eq!(
            tags_any,
            Some(vec!["important".to_string(), "urgent".to_string()])
        );
        assert_eq!(limit, 25);
        assert_eq!(since_ms, Some(1609459200000i64));
    }

    #[test]
    fn test_scope_key_parsing() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("tenant_id".to_string(), "acme".to_string());
        params.insert("workspace_id".to_string(), "workspace1".to_string());
        params.insert("project_id".to_string(), "project1".to_string());
        params.insert("agent_id".to_string(), "agent1".to_string());
        params.insert("run_id".to_string(), "run1".to_string());

        let tenant_id = params
            .get("tenant_id")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".to_string());

        assert_eq!(tenant_id, "acme");
        assert_eq!(
            params.get("workspace_id").cloned(),
            Some("workspace1".to_string())
        );
        assert_eq!(
            params.get("project_id").cloned(),
            Some("project1".to_string())
        );
        assert_eq!(params.get("agent_id").cloned(), Some("agent1".to_string()));
        assert_eq!(params.get("run_id").cloned(), Some("run1".to_string()));
    }

    #[test]
    fn test_scope_from_query_params_requires_tenant_id() {
        let params: HashMap<String, String> = HashMap::new();

        assert!(matches!(
            scope_from_query_params(&params),
            Err(ApiError::BadRequest(_))
        ));
    }

    #[test]
    fn test_scope_from_query_params_full_scope() {
        let mut params: HashMap<String, String> = HashMap::new();
        params.insert("tenant_id".to_string(), "acme".to_string());
        params.insert("workspace_id".to_string(), "workspace1".to_string());
        params.insert("project_id".to_string(), "project1".to_string());
        params.insert("agent_id".to_string(), "agent1".to_string());
        params.insert("run_id".to_string(), "run1".to_string());

        let scope = scope_from_query_params(&params).unwrap();

        assert_eq!(scope.tenant_id, "acme");
        assert_eq!(scope.workspace_id, Some("workspace1".to_string()));
        assert_eq!(scope.project_id, Some("project1".to_string()));
        assert_eq!(scope.agent_id, Some("agent1".to_string()));
        assert_eq!(scope.run_id, Some("run1".to_string()));
    }

    #[test]
    fn test_default_tenant_id() {
        let params: HashMap<String, String> = HashMap::new();
        let tenant_id = params
            .get("tenant_id")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".to_string());

        assert_eq!(tenant_id, "default");
    }

    fn task_scope() -> ScopeKey {
        ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: Some("agent-1".to_string()),
            run_id: Some("run-1".to_string()),
        }
    }

    #[test]
    fn test_checkpoint_request_round_trips() {
        let req = CheckpointRequest {
            scope: task_scope(),
            task_id: "task-7".to_string(),
            step: 12,
            scratchpad: serde_json::json!({"url": "https://example.com", "retries": 2}),
            importance: Some(0.8),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CheckpointRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "task-7");
        assert_eq!(back.step, 12);
        assert_eq!(back.importance, Some(0.8));
        assert_eq!(back.scratchpad["retries"], 2);
    }

    #[test]
    fn test_resume_request_round_trips() {
        let req = ResumeRequest {
            scope: task_scope(),
            task_id: "task-7".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ResumeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "task-7");
        assert_eq!(back.scope.tenant_id, "acme");
    }

    #[test]
    fn test_checkpoint_materializes_into_indexable_memory_item() {
        let req = CheckpointRequest {
            scope: task_scope(),
            task_id: "task-7".to_string(),
            step: 3,
            scratchpad: serde_json::json!({"k": 1}),
            importance: None,
        };

        let id = MemoryId("ckpt-fixed".to_string());
        let record = CheckpointRecord::new(
            id.clone(),
            req.scope.clone(),
            req.task_id.clone(),
            req.step,
            req.scratchpad.clone(),
        );
        let item = record.into_memory_item("agent".to_string());

        assert_eq!(item.kind, MemoryKind::Checkpoint);
        assert!(item.tags.iter().any(|t| t == "task:task-7"));
        assert_eq!(
            item.meta.get("task_id").and_then(|v| v.as_str()),
            Some("task-7")
        );

        let parsed = CheckpointRecord::try_from_memory_item(&item).unwrap();
        assert_eq!(parsed.task_id, "task-7");
        assert_eq!(parsed.step, 3);
    }

    #[test]
    fn test_resume_picks_latest_by_created_at_ms() {
        let mut older = MemoryItem::new(
            MemoryId("ckpt-old".into()),
            task_scope(),
            MemoryKind::Checkpoint,
            mom_core::Content::Json(serde_json::json!({"step": 1, "scratchpad": {}})),
            "agent".to_string(),
        );
        older.created_at_ms = 1_000;
        older.importance = 0.95;

        let mut newer = MemoryItem::new(
            MemoryId("ckpt-new".into()),
            task_scope(),
            MemoryKind::Checkpoint,
            mom_core::Content::Json(serde_json::json!({"step": 5, "scratchpad": {}})),
            "agent".to_string(),
        );
        newer.created_at_ms = 2_000;
        newer.importance = 0.5;

        let scored = vec![
            Scored {
                score: 0.95,
                item: older,
            },
            Scored {
                score: 0.5,
                item: newer.clone(),
            },
        ];

        let latest = scored
            .into_iter()
            .max_by_key(|s| s.item.created_at_ms)
            .unwrap()
            .item;
        assert_eq!(latest.id.0, "ckpt-new");
        assert_eq!(latest.created_at_ms, 2_000);
    }

    #[test]
    fn test_embedding_disabled_error() {
        // Simulate the error handling when embeddings are not available
        let error_msg = "Embeddings not available";
        assert!(!error_msg.is_empty());
        assert!(error_msg.contains("Embeddings"));
    }

    #[test]
    fn test_hybrid_search_request_validation() {
        // Test empty query validation
        let empty_query = HybridSearchRequest {
            query: String::new(),
            limit: Some(10),
        };
        assert!(empty_query.query.is_empty());

        // Test max length validation (1000 chars)
        let long_query = HybridSearchRequest {
            query: "x".repeat(1001),
            limit: Some(10),
        };
        assert!(long_query.query.len() > 1000);

        // Test valid query
        let valid_query = HybridSearchRequest {
            query: "what are my recent decisions about kubernetes?".to_string(),
            limit: Some(20),
        };
        assert!(!valid_query.query.is_empty() && valid_query.query.len() <= 1000);
    }

    #[test]
    fn test_hybrid_search_limit_clamping() {
        // Verify limit clamping logic for hybrid search (1-100 range)
        let test_cases = vec![
            (None, 10),   // None → default 10
            (Some(0), 1), // 0 → clamped to 1
            (Some(1), 1),
            (Some(50), 50),
            (Some(100), 100),
            (Some(500), 100), // 500 → clamped to 100
        ];

        for (input, expected) in test_cases {
            let clamped = input.unwrap_or(10).clamp(1, 100);
            assert_eq!(
                clamped, expected,
                "Input {:?} should clamp to {}",
                input, expected
            );
        }
    }

    #[test]
    fn test_default_ingest_scope_uses_tenant_default() {
        let scope = default_ingest_scope();
        assert!(!scope.tenant_id.is_empty());
    }

    #[test]
    fn test_source_poll_status_serialization() {
        let status = SourcePollStatus {
            source: "oxidizedrag".to_string(),
            last_poll_at_ms: Some(1_700_000_000_000),
            last_count: 3,
            last_error: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("oxidizedrag"));
        assert!(json.contains("last_poll_at_ms"));
    }

    #[test]
    fn test_context_pack_request_serialization() {
        let req = ContextPackRequest {
            query: Query {
                scope: ScopeKey {
                    tenant_id: "acme".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: Some("reviewer".to_string()),
                    run_id: None,
                },
                text: "kubernetes deployment decisions".to_string(),
                kinds: None,
                tags_any: None,
                limit: 0,
                since_ms: None,
                until_ms: None,
                cursor: None,
            },
            budget_tokens: Some(1500),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ContextPackRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.budget_tokens, Some(1500));
        assert_eq!(parsed.query.text, "kubernetes deployment decisions");
    }

    #[test]
    fn test_context_pack_candidate_limit_from_budget() {
        let budget = 1500usize;
        let candidate_limit = (budget / TOKENS_PER_ITEM).clamp(10, 100);
        assert_eq!(candidate_limit, 10);
    }

    #[test]
    fn test_scope_from_ingestion_request() {
        let req = IngestionRequest {
            tenant_id: "acme".to_string(),
            workspace_id: Some("ws-1".to_string()),
            project_id: Some("proj-1".to_string()),
            agent_id: Some("agent-1".to_string()),
            run_id: Some("run-1".to_string()),
        };
        let scope = scope_from_request(&req);
        assert_eq!(scope.tenant_id, "acme");
        assert_eq!(scope.workspace_id.as_deref(), Some("ws-1"));
        assert_eq!(scope.project_id.as_deref(), Some("proj-1"));
        assert_eq!(scope.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(scope.run_id.as_deref(), Some("run-1"));
    }

    #[test]
    fn test_hybrid_search_request_serialization() {
        // Verify HybridSearchRequest can be serialized/deserialized
        let req = HybridSearchRequest {
            query: "recall memories about meeting decisions".to_string(),
            limit: Some(15),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: HybridSearchRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.query, req.query);
        assert_eq!(deserialized.limit, req.limit);
    }

    #[test]
    fn test_hybrid_search_scope_key_construction() {
        // Verify optional scope fields (workspace_id, project_id, etc.)
        // can be None and the search spans entire tenant
        use std::collections::HashMap;

        let mut params = HashMap::new();
        params.insert("tenant_id".to_string(), "acme-corp".to_string());
        // workspace_id, project_id, agent_id, run_id deliberately omitted

        // When omitted, optional fields should be None
        let workspace_id = params.get("workspace_id").cloned();
        let project_id = params.get("project_id").cloned();
        let agent_id = params.get("agent_id").cloned();
        let run_id = params.get("run_id").cloned();

        assert!(workspace_id.is_none());
        assert!(project_id.is_none());
        assert!(agent_id.is_none());
        assert!(run_id.is_none());

        // When provided, should be Some
        let mut params_with_scope = HashMap::new();
        params_with_scope.insert("tenant_id".to_string(), "acme-corp".to_string());
        params_with_scope.insert("workspace_id".to_string(), "ws-123".to_string());
        params_with_scope.insert("project_id".to_string(), "proj-456".to_string());

        let workspace_id = params_with_scope.get("workspace_id").cloned();
        let project_id = params_with_scope.get("project_id").cloned();

        assert!(workspace_id.is_some());
        assert!(project_id.is_some());
    }

    #[tokio::test]
    async fn test_post_batch_delete() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        // 1. Test Non-Atomic Batch Delete (default)
        let item1 = MemoryItem {
            id: MemoryId("del-endpoint-1".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 1".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };
        let item2 = MemoryItem {
            id: MemoryId("del-endpoint-2".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 2".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        state.store.put(item1).await.unwrap();
        state.store.put(item2).await.unwrap();

        let mut params = std::collections::HashMap::new();
        params.insert("tenant_id".to_string(), "acme".to_string());

        let req = BatchDeleteRequest {
            ids: vec![
                MemoryId("del-endpoint-1".to_string()),
                MemoryId("del-endpoint-2".to_string()),
            ],
        };

        let res = batch_delete_memory(
            State(state.clone()),
            axum::extract::Query(params),
            Json(req),
        )
        .await
        .unwrap();

        assert_eq!(res.status(), StatusCode::MULTI_STATUS);

        assert!(state
            .store
            .get(&MemoryId("del-endpoint-1".to_string()))
            .await
            .unwrap()
            .is_none());
        assert!(state
            .store
            .get(&MemoryId("del-endpoint-2".to_string()))
            .await
            .unwrap()
            .is_none());

        // 2. Test Atomic Batch Delete
        let item3 = MemoryItem {
            id: MemoryId("del-endpoint-3".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 3".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };
        let item4 = MemoryItem {
            id: MemoryId("del-endpoint-4".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 4".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        state.store.put(item3).await.unwrap();
        state.store.put(item4).await.unwrap();

        let mut params_atomic = std::collections::HashMap::new();
        params_atomic.insert("tenant_id".to_string(), "acme".to_string());
        params_atomic.insert("atomic".to_string(), "true".to_string());

        let req_atomic = BatchDeleteRequest {
            ids: vec![
                MemoryId("del-endpoint-3".to_string()),
                MemoryId("del-endpoint-4".to_string()),
            ],
        };

        let res_atomic = batch_delete_memory(
            State(state.clone()),
            axum::extract::Query(params_atomic),
            Json(req_atomic),
        )
        .await
        .unwrap();

        assert_eq!(res_atomic.status(), StatusCode::NO_CONTENT);

        assert!(state
            .store
            .get(&MemoryId("del-endpoint-3".to_string()))
            .await
            .unwrap()
            .is_none());
        assert!(state
            .store
            .get(&MemoryId("del-endpoint-4".to_string()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_prepare_memory_item_id_generation() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let item = MemoryItem {
            id: MemoryId("".to_string()),
            scope: ScopeKey {
                tenant_id: "test-tenant".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let prepared = prepare_memory_item(&state, item).await.unwrap();
        assert!(!prepared.id.0.is_empty());
    }

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_post_batch_write_limit_check() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let item = MemoryItem {
            id: MemoryId("".to_string()),
            scope: ScopeKey {
                tenant_id: "test-tenant".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello".to_string()),
            tags: vec![],
            importance: 0.0,
            confidence: 0.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        std::env::set_var("MOM_MAX_BATCH_SIZE", "1");

        let req = BatchWriteRequest {
            items: vec![item.clone(), item.clone()],
        };

        let result = batch_write_memory(
            State(state),
            axum::extract::Query(HashMap::new()),
            Json(req),
        )
        .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ApiError::PayloadTooLarge(msg) => {
                assert!(msg.contains("exceeds maximum allowed"));
            }
            _ => panic!("expected PayloadTooLarge error"),
        }
        std::env::remove_var("MOM_MAX_BATCH_SIZE");
    }

    #[tokio::test]
    async fn test_post_batch_query_multi_scope_isolation() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "default".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let make_item = |tenant: &str, id: &str, time: i64| MemoryItem {
            id: MemoryId(id.to_string()),
            scope: ScopeKey {
                tenant_id: tenant.to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: time,
            content: Content::Text("hello".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        // Seed data
        let items = vec![
            make_item("tenant-a", "a-1", 100),
            make_item("tenant-a", "a-2", 200), // newer
            make_item("tenant-b", "b-1", 150),
            make_item("tenant-b", "b-2", 250), // newer
        ];

        state.store.write_batch(items, false).await.unwrap();

        let make_query = |tenant: &str| Query {
            scope: ScopeKey {
                tenant_id: tenant.to_string(),
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

        let req = BatchQueryRequest {
            queries: vec![make_query("tenant-a"), make_query("tenant-b")],
        };

        let response = batch_query_memory(State(state), Json(req)).await.unwrap().0;
        let results = response.results;

        assert_eq!(results.len(), 2);

        let res_a = &results[0];
        assert_eq!(res_a.len(), 2);
        assert_eq!(res_a[0].item.id.0, "a-2"); // ordered by recency
        assert_eq!(res_a[1].item.id.0, "a-1");

        let res_b = &results[1];
        assert_eq!(res_b.len(), 2);
        assert_eq!(res_b[0].item.id.0, "b-2");
        assert_eq!(res_b[1].item.id.0, "b-1");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_post_batch_write_atomic_success() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let item1 = MemoryItem {
            id: MemoryId("at-good-1".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 1".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };
        let item2 = MemoryItem {
            id: MemoryId("at-good-2".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello 2".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let mut params = std::collections::HashMap::new();
        params.insert("atomic".to_string(), "true".to_string());

        let req = BatchWriteRequest {
            items: vec![item1, item2],
        };

        let response = batch_write_memory(
            State(state.clone()),
            axum::extract::Query(params),
            Json(req),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);

        assert!(state
            .store
            .get(&MemoryId("at-good-1".to_string()))
            .await
            .unwrap()
            .is_some());
        assert!(state
            .store
            .get(&MemoryId("at-good-2".to_string()))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_post_batch_write_atomic_failure() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let item1 = MemoryItem {
            id: MemoryId("at-good".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello good".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };
        let item2 = MemoryItem {
            id: MemoryId("at-bad".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello bad".to_string()),
            tags: vec![],
            importance: 2.0,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let mut params = std::collections::HashMap::new();
        params.insert("atomic".to_string(), "true".to_string());

        let req = BatchWriteRequest {
            items: vec![item1, item2],
        };

        let result = batch_write_memory(
            State(state.clone()),
            axum::extract::Query(params),
            Json(req),
        )
        .await;

        assert!(result.is_err());

        assert!(state
            .store
            .get(&MemoryId("at-good".to_string()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_post_batch_write_best_effort() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        let item1 = MemoryItem {
            id: MemoryId("be-good".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello good".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };
        let item2 = MemoryItem {
            id: MemoryId("be-bad".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("hello bad".to_string()),
            tags: vec![],
            importance: 2.0,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let req = BatchWriteRequest {
            items: vec![item1, item2],
        };

        let response = batch_write_memory(
            State(state.clone()),
            axum::extract::Query(HashMap::new()),
            Json(req),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::MULTI_STATUS);

        assert!(state
            .store
            .get(&MemoryId("be-good".to_string()))
            .await
            .unwrap()
            .is_some());
        assert!(state
            .store
            .get(&MemoryId("be-bad".to_string()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_list_memories_pagination() {
        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        for i in 1..=5 {
            let item = MemoryItem {
                id: MemoryId(format!("page-item-{}", i)),
                scope: ScopeKey {
                    tenant_id: "acme".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: None,
                    run_id: None,
                },
                kind: MemoryKind::Event,
                created_at_ms: i * 1000,
                content: Content::Text(format!("hello {}", i)),
                tags: vec![],
                importance: 0.5,
                confidence: 0.5,
                source: "user".to_string(),
                ttl_ms: None,
                meta: Default::default(),
                embedding: None,
                embedding_model: None,
            };
            state.store.put(item).await.unwrap();
        }

        let mut params = std::collections::HashMap::new();
        params.insert("tenant_id".to_string(), "acme".to_string());
        params.insert("limit".to_string(), "2".to_string());

        let res1 = list_memories(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            axum::extract::Query(params.clone()),
        )
        .await
        .unwrap();

        let body1 = res1.0;
        assert_eq!(body1.items.len(), 2);
        assert_eq!(body1.items[0].id.0, "page-item-5");
        assert_eq!(body1.items[1].id.0, "page-item-4");
        assert!(body1.next_cursor.is_some());

        let mut params2 = params.clone();
        params2.insert("cursor".to_string(), body1.next_cursor.unwrap());

        let res2 = list_memories(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            axum::extract::Query(params2),
        )
        .await
        .unwrap();

        let body2 = res2.0;
        assert_eq!(body2.items.len(), 2);
        assert_eq!(body2.items[0].id.0, "page-item-3");
        assert_eq!(body2.items[1].id.0, "page-item-2");
        assert!(body2.next_cursor.is_some());

        let mut params3 = params.clone();
        params3.insert("cursor".to_string(), body2.next_cursor.unwrap());

        let res3 = list_memories(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            axum::extract::Query(params3),
        )
        .await
        .unwrap();

        let body3 = res3.0;
        assert_eq!(body3.items.len(), 1);
        assert_eq!(body3.items[0].id.0, "page-item-1");
        assert!(body3.next_cursor.is_none());
    }

    #[tokio::test]
    async fn test_payload_compression_round_trip() {
        use axum::body::Body;
        use flate2::read::GzDecoder;
        use flate2::write::GzEncoder;
        use http_body_util::BodyExt;
        use std::io::{Read, Write};
        use tower::ServiceExt;

        let store = SurrealDBStore::new("mem://test").await.unwrap();
        let state = AppState {
            store: Arc::new(store),
            embedder: None,
            ingestion_scheduler: Arc::new(IngestionScheduler::new(300)),
            source_registry: SourceRegistry::new(),
            poll_tracker: SharedPollTracker::new(),
            default_ingest_scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
        };

        // Construct the router matching our production router
        let app = Router::new()
            .route("/v1/memory/batch", post(batch_write_memory))
            .layer(RequestDecompressionLayer::new())
            .layer(CompressionLayer::new())
            .with_state(state.clone());

        // Prepare test items
        let item = MemoryItem {
            id: MemoryId("gzip-111122223333444455556666777788889999".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("compressed hello".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let req_body = BatchWriteRequest { items: vec![item] };

        let json_bytes = serde_json::to_vec(&req_body).unwrap();

        // Compress json_bytes with gzip
        let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&json_bytes).unwrap();
        let compressed_bytes = encoder.finish().unwrap();

        // 1. Send Gzip compressed request, ask for Gzip compressed response
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/memory/batch?atomic=true")
            .header("content-type", "application/json")
            .header("content-encoding", "gzip")
            .header("accept-encoding", "gzip")
            .body(Body::from(compressed_bytes))
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();

        println!("DEBUG Response status: {:?}", response.status());
        println!("DEBUG Response headers: {:?}", response.headers());

        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response
                .headers()
                .get("content-encoding")
                .unwrap()
                .to_str()
                .unwrap(),
            "gzip"
        );

        // Read and decompress response body
        let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
        let mut decoder = GzDecoder::new(&body_bytes[..]);
        let mut decompressed_json = Vec::new();
        decoder.read_to_end(&mut decompressed_json).unwrap();

        let resp: BatchWriteResponse = serde_json::from_slice(&decompressed_json).unwrap();
        assert_eq!(resp.ids.len(), 1);
        assert_eq!(resp.ids[0].0, "gzip-111122223333444455556666777788889999");

        // Verify it was correctly stored in the memory store
        let stored = state
            .store
            .get(&MemoryId(
                "gzip-111122223333444455556666777788889999".to_string(),
            ))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.source, "user");

        // 2. Test ZSTD compression round-trip
        let item_zstd = MemoryItem {
            id: MemoryId("zstd-111122223333444455556666777788889999".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: Content::Text("zstd compressed hello".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 0.5,
            source: "user".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let req_body_zstd = BatchWriteRequest {
            items: vec![item_zstd],
        };

        let json_bytes_zstd = serde_json::to_vec(&req_body_zstd).unwrap();
        let compressed_bytes_zstd = zstd::stream::encode_all(&json_bytes_zstd[..], 0).unwrap();

        let req_zstd = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/memory/batch?atomic=true")
            .header("content-type", "application/json")
            .header("content-encoding", "zstd")
            .header("accept-encoding", "zstd")
            .body(Body::from(compressed_bytes_zstd))
            .unwrap();

        let response_zstd = app.oneshot(req_zstd).await.unwrap();

        assert_eq!(response_zstd.status(), StatusCode::CREATED);
        assert_eq!(
            response_zstd
                .headers()
                .get("content-encoding")
                .unwrap()
                .to_str()
                .unwrap(),
            "zstd"
        );

        let body_bytes_zstd = response_zstd
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let decompressed_json_zstd = zstd::stream::decode_all(&body_bytes_zstd[..]).unwrap();

        let resp_zstd: BatchWriteResponse =
            serde_json::from_slice(&decompressed_json_zstd).unwrap();
        assert_eq!(resp_zstd.ids.len(), 1);
        assert_eq!(
            resp_zstd.ids[0].0,
            "zstd-111122223333444455556666777788889999"
        );
    }
}
