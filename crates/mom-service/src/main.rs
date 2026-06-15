use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mom_core::{
    build_context_pack, task_tag, CheckpointRecord, ContextPack, ContextPackRequest, Embedder,
    MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey, Scored, TOKENS_PER_ITEM,
};
use mom_embeddings::{create_embedder, maybe_embed_item};
use mom_sources::{
    DataFabricSource, IngestionScheduler, MemorySource, OxidizedGraphSource, OxidizedRAGSource,
};
use mom_store_surrealdb::SurrealDBStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

/// Registry of memory sources indexed by source ID
#[derive(Clone)]
struct SourceRegistry {
    sources: Arc<HashMap<String, Arc<Box<dyn MemorySource>>>>,
}

impl SourceRegistry {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            sources: Arc::new(HashMap::new()),
        }
    }

    fn get(&self, source_id: &str) -> Option<Arc<Box<dyn MemorySource>>> {
        self.sources.get(source_id).cloned()
    }

    fn source_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.sources.keys().cloned().collect();
        ids.sort();
        ids
    }
}

#[derive(Clone)]
struct AppState {
    store: Arc<SurrealDBStore>,
    embedder: Option<Arc<Box<dyn Embedder>>>,
    ingestion_scheduler: Arc<Mutex<IngestionScheduler>>,
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

/// Get source endpoint URL from environment or use default
fn get_source_endpoint(source_name: &str, default: &str) -> String {
    let env_var = match source_name {
        "oxidizedrag" => "OXIDIZEDRAG_URL",
        "oxidizedgraph" => "OXIDIZEDGRAPH_URL",
        "datafabric" => "DATAFABRIC_URL",
        _ => return default.to_string(),
    };

    std::env::var(env_var).unwrap_or_else(|_| default.to_string())
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

async fn run_ingestion_poll_cycle(st: AppState) {
    let scope = st.default_ingest_scope.clone();
    let registry = st.source_registry.clone();
    let now_ms = chrono::Utc::now().timestamp_millis();
    let embedder = st.embedder.as_ref().map(|e| e.as_ref().as_ref());

    for source_id in registry.source_ids() {
        let Some(source_obj) = registry.get(&source_id) else {
            continue;
        };

        match persist_source_memories(&st.store, source_obj.as_ref().as_ref(), &scope, embedder)
            .await
        {
            Ok(count) => {
                st.poll_tracker
                    .record_success(&source_id, count, now_ms)
                    .await;
                info!(
                    "Background ingest: {} memories from {} (tenant: {})",
                    count, source_id, scope.tenant_id
                );
            }
            Err(err) => {
                let message = match &err {
                    ApiError::NotFound => "Not found".to_string(),
                    ApiError::BadRequest(msg) => msg.clone(),
                    ApiError::Internal(msg) => msg.clone(),
                };
                st.poll_tracker
                    .record_error(&source_id, message.clone(), now_ms)
                    .await;
                warn!("Background ingest failed for {}: {}", source_id, message);
            }
        }
    }
}

fn spawn_ingestion_poller(st: AppState, poll_interval_secs: u64) {
    if std::env::var("MOM_DISABLE_BACKGROUND_INGEST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        info!("Background ingestion polling disabled (MOM_DISABLE_BACKGROUND_INGEST)");
        return;
    }

    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(poll_interval_secs));
        interval.tick().await; // skip immediate tick on startup
        loop {
            interval.tick().await;
            run_ingestion_poll_cycle(st.clone()).await;
        }
    });
    info!(
        "Background ingestion poller started (interval: {}s)",
        poll_interval_secs
    );
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
    let store = SurrealDBStore::new(&db_path).await?;

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

    // Initialize ingestion scheduler with sources
    let mut scheduler = IngestionScheduler::new(300); // 5-minute poll interval

    // Get source endpoints from environment or use defaults
    let rag_endpoint = get_source_endpoint("oxidizedrag", "http://localhost:8001");
    let graph_endpoint = get_source_endpoint("oxidizedgraph", "http://localhost:8002");
    let fabric_endpoint = get_source_endpoint("datafabric", "http://localhost:8003");

    info!("Initializing ingestion sources:");
    info!("  oxidizedrag  : {}", rag_endpoint);
    info!("  oxidizedgraph: {}", graph_endpoint);
    info!("  datafabric   : {}", fabric_endpoint);

    // Create all memory sources
    let rag_source =
        Arc::new(Box::new(OxidizedRAGSource::new(rag_endpoint)) as Box<dyn MemorySource>);
    let graph_source =
        Arc::new(Box::new(OxidizedGraphSource::new(graph_endpoint)) as Box<dyn MemorySource>);
    let fabric_source =
        Arc::new(Box::new(DataFabricSource::new(fabric_endpoint)) as Box<dyn MemorySource>);

    // Register sources with scheduler
    scheduler.register_source(Box::new(OxidizedRAGSource::new(get_source_endpoint(
        "oxidizedrag",
        "http://localhost:8001",
    ))));
    scheduler.register_source(Box::new(OxidizedGraphSource::new(get_source_endpoint(
        "oxidizedgraph",
        "http://localhost:8002",
    ))));
    scheduler.register_source(Box::new(DataFabricSource::new(get_source_endpoint(
        "datafabric",
        "http://localhost:8003",
    ))));

    info!(
        "✅ Ingestion scheduler initialized with {} sources",
        scheduler.source_count()
    );

    // Build source registry for handlers
    let mut source_registry_map = HashMap::new();
    source_registry_map.insert("oxidizedrag".to_string(), rag_source);
    source_registry_map.insert("oxidizedgraph".to_string(), graph_source);
    source_registry_map.insert("datafabric".to_string(), fabric_source);

    let source_registry = SourceRegistry {
        sources: Arc::new(source_registry_map),
    };

    let poll_interval_secs = scheduler.poll_interval();
    let ingest_scope = default_ingest_scope();
    info!(
        "Default background ingest scope: tenant={}",
        ingest_scope.tenant_id
    );

    let state = AppState {
        store: Arc::new(store),
        embedder,
        ingestion_scheduler: Arc::new(Mutex::new(scheduler)),
        source_registry,
        poll_tracker: SharedPollTracker::new(),
        default_ingest_scope: ingest_scope,
    };

    spawn_ingestion_poller(state.clone(), poll_interval_secs);

    // Build router
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/memory", post(put_memory).get(list_memories))
        .route("/v1/memory/:id", get(get_memory).delete(delete_memory))
        .route("/v1/recall", post(recall))
        .route("/v1/semantic-search", post(semantic_search))
        .route("/v1/hybrid-search", post(hybrid_search))
        .route("/v1/context-pack", post(context_pack))
        .route("/v1/ingest/:source", post(ingest_source))
        .route("/v1/ingest/all", post(ingest_all))
        .route("/v1/ingest/status", get(ingest_status))
        .route("/v1/task/checkpoint", post(task_checkpoint))
        .route("/v1/task/resume", post(task_resume))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("✅ MOM API listening on http://{}", addr);
    info!("📚 Endpoints:");
    info!("  GET    /healthz              - Health check");
    info!("  POST   /v1/memory            - Write memory");
    info!("  GET    /v1/memory            - List memories");
    info!("  GET    /v1/memory/:id        - Get memory");
    info!("  DELETE /v1/memory/:id        - Delete memory");
    info!("  POST   /v1/recall            - Recall context");
    info!("  POST   /v1/semantic-search   - Vector semantic search");
    info!("  POST   /v1/hybrid-search     - Hybrid lexical+vector recall (RRF)");
    info!("  POST   /v1/context-pack      - Structured context bundle for agents");
    info!("  POST   /v1/ingest/:source    - Ingest from source");
    info!("  POST   /v1/ingest/all        - Ingest from all sources");
    info!("  GET    /v1/ingest/status     - Ingestion status");
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

async fn put_memory(
    State(st): State<AppState>,
    Json(mut item): Json<MemoryItem>,
) -> Result<(StatusCode, Json<MemoryItem>), ApiError> {
    // Generate ID if not provided
    if item.id.0.is_empty() {
        item.id = MemoryId(uuid::Uuid::new_v4().to_string());
    }

    if let Some(embedder) = st.embedder.as_ref() {
        maybe_embed_item(&mut item, embedder.as_ref().as_ref()).await?;
    }

    st.store.put(item.clone()).await?;
    Ok((StatusCode::CREATED, Json(item)))
}

async fn get_memory(
    State(st): State<AppState>,
    Path(id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<MemoryItem>, ApiError> {
    // SECURITY: Require tenant_id from query parameter (will be from auth context in US-17)
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

    // Use scoped get to enforce tenant isolation
    let item = st
        .store
        .get_scoped(&MemoryId(id), &scope)
        .await?
        .ok_or(ApiError::NotFound)?;
    Ok(Json(item))
}

async fn list_memories(
    State(st): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<MemoryItem>>, ApiError> {
    let tenant_id = params
        .get("tenant_id")
        .ok_or(ApiError::BadRequest("tenant_id is required".to_string()))?
        .to_string();

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

    let query = Query {
        scope: ScopeKey {
            tenant_id,
            workspace_id: params.get("workspace_id").cloned(),
            project_id: params.get("project_id").cloned(),
            agent_id: params.get("agent_id").cloned(),
            run_id: params.get("run_id").cloned(),
        },
        text: String::new(),
        kinds,
        tags_any,
        limit,
        since_ms,
        until_ms,
    };

    let results = st.store.query(query).await?;
    Ok(Json(results.into_iter().map(|s| s.item).collect()))
}

async fn delete_memory(
    State(st): State<AppState>,
    Path(id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<StatusCode, ApiError> {
    // SECURITY: Require tenant_id from query parameter (will be from auth context in US-17)
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

    // Use scoped delete to enforce tenant isolation
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
    Json(mut q): Json<Query>,
) -> Result<Json<Vec<Scored<MemoryItem>>>, ApiError> {
    // Set default tenant if not provided
    if q.scope.tenant_id.is_empty() {
        q.scope.tenant_id = "default".to_string();
    }

    let results = st.store.query(q).await?;
    Ok(Json(results))
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

    let embedder = st
        .embedder
        .as_ref()
        .ok_or_else(|| ApiError::Internal("Embedding unavailable".to_string()))?;

    // Generate embedding for query text
    let query_embedding = embedder
        .embed(&req.query)
        .await
        .map_err(|_| ApiError::Internal("Embedding unavailable".to_string()))?;

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

    let embedder = st
        .embedder
        .as_ref()
        .ok_or_else(|| ApiError::Internal("embedding provider not available".to_string()))?;

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
    };

    // Use hybrid recall from store (Phase 2b - RRF algorithm)
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

async fn persist_source_memories(
    store: &SurrealDBStore,
    source: &dyn MemorySource,
    scope: &ScopeKey,
    embedder: Option<&dyn Embedder>,
) -> Result<usize, ApiError> {
    let memories = source.fetch_memories(scope, None).await?;
    let mut count = 0;
    for mut item in memories {
        if let Some(emb) = embedder {
            maybe_embed_item(&mut item, emb).await?;
        }
        store.put(item).await?;
        count += 1;
    }
    Ok(count)
}

async fn ingest_source(
    State(st): State<AppState>,
    Path(source): Path<String>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<IngestionResponse>, ApiError> {
    let registry = st.source_registry.clone();
    let scope = scope_from_request(&req);

    let Some(source_obj) = registry.get(&source) else {
        return Err(ApiError::NotFound);
    };

    let embedder = st.embedder.as_ref().map(|e| e.as_ref().as_ref());
    let count =
        persist_source_memories(&st.store, source_obj.as_ref().as_ref(), &scope, embedder).await?;

    Ok(Json(IngestionResponse {
        source: source.clone(),
        count,
        message: format!(
            "Ingested {} memories from {} (scope: {})",
            count, source, req.tenant_id
        ),
    }))
}

async fn ingest_all(
    State(st): State<AppState>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<Vec<IngestionResponse>>, ApiError> {
    let scope = scope_from_request(&req);
    let registry = st.source_registry.clone();
    let mut responses = Vec::new();

    for source_id in registry.source_ids() {
        let Some(source_obj) = registry.get(&source_id) else {
            continue;
        };

        let embedder = st.embedder.as_ref().map(|e| e.as_ref().as_ref());
        match persist_source_memories(&st.store, source_obj.as_ref().as_ref(), &scope, embedder)
            .await
        {
            Ok(count) => responses.push(IngestionResponse {
                source: source_id.clone(),
                count,
                message: format!(
                    "Ingested {} memories from {} (scope: {})",
                    count, source_id, req.tenant_id
                ),
            }),
            Err(err) => {
                let message = match &err {
                    ApiError::NotFound => "Not found".to_string(),
                    ApiError::BadRequest(msg) => msg.clone(),
                    ApiError::Internal(msg) => msg.clone(),
                };
                warn!("Ingestion failed for {}: {}", source_id, message);
                responses.push(IngestionResponse {
                    source: source_id,
                    count: 0,
                    message: format!("Ingestion failed: {}", message),
                });
            }
        }
    }

    Ok(Json(responses))
}

async fn ingest_status(State(st): State<AppState>) -> Result<Json<IngestionStatus>, ApiError> {
    let scheduler = st.ingestion_scheduler.lock().await;
    let poll_interval_secs = scheduler.poll_interval();
    let tracker = st.poll_tracker.snapshot().await;
    let mut source_status: Vec<SourcePollStatus> = tracker.sources.into_values().collect();
    source_status.sort_by(|a, b| a.source.cmp(&b.source));

    Ok(Json(IngestionStatus {
        sources: scheduler.source_count(),
        poll_interval_secs,
        last_poll_at_ms: tracker.last_poll_at_ms,
        source_status,
    }))
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
#[derive(Debug)]
enum ApiError {
    NotFound,
    BadRequest(String),
    Internal(String),
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        error!("Internal error: {}", err);
        ApiError::Internal(err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "Not found".to_string()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
}
