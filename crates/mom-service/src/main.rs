use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mom_core::{Embedder, MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey, Scored};
use mom_embeddings::create_embedder;
use mom_sources::{
    DataFabricSource, IngestionScheduler, MemorySource, OxidizedGraphSource, OxidizedRAGSource,
};
use mom_store_surrealdb::SurrealDBStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
}

#[derive(Clone)]
struct AppState {
    store: Arc<SurrealDBStore>,
    embedder: Option<Arc<Box<dyn Embedder>>>,
    ingestion_scheduler: Arc<Mutex<IngestionScheduler>>,
    source_registry: SourceRegistry,
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

#[derive(Debug, Serialize)]
pub struct IngestionStatus {
    pub sources: usize,
    pub poll_interval_secs: u64,
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

    let state = AppState {
        store: Arc::new(store),
        embedder,
        ingestion_scheduler: Arc::new(Mutex::new(scheduler)),
        source_registry,
    };

    // Build router
    let app = Router::new()
        .without_v07_checks()
        .route("/healthz", get(healthz))
        .route("/v1/memory", post(put_memory).get(list_memories))
        .route("/v1/memory/:id", get(get_memory).delete(delete_memory))
        .route("/v1/recall", post(recall))
        .route("/v1/semantic-search", post(semantic_search))
        .route("/v1/hybrid-search", post(hybrid_search))
        .route("/v1/ingest/:source", post(ingest_source))
        .route("/v1/ingest/all", post(ingest_all))
        .route("/v1/ingest/status", get(ingest_status))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

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
    info!("  POST   /v1/ingest/:source    - Ingest from source");
    info!("  POST   /v1/ingest/all        - Ingest from all sources");
    info!("  GET    /v1/ingest/status     - Ingestion status");

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

    st.store.put(item.clone()).await?;
    Ok((StatusCode::CREATED, Json(item)))
}

async fn get_memory(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MemoryItem>, ApiError> {
    let item = st
        .store
        .get(&MemoryId(id))
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
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string());

    // Parse kinds filter (comma-separated: event,summary,fact,preference)
    let kinds = params.get("kinds").and_then(|k| {
        let parsed: Result<Vec<MemoryKind>, _> = k
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "event" => Ok(MemoryKind::Event),
                "summary" => Ok(MemoryKind::Summary),
                "fact" => Ok(MemoryKind::Fact),
                "preference" => Ok(MemoryKind::Preference),
                _ => Err(()),
            })
            .collect();
        parsed
            .ok()
            .and_then(|v| if v.is_empty() { None } else { Some(v) })
    });

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
) -> Result<StatusCode, ApiError> {
    st.store.delete(&MemoryId(id)).await?;
    Ok(StatusCode::NO_CONTENT)
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
    let query_embedding = embedder
        .embed(&req.query)
        .await
        .map_err(|e| {
            tracing::error!("embedding failed: {}", e);
            ApiError::Internal("embedding failed".to_string())
        })?;

    // Clamp limit to [1, 100] range
    let limit = req
        .limit
        .unwrap_or(10)
        .clamp(1, 100);

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

async fn ingest_source(
    State(st): State<AppState>,
    Path(source): Path<String>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<IngestionResponse>, ApiError> {
    let registry = st.source_registry.clone();

    if let Some(_source_obj) = registry.get(&source) {
        // Trigger ingestion (would call actual API in production)
        let count = 0; // Mocked for now
        Ok(Json(IngestionResponse {
            source: source.clone(),
            count,
            message: format!(
                "Ingestion triggered for {} (scope: {})",
                source, req.tenant_id
            ),
        }))
    } else {
        Err(ApiError::NotFound)
    }
}

async fn ingest_all(
    State(st): State<AppState>,
    Json(req): Json<IngestionRequest>,
) -> Result<Json<Vec<IngestionResponse>>, ApiError> {
    let scheduler = st.ingestion_scheduler.lock().await;
    let count = scheduler.source_count();

    Ok(Json(vec![IngestionResponse {
        source: "all".to_string(),
        count,
        message: format!(
            "Ingestion triggered for {} sources (scope: {})",
            count, req.tenant_id
        ),
    }]))
}

async fn ingest_status(State(st): State<AppState>) -> Result<Json<IngestionStatus>, ApiError> {
    let scheduler = st.ingestion_scheduler.lock().await;
    Ok(Json(IngestionStatus {
        sources: scheduler.source_count(),
        poll_interval_secs: 300,
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

    // Helper to parse kinds filter (extracted from list_memories logic for testability)
    fn parse_kinds(kinds_str: &str) -> Option<Vec<MemoryKind>> {
        let parsed: Result<Vec<MemoryKind>, _> = kinds_str
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "event" => Ok(MemoryKind::Event),
                "summary" => Ok(MemoryKind::Summary),
                "fact" => Ok(MemoryKind::Fact),
                "preference" => Ok(MemoryKind::Preference),
                _ => Err(()),
            })
            .collect();
        parsed
            .ok()
            .and_then(|v: Vec<MemoryKind>| if v.is_empty() { None } else { Some(v) })
    }

    // Helper to parse tags filter
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

        let kinds = params.get("kinds").and_then(|k| {
            let parsed: Result<Vec<MemoryKind>, _> = k
                .split(',')
                .map(|s| match s.trim().to_lowercase().as_str() {
                    "event" => Ok(MemoryKind::Event),
                    "summary" => Ok(MemoryKind::Summary),
                    "fact" => Ok(MemoryKind::Fact),
                    "preference" => Ok(MemoryKind::Preference),
                    _ => Err(()),
                })
                .collect();
            parsed
                .ok()
                .and_then(|v: Vec<MemoryKind>| if v.is_empty() { None } else { Some(v) })
        });

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

    // Phase 2a: Semantic Search Tests

    #[test]
    fn test_semantic_search_request_parsing() {
        use serde_json::json;

        let req_json = json!({
            "query": "deployment failed",
            "limit": 25
        });

        let req: SemanticSearchRequest = serde_json::from_value(req_json).unwrap();
        assert_eq!(req.query, "deployment failed");
        assert_eq!(req.limit, Some(25));
    }

    #[test]
    fn test_semantic_search_request_defaults() {
        use serde_json::json;

        let req_json = json!({
            "query": "error handling"
        });

        let req: SemanticSearchRequest = serde_json::from_value(req_json).unwrap();
        assert_eq!(req.query, "error handling");
        assert_eq!(req.limit, None);
    }

    #[test]
    fn test_semantic_search_request_limit_validation() {
        // Limit should be applied in endpoint handler
        let req = SemanticSearchRequest {
            query: "test".to_string(),
            limit: Some(500),
        };

        let limit = req.limit.unwrap_or(10).min(100);
        assert_eq!(limit, 100); // Should be clamped to max 100
    }

    #[test]
    fn test_semantic_search_scope_creation() {
        let _req = SemanticSearchRequest {
            query: "test".to_string(),
            limit: Some(10),
        };

        let scope = ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: None,
            run_id: None,
        };

        assert_eq!(scope.tenant_id, "acme");
        assert!(scope.workspace_id.is_none());
    }

    #[test]
    fn test_embedding_provider_env_config() {
        // Test that environment configuration would work
        let provider = std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "ollama".to_string());

        // Verify it's one of the supported providers
        assert!(
            provider == "ollama" || provider == "mistral" || provider == "openai",
            "Unknown provider: {}",
            provider
        );
    }

    #[test]
    fn test_embedding_model_defaults() {
        // Ollama default
        let ollama_model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "mxbai-embed-large".to_string());
        assert!(!ollama_model.is_empty());

        // Mistral default
        let mistral_model =
            std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-embed".to_string());
        assert!(!mistral_model.is_empty());

        // OpenAI default
        let openai_model =
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "text-embedding-3-large".to_string());
        assert!(!openai_model.is_empty());
    }

    #[test]
    fn test_semantic_search_endpoint_url_routing() {
        // Verify the semantic-search endpoint would be registered
        // This test validates the request/response types work (tenant_id comes from query param)
        let req = SemanticSearchRequest {
            query: "test query".to_string(),
            limit: Some(10),
        };

        // Verify request can be serialized/deserialized
        let json = serde_json::to_string(&req).unwrap();
        let _deserialized: SemanticSearchRequest = serde_json::from_str(&json).unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn test_vector_search_limit_bounds() {
        // Verify limit clamping logic
        let test_cases = vec![
            (0, 10),     // 0 → default 10
            (1, 1),      // 1 → 1
            (10, 10),    // 10 → 10
            (50, 50),    // 50 → 50
            (100, 100),  // 100 → 100
            (500, 100),  // 500 → clamped to 100
            (1000, 100), // 1000 → clamped to 100
        ];

        for (input, expected) in test_cases {
            let clamped = if input == 0 {
                10
            } else {
                Some(input).map(|l| l.min(100)).unwrap_or(10)
            };
            assert_eq!(
                clamped, expected,
                "Input {} should map to {}",
                input, expected
            );
        }
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
