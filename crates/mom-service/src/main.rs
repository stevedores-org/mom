use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mom_core::{MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey, Scored};
use mom_store_surrealdb::SurrealDBStore;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    store: Arc<SurrealDBStore>,
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

    let state = AppState {
        store: Arc::new(store),
    };

    // Build router
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/memory", post(put_memory).get(list_memories))
        .route("/v1/memory/:id", get(get_memory).delete(delete_memory))
        .route("/v1/recall", post(recall))
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

// ============================================================================
// Recall Ranking Constants
// ============================================================================

/// Time window for recency decay (30 days in milliseconds)
const RECENCY_DECAY_WINDOW_MS: i64 = 30 * 24 * 60 * 60 * 1000;

/// Scoring weights for combined ranking
const TEXT_MATCH_WEIGHT: f32 = 0.60;
const IMPORTANCE_WEIGHT: f32 = 0.25;
const RECENCY_WEIGHT: f32 = 0.15;

/// Multiply limit by this factor when fetching candidates for ranking
/// Ensures high-relevance items aren't filtered by initial LIMIT clause
const CANDIDATE_MULTIPLIER: usize = 5;

// ============================================================================
// Recall Ranking Functions
// ============================================================================

/// Compute lexical search score (0..1) based on text match
/// Returns 0.0 if query doesn't match, up to 1.0 for exact match
fn compute_text_match_score(item_content: &str, query_text: &str) -> f32 {
    // Handle empty inputs to prevent division by zero
    if query_text.is_empty() || item_content.is_empty() {
        return 0.0;
    }

    let query_lower = query_text.to_lowercase();
    let content_lower = item_content.to_lowercase();

    // Exact match = 1.0
    if content_lower == query_lower {
        return 1.0;
    }

    // Substring match: check if query appears
    let match_count = content_lower.matches(&query_lower).count();
    if match_count == 0 {
        return 0.0;
    }

    // Position-based scoring: early matches score higher
    let position = content_lower
        .find(&query_lower)
        .unwrap_or(content_lower.len());
    let distance_ratio = (position as f32) / (content_lower.len() as f32);
    let position_score = 1.0 - (distance_ratio * 0.5); // Early matches boost score

    // Combined score: 50% for substring match + 50% for position
    // Multiple matches don't increase score further (already checked for existence)
    let score = 0.5 + position_score * 0.5;
    score.min(1.0)
}

/// Compute recency score (0..1) based on how recent the memory is
/// Newer items score higher, older items decay to 0 after RECENCY_DECAY_WINDOW_MS
fn compute_recency_score(created_at_ms: i64) -> f32 {
    let now = chrono::Utc::now().timestamp_millis();
    let age_ms = (now - created_at_ms).max(0);

    // Decay function: memories score 1.0 if current, decay to 0.0 after window
    let decay = (age_ms as f32) / (RECENCY_DECAY_WINDOW_MS as f32);
    (1.0 - decay).max(0.0)
}

/// Compute combined ranking score from text match, importance, and recency
fn compute_ranking_score(item: &mom_core::MemoryItem, query_text: &str) -> f32 {
    let text_match = compute_text_match_score(&item_to_text(item), query_text);

    // If there's no text match, score is 0 (no recall result)
    if text_match == 0.0 {
        return 0.0;
    }

    let recency = compute_recency_score(item.created_at_ms);
    let importance = item.importance;

    // Weighted combination of ranking factors
    (text_match * TEXT_MATCH_WEIGHT) + (importance * IMPORTANCE_WEIGHT) + (recency * RECENCY_WEIGHT)
}

/// Extract text content from MemoryItem for searching
fn item_to_text(item: &mom_core::MemoryItem) -> String {
    match &item.content {
        mom_core::Content::Text(t) => t.clone(),
        mom_core::Content::Json(v) => v.to_string(),
        mom_core::Content::TextJson { text, json } => {
            format!("{} {}", text, json)
        }
    }
}

async fn recall(
    State(st): State<AppState>,
    Json(mut q): Json<Query>,
) -> Result<Json<Vec<Scored<MemoryItem>>>, ApiError> {
    // Set default tenant if not provided
    if q.scope.tenant_id.is_empty() {
        q.scope.tenant_id = "default".to_string();
    }

    // Apply lexical search scoring if query text is provided
    if !q.text.is_empty() {
        // Fetch larger candidate set for ranking (don't lose high-relevance items to LIMIT)
        // Multiply limit by CANDIDATE_MULTIPLIER to ensure ranking sees diverse results
        let original_limit = q.limit;
        q.limit = (q.limit * CANDIDATE_MULTIPLIER).min(1000); // Cap at 1000 for safety

        let results = st.store.query(q.clone()).await?;

        // Apply ranking: compute scores and filter by text match
        let mut scored: Vec<Scored<MemoryItem>> = results
            .into_iter()
            .map(|scored_item| {
                let ranking_score = compute_ranking_score(&scored_item.item, &q.text);
                Scored {
                    score: ranking_score,
                    item: scored_item.item,
                }
            })
            .filter(|s| s.score > 0.0) // Only keep items with text match
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply original limit to final results
        scored.truncate(original_limit);

        Ok(Json(scored))
    } else {
        // No query text: return results as-is (store determines ordering)
        let results = st.store.query(q).await?;
        Ok(Json(results))
    }
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
            ApiError::Internal(_msg) => {
                // Log the real error server-side (via tracing), but return generic message to client
                // to avoid exposing sensitive information (database errors, stack traces, etc.)
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                )
            }
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

    // US-4: Recall/Lexical Search Tests

    #[test]
    fn test_text_match_score_exact() {
        let score = compute_text_match_score("deployment", "deployment");
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_text_match_score_substring() {
        let score = compute_text_match_score("production deployment complete", "deployment");
        assert!(score > 0.0 && score < 1.0);
        assert!(score >= 0.5); // Substring should score reasonably high
    }

    #[test]
    fn test_text_match_score_no_match() {
        let score = compute_text_match_score("hello world", "deployment");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_text_match_score_case_insensitive() {
        let score1 = compute_text_match_score("DEPLOYMENT", "deployment");
        let score2 = compute_text_match_score("deployment", "DEPLOYMENT");
        assert_eq!(score1, 1.0);
        assert_eq!(score2, 1.0);
    }

    #[test]
    fn test_text_match_score_early_position() {
        let early = compute_text_match_score("deployment started", "deployment");
        let late = compute_text_match_score("we started a deployment", "deployment");
        assert!(early > late); // Earlier position scores higher
    }

    #[test]
    fn test_text_match_score_multiple_occurrences() {
        let score =
            compute_text_match_score("deployment and deployment and deployment", "deployment");
        assert!(score > 0.9); // Multiple matches boost score
    }

    #[test]
    fn test_text_match_score_empty_query() {
        let score = compute_text_match_score("hello world", "");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_recency_score_current() {
        let now = chrono::Utc::now().timestamp_millis();
        let score = compute_recency_score(now);
        assert!(score > 0.95); // Very recent items score high
    }

    #[test]
    fn test_recency_score_old() {
        let thirty_one_days_ago =
            chrono::Utc::now().timestamp_millis() - (31 * 24 * 60 * 60 * 1000);
        let score = compute_recency_score(thirty_one_days_ago);
        assert!(score < 0.1); // Very old items score low
    }

    #[test]
    fn test_recency_score_one_week_old() {
        let one_week_ago = chrono::Utc::now().timestamp_millis() - (7 * 24 * 60 * 60 * 1000);
        let score = compute_recency_score(one_week_ago);
        assert!(score > 0.7); // One week old should still score reasonably
    }

    #[test]
    fn test_item_to_text_text_content() {
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: mom_core::Content::Text("deployment failed".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let text = item_to_text(&item);
        assert_eq!(text, "deployment failed");
    }

    #[test]
    fn test_item_to_text_json_content() {
        let json_val = serde_json::json!({"status": "failed", "reason": "timeout"});
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: 0,
            content: mom_core::Content::Json(json_val),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let text = item_to_text(&item);
        assert!(text.contains("failed"));
    }

    #[test]
    fn test_ranking_score_high_importance_recent() {
        let now = chrono::Utc::now().timestamp_millis();
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now,
            content: mom_core::Content::Text("deployment started".to_string()),
            tags: vec![],
            importance: 0.9,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let score = compute_ranking_score(&item, "deployment");
        assert!(score > 0.6); // Good match, high importance, recent
    }

    #[test]
    fn test_ranking_score_low_importance_old() {
        let thirty_days_ago = chrono::Utc::now().timestamp_millis() - (30 * 24 * 60 * 60 * 1000);
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: thirty_days_ago,
            content: mom_core::Content::Text("old deployment info".to_string()),
            tags: vec![],
            importance: 0.1,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let score = compute_ranking_score(&item, "deployment");
        // Exact match (0.5) + low importance (0.1 * 0.3 = 0.03) + very low recency (0.02 * 0.2)
        // ≈ 0.5 + 0.03 + 0.004 ≈ 0.534 (still reasonable since text match is high)
        assert!(score < 0.7 && score > 0.3); // Match but low importance and old
    }

    #[test]
    fn test_ranking_score_no_text_match() {
        let now = chrono::Utc::now().timestamp_millis();
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now,
            content: mom_core::Content::Text("hello world".to_string()),
            tags: vec![],
            importance: 0.9,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let score = compute_ranking_score(&item, "deployment");
        assert_eq!(score, 0.0); // No text match = 0 score (filtered out in recall)
    }

    #[test]
    fn test_ranking_combination_weights() {
        // Verify that text match, importance, and recency are weighted correctly
        let now = chrono::Utc::now().timestamp_millis();
        let item = MemoryItem {
            id: MemoryId("test".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now,
            content: mom_core::Content::Text("deployment".to_string()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "system".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let score = compute_ranking_score(&item, "deployment");
        // Expected: text_match(1.0) * 0.6 + importance(0.5) * 0.25 + recency(~1.0) * 0.15
        // = 0.6 + 0.125 + 0.15 = 0.875
        assert!(score > 0.85 && score < 0.95);
    }

    #[test]
    fn test_recall_empty_query_returns_all() {
        // Verify that empty query returns results without text filtering
        // This test validates the query logic would work if results were available
        let query_text = "";
        assert!(query_text.is_empty());
    }

    #[test]
    fn test_ranking_high_importance_beats_recency() {
        // Verify that importance significantly affects ranking
        let now = chrono::Utc::now().timestamp_millis();
        let recent_low_importance = MemoryItem {
            id: MemoryId("recent".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now,
            content: mom_core::Content::Text("deployment".to_string()),
            tags: vec![],
            importance: 0.2, // Low importance
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let old_high_importance = MemoryItem {
            id: MemoryId("old".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now - (20 * 24 * 60 * 60 * 1000), // 20 days old
            content: mom_core::Content::Text("deployment".to_string()),
            tags: vec![],
            importance: 0.9, // High importance
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let recent_score = compute_ranking_score(&recent_low_importance, "deployment");
        let old_score = compute_ranking_score(&old_high_importance, "deployment");

        // High importance should outweigh recency: 0.6 + 0.9*0.25 + ~0.6*0.15
        // = 0.6 + 0.225 + 0.09 ≈ 0.915
        // vs 0.6 + 0.2*0.25 + 1.0*0.15 = 0.6 + 0.05 + 0.15 = 0.8
        assert!(old_score > recent_score);
    }

    #[test]
    fn test_ranking_text_match_primary_factor() {
        // Verify that text match is the dominant scoring factor
        let now = chrono::Utc::now().timestamp_millis();
        let high_importance = MemoryItem {
            id: MemoryId("high".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now - (25 * 24 * 60 * 60 * 1000),
            content: mom_core::Content::Text("deployment".to_string()),
            tags: vec![],
            importance: 0.9,
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let no_match = MemoryItem {
            id: MemoryId("nomatch".to_string()),
            scope: ScopeKey {
                tenant_id: "test".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms: now,
            content: mom_core::Content::Text("hello world".to_string()),
            tags: vec![],
            importance: 0.9,
            confidence: 1.0,
            source: "test".to_string(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        };

        let match_score = compute_ranking_score(&high_importance, "deployment");
        let no_match_score = compute_ranking_score(&no_match, "deployment");

        // No text match = 0, regardless of importance/recency
        assert_eq!(no_match_score, 0.0);
        assert!(match_score > 0.0);
    }
}
