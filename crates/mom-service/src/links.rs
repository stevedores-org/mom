//! HTTP handlers for memory graph links (US-11).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use mom_core::{
    MemoryId, MemoryLink, MemoryLinkId, MemoryLinkStore, RelationshipType, TraversalStep,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{scope_from_query_params_with_headers, ApiError, AppState};
use crate::tenant::audit_tenant_access;

#[derive(Debug, Deserialize)]
pub struct CreateLinkRequest {
    pub src_memory_id: String,
    pub dst_memory_id: String,
    pub rel: String,
    pub weight: Option<f32>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateLinkRequest {
    pub weight: Option<f32>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct TraverseResponse {
    pub from: MemoryId,
    pub rel: Option<String>,
    pub steps: Vec<TraversalStep>,
}

pub async fn create_link(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<CreateLinkRequest>,
) -> Result<(StatusCode, Json<MemoryLink>), ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    let rel = RelationshipType::parse(&req.rel)
        .ok_or_else(|| ApiError::BadRequest(format!("invalid rel: {}", req.rel)))?;

    let link = MemoryLink {
        id: MemoryLinkId(uuid::Uuid::new_v4().to_string()),
        tenant_id: scope.tenant_id.clone(),
        src: MemoryId(req.src_memory_id),
        dst: MemoryId(req.dst_memory_id),
        rel,
        weight: req.weight.unwrap_or(1.0),
        confidence: req.confidence.unwrap_or(1.0),
        created_at_ms: chrono::Utc::now().timestamp_millis(),
    };

    audit_tenant_access("link_create", &scope.tenant_id, &link.id.0);
    st.store.put_link(link.clone()).await?;
    Ok((StatusCode::CREATED, Json(link)))
}

pub async fn update_link(
    State(st): State<AppState>,
    Path(link_id): Path<String>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<UpdateLinkRequest>,
) -> Result<Json<MemoryLink>, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    let existing = st
        .store
        .get_link(&scope.tenant_id, &MemoryLinkId(link_id.clone()))
        .await?
        .ok_or(ApiError::NotFound)?;

    let updated = MemoryLink {
        weight: req.weight.unwrap_or(existing.weight),
        confidence: req.confidence.unwrap_or(existing.confidence),
        ..existing
    };

    audit_tenant_access("link_update", &scope.tenant_id, &link_id);
    st.store.update_link(updated.clone()).await?;
    Ok(Json(updated))
}

pub async fn delete_link(
    State(st): State<AppState>,
    Path(link_id): Path<String>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<StatusCode, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    audit_tenant_access("link_delete", &scope.tenant_id, &link_id);
    st.store
        .delete_link(&scope.tenant_id, &MemoryLinkId(link_id))
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn traverse_links(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<TraverseResponse>, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    let from = params
        .get("from")
        .ok_or_else(|| ApiError::BadRequest("from is required".to_string()))?;
    let rel = params
        .get("rel")
        .map(|s| s.as_str())
        .and_then(RelationshipType::parse);
    let max_depth = params
        .get("depth")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    audit_tenant_access("link_traverse", &scope.tenant_id, from);
    let steps = st
        .store
        .traverse(&scope.tenant_id, &MemoryId(from.clone()), rel, max_depth)
        .await?;

    Ok(Json(TraverseResponse {
        from: MemoryId(from.clone()),
        rel: rel.map(|r| r.as_str().to_string()),
        steps,
    }))
}

pub async fn list_conflicts(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<MemoryLink>>, ApiError> {
    let scope = scope_from_query_params_with_headers(&params, &headers)?;
    let memory_id = params.get("memory_id").map(|id| MemoryId(id.clone()));

    audit_tenant_access("link_conflicts", &scope.tenant_id, "contradicts");
    let links = st
        .store
        .find_contradictions(&scope.tenant_id, memory_id.as_ref())
        .await?;
    Ok(Json(links))
}
