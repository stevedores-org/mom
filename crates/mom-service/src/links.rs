//! HTTP handlers for memory graph links (US-11).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use mom_core::{
    validate_link_metadata, MemoryId, MemoryLink, MemoryLinkId, MemoryLinkStore, RelationshipType,
    TraversalStep,
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

fn map_link_store_err(err: anyhow::Error) -> ApiError {
    let msg = err.to_string();
    if msg.contains("not found in tenant") || msg.contains("weight and confidence must be") {
        ApiError::BadRequest(msg)
    } else if msg == "link not found" {
        ApiError::NotFound
    } else {
        err.into()
    }
}

fn parse_optional_rel(
    params: &HashMap<String, String>,
) -> Result<Option<RelationshipType>, ApiError> {
    match params.get("rel") {
        None => Ok(None),
        Some(value) => RelationshipType::parse(value)
            .ok_or_else(|| ApiError::BadRequest(format!("invalid rel: {value}")))
            .map(Some),
    }
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

    let weight = req.weight.unwrap_or(1.0);
    let confidence = req.confidence.unwrap_or(1.0);
    validate_link_metadata(weight, confidence).map_err(map_link_store_err)?;

    let link = MemoryLink {
        id: MemoryLinkId(uuid::Uuid::new_v4().to_string()),
        tenant_id: scope.tenant_id.clone(),
        src: MemoryId(req.src_memory_id),
        dst: MemoryId(req.dst_memory_id),
        rel,
        weight,
        confidence,
        created_at_ms: chrono::Utc::now().timestamp_millis(),
    };

    audit_tenant_access("link_create", &scope.tenant_id, &link.id.0);
    st.store
        .put_link(link.clone())
        .await
        .map_err(map_link_store_err)?;
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

    let weight = req.weight.unwrap_or(existing.weight);
    let confidence = req.confidence.unwrap_or(existing.confidence);
    validate_link_metadata(weight, confidence).map_err(map_link_store_err)?;

    let updated = MemoryLink {
        weight,
        confidence,
        ..existing
    };

    audit_tenant_access("link_update", &scope.tenant_id, &link_id);
    st.store
        .update_link(updated.clone())
        .await
        .map_err(map_link_store_err)?;
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
    let rel = parse_optional_rel(&params)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_optional_rel_accepts_absent_or_valid() {
        let mut params = HashMap::new();
        assert!(parse_optional_rel(&params).unwrap().is_none());

        params.insert("rel".into(), "causal".into());
        assert_eq!(
            parse_optional_rel(&params).unwrap(),
            Some(RelationshipType::Causal)
        );
    }

    #[test]
    fn parse_optional_rel_rejects_invalid() {
        let mut params = HashMap::new();
        params.insert("rel".into(), "not-a-rel".into());
        assert!(matches!(
            parse_optional_rel(&params),
            Err(ApiError::BadRequest(_))
        ));
    }

    #[test]
    fn map_link_store_err_classifies_client_errors() {
        assert!(matches!(
            map_link_store_err(anyhow::anyhow!("source memory not found in tenant")),
            ApiError::BadRequest(_)
        ));
        assert!(matches!(
            map_link_store_err(anyhow::anyhow!("weight and confidence must be in 0..=1")),
            ApiError::BadRequest(_)
        ));
        assert!(matches!(
            map_link_store_err(anyhow::anyhow!("link not found")),
            ApiError::NotFound
        ));
    }
}
