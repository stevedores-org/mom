//! Tenant isolation helpers for HTTP handlers (US-7 / lornu-ai/mom#11).

use axum::http::HeaderMap;
use mom_core::{require_tenant_id, MemoryItem, ScopeKey};
use tracing::info;

use super::ApiError;

const TENANT_HEADER: &str = "x-tenant-id";

/// Resolve tenant from query params, optionally enforcing a matching header.
pub fn resolve_tenant_scope(
    params: &std::collections::HashMap<String, String>,
    headers: &HeaderMap,
) -> Result<ScopeKey, ApiError> {
    let tenant_id = params
        .get("tenant_id")
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))?
        .to_string();

    require_tenant_id(&tenant_id)
        .map_err(|_| ApiError::BadRequest("tenant_id is required".to_string()))?;

    if let Some(header_tenant) = headers.get(TENANT_HEADER).and_then(|v| v.to_str().ok()) {
        if !header_tenant.trim().is_empty() && header_tenant != tenant_id {
            return Err(ApiError::BadRequest(
                "tenant_id query param does not match X-Tenant-ID header".to_string(),
            ));
        }
    }

    Ok(ScopeKey {
        tenant_id,
        workspace_id: params.get("workspace_id").cloned(),
        project_id: params.get("project_id").cloned(),
        agent_id: params.get("agent_id").cloned(),
        run_id: params.get("run_id").cloned(),
    })
}

/// Validate a memory write carries a tenant and matches optional header context.
pub fn validate_memory_write(item: &MemoryItem, headers: &HeaderMap) -> Result<(), ApiError> {
    require_tenant_id(&item.scope.tenant_id)
        .map_err(|_| ApiError::BadRequest("tenant_id is required on memory writes".to_string()))?;

    if let Some(header_tenant) = headers.get(TENANT_HEADER).and_then(|v| v.to_str().ok()) {
        if !header_tenant.trim().is_empty() && header_tenant != item.scope.tenant_id {
            return Err(ApiError::BadRequest(
                "memory scope tenant_id does not match X-Tenant-ID header".to_string(),
            ));
        }
    }

    Ok(())
}

/// Structured audit log for tenant-scoped operations (until US-17 auth lands).
pub fn audit_tenant_access(operation: &str, tenant_id: &str, resource: &str) {
    info!(
        audit = "tenant_access",
        operation, tenant_id, resource, "tenant-scoped memory access"
    );
}
