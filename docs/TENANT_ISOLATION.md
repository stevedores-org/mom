# Tenant Isolation Model (US-7)

MOM enforces strict multi-tenant isolation at the API, query-builder, and database layers.

## Scope model

Every memory item carries a `ScopeKey`:

- `tenant_id` (required) — organization boundary; all reads and writes must specify it
- `workspace_id`, `project_id`, `agent_id`, `run_id` (optional) — sub-scope filters within a tenant

Cross-tenant access returns **empty results** or **404**, never another tenant's data.

## API layer

| Endpoint | Tenant source |
|---|---|
| `POST /v1/memory` | Required in `item.scope.tenant_id` |
| `GET /v1/memory` | Required query param `tenant_id` |
| `GET/DELETE /v1/memory/:id` | Required query param `tenant_id` |
| `POST /v1/recall` | Required query param `tenant_id` |
| `POST /v1/semantic-search` | Required query param `tenant_id` |
| `POST /v1/hybrid-search` | Required query param `tenant_id` |
| `POST /v1/ingest/*` | Required query param `tenant_id` |

Optional header `X-Tenant-ID` must match the query/body tenant when present (interim until US-17 auth).

## Storage layer

- SurrealDB `memory_items` table uses `memory_id` as the logical identifier (record id is derived from it).
- All queries include `WHERE tenant_id = $tenant`.
- Sub-scope fields are appended to the WHERE clause when set on the query scope.
- `PERMISSIONS` on the table enforce tenant match on select/create/update/delete when scope variables are supplied.

## Audit logging

Tenant-scoped operations emit structured logs:

```
audit=tenant_access operation=read tenant_id=acme resource=mem-123
```

## Error handling

Client responses never include internal database errors or cross-tenant existence hints. Internal failures return a generic `internal server error` while details are logged server-side.

## Testing

Integration tests in `mom-store-surrealdb` verify:

- Tenant B queries return empty when only Tenant A has data
- Cross-tenant scoped get returns `None`
- Blank `tenant_id` writes are rejected

## Future (US-17)

Authentication middleware will populate tenant context from verified credentials, replacing the optional `X-Tenant-ID` header check.
