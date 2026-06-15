//! MOM Core - Stable kernel API for event-sourced memory
//!
//! This is the minimal "MOM contract" - everything depends on it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryId(pub String);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Event,
    Summary,
    Fact,
    Preference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeKey {
    pub tenant_id: String,
    pub workspace_id: Option<String>,
    pub project_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Json(serde_json::Value),
    TextJson {
        text: String,
        json: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: MemoryId,
    pub scope: ScopeKey,
    pub kind: MemoryKind,
    pub created_at_ms: i64,
    pub content: Content,
    pub tags: Vec<String>,

    // ranking knobs
    pub importance: f32, // 0..1
    pub confidence: f32, // 0..1

    // provenance / safety
    pub source: String, // "user" | "tool" | "agent" | "system"
    pub ttl_ms: Option<i64>,
    pub meta: BTreeMap<String, serde_json::Value>,

    // semantic search (Phase 2)
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub scope: ScopeKey,
    pub text: String,
    pub kinds: Option<Vec<MemoryKind>>,
    pub tags_any: Option<Vec<String>>,
    pub limit: usize,

    // optional: time bounds (ms since epoch)
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scored<T> {
    pub score: f32,
    pub item: T,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipType {
    Causal,
    DerivedFrom,
    Contradicts,
    SameAs,
    References,
}

impl RelationshipType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Causal => "causal",
            Self::DerivedFrom => "derived_from",
            Self::Contradicts => "contradicts",
            Self::SameAs => "same_as",
            Self::References => "references",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "causal" => Some(Self::Causal),
            "derived_from" => Some(Self::DerivedFrom),
            "contradicts" => Some(Self::Contradicts),
            "same_as" => Some(Self::SameAs),
            "references" => Some(Self::References),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryLinkId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryLink {
    pub id: MemoryLinkId,
    pub tenant_id: String,
    pub src: MemoryId,
    pub dst: MemoryId,
    pub rel: RelationshipType,
    pub weight: f32,
    pub confidence: f32,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraversalStep {
    pub memory_id: MemoryId,
    pub depth: usize,
    pub link: MemoryLink,
}

/// Validates link `weight` and `confidence` are within 0..=1.
pub fn validate_link_metadata(weight: f32, confidence: f32) -> anyhow::Result<()> {
    if !(0.0..=1.0).contains(&weight) || !(0.0..=1.0).contains(&confidence) {
        anyhow::bail!("weight and confidence must be in 0..=1");
    }
    Ok(())
}

/// Graph edge storage for semantic memory relationships (US-11).
#[async_trait::async_trait]
pub trait MemoryLinkStore: Send + Sync {
    async fn put_link(&self, link: MemoryLink) -> anyhow::Result<()>;
    async fn update_link(&self, link: MemoryLink) -> anyhow::Result<()>;
    async fn delete_link(&self, tenant_id: &str, link_id: &MemoryLinkId) -> anyhow::Result<()>;
    async fn get_link(
        &self,
        tenant_id: &str,
        link_id: &MemoryLinkId,
    ) -> anyhow::Result<Option<MemoryLink>>;
    async fn list_links_from(
        &self,
        tenant_id: &str,
        src: &MemoryId,
        rel: Option<RelationshipType>,
    ) -> anyhow::Result<Vec<MemoryLink>>;
    async fn traverse(
        &self,
        tenant_id: &str,
        from: &MemoryId,
        rel: Option<RelationshipType>,
        max_depth: usize,
    ) -> anyhow::Result<Vec<TraversalStep>>;
    async fn find_contradictions(
        &self,
        tenant_id: &str,
        memory_id: Option<&MemoryId>,
    ) -> anyhow::Result<Vec<MemoryLink>>;
}

/// Returns an error when `tenant_id` is missing or blank.
pub fn require_tenant_id(tenant_id: &str) -> anyhow::Result<()> {
    if tenant_id.trim().is_empty() {
        anyhow::bail!("tenant_id is required");
    }
    Ok(())
}

/// Validates that a query scope carries a non-empty tenant identifier.
pub fn require_query_scope(scope: &ScopeKey) -> anyhow::Result<()> {
    require_tenant_id(&scope.tenant_id)
}

/// Returns `true` if `item_scope` satisfies the predicate expressed by
/// `query_scope`. `tenant_id` is always compared by equality; each of
/// `workspace_id` / `project_id` / `agent_id` / `run_id` is compared by
/// equality only when the query scope has it set — fields left as `None`
/// on the query side don't constrain the match. Matches the semantics
/// `MemoryStore::query` uses for the same fields so point-lookup and
/// search behave identically.
pub fn scope_matches(item_scope: &ScopeKey, query_scope: &ScopeKey) -> bool {
    if item_scope.tenant_id != query_scope.tenant_id {
        return false;
    }
    fn opt_matches(item: &Option<String>, query: &Option<String>) -> bool {
        match query {
            Some(q) => item.as_ref() == Some(q),
            None => true,
        }
    }
    opt_matches(&item_scope.workspace_id, &query_scope.workspace_id)
        && opt_matches(&item_scope.project_id, &query_scope.project_id)
        && opt_matches(&item_scope.agent_id, &query_scope.agent_id)
        && opt_matches(&item_scope.run_id, &query_scope.run_id)
}

/// Core storage trait - implement this for new backends
#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    async fn put(&self, item: MemoryItem) -> anyhow::Result<()>;
    async fn get(&self, id: &MemoryId) -> anyhow::Result<Option<MemoryItem>>;
    async fn query(&self, q: Query) -> anyhow::Result<Vec<Scored<MemoryItem>>>;
    async fn delete(&self, id: &MemoryId) -> anyhow::Result<()>;

    /// Scope-aware get: retrieves an item only if it belongs to the specified scope.
    ///
    /// SECURITY: enforces multi-tenant **and** sub-scope isolation. An item
    /// matches if `tenant_id` is equal AND every sub-scope field that the
    /// query scope sets (workspace / project / agent / run) is equal on the
    /// item. Sub-scope fields the query scope leaves as `None` are
    /// unconstrained — this matches the semantics already used by
    /// [`MemoryStore::query`] so the same scope value behaves consistently
    /// across point-lookup and search APIs.
    ///
    /// Returns `None` if the item doesn't exist or doesn't satisfy the scope
    /// predicate.
    async fn get_scoped(
        &self,
        id: &MemoryId,
        scope: &ScopeKey,
    ) -> anyhow::Result<Option<MemoryItem>> {
        if let Some(item) = self.get(id).await? {
            if scope_matches(&item.scope, scope) {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    /// Scope-aware delete: deletes an item only if it belongs to the
    /// specified scope. SECURITY semantics as for [`get_scoped`].
    /// Returns `Ok(())` whether item exists or not (idempotent).
    async fn delete_scoped(&self, id: &MemoryId, scope: &ScopeKey) -> anyhow::Result<()> {
        if let Some(item) = self.get(id).await? {
            if scope_matches(&item.scope, scope) {
                self.delete(id).await?;
            }
        }
        Ok(())
    }

    /// Vector-based semantic search (Phase 2)
    async fn vector_recall(
        &self,
        _query_embedding: &[f32],
        _scope: &ScopeKey,
        _limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        // Default implementation returns empty - implementations can override
        Ok(Vec::new())
    }

    /// Hybrid recall combining lexical + semantic search with RRF fusion (Phase 2)
    async fn hybrid_recall(
        &self,
        _q: Query,
        _query_embedding: &[f32],
        _limit: usize,
    ) -> anyhow::Result<Vec<Scored<MemoryItem>>> {
        // Default implementation returns empty - implementations can override
        Ok(Vec::new())
    }
}

/// Optional: embedder for semantic search (plug in later)
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, input: &str) -> anyhow::Result<Vec<f32>>;
    fn dims(&self) -> usize;
    fn model_id(&self) -> &str;
}

impl MemoryItem {
    pub fn new(
        id: MemoryId,
        scope: ScopeKey,
        kind: MemoryKind,
        content: Content,
        source: String,
    ) -> Self {
        Self {
            id,
            scope,
            kind,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            content,
            tags: Vec::new(),
            importance: 0.5,
            confidence: 1.0,
            source,
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relationship_type_roundtrip() {
        assert_eq!(
            RelationshipType::parse("derived_from"),
            Some(RelationshipType::DerivedFrom)
        );
        assert_eq!(RelationshipType::Contradicts.as_str(), "contradicts");
    }

    #[test]
    fn validate_link_metadata_rejects_out_of_range() {
        assert!(validate_link_metadata(1.1, 0.5).is_err());
        assert!(validate_link_metadata(0.5, -0.1).is_err());
        assert!(validate_link_metadata(0.0, 1.0).is_ok());
    }

    #[test]
    fn require_tenant_id_rejects_blank() {
        assert!(require_tenant_id("").is_err());
        assert!(require_tenant_id("   ").is_err());
    }

    #[test]
    fn require_tenant_id_accepts_non_blank() {
        assert!(require_tenant_id("acme").is_ok());
    }

    #[test]
    fn test_memory_item_new() {
        let item = MemoryItem::new(
            MemoryId("test-1".to_string()),
            ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            MemoryKind::Event,
            Content::Text("Hello world".to_string()),
            "user".to_string(),
        );

        assert_eq!(item.id.0, "test-1");
        assert_eq!(item.kind, MemoryKind::Event);
        assert_eq!(item.importance, 0.5);
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
    fn scope_matches_rejects_cross_tenant() {
        let item = scope("acme", Some("w1"), None, None, None);
        let query = scope("globex", Some("w1"), None, None, None);
        assert!(!scope_matches(&item, &query));
    }

    #[test]
    fn scope_matches_enforces_workspace_isolation() {
        let item = scope("acme", Some("workspace-a"), None, None, None);
        let query = scope("acme", Some("workspace-b"), None, None, None);
        assert!(
            !scope_matches(&item, &query),
            "an item in workspace-a must NOT satisfy a query for workspace-b in the same tenant"
        );
    }

    #[test]
    fn scope_matches_enforces_project_isolation() {
        let item = scope("acme", Some("w1"), Some("proj-a"), None, None);
        let query = scope("acme", Some("w1"), Some("proj-b"), None, None);
        assert!(!scope_matches(&item, &query));
    }

    #[test]
    fn scope_matches_enforces_agent_and_run_isolation() {
        let item = scope("acme", Some("w1"), None, Some("agent-a"), Some("run-1"));
        let same_agent_diff_run = scope("acme", Some("w1"), None, Some("agent-a"), Some("run-2"));
        let diff_agent = scope("acme", Some("w1"), None, Some("agent-b"), Some("run-1"));
        assert!(!scope_matches(&item, &same_agent_diff_run));
        assert!(!scope_matches(&item, &diff_agent));
    }

    #[test]
    fn scope_matches_treats_none_in_query_as_unconstrained() {
        let item = scope(
            "acme",
            Some("w1"),
            Some("proj-a"),
            Some("agent-a"),
            Some("run-1"),
        );
        // Query at the workspace level — should match items further-scoped within it.
        let workspace_query = scope("acme", Some("w1"), None, None, None);
        assert!(scope_matches(&item, &workspace_query));
        // Query at the tenant level — should match anything in the tenant.
        let tenant_query = scope("acme", None, None, None, None);
        assert!(scope_matches(&item, &tenant_query));
    }

    #[test]
    fn scope_matches_requires_item_to_carry_field_query_asks_for() {
        // Item is workspace-broad (None); query asks for a specific workspace.
        // The item should NOT satisfy the predicate — otherwise a tenant-wide
        // record would leak into every workspace-scoped read.
        let item = scope("acme", None, None, None, None);
        let workspace_query = scope("acme", Some("w1"), None, None, None);
        assert!(!scope_matches(&item, &workspace_query));
    }
}
