//! data-fabric Connector - Task records and knowledge base
//!
//! Ingests task records, modifications, and durable facts from data-fabric
//! as memory events and facts.

use crate::MemorySource;
use anyhow::Result;
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use std::collections::BTreeMap;

/// Memory source for data-fabric task records and knowledge
///
/// Fetches task execution records, modifications, and validated facts
/// and converts them to MOM memory items.
pub struct DataFabricSource {
    /// URL endpoint for data-fabric API
    #[allow(dead_code)]
    endpoint: String,
    /// API key if required
    api_key: Option<String>,
}

impl DataFabricSource {
    /// Create a new data-fabric connector
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            api_key: None,
        }
    }

    /// Set API key for authentication
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }
}

#[async_trait]
impl MemorySource for DataFabricSource {
    fn source_id(&self) -> &str {
        "datafabric"
    }

    fn description(&self) -> &str {
        "Task records, modifications, and validated facts from data-fabric"
    }

    async fn fetch_memories(
        &self,
        scope: &ScopeKey,
        since: Option<i64>,
    ) -> Result<Vec<MemoryItem>> {
        // Phase 2c.2: Implement actual data-fabric API integration
        // For now, return empty (stub implementation)
        //
        // Real implementation would:
        // 1. Call data-fabric API with scope (workspace, project, entity_id)
        // 2. Fetch task records and modification history
        // 3. Transform to MemoryItems:
        //    - Task execution → Event or Fact
        //    - File modifications → Event
        //    - Validated facts → Fact
        //    - Policies/preferences → Preference
        //    - Knowledge base entries → Fact
        // 4. Apply scope filtering (workspace, project, since timestamp)
        // 5. Set confidence based on validation status
        // 6. Track provenance (who/what created the fact)

        let mut memories = Vec::new();

        // Example: Task completion memory structure
        let example_memory = MemoryItem {
            id: MemoryId(format!(
                "datafabric:{}:{}:task:1",
                scope.workspace_id.as_deref().unwrap_or("unknown"),
                scope.project_id.as_deref().unwrap_or("unknown")
            )),
            scope: scope.clone(),
            kind: MemoryKind::Fact,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            content: Content::TextJson {
                text: "Example task record (placeholder)".to_string(),
                json: serde_json::json!({
                    "task_type": "build",
                    "status": "completed",
                    "placeholder": true
                }),
            },
            tags: vec!["task".to_string(), "data-fabric".to_string()],
            importance: 0.6,
            confidence: 1.0, // High confidence for validated facts
            source: self.source_id().to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        };

        if since.is_none() {
            // Only return example on full fetch, not incremental
            memories.push(example_memory);
        }

        Ok(memories)
    }

    async fn health_check(&self) -> Result<()> {
        // Phase 2c.2: Implement actual health check
        // For now, assume healthy
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datafabric_source_creation() {
        let source = DataFabricSource::new("http://localhost:8080".to_string());
        assert_eq!(source.source_id(), "datafabric");
        assert!(source.api_key.is_none());
    }

    #[test]
    fn test_datafabric_with_api_key() {
        let source = DataFabricSource::new("http://localhost:8080".to_string())
            .with_api_key("secret789".to_string());
        assert_eq!(source.api_key.as_deref(), Some("secret789"));
    }

    #[tokio::test]
    async fn test_datafabric_fetch_memories() {
        let source = DataFabricSource::new("http://localhost:8080".to_string());
        let scope = ScopeKey {
            tenant_id: "test".to_string(),
            workspace_id: Some("repo".to_string()),
            project_id: Some("ci".to_string()),
            agent_id: None,
            run_id: Some("20260305".to_string()),
        };

        let memories = source.fetch_memories(&scope, None).await.unwrap();
        // Current stub returns example memory
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].kind, MemoryKind::Fact);
        // data-fabric facts have high confidence
        assert_eq!(memories[0].confidence, 1.0);
    }
}
