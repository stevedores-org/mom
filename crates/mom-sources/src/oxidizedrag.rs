//! oxidizedRAG Connector - Code understanding and analysis
//!
//! Ingests code analysis results from oxidizedRAG as memory facts and events.
//! Provides code context, dependencies, and patterns as structured memories.

use crate::MemorySource;
use anyhow::Result;
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use std::collections::BTreeMap;

/// Memory source for oxidizedRAG code analysis
///
/// Fetches code analysis results (AST patterns, semantic meanings, dependencies)
/// and converts them to MOM memory items.
pub struct OxidizedRAGSource {
    /// URL endpoint for oxidizedRAG API
    #[allow(dead_code)]
    endpoint: String,
    /// API key if required
    api_key: Option<String>,
}

impl OxidizedRAGSource {
    /// Create a new oxidizedRAG connector
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
impl MemorySource for OxidizedRAGSource {
    fn source_id(&self) -> &str {
        "oxidizedrag"
    }

    fn description(&self) -> &str {
        "Code analysis and pattern extraction from oxidizedRAG"
    }

    async fn fetch_memories(
        &self,
        scope: &ScopeKey,
        since: Option<i64>,
    ) -> Result<Vec<MemoryItem>> {
        // Phase 2c.2: Implement actual oxidizedRAG API integration
        // For now, return empty (stub implementation)
        //
        // Real implementation would:
        // 1. Call oxidizedRAG API with scope (repo, file, language)
        // 2. Fetch code analysis results
        // 3. Transform to MemoryItems:
        //    - Function definitions → Fact
        //    - Cross-file calls → Fact
        //    - Patterns detected → Summary
        //    - File modifications → Event
        // 4. Apply scope filtering (since timestamp)
        // 5. Attach embeddings via embedder

        let mut memories = Vec::new();

        // Example: Code analysis memory structure
        let example_memory = MemoryItem {
            id: MemoryId(format!(
                "oxidizedrag:{}:{}:example",
                scope.workspace_id.as_deref().unwrap_or("unknown"),
                scope.project_id.as_deref().unwrap_or("unknown")
            )),
            scope: scope.clone(),
            kind: MemoryKind::Fact,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            content: Content::TextJson {
                text: "Example code analysis memory (placeholder)".to_string(),
                json: serde_json::json!({
                    "analysis_type": "code",
                    "placeholder": true
                }),
            },
            tags: vec!["code-analysis".to_string(), "oxidizedrag".to_string()],
            importance: 0.7,
            confidence: 0.8,
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
    fn test_oxidizedrag_source_creation() {
        let source = OxidizedRAGSource::new("http://localhost:8080".to_string());
        assert_eq!(source.source_id(), "oxidizedrag");
        assert!(source.api_key.is_none());
    }

    #[test]
    fn test_oxidizedrag_with_api_key() {
        let source = OxidizedRAGSource::new("http://localhost:8080".to_string())
            .with_api_key("secret123".to_string());
        assert_eq!(source.api_key.as_deref(), Some("secret123"));
    }

    #[tokio::test]
    async fn test_oxidizedrag_fetch_memories() {
        let source = OxidizedRAGSource::new("http://localhost:8080".to_string());
        let scope = ScopeKey {
            tenant_id: "test".to_string(),
            workspace_id: Some("repo".to_string()),
            project_id: Some("main".to_string()),
            agent_id: None,
            run_id: None,
        };

        let memories = source.fetch_memories(&scope, None).await.unwrap();
        // Current stub returns example memory
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].kind, MemoryKind::Fact);
    }
}
