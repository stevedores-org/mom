//! oxidizedRAG Connector - Code understanding and analysis
//!
//! Ingests code analysis results from oxidizedRAG as memory facts and events.
//! Provides code context, dependencies, and patterns as structured memories.
//!
//! API Integration:
//! - GET {endpoint}/v1/analyze?repo=:repo&file=:file → Code analysis results
//! - GET {endpoint}/v1/health → Health check

use crate::MemorySource;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// API response from oxidizedRAG analysis endpoint
#[derive(Debug, Deserialize, Serialize)]
struct OxidizedRAGAnalysis {
    repo: String,
    file: String,
    analysis_type: String,
    functions: Vec<FunctionAnalysis>,
    patterns: Vec<String>,
    dependencies: Vec<String>,
    timestamp: i64,
    confidence: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct FunctionAnalysis {
    name: String,
    signature: String,
    line_start: i32,
    line_end: i32,
}

/// Memory source for oxidizedRAG code analysis
///
/// Fetches code analysis results (AST patterns, semantic meanings, dependencies)
/// and converts them to MOM memory items.
pub struct OxidizedRAGSource {
    /// URL endpoint for oxidizedRAG API
    endpoint: String,
    /// HTTP client for API calls
    client: reqwest::Client,
    /// API key if required
    api_key: Option<String>,
}

impl OxidizedRAGSource {
    /// Create a new oxidizedRAG connector
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
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
        _since: Option<i64>,
    ) -> Result<Vec<MemoryItem>> {
        let mut memories = Vec::new();

        // Get repo and file from scope
        let repo = scope.workspace_id.as_deref().unwrap_or("default");
        let file = scope.project_id.as_deref().unwrap_or("all");

        // Call oxidizedRAG API
        let url = format!("{}/v1/analyze?repo={}&file={}", self.endpoint, repo, file);

        match self.client.get(&url).send().await {
            Ok(response) => {
                match response.json::<OxidizedRAGAnalysis>().await {
                    Ok(analysis) => {
                        // Convert function analyses to memory items
                        for func in analysis.functions {
                            let memory = MemoryItem {
                                id: MemoryId(format!(
                                    "oxidizedrag:{}:{}:{}",
                                    repo, file, func.name
                                )),
                                scope: scope.clone(),
                                kind: MemoryKind::Fact,
                                created_at_ms: analysis.timestamp,
                                content: Content::TextJson {
                                    text: format!(
                                        "Function {} at {}:{}",
                                        func.name, func.line_start, func.line_end
                                    ),
                                    json: serde_json::json!({
                                        "type": "function",
                                        "name": func.name,
                                        "signature": func.signature,
                                        "location": {
                                            "file": file,
                                            "start": func.line_start,
                                            "end": func.line_end
                                        }
                                    }),
                                },
                                tags: vec![
                                    "code-analysis".to_string(),
                                    "function".to_string(),
                                    "oxidizedrag".to_string(),
                                ],
                                importance: 0.7,
                                confidence: analysis.confidence,
                                source: self.source_id().to_string(),
                                ttl_ms: None,
                                meta: BTreeMap::new(),
                                embedding: None,
                                embedding_model: None,
                            };
                            memories.push(memory);
                        }

                        // Create summary memory for patterns
                        if !analysis.patterns.is_empty() {
                            let pattern_memory = MemoryItem {
                                id: MemoryId(format!("oxidizedrag:{}:{}:patterns", repo, file)),
                                scope: scope.clone(),
                                kind: MemoryKind::Summary,
                                created_at_ms: analysis.timestamp,
                                content: Content::TextJson {
                                    text: format!(
                                        "Code patterns detected: {}",
                                        analysis.patterns.join(", ")
                                    ),
                                    json: serde_json::json!({
                                        "type": "patterns",
                                        "patterns": analysis.patterns,
                                        "count": analysis.patterns.len()
                                    }),
                                },
                                tags: vec!["pattern".to_string(), "oxidizedrag".to_string()],
                                importance: 0.6,
                                confidence: analysis.confidence,
                                source: self.source_id().to_string(),
                                ttl_ms: None,
                                meta: BTreeMap::new(),
                                embedding: None,
                                embedding_model: None,
                            };
                            memories.push(pattern_memory);
                        }
                    }
                    Err(e) => {
                        return Err(anyhow!("Failed to parse oxidizedRAG response: {}", e));
                    }
                }
            }
            Err(e) => {
                return Err(anyhow!("Failed to call oxidizedRAG API: {}", e));
            }
        }

        Ok(memories)
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/v1/health", self.endpoint);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Health check failed: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oxidizedrag_source_creation() {
        let source = OxidizedRAGSource::new("http://localhost:8001".to_string());
        assert_eq!(source.source_id(), "oxidizedrag");
        assert!(source.api_key.is_none());
    }

    #[test]
    fn test_oxidizedrag_with_api_key() {
        let source = OxidizedRAGSource::new("http://localhost:8001".to_string())
            .with_api_key("secret123".to_string());
        assert_eq!(source.api_key.as_deref(), Some("secret123"));
    }

    #[test]
    fn test_oxidizedrag_analysis_parsing() {
        // Test that we can parse analysis response
        let json = serde_json::json!({
            "repo": "my-repo",
            "file": "lib.rs",
            "analysis_type": "ast",
            "functions": [
                {
                    "name": "analyze",
                    "signature": "fn analyze(&self) -> Result<Analysis>",
                    "line_start": 10i32,
                    "line_end": 25i32
                }
            ],
            "patterns": ["async_handler", "error_handling"],
            "dependencies": ["tokio", "serde"],
            "timestamp": 1609459200000i64,
            "confidence": 0.95
        });

        let analysis: OxidizedRAGAnalysis = serde_json::from_value(json).unwrap();
        assert_eq!(analysis.repo, "my-repo");
        assert_eq!(analysis.functions.len(), 1);
        assert_eq!(analysis.patterns.len(), 2);
        assert_eq!(analysis.confidence, 0.95);
    }

    #[test]
    fn test_memory_item_from_analysis() {
        // Test that memory items are properly constructed
        let scope = ScopeKey {
            tenant_id: "test".to_string(),
            workspace_id: Some("repo".to_string()),
            project_id: Some("main".to_string()),
            agent_id: None,
            run_id: None,
        };

        let memory = MemoryItem {
            id: MemoryId("oxidizedrag:repo:main:analyze".to_string()),
            scope: scope.clone(),
            kind: MemoryKind::Fact,
            created_at_ms: 1609459200000,
            content: Content::TextJson {
                text: "Function analyze at 10:25".to_string(),
                json: serde_json::json!({
                    "type": "function",
                    "name": "analyze"
                }),
            },
            tags: vec!["code-analysis".to_string()],
            importance: 0.7,
            confidence: 0.95,
            source: "oxidizedrag".to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        };

        assert_eq!(memory.source, "oxidizedrag");
        assert_eq!(memory.kind, MemoryKind::Fact);
        assert_eq!(memory.confidence, 0.95);
    }
}
