//! Source connectors for oxidizedRAG, oxidizedgraph, and data-fabric

use crate::SourceConfig;
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

/// Trait for source connectors
#[async_trait]
pub trait SourceConnector: Send + Sync {
    /// Fetch memories from the source
    async fn fetch_memories(&self, config: &SourceConfig) -> anyhow::Result<Vec<MemoryItem>>;

    /// Health check
    async fn health_check(&self, config: &SourceConfig) -> anyhow::Result<bool>;

    /// Get connector name
    fn name(&self) -> &str;
}

// ============================================================================
// oxidizedRAG Connector - Code Analysis Memories
// ============================================================================

/// Connector for oxidizedRAG code analysis system
pub struct OxidizedRAGConnector {
    client: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct RAGQueryResult {
    query: String,
    results: Vec<RAGCodeResult>,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RAGCodeResult {
    file_path: String,
    snippet: String,
    pattern: String,
    confidence: f32,
}

impl OxidizedRAGConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn fetch_recent_queries(
        &self,
        endpoint: &str,
        api_key: &str,
    ) -> anyhow::Result<Vec<RAGQueryResult>> {
        let url = format!("{}/api/v1/queries/recent", endpoint);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        let queries = response.json::<Vec<RAGQueryResult>>().await?;
        Ok(queries)
    }
}

impl Default for OxidizedRAGConnector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceConnector for OxidizedRAGConnector {
    async fn fetch_memories(&self, config: &SourceConfig) -> anyhow::Result<Vec<MemoryItem>> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API key required for oxidizedRAG"))?;

        let queries = self
            .fetch_recent_queries(&config.endpoint, api_key)
            .await?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as i64;

        let memories: Vec<MemoryItem> = queries
            .into_iter()
            .flat_map(|query| {
                query
                    .results
                    .into_iter()
                    .map(move |result| {
                        let memory_id = MemoryId(format!("rag-{}-{}", query.timestamp, &result.file_path));

                        MemoryItem {
                            id: memory_id,
                            scope: ScopeKey {
                                tenant_id: "system".to_string(),
                                workspace_id: Some("code".to_string()),
                                project_id: None,
                                agent_id: None,
                                run_id: None,
                            },
                            kind: MemoryKind::Fact,
                            created_at_ms: now,
                            content: Content::TextJson {
                                text: format!("Code pattern: {} in {}", result.pattern, result.file_path),
                                json: serde_json::json!({
                                    "source": "oxidizedRAG",
                                    "file_path": result.file_path,
                                    "pattern": result.pattern,
                                    "snippet": result.snippet,
                                    "query": query.query,
                                }),
                            },
                            tags: vec!["code".to_string(), "pattern".to_string()],
                            importance: 0.7,
                            confidence: result.confidence,
                            source: "oxidizedRAG".to_string(),
                            ttl_ms: None,
                            meta: Default::default(),
                            embedding: None,
                            embedding_model: None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        debug!("OxidizedRAG connector fetched {} memories", memories.len());
        Ok(memories)
    }

    async fn health_check(&self, config: &SourceConfig) -> anyhow::Result<bool> {
        let api_key = config.api_key.as_ref().ok_or_else(|| anyhow::anyhow!("No API key"))?;
        let url = format!("{}/api/v1/health", config.endpoint);

        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
        {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    fn name(&self) -> &str {
        "oxidizedRAG"
    }
}

// ============================================================================
// oxidizedgraph Connector - Workflow Execution Memories
// ============================================================================

/// Connector for oxidizedgraph workflow system
pub struct OxidizedGraphConnector {
    client: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkflowExecution {
    workflow_id: String,
    execution_id: String,
    status: String,
    decisions: Vec<WorkflowDecision>,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkflowDecision {
    decision_id: String,
    description: String,
    approved: bool,
    timestamp: i64,
}

impl OxidizedGraphConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn fetch_recent_executions(
        &self,
        endpoint: &str,
        api_key: &str,
    ) -> anyhow::Result<Vec<WorkflowExecution>> {
        let url = format!("{}/api/v1/executions/recent", endpoint);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        let executions = response.json::<Vec<WorkflowExecution>>().await?;
        Ok(executions)
    }
}

impl Default for OxidizedGraphConnector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceConnector for OxidizedGraphConnector {
    async fn fetch_memories(&self, config: &SourceConfig) -> anyhow::Result<Vec<MemoryItem>> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API key required for oxidizedgraph"))?;

        let executions = self
            .fetch_recent_executions(&config.endpoint, api_key)
            .await?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as i64;

        let memories: Vec<MemoryItem> = executions
            .into_iter()
            .flat_map(|execution| {
                execution
                    .decisions
                    .into_iter()
                    .map(move |decision| {
                        let memory_id = MemoryId(format!(
                            "graph-{}-{}",
                            execution.execution_id, decision.decision_id
                        ));

                        MemoryItem {
                            id: memory_id,
                            scope: ScopeKey {
                                tenant_id: "system".to_string(),
                                workspace_id: Some("workflow".to_string()),
                                project_id: None,
                                agent_id: None,
                                run_id: Some(execution.execution_id.clone()),
                            },
                            kind: MemoryKind::Fact,
                            created_at_ms: now,
                            content: Content::TextJson {
                                text: format!(
                                    "Workflow decision: {} ({})",
                                    decision.description,
                                    if decision.approved { "approved" } else { "rejected" }
                                ),
                                json: serde_json::json!({
                                    "source": "oxidizedgraph",
                                    "workflow_id": execution.workflow_id,
                                    "execution_id": execution.execution_id,
                                    "decision_id": decision.decision_id,
                                    "description": decision.description,
                                    "approved": decision.approved,
                                    "status": execution.status,
                                }),
                            },
                            tags: vec!["workflow".to_string(), "decision".to_string()],
                            importance: if decision.approved { 0.8 } else { 0.6 },
                            confidence: 0.95,
                            source: "oxidizedgraph".to_string(),
                            ttl_ms: None,
                            meta: Default::default(),
                            embedding: None,
                            embedding_model: None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        debug!("OxidizedGraph connector fetched {} memories", memories.len());
        Ok(memories)
    }

    async fn health_check(&self, config: &SourceConfig) -> anyhow::Result<bool> {
        let api_key = config.api_key.as_ref().ok_or_else(|| anyhow::anyhow!("No API key"))?;
        let url = format!("{}/api/v1/health", config.endpoint);

        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
        {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    fn name(&self) -> &str {
        "oxidizedgraph"
    }
}

// ============================================================================
// data-fabric Connector - Task & Knowledge Memories
// ============================================================================

/// Connector for data-fabric task system
pub struct DataFabricConnector {
    client: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct TaskRecord {
    task_id: String,
    description: String,
    status: String,
    decisions: Vec<TaskDecision>,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TaskDecision {
    decision_id: String,
    content: String,
    outcome: String,
}

impl DataFabricConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn fetch_recent_tasks(
        &self,
        endpoint: &str,
        api_key: &str,
    ) -> anyhow::Result<Vec<TaskRecord>> {
        let url = format!("{}/api/v1/tasks/recent", endpoint);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        let tasks = response.json::<Vec<TaskRecord>>().await?;
        Ok(tasks)
    }
}

impl Default for DataFabricConnector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceConnector for DataFabricConnector {
    async fn fetch_memories(&self, config: &SourceConfig) -> anyhow::Result<Vec<MemoryItem>> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API key required for data-fabric"))?;

        let tasks = self
            .fetch_recent_tasks(&config.endpoint, api_key)
            .await?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as i64;

        let memories: Vec<MemoryItem> = tasks
            .into_iter()
            .flat_map(|task| {
                task.decisions
                    .into_iter()
                    .map(move |decision| {
                        let memory_id = MemoryId(format!("df-{}-{}", task.task_id, decision.decision_id));

                        MemoryItem {
                            id: memory_id,
                            scope: ScopeKey {
                                tenant_id: "system".to_string(),
                                workspace_id: Some("tasks".to_string()),
                                project_id: None,
                                agent_id: None,
                                run_id: Some(task.task_id.clone()),
                            },
                            kind: MemoryKind::Fact,
                            created_at_ms: now,
                            content: Content::TextJson {
                                text: format!("Task decision: {} ({})", decision.content, decision.outcome),
                                json: serde_json::json!({
                                    "source": "data-fabric",
                                    "task_id": task.task_id,
                                    "decision_id": decision.decision_id,
                                    "description": task.description,
                                    "decision_content": decision.content,
                                    "outcome": decision.outcome,
                                    "task_status": task.status,
                                }),
                            },
                            tags: vec!["task".to_string(), "knowledge".to_string()],
                            importance: 0.85,
                            confidence: 0.9,
                            source: "data-fabric".to_string(),
                            ttl_ms: None,
                            meta: Default::default(),
                            embedding: None,
                            embedding_model: None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        debug!("DataFabric connector fetched {} memories", memories.len());
        Ok(memories)
    }

    async fn health_check(&self, config: &SourceConfig) -> anyhow::Result<bool> {
        let api_key = config.api_key.as_ref().ok_or_else(|| anyhow::anyhow!("No API key"))?;
        let url = format!("{}/api/v1/health", config.endpoint);

        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
        {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    fn name(&self) -> &str {
        "data-fabric"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OxidizedRAG Tests
    #[test]
    fn test_oxidizedrag_connector_creation() {
        let connector = OxidizedRAGConnector::new();
        assert_eq!(connector.name(), "oxidizedRAG");
    }

    #[test]
    fn test_oxidizedrag_default() {
        let _connector = OxidizedRAGConnector::default();
    }

    #[tokio::test]
    async fn test_oxidizedrag_health_check_no_api_key() {
        let connector = OxidizedRAGConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "oxidizedrag".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.health_check(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_oxidizedrag_fetch_memories_no_api_key() {
        let connector = OxidizedRAGConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "oxidizedrag".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.fetch_memories(&config).await;
        assert!(result.is_err());
    }

    // OxidizedGraph Tests
    #[test]
    fn test_oxidizedgraph_connector_creation() {
        let connector = OxidizedGraphConnector::new();
        assert_eq!(connector.name(), "oxidizedgraph");
    }

    #[test]
    fn test_oxidizedgraph_default() {
        let _connector = OxidizedGraphConnector::default();
    }

    #[tokio::test]
    async fn test_oxidizedgraph_health_check_no_api_key() {
        let connector = OxidizedGraphConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "oxidizedgraph".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.health_check(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_oxidizedgraph_fetch_memories_no_api_key() {
        let connector = OxidizedGraphConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "oxidizedgraph".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.fetch_memories(&config).await;
        assert!(result.is_err());
    }

    // DataFabric Tests
    #[test]
    fn test_datafabric_connector_creation() {
        let connector = DataFabricConnector::new();
        assert_eq!(connector.name(), "data-fabric");
    }

    #[test]
    fn test_datafabric_default() {
        let _connector = DataFabricConnector::default();
    }

    #[tokio::test]
    async fn test_datafabric_health_check_no_api_key() {
        let connector = DataFabricConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "data-fabric".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.health_check(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_datafabric_fetch_memories_no_api_key() {
        let connector = DataFabricConnector::new();
        let config = SourceConfig {
            source_id: "test".to_string(),
            source_type: "data-fabric".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: None,
            poll_interval_secs: 60,
            enabled: true,
        };

        let result = connector.fetch_memories(&config).await;
        assert!(result.is_err());
    }
}
