//! data-fabric Connector - Task records and knowledge base
//!
//! Ingests task records, modifications, and durable facts from data-fabric
//! as memory events and facts.
//!
//! API Integration:
//! - GET {endpoint}/v1/tasks?workspace=:workspace_id&since=:timestamp → Task records
//! - GET {endpoint}/v1/facts?workspace=:workspace_id → Validated facts
//! - GET {endpoint}/v1/health → Health check

use crate::MemorySource;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// API response from data-fabric tasks endpoint
#[derive(Debug, Deserialize, Serialize)]
struct TaskRecord {
    id: String,
    task_type: String,
    status: String,
    title: String,
    description: String,
    started_at: i64,
    completed_at: Option<i64>,
    owner: String,
    priority: i32,
}

#[derive(Debug, Deserialize, Serialize)]
struct FactRecord {
    id: String,
    content: String,
    category: String,
    validation_status: String,
    created_at: i64,
    confidence: f32,
}

/// Memory source for data-fabric task records and knowledge
///
/// Fetches task execution records, modifications, and validated facts
/// and converts them to MOM memory items.
pub struct DataFabricSource {
    /// URL endpoint for data-fabric API
    endpoint: String,
    /// HTTP client for API calls
    client: reqwest::Client,
    /// API key if required
    api_key: Option<String>,
}

impl DataFabricSource {
    /// Create a new data-fabric connector
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
        let mut memories = Vec::new();

        let workspace_id = scope.workspace_id.as_deref().unwrap_or("default");

        // Build URL with optional since parameter
        let url = match since {
            Some(ts) => format!(
                "{}/v1/tasks?workspace={}&since={}",
                self.endpoint, workspace_id, ts
            ),
            None => format!("{}/v1/tasks?workspace={}", self.endpoint, workspace_id),
        };

        // Fetch task records
        match self.client.get(&url).send().await {
            Ok(response) => match response.json::<Vec<TaskRecord>>().await {
                Ok(tasks) => {
                    for task in tasks {
                        let memory = MemoryItem {
                            id: MemoryId(format!("datafabric:task:{}", task.id)),
                            scope: scope.clone(),
                            kind: if task.status == "completed" {
                                MemoryKind::Fact
                            } else {
                                MemoryKind::Event
                            },
                            created_at_ms: task.started_at,
                            content: Content::TextJson {
                                text: format!("{}: {}", task.task_type, task.title),
                                json: serde_json::json!({
                                    "type": "task",
                                    "task_type": task.task_type,
                                    "status": task.status,
                                    "title": task.title,
                                    "description": task.description,
                                    "priority": task.priority,
                                    "owner": task.owner
                                }),
                            },
                            tags: vec![
                                "task".to_string(),
                                task.task_type.clone(),
                                format!("priority:{}", task.priority),
                                "datafabric".to_string(),
                            ],
                            importance: (task.priority as f32) / 10.0,
                            confidence: if task.status == "completed" { 1.0 } else { 0.7 },
                            source: self.source_id().to_string(),
                            ttl_ms: None,
                            meta: BTreeMap::new(),
                            embedding: None,
                            embedding_model: None,
                        };
                        memories.push(memory);
                    }
                }
                Err(e) => {
                    return Err(anyhow!("Failed to parse data-fabric tasks response: {}", e));
                }
            },
            Err(e) => {
                return Err(anyhow!("Failed to call data-fabric API: {}", e));
            }
        }

        // Optionally fetch validated facts
        let facts_url = format!("{}/v1/facts?workspace={}", self.endpoint, workspace_id);

        if let Ok(response) = self.client.get(&facts_url).send().await {
            if let Ok(facts) = response.json::<Vec<FactRecord>>().await {
                for fact in facts {
                    let memory = MemoryItem {
                        id: MemoryId(format!("datafabric:fact:{}", fact.id)),
                        scope: scope.clone(),
                        kind: MemoryKind::Fact,
                        created_at_ms: fact.created_at,
                        content: Content::TextJson {
                            text: fact.content.clone(),
                            json: serde_json::json!({
                                "type": "fact",
                                "category": fact.category,
                                "validation_status": fact.validation_status
                            }),
                        },
                        tags: vec![
                            "fact".to_string(),
                            fact.category,
                            "validated".to_string(),
                            "datafabric".to_string(),
                        ],
                        importance: 0.8,
                        confidence: fact.confidence.max(0.9), // facts have high confidence
                        source: self.source_id().to_string(),
                        ttl_ms: None,
                        meta: BTreeMap::new(),
                        embedding: None,
                        embedding_model: None,
                    };
                    memories.push(memory);
                }
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
    fn test_datafabric_source_creation() {
        let source = DataFabricSource::new("http://localhost:8003".to_string());
        assert_eq!(source.source_id(), "datafabric");
        assert!(source.api_key.is_none());
    }

    #[test]
    fn test_datafabric_with_api_key() {
        let source = DataFabricSource::new("http://localhost:8003".to_string())
            .with_api_key("secret789".to_string());
        assert_eq!(source.api_key.as_deref(), Some("secret789"));
    }

    #[test]
    fn test_task_record_parsing() {
        let json = serde_json::json!({
            "id": "task-001",
            "task_type": "build",
            "status": "completed",
            "title": "Compile project",
            "description": "Full project compilation",
            "started_at": 1609459200000i64,
            "completed_at": 1609459300000i64,
            "owner": "ci-agent",
            "priority": 8
        });

        let task: TaskRecord = serde_json::from_value(json).unwrap();
        assert_eq!(task.id, "task-001");
        assert_eq!(task.status, "completed");
        assert_eq!(task.priority, 8);
    }

    #[test]
    fn test_fact_record_parsing() {
        let json = serde_json::json!({
            "id": "fact-001",
            "content": "API endpoint accepts POST requests",
            "category": "api-spec",
            "validation_status": "validated",
            "created_at": 1609459200000i64,
            "confidence": 0.98
        });

        let fact: FactRecord = serde_json::from_value(json).unwrap();
        assert_eq!(fact.id, "fact-001");
        assert_eq!(fact.validation_status, "validated");
        assert!(fact.confidence >= 0.9);
    }

    #[test]
    fn test_memory_item_from_task() {
        let scope = ScopeKey {
            tenant_id: "test".to_string(),
            workspace_id: Some("repo".to_string()),
            project_id: Some("ci".to_string()),
            agent_id: None,
            run_id: Some("20260305".to_string()),
        };

        let memory = MemoryItem {
            id: MemoryId("datafabric:task:task-001".to_string()),
            scope: scope.clone(),
            kind: MemoryKind::Fact,
            created_at_ms: 1609459200000,
            content: Content::TextJson {
                text: "build: Compile project".to_string(),
                json: serde_json::json!({"type": "task"}),
            },
            tags: vec!["task".to_string(), "build".to_string()],
            importance: 0.8,
            confidence: 1.0,
            source: "datafabric".to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        };

        assert_eq!(memory.source, "datafabric");
        assert_eq!(memory.kind, MemoryKind::Fact);
        assert_eq!(memory.confidence, 1.0);
    }
}
