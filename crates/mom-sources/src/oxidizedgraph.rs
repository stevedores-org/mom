//! oxidizedgraph Connector - Workflow orchestration and decisions
//!
//! Ingests agent workflow executions, state transitions, and decisions
//! as memory events and facts.
//!
//! API Integration:
//! - GET {endpoint}/v1/traces?agent=:agent_id&run=:run_id → Workflow traces
//! - GET {endpoint}/v1/decisions?agent=:agent_id → Decision log
//! - GET {endpoint}/v1/health → Health check

use crate::MemorySource;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// API response from oxidizedgraph traces endpoint
#[derive(Debug, Deserialize, Serialize)]
struct WorkflowTrace {
    agent_id: String,
    run_id: String,
    decisions: Vec<Decision>,
    state_transitions: Vec<StateTransition>,
    timestamp: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Decision {
    id: String,
    decision_type: String,
    action: String,
    confidence: f32,
    reasoning: String,
    timestamp: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct StateTransition {
    from_state: String,
    to_state: String,
    trigger: String,
    timestamp: i64,
}

/// Memory source for oxidizedgraph workflow execution
///
/// Fetches agent workflow traces, decision logs, and execution state
/// and converts them to MOM memory items.
pub struct OxidizedGraphSource {
    /// URL endpoint for oxidizedgraph API
    endpoint: String,
    /// HTTP client for API calls
    client: reqwest::Client,
    /// API key if required
    api_key: Option<String>,
}

impl OxidizedGraphSource {
    /// Create a new oxidizedgraph connector
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
impl MemorySource for OxidizedGraphSource {
    fn source_id(&self) -> &str {
        "oxidizedgraph"
    }

    fn description(&self) -> &str {
        "Agent workflow executions, decisions, and state transitions from oxidizedgraph"
    }

    async fn fetch_memories(
        &self,
        scope: &ScopeKey,
        _since: Option<i64>,
    ) -> Result<Vec<MemoryItem>> {
        let mut memories = Vec::new();

        let agent_id = scope.agent_id.as_deref().unwrap_or("default");
        let run_id = scope.run_id.as_deref().unwrap_or("latest");

        // Call oxidizedgraph API for traces
        let url = format!(
            "{}/v1/traces?agent={}&run={}",
            self.endpoint, agent_id, run_id
        );

        match self.client.get(&url).send().await {
            Ok(response) => {
                match response.json::<WorkflowTrace>().await {
                    Ok(trace) => {
                        // Convert decisions to memory items
                        for decision in trace.decisions {
                            let memory = MemoryItem {
                                id: MemoryId(format!(
                                    "oxidizedgraph:{}:{}:decision:{}",
                                    agent_id, run_id, decision.id
                                )),
                                scope: scope.clone(),
                                kind: MemoryKind::Fact,
                                created_at_ms: decision.timestamp,
                                content: Content::TextJson {
                                    text: format!(
                                        "Decision: {} - {}",
                                        decision.decision_type, decision.action
                                    ),
                                    json: serde_json::json!({
                                        "type": "decision",
                                        "decision_type": decision.decision_type,
                                        "action": decision.action,
                                        "reasoning": decision.reasoning
                                    }),
                                },
                                tags: vec![
                                    "workflow".to_string(),
                                    "decision".to_string(),
                                    "oxidizedgraph".to_string(),
                                ],
                                importance: 0.8,
                                confidence: decision.confidence,
                                source: self.source_id().to_string(),
                                ttl_ms: None,
                                meta: BTreeMap::new(),
                                embedding: None,
                                embedding_model: None,
                            };
                            memories.push(memory);
                        }

                        // Convert state transitions to memory items
                        for (idx, transition) in trace.state_transitions.iter().enumerate() {
                            let memory = MemoryItem {
                                id: MemoryId(format!(
                                    "oxidizedgraph:{}:{}:transition:{}",
                                    agent_id, run_id, idx
                                )),
                                scope: scope.clone(),
                                kind: MemoryKind::Event,
                                created_at_ms: transition.timestamp,
                                content: Content::TextJson {
                                    text: format!(
                                        "State transition: {} → {} ({})",
                                        transition.from_state,
                                        transition.to_state,
                                        transition.trigger
                                    ),
                                    json: serde_json::json!({
                                        "type": "state_transition",
                                        "from": transition.from_state,
                                        "to": transition.to_state,
                                        "trigger": transition.trigger
                                    }),
                                },
                                tags: vec![
                                    "state".to_string(),
                                    "transition".to_string(),
                                    "oxidizedgraph".to_string(),
                                ],
                                importance: 0.6,
                                confidence: 1.0,
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
                        return Err(anyhow!("Failed to parse oxidizedgraph response: {}", e));
                    }
                }
            }
            Err(e) => {
                return Err(anyhow!("Failed to call oxidizedgraph API: {}", e));
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
    fn test_oxidizedgraph_source_creation() {
        let source = OxidizedGraphSource::new("http://localhost:8002".to_string());
        assert_eq!(source.source_id(), "oxidizedgraph");
        assert!(source.api_key.is_none());
    }

    #[test]
    fn test_oxidizedgraph_with_api_key() {
        let source = OxidizedGraphSource::new("http://localhost:8002".to_string())
            .with_api_key("secret456".to_string());
        assert_eq!(source.api_key.as_deref(), Some("secret456"));
    }

    #[test]
    fn test_workflow_trace_parsing() {
        let json = serde_json::json!({
            "agent_id": "code-reviewer",
            "run_id": "run-001",
            "decisions": [
                {
                    "id": "d1",
                    "decision_type": "approval",
                    "action": "approve_pr",
                    "confidence": 0.95,
                    "reasoning": "Code quality is good",
                    "timestamp": 1609459200000i64
                }
            ],
            "state_transitions": [
                {
                    "from_state": "reviewing",
                    "to_state": "approved",
                    "trigger": "quality_check_passed",
                    "timestamp": 1609459200000i64
                }
            ],
            "timestamp": 1609459200000i64
        });

        let trace: WorkflowTrace = serde_json::from_value(json).unwrap();
        assert_eq!(trace.agent_id, "code-reviewer");
        assert_eq!(trace.decisions.len(), 1);
        assert_eq!(trace.state_transitions.len(), 1);
        assert_eq!(trace.decisions[0].confidence, 0.95);
    }

    #[test]
    fn test_memory_item_from_decision() {
        let scope = ScopeKey {
            tenant_id: "test".to_string(),
            workspace_id: Some("workspace".to_string()),
            project_id: Some("project".to_string()),
            agent_id: Some("agent:code-reviewer".to_string()),
            run_id: Some("run:20260305".to_string()),
        };

        let memory = MemoryItem {
            id: MemoryId("oxidizedgraph:agent:run:decision:d1".to_string()),
            scope: scope.clone(),
            kind: MemoryKind::Fact,
            created_at_ms: 1609459200000,
            content: Content::TextJson {
                text: "Decision: approval - approve_pr".to_string(),
                json: serde_json::json!({
                    "type": "decision"
                }),
            },
            tags: vec!["workflow".to_string(), "decision".to_string()],
            importance: 0.8,
            confidence: 0.95,
            source: "oxidizedgraph".to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        };

        assert_eq!(memory.source, "oxidizedgraph");
        assert_eq!(memory.kind, MemoryKind::Fact);
        assert_eq!(memory.confidence, 0.95);
    }
}
