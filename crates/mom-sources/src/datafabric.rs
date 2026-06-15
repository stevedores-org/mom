//! data-fabric Connector - Task records and knowledge base
//!
//! Ingests task records, modifications, and durable facts from data-fabric
//! as memory events and facts.
//!
//! API Integration:
//! - GET {endpoint}/v1/tasks?workspace=:workspace_id&since=:timestamp → Task records
//! - GET {endpoint}/v1/facts?workspace=:workspace_id → Validated facts
//! - GET {endpoint}/v1/health → Health check

use crate::http::{apply_api_key, build_http_client, send_with_retry};
use crate::MemorySource;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Map an upstream task priority (data-fabric uses an unbounded `i32`) onto
/// `MemoryItem::importance`'s documented `0..=1` range. Treats `priority`
/// as a 0–10 scale (matching the existing intent of `priority / 10.0`) and
/// clamps any value outside that to the boundary. Negative priorities map
/// to 0, priorities ≥ 10 cap at 1.0 so the SurrealDB schema's
/// `ASSERT $value >= 0 AND $value <= 1` invariant is never violated.
/// Tracking stevedores-org/mom#3 item #12.
fn priority_to_importance(priority: i32) -> f32 {
    let p = priority.clamp(0, 10) as f32;
    p / 10.0
}

/// Compute the confidence we store for a data-fabric **fact**. Facts are
/// labeled "validated" upstream so we floor at 0.7 (still in the "high"
/// band) rather than blindly taking the upstream value; we then clamp to
/// `0..=1` so a malformed upstream confidence (NaN, negative, > 1.0)
/// can't propagate. Replaces the prior `fact.confidence.max(0.9)` thinko
/// which **raised** low-confidence facts up to 0.9, fabricating
/// confidence the source didn't claim. Tracking stevedores-org/mom#3
/// item #11.
fn floor_fact_confidence(upstream: f32) -> f32 {
    // Reject NaN before clamp: NaN.max(0.7) returns NaN on some libcs.
    let cleaned = if upstream.is_nan() { 0.7 } else { upstream };
    cleaned.max(0.7).clamp(0.0, 1.0)
}

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
            client: build_http_client().unwrap_or_else(|_| reqwest::Client::new()),
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
        let api_key = self.api_key.clone();
        match send_with_retry(|| apply_api_key(self.client.get(&url), &api_key)).await {
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
                            // MemoryItem::importance is documented `0..=1`; the
                            // upstream `priority` is an i32 with no documented
                            // upper bound, so a runaway value would silently
                            // break the invariant (and the SurrealDB schema's
                            // ASSERT). Clamp before constructing. Tracking #3
                            // item #12.
                            importance: priority_to_importance(task.priority),
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

        if let Ok(response) =
            send_with_retry(|| apply_api_key(self.client.get(&facts_url), &api_key)).await
        {
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
                        // `.max(0.9)` was a thinko — it raises every fact
                        // whose upstream confidence is below 0.9 *up* to
                        // 0.9, silently fabricating confidence the source
                        // didn't claim. Floor facts at 0.7 instead (still
                        // "high" without overstating), then clamp to the
                        // documented `0..=1` range so an upstream that
                        // violates the contract doesn't propagate.
                        // Tracking #3 item #11.
                        confidence: floor_fact_confidence(fact.confidence),
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
        let api_key = self.api_key.clone();
        send_with_retry(|| apply_api_key(self.client.get(&url), &api_key))
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

    // -------------------------------------------------------------------
    // Item #12 — `importance: (priority as f32) / 10.0` could exceed 1.0
    // -------------------------------------------------------------------

    #[test]
    fn priority_to_importance_clamps_above_ten_to_one() {
        // An upstream that returns priority = 11 (e.g. a hot-fix bump)
        // would silently violate MemoryItem::importance's `0..=1`
        // invariant and trip the SurrealDB schema's ASSERT on insert.
        assert_eq!(priority_to_importance(11), 1.0);
        assert_eq!(priority_to_importance(1_000_000), 1.0);
        assert_eq!(priority_to_importance(i32::MAX), 1.0);
    }

    #[test]
    fn priority_to_importance_clamps_negative_to_zero() {
        // Same risk on the other end — a buggy upstream returning a
        // negative priority shouldn't produce a negative importance.
        assert_eq!(priority_to_importance(-1), 0.0);
        assert_eq!(priority_to_importance(i32::MIN), 0.0);
    }

    #[test]
    fn priority_to_importance_maps_in_range_linearly() {
        assert_eq!(priority_to_importance(0), 0.0);
        assert_eq!(priority_to_importance(5), 0.5);
        assert_eq!(priority_to_importance(8), 0.8);
        assert_eq!(priority_to_importance(10), 1.0);
    }

    // -------------------------------------------------------------------
    // Item #11 — `fact.confidence.max(0.9)` raised low-confidence facts
    // -------------------------------------------------------------------

    #[test]
    fn floor_fact_confidence_does_not_inflate_high_upstream_values() {
        // Upstream said 0.95 — store 0.95. The prior code's `.max(0.9)`
        // would have kept it; this test guards the no-op path so a
        // future refactor doesn't accidentally cap.
        assert_eq!(floor_fact_confidence(0.95), 0.95);
        assert_eq!(floor_fact_confidence(1.0), 1.0);
    }

    #[test]
    fn floor_fact_confidence_floors_low_upstream_at_seven_tenths() {
        // Was: `.max(0.9)` — would have returned 0.9. New behavior
        // floors at 0.7 (still high) without fabricating 0.2 worth
        // of confidence the source didn't claim.
        assert_eq!(floor_fact_confidence(0.5), 0.7);
        assert_eq!(floor_fact_confidence(0.0), 0.7);
    }

    #[test]
    fn floor_fact_confidence_clamps_out_of_range_inputs() {
        // A malformed upstream confidence (> 1.0 or negative) must
        // not propagate to the SurrealDB schema's ASSERT.
        assert_eq!(floor_fact_confidence(1.5), 1.0);
        assert_eq!(floor_fact_confidence(-0.2), 0.7);
    }

    #[test]
    fn floor_fact_confidence_handles_nan_gracefully() {
        // NaN.max(0.7) returns NaN on some libcs, which would silently
        // poison every downstream computation. Treat as missing.
        assert_eq!(floor_fact_confidence(f32::NAN), 0.7);
    }
}
