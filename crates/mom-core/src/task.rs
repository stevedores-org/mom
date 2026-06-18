//! Typed wrappers over `MemoryItem` for the `Task` and `Checkpoint`
//! memory kinds.
//!
//! The kernel stores agent task tracking and durable-execution checkpoints
//! as generic `MemoryItem`s with `kind = Task` / `Checkpoint`. The structured
//! fields (status, scratchpad, dependency edges) live inside `Content::Json`
//! and `meta`. These wrappers codify the convention so consumers don't have
//! to re-derive it.

use crate::{Content, MemoryId, MemoryItem, MemoryKind, ScopeKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Meta key under which a `Checkpoint` records the originating task id.
pub const META_TASK_ID: &str = "task_id";

/// Tag prefix used to index checkpoints by their owning task so lookups
/// can ride on the existing `tags_any` query path without needing a new
/// store primitive.
pub const TAG_TASK_PREFIX: &str = "task:";

/// Build the tag value for a given task id.
pub fn task_tag(task_id: &str) -> String {
    format!("{}{}", TAG_TASK_PREFIX, task_id)
}

/// High-level state of an agent task.
///
/// Stored under the `"status"` key inside the task's `Content::Json` body.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Blocked,
    Completed,
    Failed,
}

/// Typed view over a `MemoryItem` whose `kind == MemoryKind::Task`.
///
/// Construct via [`TaskRecord::new`] or parse from an existing item via
/// [`TaskRecord::try_from_memory_item`]. The structured fields are mirrored
/// into the inner `MemoryItem`'s `Content::Json` so downstream consumers
/// that only know about `MemoryItem` still see the same data.
#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: MemoryId,
    pub scope: ScopeKey,
    pub status: TaskStatus,
    pub description: String,
    pub depends_on: Vec<String>,
    pub importance: f32,
}

/// Typed view over a `MemoryItem` whose `kind == MemoryKind::Checkpoint`.
///
/// A checkpoint references its originating task via `meta["task_id"]` and
/// is also tagged with `task:<task_id>` so it can be looked up via the
/// existing `tags_any` query path.
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    pub id: MemoryId,
    pub scope: ScopeKey,
    pub task_id: String,
    pub step: i64,
    pub scratchpad: Value,
    pub importance: f32,
}

/// Reasons a `MemoryItem` cannot be interpreted as a typed record.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TaskParseError {
    #[error("wrong memory kind: expected {expected:?}, got {actual:?}")]
    WrongKind {
        expected: MemoryKind,
        actual: MemoryKind,
    },
    #[error("content is not Content::Json (was {got})")]
    NotJsonContent { got: &'static str },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("field {0} has wrong type")]
    WrongFieldType(&'static str),
}

impl TaskRecord {
    pub fn new(id: MemoryId, scope: ScopeKey, status: TaskStatus, description: String) -> Self {
        Self {
            id,
            scope,
            status,
            description,
            depends_on: Vec::new(),
            importance: 0.5,
        }
    }

    pub fn with_depends_on(mut self, depends_on: Vec<String>) -> Self {
        self.depends_on = depends_on;
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Materialize this record into a `MemoryItem` suitable for `MemoryStore::put`.
    pub fn into_memory_item(self, source: String) -> MemoryItem {
        let content = Content::Json(serde_json::json!({
            "status": self.status,
            "description": self.description,
            "depends_on": self.depends_on,
        }));
        let mut item = MemoryItem::new(self.id, self.scope, MemoryKind::Task, content, source);
        item.importance = self.importance;
        item
    }

    pub fn try_from_memory_item(item: &MemoryItem) -> Result<Self, TaskParseError> {
        if item.kind != MemoryKind::Task {
            return Err(TaskParseError::WrongKind {
                expected: MemoryKind::Task,
                actual: item.kind,
            });
        }
        let json = match &item.content {
            Content::Json(v) => v,
            Content::Text(_) => return Err(TaskParseError::NotJsonContent { got: "text" }),
            Content::TextJson { .. } => {
                return Err(TaskParseError::NotJsonContent { got: "text_json" })
            }
        };

        let status: TaskStatus = serde_json::from_value(
            json.get("status")
                .ok_or(TaskParseError::MissingField("status"))?
                .clone(),
        )
        .map_err(|_| TaskParseError::WrongFieldType("status"))?;

        let description = json
            .get("description")
            .and_then(Value::as_str)
            .ok_or(TaskParseError::MissingField("description"))?
            .to_string();

        let depends_on = json
            .get("depends_on")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            id: item.id.clone(),
            scope: item.scope.clone(),
            status,
            description,
            depends_on,
            importance: item.importance,
        })
    }
}

impl CheckpointRecord {
    pub fn new(
        id: MemoryId,
        scope: ScopeKey,
        task_id: String,
        step: i64,
        scratchpad: Value,
    ) -> Self {
        Self {
            id,
            scope,
            task_id,
            step,
            scratchpad,
            importance: 0.5,
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn into_memory_item(self, source: String) -> MemoryItem {
        let content = Content::Json(serde_json::json!({
            "step": self.step,
            "scratchpad": self.scratchpad,
        }));
        let mut item =
            MemoryItem::new(self.id, self.scope, MemoryKind::Checkpoint, content, source);
        item.importance = self.importance;
        item.tags.push(task_tag(&self.task_id));

        let mut meta: BTreeMap<String, Value> = BTreeMap::new();
        meta.insert(
            META_TASK_ID.to_string(),
            Value::String(self.task_id.clone()),
        );
        item.meta = meta;
        item
    }

    pub fn try_from_memory_item(item: &MemoryItem) -> Result<Self, TaskParseError> {
        if item.kind != MemoryKind::Checkpoint {
            return Err(TaskParseError::WrongKind {
                expected: MemoryKind::Checkpoint,
                actual: item.kind,
            });
        }
        let json = match &item.content {
            Content::Json(v) => v,
            Content::Text(_) => return Err(TaskParseError::NotJsonContent { got: "text" }),
            Content::TextJson { .. } => {
                return Err(TaskParseError::NotJsonContent { got: "text_json" })
            }
        };

        let task_id = item
            .meta
            .get(META_TASK_ID)
            .and_then(Value::as_str)
            .ok_or(TaskParseError::MissingField("meta.task_id"))?
            .to_string();

        let step = json
            .get("step")
            .and_then(Value::as_i64)
            .ok_or(TaskParseError::MissingField("step"))?;

        let scratchpad = json
            .get("scratchpad")
            .cloned()
            .ok_or(TaskParseError::MissingField("scratchpad"))?;

        Ok(Self {
            id: item.id.clone(),
            scope: item.scope.clone(),
            task_id,
            step,
            scratchpad,
            importance: item.importance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scope() -> ScopeKey {
        ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: Some("agent-1".to_string()),
            run_id: Some("run-1".to_string()),
        }
    }

    #[test]
    fn task_status_round_trips_snake_case() {
        let s = serde_json::to_string(&TaskStatus::InProgress).unwrap();
        assert_eq!(s, "\"in_progress\"");
        let back: TaskStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(back, TaskStatus::InProgress);
    }

    #[test]
    fn task_record_round_trips_through_memory_item() {
        let record = TaskRecord::new(
            MemoryId("task-1".into()),
            test_scope(),
            TaskStatus::InProgress,
            "Compile the report".into(),
        )
        .with_depends_on(vec!["task-0".into()])
        .with_importance(0.9);

        let item = record.clone().into_memory_item("agent".into());
        assert_eq!(item.kind, MemoryKind::Task);
        assert_eq!(item.importance, 0.9);

        let parsed = TaskRecord::try_from_memory_item(&item).unwrap();
        assert_eq!(parsed.id, record.id);
        assert_eq!(parsed.status, TaskStatus::InProgress);
        assert_eq!(parsed.description, "Compile the report");
        assert_eq!(parsed.depends_on, vec!["task-0"]);
        assert_eq!(parsed.importance, 0.9);
    }

    #[test]
    fn task_record_rejects_non_task_items() {
        let item = MemoryItem::new(
            MemoryId("e".into()),
            test_scope(),
            MemoryKind::Event,
            Content::Text("hi".into()),
            "user".into(),
        );
        let err = TaskRecord::try_from_memory_item(&item).unwrap_err();
        assert!(matches!(err, TaskParseError::WrongKind { .. }));
    }

    #[test]
    fn task_record_rejects_text_content() {
        let mut item = MemoryItem::new(
            MemoryId("t".into()),
            test_scope(),
            MemoryKind::Task,
            Content::Text("not json".into()),
            "agent".into(),
        );
        item.kind = MemoryKind::Task;
        let err = TaskRecord::try_from_memory_item(&item).unwrap_err();
        assert!(matches!(err, TaskParseError::NotJsonContent { .. }));
    }

    #[test]
    fn checkpoint_record_round_trips_through_memory_item() {
        let record = CheckpointRecord::new(
            MemoryId("ckpt-1".into()),
            test_scope(),
            "task-1".into(),
            4,
            serde_json::json!({"scratch": {"k": 1}}),
        )
        .with_importance(0.8);

        let item = record.clone().into_memory_item("agent".into());
        assert_eq!(item.kind, MemoryKind::Checkpoint);
        assert_eq!(item.importance, 0.8);
        assert!(item.tags.contains(&task_tag("task-1")));
        assert_eq!(
            item.meta.get(META_TASK_ID).and_then(Value::as_str),
            Some("task-1")
        );

        let parsed = CheckpointRecord::try_from_memory_item(&item).unwrap();
        assert_eq!(parsed.task_id, "task-1");
        assert_eq!(parsed.step, 4);
        assert_eq!(parsed.scratchpad["scratch"]["k"], 1);
    }

    #[test]
    fn checkpoint_record_missing_task_id_meta_fails() {
        let mut item = MemoryItem::new(
            MemoryId("c".into()),
            test_scope(),
            MemoryKind::Checkpoint,
            Content::Json(serde_json::json!({"step": 1, "scratchpad": {}})),
            "agent".into(),
        );
        item.meta = BTreeMap::new();
        let err = CheckpointRecord::try_from_memory_item(&item).unwrap_err();
        assert_eq!(err, TaskParseError::MissingField("meta.task_id"));
    }

    #[test]
    fn checkpoint_record_missing_step_fails() {
        let mut item = MemoryItem::new(
            MemoryId("c".into()),
            test_scope(),
            MemoryKind::Checkpoint,
            Content::Json(serde_json::json!({"scratchpad": {}})),
            "agent".into(),
        );
        item.meta
            .insert(META_TASK_ID.into(), Value::String("task-1".into()));
        let err = CheckpointRecord::try_from_memory_item(&item).unwrap_err();
        assert_eq!(err, TaskParseError::MissingField("step"));
    }

    #[test]
    fn task_tag_helper_is_consistent() {
        assert_eq!(task_tag("task-42"), "task:task-42");
    }
}
