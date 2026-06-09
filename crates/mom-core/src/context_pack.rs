//! Context packs: structured memory bundles sized for agent token budgets.

use crate::{Content, MemoryItem, MemoryKind, Scored};
use serde::{Deserialize, Serialize};

/// Rough token estimate per memory item (issue #19 spec).
pub const TOKENS_PER_ITEM: usize = 150;

/// Default token budget when the client omits `budget_tokens`.
pub const DEFAULT_BUDGET_TOKENS: usize = 3000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackRequest {
    pub query: crate::Query,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub memory_id: String,
    pub source: String,
    pub kind: MemoryKind,
    pub created_at_ms: i64,
    pub score: f32,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub highlights: Vec<Scored<MemoryItem>>,
    pub summaries: Vec<Scored<MemoryItem>>,
    pub facts: Vec<Scored<MemoryItem>>,
    pub citations: Vec<Citation>,
    pub estimated_tokens: usize,
    pub budget_tokens: usize,
}

/// Extract a short text preview from memory content for citations.
pub fn content_preview(content: &Content, max_len: usize) -> String {
    let text = match content {
        Content::Text(t) => t.clone(),
        Content::TextJson { text, .. } => text.clone(),
        Content::Json(v) => v.to_string(),
    };
    if text.chars().count() <= max_len {
        text
    } else {
        text.chars().take(max_len).collect::<String>() + "…"
    }
}

/// Partition scored memories into a token-budgeted context pack.
pub fn build_context_pack(
    items: Vec<Scored<MemoryItem>>,
    budget_tokens: Option<usize>,
) -> ContextPack {
    let budget = budget_tokens
        .unwrap_or(DEFAULT_BUDGET_TOKENS)
        .max(TOKENS_PER_ITEM);
    let max_items = budget / TOKENS_PER_ITEM;

    let mut highlights = Vec::new();
    let mut summaries = Vec::new();
    let mut facts = Vec::new();
    let mut citations = Vec::new();
    let mut included = 0usize;

    for scored in items {
        if included >= max_items {
            break;
        }

        let bucket = match scored.item.kind {
            MemoryKind::Event | MemoryKind::Task => Some(&mut highlights),
            MemoryKind::Summary | MemoryKind::Checkpoint => Some(&mut summaries),
            MemoryKind::Fact | MemoryKind::Preference => Some(&mut facts),
        };

        if let Some(vec) = bucket {
            citations.push(Citation {
                memory_id: scored.item.id.0.clone(),
                source: scored.item.source.clone(),
                kind: scored.item.kind,
                created_at_ms: scored.item.created_at_ms,
                score: scored.score,
                preview: content_preview(&scored.item.content, 120),
            });
            vec.push(scored);
            included += 1;
        }
    }

    ContextPack {
        highlights,
        summaries,
        facts,
        citations,
        estimated_tokens: included * TOKENS_PER_ITEM,
        budget_tokens: budget,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryId, ScopeKey};
    use std::collections::BTreeMap;

    fn sample_item(kind: MemoryKind, id: &str, importance: f32) -> Scored<MemoryItem> {
        Scored {
            score: importance,
            item: MemoryItem {
                id: MemoryId(id.to_string()),
                scope: ScopeKey {
                    tenant_id: "acme".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: Some("agent-1".to_string()),
                    run_id: None,
                },
                kind,
                created_at_ms: 1_700_000_000_000,
                content: Content::Text(format!("content for {id}")),
                tags: vec![],
                importance,
                confidence: 0.9,
                source: "agent".to_string(),
                ttl_ms: None,
                meta: BTreeMap::new(),
                embedding: None,
                embedding_model: None,
            },
        }
    }

    #[test]
    fn partitions_by_kind() {
        let items = vec![
            sample_item(MemoryKind::Event, "e1", 0.9),
            sample_item(MemoryKind::Summary, "s1", 0.8),
            sample_item(MemoryKind::Fact, "f1", 0.95),
            sample_item(MemoryKind::Preference, "p1", 0.7),
        ];
        let pack = build_context_pack(items, Some(600));
        assert_eq!(pack.highlights.len(), 1);
        assert_eq!(pack.summaries.len(), 1);
        assert_eq!(pack.facts.len(), 2);
        assert_eq!(pack.citations.len(), 4);
        assert_eq!(pack.estimated_tokens, 4 * TOKENS_PER_ITEM);
    }

    #[test]
    fn respects_token_budget() {
        let items: Vec<_> = (0..10)
            .map(|i| sample_item(MemoryKind::Event, &format!("e{i}"), 0.5))
            .collect();
        let pack = build_context_pack(items, Some(300));
        assert_eq!(pack.highlights.len(), 2);
        assert_eq!(pack.estimated_tokens, 300);
    }

    #[test]
    fn content_preview_truncates() {
        let long = "a".repeat(200);
        let preview = content_preview(&Content::Text(long), 50);
        assert!(preview.chars().count() <= 51);
        assert!(preview.ends_with('…'));
    }
}
