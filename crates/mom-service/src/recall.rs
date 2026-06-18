//! Lexical recall ranking for `/v1/recall`.

use mom_core::{Content, MemoryItem, Query, Scored};

const RECENCY_DECAY_WINDOW_MS: i64 = 30 * 24 * 60 * 60 * 1000;
const TEXT_MATCH_WEIGHT: f32 = 0.60;
const IMPORTANCE_WEIGHT: f32 = 0.25;
const RECENCY_WEIGHT: f32 = 0.15;
const CANDIDATE_MULTIPLIER: usize = 5;

pub fn rank_recall_results(q: Query, results: Vec<Scored<MemoryItem>>) -> Vec<Scored<MemoryItem>> {
    if q.text.is_empty() {
        return results;
    }

    let original_limit = q.limit.max(1);
    let mut scored: Vec<Scored<MemoryItem>> = results
        .into_iter()
        .map(|scored_item| {
            let ranking_score = compute_ranking_score(&scored_item.item, &q.text);
            Scored {
                score: ranking_score,
                item: scored_item.item,
            }
        })
        .filter(|s| s.score > 0.0)
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(original_limit);
    scored
}

pub fn recall_candidate_limit(limit: usize) -> usize {
    limit.max(1).saturating_mul(CANDIDATE_MULTIPLIER).min(1000)
}

fn compute_text_match_score(item_content: &str, query_text: &str) -> f32 {
    if query_text.is_empty() || item_content.is_empty() {
        return 0.0;
    }

    let query_lower = query_text.to_lowercase();
    let content_lower = item_content.to_lowercase();

    if content_lower == query_lower {
        return 1.0;
    }

    if !content_lower.contains(&query_lower) {
        return 0.0;
    }

    let position = content_lower
        .find(&query_lower)
        .unwrap_or(content_lower.len());
    let distance_ratio = position as f32 / content_lower.len().max(1) as f32;
    let position_score = 1.0 - (distance_ratio * 0.5);
    (0.5 + position_score * 0.5).min(1.0)
}

fn compute_recency_score(created_at_ms: i64) -> f32 {
    let now = chrono::Utc::now().timestamp_millis();
    let age_ms = (now - created_at_ms).max(0);
    let decay = age_ms as f32 / RECENCY_DECAY_WINDOW_MS as f32;
    (1.0 - decay).max(0.0)
}

fn compute_ranking_score(item: &MemoryItem, query_text: &str) -> f32 {
    let text_match = compute_text_match_score(&item_to_text(item), query_text);
    if text_match == 0.0 {
        return 0.0;
    }

    let recency = compute_recency_score(item.created_at_ms);
    (text_match * TEXT_MATCH_WEIGHT)
        + (item.importance * IMPORTANCE_WEIGHT)
        + (recency * RECENCY_WEIGHT)
}

fn item_to_text(item: &MemoryItem) -> String {
    match &item.content {
        Content::Text(t) => t.clone(),
        Content::Json(v) => v.to_string(),
        Content::TextJson { text, json } => format!("{text} {}", json),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mom_core::{MemoryId, MemoryKind, ScopeKey};

    fn sample_item(text: &str, importance: f32, created_at_ms: i64) -> MemoryItem {
        MemoryItem {
            id: MemoryId("m1".into()),
            scope: ScopeKey {
                tenant_id: "tenant".into(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind: MemoryKind::Event,
            created_at_ms,
            content: Content::Text(text.into()),
            tags: vec![],
            importance,
            confidence: 1.0,
            source: "test".into(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        }
    }

    #[test]
    fn ranks_text_match_above_non_match() {
        let q = Query {
            scope: ScopeKey {
                tenant_id: "tenant".into(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            text: "kubernetes".into(),
            kinds: None,
            tags_any: None,
            limit: 1,
            since_ms: None,
            until_ms: None,
            cursor: None,
        };

        let results = vec![
            Scored {
                score: 0.1,
                item: sample_item("unrelated note", 0.9, 1),
            },
            Scored {
                score: 0.1,
                item: sample_item("kubernetes rollout failed", 0.5, 2),
            },
        ];

        let ranked = rank_recall_results(q, results);
        assert_eq!(ranked.len(), 1);
        assert!(matches!(
            &ranked[0].item.content,
            Content::Text(t) if t.contains("kubernetes")
        ));
    }
}
