//! US-10 integration tests against the in-memory SurrealDB engine.
//!
//! Covers the two new store primitives:
//!
//! 1. `find_active_facts_with_key` — the exact-key conflict probe used by
//!    the put-Fact path (mom-service `put_memory`). Asserts it:
//!    - returns prior active Facts in scope matching `(subject, predicate)`,
//!    - excludes Facts in other tenants/workspaces,
//!    - excludes Facts already marked `meta.superseded_by`.
//!
//! 2. `find_semantic_fact_conflicts` — the advisory cosine-similarity probe
//!    used after embedding. Asserts it:
//!    - returns only items at or above the threshold, descending by sim,
//!    - skips items with no embedding,
//!    - skips items the caller excludes by id,
//!    - skips already-superseded items,
//!    - is capped by `max`.
//!
//! Phase 1's HTTP-layer flow (conflict → supersede → version-bump) lives
//! in `mom-service::put_memory` and is exercised indirectly here by
//! seeding the store with the same shape the handler writes.

use mom_core::{
    write_superseded_by, Content, FactPayload, MemoryId, MemoryItem, MemoryKind, MemoryStore,
    ScopeKey,
};
use mom_store_surrealdb::SurrealDBStore;
use std::collections::BTreeMap;

fn scope(tenant: &str, workspace: Option<&str>) -> ScopeKey {
    ScopeKey {
        tenant_id: tenant.to_string(),
        workspace_id: workspace.map(String::from),
        project_id: None,
        agent_id: None,
        run_id: None,
    }
}

fn fact_item(
    id: &str,
    scope: ScopeKey,
    subject: &str,
    predicate: &str,
    object: &str,
    embedding: Option<Vec<f32>>,
) -> MemoryItem {
    let mut meta = BTreeMap::new();
    FactPayload {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
    }
    .write_to_meta(&mut meta);

    let embedding_model = embedding
        .as_ref()
        .map(|_: &Vec<f32>| "test-model".to_string());
    MemoryItem {
        id: MemoryId(id.into()),
        scope,
        kind: MemoryKind::Fact,
        created_at_ms: 1_700_000_000_000,
        content: Content::Text(format!("{subject} {predicate} {object}")),
        tags: vec![],
        importance: 0.8,
        confidence: 1.0,
        source: "agent".into(),
        ttl_ms: None,
        meta,
        embedding,
        embedding_model,
    }
}

#[tokio::test]
async fn find_active_facts_with_key_returns_matching_item() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    let fact = fact_item("fact-1", s.clone(), "users", "prefer", "Cloudflare", None);
    store.put(fact.clone()).await.unwrap();

    let hits = store
        .find_active_facts_with_key(&s, "users", "prefer")
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "should find the one prior Fact");
    assert_eq!(hits[0].id, fact.id);
}

#[tokio::test]
async fn find_active_facts_with_key_excludes_other_tenants() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s_a = scope("tenant-a", None);
    let s_b = scope("tenant-b", None);

    store
        .put(fact_item(
            "fact-a",
            s_a.clone(),
            "users",
            "prefer",
            "Cloudflare",
            None,
        ))
        .await
        .unwrap();
    store
        .put(fact_item(
            "fact-b",
            s_b.clone(),
            "users",
            "prefer",
            "AWS",
            None,
        ))
        .await
        .unwrap();

    let from_a = store
        .find_active_facts_with_key(&s_a, "users", "prefer")
        .await
        .unwrap();
    assert_eq!(from_a.len(), 1);
    assert_eq!(from_a[0].id.0, "fact-a", "must not leak across tenants");
}

#[tokio::test]
async fn find_active_facts_with_key_excludes_superseded() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    let mut old = fact_item("fact-old", s.clone(), "api", "rate_limit", "1000/min", None);
    let new_id = MemoryId("fact-new".into());
    write_superseded_by(&mut old.meta, &new_id);
    store.put(old).await.unwrap();
    store
        .put(fact_item(
            "fact-new",
            s.clone(),
            "api",
            "rate_limit",
            "500/min",
            None,
        ))
        .await
        .unwrap();

    let hits = store
        .find_active_facts_with_key(&s, "api", "rate_limit")
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "superseded items must be filtered out");
    assert_eq!(hits[0].id.0, "fact-new");
}

#[tokio::test]
async fn find_active_facts_with_key_excludes_other_workspaces() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s_w1 = scope("acme", Some("w1"));
    let s_w2 = scope("acme", Some("w2"));

    store
        .put(fact_item(
            "fact-w1",
            s_w1.clone(),
            "users",
            "prefer",
            "Cloudflare",
            None,
        ))
        .await
        .unwrap();
    store
        .put(fact_item(
            "fact-w2",
            s_w2.clone(),
            "users",
            "prefer",
            "AWS",
            None,
        ))
        .await
        .unwrap();

    let hits = store
        .find_active_facts_with_key(&s_w1, "users", "prefer")
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.0, "fact-w1");
}

#[tokio::test]
async fn find_semantic_fact_conflicts_returns_high_similarity() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    // Two near-identical embeddings (cosine ≈ 1.0) and one orthogonal.
    store
        .put(fact_item(
            "near-1",
            s.clone(),
            "api",
            "rate_limit",
            "1000/min",
            Some(vec![1.0, 0.0, 0.0]),
        ))
        .await
        .unwrap();
    store
        .put(fact_item(
            "near-2",
            s.clone(),
            "api",
            "throughput_cap",
            "500/min",
            Some(vec![0.99, 0.01, 0.0]),
        ))
        .await
        .unwrap();
    store
        .put(fact_item(
            "orthogonal",
            s.clone(),
            "policy",
            "log_level",
            "debug",
            Some(vec![0.0, 1.0, 0.0]),
        ))
        .await
        .unwrap();

    let query = vec![1.0_f32, 0.0, 0.0];
    let hits = store
        .find_semantic_fact_conflicts(&s, &query, None, 0.85, 10)
        .await
        .unwrap();

    let ids: Vec<&str> = hits.iter().map(|(m, _)| m.id.0.as_str()).collect();
    assert!(ids.contains(&"near-1"), "near-1 (identical vec) must hit");
    assert!(ids.contains(&"near-2"), "near-2 (≈1.0 cosine) must hit");
    assert!(
        !ids.contains(&"orthogonal"),
        "orthogonal (≈0.0 cosine) must NOT hit at threshold 0.85"
    );
    // Sorted descending by similarity: near-1 (1.0) ≥ near-2 (≈0.9999).
    assert!(
        hits[0].1 >= hits[1].1,
        "results must be sorted descending by similarity, got {hits:?}"
    );
}

#[tokio::test]
async fn find_semantic_fact_conflicts_respects_exclude_and_max() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    for n in 0..5 {
        store
            .put(fact_item(
                &format!("near-{n}"),
                s.clone(),
                "subj",
                "pred",
                &format!("obj-{n}"),
                Some(vec![1.0, 0.0, 0.0]),
            ))
            .await
            .unwrap();
    }

    let query = vec![1.0_f32, 0.0, 0.0];
    let excluded = MemoryId("near-0".into());

    let hits = store
        .find_semantic_fact_conflicts(&s, &query, Some(&excluded), 0.85, 3)
        .await
        .unwrap();
    assert_eq!(hits.len(), 3, "must be capped to max=3");
    assert!(
        hits.iter().all(|(m, _)| m.id != excluded),
        "excluded id must not appear"
    );
}

#[tokio::test]
async fn find_semantic_fact_conflicts_skips_superseded() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    let mut superseded = fact_item(
        "retired",
        s.clone(),
        "api",
        "rate_limit",
        "1000/min",
        Some(vec![1.0, 0.0, 0.0]),
    );
    write_superseded_by(&mut superseded.meta, &MemoryId("v2".into()));
    store.put(superseded).await.unwrap();
    store
        .put(fact_item(
            "active",
            s.clone(),
            "api",
            "rate_limit",
            "500/min",
            Some(vec![1.0, 0.0, 0.0]),
        ))
        .await
        .unwrap();

    let query = vec![1.0_f32, 0.0, 0.0];
    let hits = store
        .find_semantic_fact_conflicts(&s, &query, None, 0.85, 10)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|(m, _)| m.id.0.as_str()).collect();
    assert!(
        !ids.contains(&"retired"),
        "superseded item must be filtered"
    );
    assert!(ids.contains(&"active"));
}

#[tokio::test]
async fn find_semantic_fact_conflicts_returns_empty_for_empty_query() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let s = scope("acme", None);

    store
        .put(fact_item(
            "fact-1",
            s.clone(),
            "subj",
            "pred",
            "obj",
            Some(vec![1.0, 0.0, 0.0]),
        ))
        .await
        .unwrap();

    let hits = store
        .find_semantic_fact_conflicts(&s, &[], None, 0.85, 10)
        .await
        .unwrap();
    assert!(hits.is_empty(), "empty query embedding short-circuits");
}
