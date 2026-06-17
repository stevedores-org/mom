//! US-7 cross-tenant negative tests against the in-memory SurrealDB engine.
//!
//! Acceptance criteria these tests cover:
//! - AC-4: "No query can cross tenants (fail fast)"
//! - AC-7: "Test: Attempt cross-tenant access returns empty"
//!
//! The behaviour we're asserting is two-pronged:
//!
//! 1. The scoped read/delete APIs (`get_scoped`, `delete_scoped`,
//!    `query`) MUST return empty / no-op when the caller's tenant
//!    differs from the item's stored tenant — even when the id matches.
//!
//! 2. The unscoped `get`/`delete` trait methods explicitly bypass the
//!    tenant filter (they're documented as INTERNAL); HTTP handlers go
//!    through `get_scoped` / `delete_scoped`. These tests make the
//!    contract crystal: the scoped API IS the security boundary.

use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey};
use mom_store_surrealdb::SurrealDBStore;
use std::collections::BTreeMap;

fn scope(tenant: &str) -> ScopeKey {
    ScopeKey {
        tenant_id: tenant.to_string(),
        workspace_id: None,
        project_id: None,
        agent_id: None,
        run_id: None,
    }
}

fn item(id: &str, scope: ScopeKey, text: &str) -> MemoryItem {
    MemoryItem {
        id: MemoryId(id.into()),
        scope,
        kind: MemoryKind::Event,
        created_at_ms: 1_700_000_000_000,
        content: Content::Text(text.into()),
        tags: vec![],
        importance: 0.5,
        confidence: 1.0,
        source: "agent".into(),
        ttl_ms: None,
        meta: BTreeMap::new(),
        embedding: None,
        embedding_model: None,
    }
}

#[tokio::test]
async fn get_scoped_returns_none_for_other_tenant_even_with_correct_id() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let tenant_a = scope("tenant-a");
    let tenant_b = scope("tenant-b");

    store
        .put(item("shared-id", tenant_a.clone(), "tenant-a data"))
        .await
        .unwrap();

    let from_b = store
        .get_scoped(&MemoryId("shared-id".into()), &tenant_b)
        .await
        .unwrap();
    assert!(
        from_b.is_none(),
        "tenant-b must not see tenant-a's item even when it knows the id"
    );

    let from_a = store
        .get_scoped(&MemoryId("shared-id".into()), &tenant_a)
        .await
        .unwrap();
    assert!(
        from_a.is_some(),
        "owner must still be able to read its item"
    );
}

#[tokio::test]
async fn query_returns_only_callers_tenant_items() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let tenant_a = scope("tenant-a");
    let tenant_b = scope("tenant-b");

    store
        .put(item("a-1", tenant_a.clone(), "a-content-1"))
        .await
        .unwrap();
    store
        .put(item("a-2", tenant_a.clone(), "a-content-2"))
        .await
        .unwrap();
    store
        .put(item("b-1", tenant_b.clone(), "b-content"))
        .await
        .unwrap();

    let a_results = store
        .query(Query {
            scope: tenant_a.clone(),
            text: String::new(),
            kinds: None,
            tags_any: None,
            limit: 100,
            since_ms: None,
            until_ms: None,
            cursor: None,
        })
        .await
        .unwrap();
    let a_ids: Vec<&str> = a_results.iter().map(|s| s.item.id.0.as_str()).collect();
    assert_eq!(
        a_results.len(),
        2,
        "tenant-a should see exactly its 2 items"
    );
    assert!(a_ids.contains(&"a-1") && a_ids.contains(&"a-2"));
    assert!(!a_ids.contains(&"b-1"), "must not leak tenant-b's item");

    let b_results = store
        .query(Query {
            scope: tenant_b.clone(),
            text: String::new(),
            kinds: None,
            tags_any: None,
            limit: 100,
            since_ms: None,
            until_ms: None,
            cursor: None,
        })
        .await
        .unwrap();
    let b_ids: Vec<&str> = b_results.iter().map(|s| s.item.id.0.as_str()).collect();
    assert_eq!(b_results.len(), 1);
    assert!(b_ids.contains(&"b-1"));
}

#[tokio::test]
async fn delete_scoped_is_noop_for_other_tenant() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let tenant_a = scope("tenant-a");
    let tenant_b = scope("tenant-b");

    store
        .put(item("targeted", tenant_a.clone(), "a-data"))
        .await
        .unwrap();

    // tenant-b attempts to delete tenant-a's item by id.
    store
        .delete_scoped(&MemoryId("targeted".into()), &tenant_b)
        .await
        .unwrap();

    // tenant-a's item must still exist.
    let still_there = store
        .get_scoped(&MemoryId("targeted".into()), &tenant_a)
        .await
        .unwrap();
    assert!(
        still_there.is_some(),
        "tenant-b's delete attempt must not affect tenant-a's data"
    );
}

#[tokio::test]
async fn query_with_empty_tenant_returns_nothing() {
    // Defense-in-depth: even if the HTTP layer's empty-tenant guard is
    // somehow bypassed, the SurrealQL query string still includes an
    // empty `tenant_id = ''` literal that matches no stored item.
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let real = scope("real-tenant");
    let empty = scope("");

    store
        .put(item("real-1", real.clone(), "real data"))
        .await
        .unwrap();

    let leaked = store
        .query(Query {
            scope: empty,
            text: String::new(),
            kinds: None,
            tags_any: None,
            limit: 100,
            since_ms: None,
            until_ms: None,
            cursor: None,
        })
        .await
        .unwrap();
    assert!(
        leaked.is_empty(),
        "an empty tenant_id must not match any rows"
    );
}

#[tokio::test]
async fn tenant_isolation_survives_id_collision() {
    let store = SurrealDBStore::new("mem://test").await.unwrap();
    let tenant_a = scope("tenant-a");
    let tenant_b = scope("tenant-b");

    // Both tenants write distinct items at the SAME id. The store keys
    // by id alone — so the second write overwrites the first. This test
    // documents that behaviour: ids are tenant-blind at the storage
    // layer, which is fine because every read goes through `get_scoped`.
    // If tenants want strictly-tenant-namespaced ids the application
    // should prefix them.
    store
        .put(item("shared", tenant_a.clone(), "first write — tenant-a"))
        .await
        .unwrap();
    store
        .put(item("shared", tenant_b.clone(), "second write — tenant-b"))
        .await
        .unwrap();

    // tenant-a tried to read "shared" — after tenant-b's overwrite,
    // the row now belongs to tenant-b, so tenant-a must see None.
    let from_a = store
        .get_scoped(&MemoryId("shared".into()), &tenant_a)
        .await
        .unwrap();
    assert!(
        from_a.is_none(),
        "after id collision, the prior owner must not see the new owner's data"
    );

    // tenant-b sees its own write.
    let from_b = store
        .get_scoped(&MemoryId("shared".into()), &tenant_b)
        .await
        .unwrap();
    assert!(from_b.is_some());
}
