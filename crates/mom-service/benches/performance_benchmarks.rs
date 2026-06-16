use criterion::{criterion_group, criterion_main, Criterion};
use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, MemoryStore, Query, ScopeKey};
use mom_store_surrealdb::SurrealDBStore;
use tokio::runtime::Runtime;

fn bench_store_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = rt.block_on(async { SurrealDBStore::new("mem://test").await.unwrap() });

    // 1. Bench per-item index
    c.bench_function("per_item_index", |b| {
        let mut counter = 0;
        b.to_async(&rt).iter(|| {
            counter += 1;
            let item = MemoryItem {
                id: MemoryId(format!("index-{}", counter)),
                scope: ScopeKey {
                    tenant_id: "acme".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: None,
                    run_id: None,
                },
                kind: MemoryKind::Event,
                created_at_ms: 1234567,
                content: Content::Text("indexing test".to_string()),
                tags: vec![],
                importance: 0.5,
                confidence: 0.5,
                source: "user".to_string(),
                ttl_ms: None,
                meta: Default::default(),
                embedding: None,
                embedding_model: None,
            };
            let store = &store;
            async move {
                store.put(item).await.unwrap();
            }
        });
    });

    // 2. Bench write 1000 items
    c.bench_function("write_1000_items", |b| {
        let items: Vec<MemoryItem> = (0..1000)
            .map(|i| MemoryItem {
                id: MemoryId(format!("batch-{}", i)),
                scope: ScopeKey {
                    tenant_id: "acme".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: None,
                    run_id: None,
                },
                kind: MemoryKind::Event,
                created_at_ms: 1234567,
                content: Content::Text(format!("batch item {}", i)),
                tags: vec![],
                importance: 0.5,
                confidence: 0.5,
                source: "user".to_string(),
                ttl_ms: None,
                meta: Default::default(),
                embedding: None,
                embedding_model: None,
            })
            .collect();

        b.to_async(&rt).iter_with_large_drop(|| {
            let store = &store;
            let items = items.clone();
            async move {
                store.write_batch(items).await.unwrap();
            }
        });
    });

    // Populate the store with 10,000 items for query and vector search benchmarks
    rt.block_on(async {
        for batch_idx in 0..10 {
            let items: Vec<MemoryItem> = (0..1000)
                .map(|i| {
                    let idx = batch_idx * 1000 + i;
                    MemoryItem {
                        id: MemoryId(format!("query-{}", idx)),
                        scope: ScopeKey {
                            tenant_id: "acme".to_string(),
                            workspace_id: None,
                            project_id: None,
                            agent_id: None,
                            run_id: None,
                        },
                        kind: MemoryKind::Event,
                        created_at_ms: idx as i64,
                        content: Content::Text(format!("query item {}", idx)),
                        tags: vec![format!("tag-{}", idx % 10)],
                        importance: 0.5,
                        confidence: 0.5,
                        source: "user".to_string(),
                        ttl_ms: None,
                        meta: Default::default(),
                        embedding: if idx % 10 == 0 {
                            Some(vec![0.1; 1536])
                        } else {
                            None
                        },
                        embedding_model: if idx % 10 == 0 {
                            Some("stub-model".to_string())
                        } else {
                            None
                        },
                    }
                })
                .collect();
            store.write_batch(items).await.unwrap();
        }
    });

    // 3. Bench query 10K
    c.bench_function("query_10k_items", |b| {
        let query = Query {
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            text: "item".to_string(),
            kinds: None,
            tags_any: None,
            limit: 10,
            since_ms: None,
            until_ms: None,
        };

        b.to_async(&rt).iter(|| {
            let store = &store;
            let query = query.clone();
            async move {
                let results = store.query(query).await.unwrap();
                assert!(!results.is_empty());
            }
        });
    });

    // 4. Bench vector search
    c.bench_function("vector_search_similarity", |b| {
        let query_embedding = vec![0.1; 1536];
        let scope = ScopeKey {
            tenant_id: "acme".to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: None,
            run_id: None,
        };

        b.to_async(&rt).iter(|| {
            let store = &store;
            let query_embedding = query_embedding.clone();
            let scope = scope.clone();
            async move {
                let results = store
                    .vector_recall(&query_embedding, &scope, 10)
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            }
        });
    });
}

criterion_group!(benches, bench_store_operations);
criterion_main!(benches);
