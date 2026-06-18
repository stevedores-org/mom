use mom_core::{Content, MemoryId, MemoryItem, MemoryKind, MemoryStore, ScopeKey};
use mom_store_surrealdb::SurrealDBStore;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting 10K concurrent writes load test...");

    let store = Arc::new(SurrealDBStore::new("mem://test").await?);
    let num_writes = 10000;

    let start_all = Instant::now();

    let mut join_set = JoinSet::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<Duration>();

    for i in 0..num_writes {
        let store = Arc::clone(&store);
        let tx = tx.clone();

        join_set.spawn(async move {
            let item = MemoryItem {
                id: MemoryId(format!("load-{}", i)),
                scope: ScopeKey {
                    tenant_id: "load-tenant".to_string(),
                    workspace_id: None,
                    project_id: None,
                    agent_id: None,
                    run_id: None,
                },
                kind: MemoryKind::Event,
                created_at_ms: chrono::Utc::now().timestamp_millis(),
                content: Content::Text(format!("load test write number {}", i)),
                tags: vec![],
                importance: 0.5,
                confidence: 0.5,
                source: "load-tester".to_string(),
                ttl_ms: None,
                meta: Default::default(),
                embedding: None,
                embedding_model: None,
            };

            let start_write = Instant::now();
            store.put(item).await.unwrap();
            let latency = start_write.elapsed();

            let _ = tx.send(latency);
        });
    }

    // Drop the original sender so the receiver closes when all spawns are done
    drop(tx);

    // Collect all latencies
    let mut latencies = Vec::with_capacity(num_writes);
    while let Some(latency) = rx.recv().await {
        latencies.push(latency);
    }

    // Wait for all tokio tasks to finish joining
    while join_set.join_next().await.is_some() {}

    let total_duration = start_all.elapsed();

    // Verify all 10,000 writes are indeed stored
    println!("Verifying store integrity...");
    let stored_sample = store.get(&MemoryId("load-9999".to_string())).await?;
    assert!(
        stored_sample.is_some(),
        "Load test data was not successfully written!"
    );

    // Sort latencies to compute percentiles
    latencies.sort();

    let p50 = latencies[num_writes / 2];
    let p90 = latencies[(num_writes as f32 * 0.90) as usize];
    let p99 = latencies[(num_writes as f32 * 0.99) as usize];
    let min = latencies[0];
    let max = latencies[num_writes - 1];

    let avg: Duration = latencies.iter().sum::<Duration>() / num_writes as u32;

    let throughput = num_writes as f64 / total_duration.as_secs_f64();

    println!("\n📊 --- Load Test Results ---");
    println!("Total writes      : {}", num_writes);
    println!("Total duration    : {:?}", total_duration);
    println!("Throughput        : {:.2} writes/sec", throughput);
    println!("Min latency       : {:?}", min);
    println!("Average latency   : {:?}", avg);
    println!("p50 (median)      : {:?}", p50);
    println!("p90               : {:?}", p90);
    println!("p99               : {:?}", p99);
    println!("Max latency       : {:?}", max);
    println!("-----------------------------\n");

    Ok(())
}
