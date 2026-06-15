use mom_grpc::proto::memory_store_service_client::MemoryStoreServiceClient;
use mom_grpc::proto::{content::ContentType, Content, MemoryItem, MemoryKind, ScopeKey};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grpc_addr = "http://127.0.0.1:50051";
    let http_addr = "http://127.0.0.1:8080/v1/memory";

    println!("Connecting to MOM gRPC server at {}...", grpc_addr);
    let mut grpc_client = match MemoryStoreServiceClient::connect(grpc_addr).await {
        Ok(c) => c,
        Err(e) => {
            println!(
                "Could not connect to gRPC server: {}. Make sure the MOM service is running.",
                e
            );
            return Ok(());
        }
    };

    let http_client = reqwest::Client::new();
    // Verify HTTP is running
    if let Err(e) = http_client
        .get("http://127.0.0.1:8080/healthz")
        .send()
        .await
    {
        println!(
            "Could not connect to HTTP server: {}. Make sure the MOM service is running.",
            e
        );
        return Ok(());
    }

    println!("Starting Benchmark: 100 Write + Get operations...");
    let iterations = 100;

    // ------------------ gRPC Write Benchmark ------------------
    let scope = ScopeKey {
        tenant_id: "bench-tenant".to_string(),
        workspace_id: Some("workspace-bench".to_string()),
        project_id: None,
        agent_id: Some("agent-bench".to_string()),
        run_id: None,
    };

    let mut grpc_latencies = Vec::new();
    let start_grpc = Instant::now();
    for i in 0..iterations {
        let item_id = format!("grpc-bench-{}", i);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let item = MemoryItem {
            id: item_id,
            scope: Some(scope.clone()),
            kind: MemoryKind::Fact as i32,
            created_at_ms: now_ms,
            content: Some(Content {
                content_type: Some(ContentType::Text(format!("Fact number {}", i))),
            }),
            tags: vec!["benchmark".to_string()],
            importance: 0.5,
            confidence: 0.9,
            source: "bench-agent".to_string(),
            ttl_ms: None,
            meta: std::collections::HashMap::new(),
            embedding: vec![],
            embedding_model: None,
        };

        let start_req = Instant::now();
        grpc_client.write(item).await?;
        grpc_latencies.push(start_req.elapsed());
    }
    let total_grpc_write = start_grpc.elapsed();

    // ------------------ HTTP Write Benchmark ------------------
    let mut http_latencies = Vec::new();
    let start_http = Instant::now();
    for i in 0..iterations {
        let item_id = format!("http-bench-{}", i);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let payload = serde_json::json!({
            "id": item_id,
            "scope": {
                "tenant_id": "bench-tenant",
                "workspace_id": "workspace-bench",
                "project_id": null,
                "agent_id": "agent-bench",
                "run_id": null
            },
            "kind": "fact",
            "created_at_ms": now_ms,
            "content": {
                "text": format!("Fact number {}", i)
            },
            "tags": ["benchmark"],
            "importance": 0.5,
            "confidence": 0.9,
            "source": "bench-agent",
            "ttl_ms": null,
            "meta": {},
            "embedding": null,
            "embedding_model": null
        });

        let start_req = Instant::now();
        let res = http_client.post(http_addr).json(&payload).send().await?;
        if !res.status().is_success() {
            println!("HTTP Write failed: {:?}", res.status());
        }
        http_latencies.push(start_req.elapsed());
    }
    let total_http_write = start_http.elapsed();

    // Calculate latency metrics
    grpc_latencies.sort();
    http_latencies.sort();

    let grpc_avg_ms = total_grpc_write.as_secs_f64() * 1000.0 / iterations as f64;
    let http_avg_ms = total_http_write.as_secs_f64() * 1000.0 / iterations as f64;

    let grpc_p95 = grpc_latencies[(iterations * 95 / 100) - 1].as_secs_f64() * 1000.0;
    let http_p95 = http_latencies[(iterations * 95 / 100) - 1].as_secs_f64() * 1000.0;

    let grpc_tps = iterations as f64 / total_grpc_write.as_secs_f64();
    let http_tps = iterations as f64 / total_http_write.as_secs_f64();

    println!(
        "\n=== Write Performance Comparison ({} runs) ===",
        iterations
    );
    println!("Protocol | Avg Latency | p95 Latency | Throughput (req/sec)");
    println!("---------|-------------|-------------|---------------------");
    println!(
        "gRPC     | {:>8.2}ms | {:>8.2}ms | {:>14.1} rps",
        grpc_avg_ms, grpc_p95, grpc_tps
    );
    println!(
        "HTTP     | {:>8.2}ms | {:>8.2}ms | {:>14.1} rps",
        http_avg_ms, http_p95, http_tps
    );
    println!("===================================================\n");

    Ok(())
}
