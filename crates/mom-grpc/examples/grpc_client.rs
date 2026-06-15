use mom_grpc::proto::memory_store_service_client::MemoryStoreServiceClient;
use mom_grpc::proto::{
    content::ContentType, Content, MemoryItem, MemoryKind, QueryRequest, ScopeKey,
};
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "http://127.0.0.1:50051";
    println!("Connecting to MOM gRPC server at {}...", addr);
    let mut client = MemoryStoreServiceClient::connect(addr).await?;
    println!("✅ Connected to MOM gRPC server!");

    // Create a shared scope
    let scope = ScopeKey {
        tenant_id: "example-tenant".to_string(),
        workspace_id: Some("workspace-1".to_string()),
        project_id: None,
        agent_id: Some("agent-1".to_string()),
        run_id: None,
    };

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    // 1. Write memory
    let item = MemoryItem {
        id: "doc-123".to_string(),
        scope: Some(scope.clone()),
        kind: MemoryKind::Fact as i32,
        created_at_ms: now_ms,
        content: Some(Content {
            content_type: Some(ContentType::Text(
                "Project Conductor is launching on Friday.".to_string(),
            )),
        }),
        tags: vec!["conductor".to_string(), "milestone".to_string()],
        importance: 0.9,
        confidence: 0.99,
        source: "agent-1".to_string(),
        ttl_ms: None,
        meta: std::collections::HashMap::new(),
        embedding: vec![],
        embedding_model: None,
    };

    println!("Writing memory item doc-123...");
    let response = client.write(item).await?;
    println!("✅ Write response: {:?}", response.into_inner());

    // 2. Get memory
    let get_req = mom_grpc::proto::MemoryId {
        id: "doc-123".to_string(),
        scope: Some(scope.clone()),
    };
    println!("Retrieving memory item doc-123...");
    let response = client.get(get_req).await?;
    println!("✅ Get response: {:?}", response.into_inner());

    // 3. Query memories
    let query_req = QueryRequest {
        scope: Some(scope.clone()),
        text: "".to_string(),
        kinds: vec![MemoryKind::Fact as i32],
        tags_any: vec!["conductor".to_string()],
        limit: 5,
        since_ms: None,
        until_ms: None,
    };
    println!("Querying memories with tag 'conductor'...");
    let mut stream = client.query(query_req).await?.into_inner();
    while let Some(scored_item) = stream.message().await? {
        println!("✅ Query item: {:?}", scored_item);
    }

    // 4. Delete memory
    let del_req = mom_grpc::proto::MemoryId {
        id: "doc-123".to_string(),
        scope: Some(scope.clone()),
    };
    println!("Deleting memory item doc-123...");
    client.delete(del_req).await?;
    println!("✅ Deleted memory doc-123 successfully");

    Ok(())
}
