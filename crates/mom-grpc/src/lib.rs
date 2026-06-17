use mom_core::{
    Content as CoreContent, MemoryId as CoreMemoryId, MemoryItem as CoreMemoryItem,
    MemoryKind as CoreMemoryKind, MemoryStore, Query as CoreQuery, ScopeKey as CoreScopeKey,
};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

// Import the generated gRPC types
pub mod proto {
    tonic::include_proto!("memory");

    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("memory_descriptor");
}

impl From<proto::ScopeKey> for CoreScopeKey {
    fn from(s: proto::ScopeKey) -> Self {
        Self {
            tenant_id: s.tenant_id,
            workspace_id: s.workspace_id,
            project_id: s.project_id,
            agent_id: s.agent_id,
            run_id: s.run_id,
        }
    }
}

impl From<CoreScopeKey> for proto::ScopeKey {
    fn from(s: CoreScopeKey) -> Self {
        Self {
            tenant_id: s.tenant_id,
            workspace_id: s.workspace_id,
            project_id: s.project_id,
            agent_id: s.agent_id,
            run_id: s.run_id,
        }
    }
}

impl From<proto::MemoryKind> for CoreMemoryKind {
    fn from(k: proto::MemoryKind) -> Self {
        match k {
            proto::MemoryKind::Event => CoreMemoryKind::Event,
            proto::MemoryKind::Summary => CoreMemoryKind::Summary,
            proto::MemoryKind::Fact => CoreMemoryKind::Fact,
            proto::MemoryKind::Preference => CoreMemoryKind::Preference,
            proto::MemoryKind::Task => CoreMemoryKind::Task,
            proto::MemoryKind::Checkpoint => CoreMemoryKind::Checkpoint,
            proto::MemoryKind::Unspecified => CoreMemoryKind::Event,
        }
    }
}

impl From<CoreMemoryKind> for proto::MemoryKind {
    fn from(k: CoreMemoryKind) -> Self {
        match k {
            CoreMemoryKind::Event => proto::MemoryKind::Event,
            CoreMemoryKind::Summary => proto::MemoryKind::Summary,
            CoreMemoryKind::Fact => proto::MemoryKind::Fact,
            CoreMemoryKind::Preference => proto::MemoryKind::Preference,
            CoreMemoryKind::Task => proto::MemoryKind::Task,
            CoreMemoryKind::Checkpoint => proto::MemoryKind::Checkpoint,
        }
    }
}

impl TryFrom<proto::Content> for CoreContent {
    type Error = Status;

    fn try_from(c: proto::Content) -> Result<Self, Self::Error> {
        let content_type = c
            .content_type
            .ok_or_else(|| Status::invalid_argument("Content type must be specified"))?;

        match content_type {
            proto::content::ContentType::Text(text) => Ok(CoreContent::Text(text)),
            proto::content::ContentType::Json(json_str) => {
                let val = serde_json::from_str(&json_str).map_err(|e| {
                    Status::invalid_argument(format!("Invalid JSON content: {}", e))
                })?;
                Ok(CoreContent::Json(val))
            }
            proto::content::ContentType::TextJson(tj) => {
                let val = serde_json::from_str(&tj.json).map_err(|e| {
                    Status::invalid_argument(format!("Invalid JSON content: {}", e))
                })?;
                Ok(CoreContent::TextJson {
                    text: tj.text,
                    json: val,
                })
            }
        }
    }
}

impl From<CoreContent> for proto::Content {
    fn from(c: CoreContent) -> Self {
        match c {
            CoreContent::Text(text) => proto::Content {
                content_type: Some(proto::content::ContentType::Text(text)),
            },
            CoreContent::Json(val) => proto::Content {
                content_type: Some(proto::content::ContentType::Json(val.to_string())),
            },
            CoreContent::TextJson { text, json } => proto::Content {
                content_type: Some(proto::content::ContentType::TextJson(proto::TextJson {
                    text,
                    json: json.to_string(),
                })),
            },
        }
    }
}

impl TryFrom<proto::MemoryItem> for CoreMemoryItem {
    type Error = Status;

    fn try_from(item: proto::MemoryItem) -> Result<Self, Self::Error> {
        let scope = item
            .scope
            .ok_or_else(|| Status::invalid_argument("Scope key must be specified"))?;
        let content = item
            .content
            .ok_or_else(|| Status::invalid_argument("Content must be specified"))?;

        let mut meta = std::collections::BTreeMap::new();
        for (k, v) in item.meta {
            let val = serde_json::from_str(&v).map_err(|e| {
                Status::invalid_argument(format!("Invalid JSON in meta field '{}': {}", k, e))
            })?;
            meta.insert(k, val);
        }

        let core_kind = proto::MemoryKind::try_from(item.kind)
            .ok()
            .map(CoreMemoryKind::from)
            .unwrap_or(CoreMemoryKind::Event);

        Ok(CoreMemoryItem {
            id: CoreMemoryId(item.id),
            scope: scope.into(),
            kind: core_kind,
            created_at_ms: item.created_at_ms,
            content: content.try_into()?,
            tags: item.tags,
            importance: item.importance,
            confidence: item.confidence,
            source: item.source,
            ttl_ms: item.ttl_ms,
            meta,
            embedding: if item.embedding.is_empty() {
                None
            } else {
                Some(item.embedding)
            },
            embedding_model: item.embedding_model,
        })
    }
}

impl From<CoreMemoryItem> for proto::MemoryItem {
    fn from(item: CoreMemoryItem) -> Self {
        let mut meta = std::collections::HashMap::new();
        for (k, v) in item.meta {
            meta.insert(k, v.to_string());
        }

        proto::MemoryItem {
            id: item.id.0,
            scope: Some(item.scope.into()),
            kind: proto::MemoryKind::from(item.kind) as i32,
            created_at_ms: item.created_at_ms,
            content: Some(proto::Content::from(item.content)),
            tags: item.tags,
            importance: item.importance,
            confidence: item.confidence,
            source: item.source,
            ttl_ms: item.ttl_ms,
            meta,
            embedding: item.embedding.unwrap_or_default(),
            embedding_model: item.embedding_model,
        }
    }
}

impl TryFrom<proto::QueryRequest> for CoreQuery {
    type Error = Status;

    fn try_from(q: proto::QueryRequest) -> Result<Self, Self::Error> {
        let scope = q
            .scope
            .ok_or_else(|| Status::invalid_argument("Scope key must be specified"))?;

        let kinds = if q.kinds.is_empty() {
            None
        } else {
            let mapped_kinds: Vec<CoreMemoryKind> = q
                .kinds
                .into_iter()
                .map(|k| {
                    proto::MemoryKind::try_from(k)
                        .ok()
                        .map(CoreMemoryKind::from)
                        .unwrap_or(CoreMemoryKind::Event)
                })
                .collect();
            Some(mapped_kinds)
        };

        let tags_any = if q.tags_any.is_empty() {
            None
        } else {
            Some(q.tags_any)
        };

        Ok(CoreQuery {
            scope: scope.into(),
            text: q.text,
            kinds,
            tags_any,
            limit: q.limit as usize,
            since_ms: q.since_ms,
            until_ms: q.until_ms,
            cursor: None,
        })
    }
}

pub struct MemoryStoreServiceGrpc {
    store: Arc<dyn MemoryStore>,
}

impl MemoryStoreServiceGrpc {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl proto::memory_store_service_server::MemoryStoreService for MemoryStoreServiceGrpc {
    async fn write(
        &self,
        request: Request<proto::MemoryItem>,
    ) -> Result<Response<proto::MemoryId>, Status> {
        let proto_item = request.into_inner();
        let core_item = CoreMemoryItem::try_from(proto_item)?;
        let id = proto::MemoryId {
            id: core_item.id.0.clone(),
            scope: Some(proto::ScopeKey::from(core_item.scope.clone())),
        };

        self.store
            .put(core_item)
            .await
            .map_err(|e| Status::internal(format!("Failed to write memory: {}", e)))?;

        Ok(Response::new(id))
    }

    async fn get(
        &self,
        request: Request<proto::MemoryId>,
    ) -> Result<Response<proto::MemoryItem>, Status> {
        let req = request.into_inner();
        let id = CoreMemoryId(req.id);

        let item = if let Some(scope) = req.scope {
            self.store.get_scoped(&id, &scope.into()).await
        } else {
            self.store.get(&id).await
        }
        .map_err(|e| Status::internal(format!("Failed to retrieve memory: {}", e)))?;

        match item {
            Some(core_item) => Ok(Response::new(proto::MemoryItem::from(core_item))),
            None => Err(Status::not_found("Memory item not found")),
        }
    }

    type QueryStream = ReceiverStream<Result<proto::ScoredMemoryItem, Status>>;

    async fn query(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<Self::QueryStream>, Status> {
        let proto_query = request.into_inner();
        let core_query = CoreQuery::try_from(proto_query)?;

        let results = self
            .store
            .query(core_query)
            .await
            .map_err(|e| Status::internal(format!("Failed to query memories: {}", e)))?;

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        tokio::spawn(async move {
            for res in results {
                let proto_scored = proto::ScoredMemoryItem {
                    score: res.score,
                    item: Some(proto::MemoryItem::from(res.item)),
                };
                if tx.send(Ok(proto_scored)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn delete(&self, request: Request<proto::MemoryId>) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let id = CoreMemoryId(req.id);

        if let Some(scope) = req.scope {
            self.store.delete_scoped(&id, &scope.into()).await
        } else {
            self.store.delete(&id).await
        }
        .map_err(|e| Status::internal(format!("Failed to delete memory: {}", e)))?;

        Ok(Response::new(()))
    }

    type RecallStream = ReceiverStream<Result<proto::ScoredMemoryItem, Status>>;

    async fn recall(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<Self::RecallStream>, Status> {
        let proto_query = request.into_inner();
        let core_query = CoreQuery::try_from(proto_query)?;

        let results = self
            .store
            .query(core_query)
            .await
            .map_err(|e| Status::internal(format!("Failed to recall memories: {}", e)))?;

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        tokio::spawn(async move {
            for res in results {
                let proto_scored = proto::ScoredMemoryItem {
                    score: res.score,
                    item: Some(proto::MemoryItem::from(res.item)),
                };
                if tx.send(Ok(proto_scored)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type BulkWriteStream = ReceiverStream<Result<proto::MemoryId, Status>>;

    async fn bulk_write(
        &self,
        request: Request<tonic::Streaming<proto::MemoryItem>>,
    ) -> Result<Response<Self::BulkWriteStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let store = self.store.clone();

        tokio::spawn(async move {
            while let Some(res) = stream.next().await {
                match res {
                    Ok(proto_item) => match CoreMemoryItem::try_from(proto_item) {
                        Ok(core_item) => {
                            let id = proto::MemoryId {
                                id: core_item.id.0.clone(),
                                scope: Some(proto::ScopeKey::from(core_item.scope.clone())),
                            };
                            if let Err(e) = store.put(core_item).await {
                                if tx
                                    .send(Err(Status::internal(format!(
                                        "Failed to write memory: {}",
                                        e
                                    ))))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            } else {
                                if tx.send(Ok(id)).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(status) => {
                            if tx.send(Err(status)).await.is_err() {
                                break;
                            }
                        }
                    },
                    Err(err) => {
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {}", err))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type BulkGetStream = ReceiverStream<Result<proto::MemoryItem, Status>>;

    async fn bulk_get(
        &self,
        request: Request<tonic::Streaming<proto::MemoryId>>,
    ) -> Result<Response<Self::BulkGetStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let store = self.store.clone();

        tokio::spawn(async move {
            while let Some(res) = stream.next().await {
                match res {
                    Ok(req) => {
                        let id = CoreMemoryId(req.id);
                        let get_res = if let Some(scope) = req.scope {
                            store.get_scoped(&id, &scope.into()).await
                        } else {
                            store.get(&id).await
                        };

                        match get_res {
                            Ok(Some(core_item)) => {
                                if tx
                                    .send(Ok(proto::MemoryItem::from(core_item)))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Ok(None) => {
                                if tx
                                    .send(Err(Status::not_found(format!(
                                        "Memory item not found: {}",
                                        id.0
                                    ))))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                if tx
                                    .send(Err(Status::internal(format!(
                                        "Failed to retrieve memory: {}",
                                        e
                                    ))))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {}", err))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

pub async fn start_grpc_server(
    store: Arc<dyn MemoryStore>,
    addr: std::net::SocketAddr,
) -> Result<(), anyhow::Error> {
    let service = MemoryStoreServiceGrpc::new(store);

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;

    tracing::info!("Starting gRPC server on {}", addr);

    tonic::transport::Server::builder()
        .accept_http1(true) // Support gRPC-Web / HTTP1
        .layer(tower_http::trace::TraceLayer::new_for_grpc())
        .add_service(proto::memory_store_service_server::MemoryStoreServiceServer::new(service))
        .add_service(reflection_service)
        .serve(addr)
        .await?;

    Ok(())
}
