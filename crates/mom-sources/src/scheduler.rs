//! Ingestion scheduler — polls registered sources and writes into MOM storage.

use crate::MemorySource;
use anyhow::{anyhow, Result};
use mom_core::{MemoryStore, ScopeKey};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct SourceStats {
    pub last_poll_at_ms: Option<i64>,
    pub last_success_count: usize,
    pub last_error: Option<String>,
    pub total_ingested: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IngestionStatusReport {
    pub poll_interval_secs: u64,
    pub loop_running: bool,
    pub sources: HashMap<String, SourceStats>,
}

pub struct IngestionScheduler {
    sources: HashMap<String, Arc<dyn MemorySource>>,
    poll_interval_secs: u64,
    stats: Arc<RwLock<HashMap<String, SourceStats>>>,
    loop_running: Arc<AtomicBool>,
}

impl IngestionScheduler {
    pub fn new(poll_interval_secs: u64) -> Self {
        Self {
            sources: HashMap::new(),
            poll_interval_secs,
            stats: Arc::new(RwLock::new(HashMap::new())),
            loop_running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn register_source(&mut self, source: Arc<dyn MemorySource>) {
        let id = source.source_id().to_string();
        self.sources.insert(id.clone(), source);
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn poll_interval(&self) -> u64 {
        self.poll_interval_secs
    }

    pub fn source_ids(&self) -> Vec<String> {
        self.sources.keys().cloned().collect()
    }

    pub async fn status(&self) -> IngestionStatusReport {
        IngestionStatusReport {
            poll_interval_secs: self.poll_interval_secs,
            loop_running: self.loop_running.load(Ordering::Relaxed),
            sources: self.stats.read().await.clone(),
        }
    }

    pub async fn ingest_source<S: MemoryStore + ?Sized>(
        &self,
        store: &S,
        source_id: &str,
        scope: &ScopeKey,
    ) -> Result<usize> {
        let source = self
            .sources
            .get(source_id)
            .ok_or_else(|| anyhow!("unknown source: {source_id}"))?
            .clone();

        let since = self
            .stats
            .read()
            .await
            .get(source_id)
            .and_then(|s| s.last_poll_at_ms);

        let now = chrono::Utc::now().timestamp_millis();
        match source.fetch_memories(scope, since).await {
            Ok(memories) => {
                let count = memories.len();
                for memory in memories {
                    store.put(memory).await?;
                }
                let mut stats = self.stats.write().await;
                let entry = stats.entry(source_id.to_string()).or_default();
                entry.last_poll_at_ms = Some(now);
                entry.last_success_count = count;
                entry.last_error = None;
                entry.total_ingested += count as u64;
                Ok(count)
            }
            Err(err) => {
                let mut stats = self.stats.write().await;
                let entry = stats.entry(source_id.to_string()).or_default();
                entry.last_success_count = 0;
                entry.last_error = Some(err.to_string());
                Err(err)
            }
        }
    }

    pub async fn ingest_all<S: MemoryStore + ?Sized>(
        &self,
        store: &S,
        scope: &ScopeKey,
    ) -> Vec<(String, Result<usize>)> {
        let mut results = Vec::new();
        for source_id in self.source_ids() {
            let outcome = self.ingest_source(store, &source_id, scope).await;
            results.push((source_id, outcome));
        }
        results
    }

    /// Spawn a background polling loop for all registered sources.
    pub fn spawn_polling_loop<S: MemoryStore + Send + Sync + 'static>(
        self: Arc<Self>,
        store: Arc<S>,
        scope: ScopeKey,
    ) -> tokio::task::JoinHandle<()> {
        self.loop_running.store(true, Ordering::Relaxed);
        tokio::spawn(async move {
            info!(
                interval_secs = self.poll_interval_secs,
                sources = self.source_count(),
                "ingestion polling loop started"
            );
            loop {
                for source_id in self.source_ids() {
                    match self.ingest_source(store.as_ref(), &source_id, &scope).await {
                        Ok(count) if count > 0 => {
                            info!(source = %source_id, count, "ingestion poll stored memories");
                        }
                        Ok(_) => {}
                        Err(err) => {
                            warn!(source = %source_id, error = %err, "ingestion poll failed");
                        }
                    }
                }
                tokio::time::sleep(Duration::from_secs(self.poll_interval_secs)).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mom_core::{MemoryId, MemoryItem};
    use std::sync::Mutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct RecordingStore {
        items: Mutex<Vec<MemoryItem>>,
    }

    #[async_trait]
    impl MemoryStore for RecordingStore {
        async fn put(&self, item: MemoryItem) -> anyhow::Result<()> {
            self.items.lock().unwrap().push(item);
            Ok(())
        }

        async fn get(&self, _id: &MemoryId) -> anyhow::Result<Option<MemoryItem>> {
            Ok(None)
        }

        async fn query(
            &self,
            _q: mom_core::Query,
        ) -> anyhow::Result<Vec<mom_core::Scored<MemoryItem>>> {
            Ok(Vec::new())
        }

        async fn delete(&self, _id: &MemoryId) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn ingest_source_fetches_and_stores_from_wiremock() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "repo": "repo",
                "file": "all",
                "analysis_type": "ast",
                "functions": [{
                    "name": "main",
                    "signature": "fn main()",
                    "line_start": 1,
                    "line_end": 2
                }],
                "patterns": [],
                "dependencies": [],
                "timestamp": 1_609_459_200_000_i64,
                "confidence": 0.9
            })))
            .mount(&server)
            .await;

        let source = Arc::new(crate::OxidizedRAGSource::new(server.uri()));
        let mut scheduler = IngestionScheduler::new(60);
        scheduler.register_source(source);

        let store = Arc::new(RecordingStore {
            items: Mutex::new(Vec::new()),
        });
        let scope = ScopeKey {
            tenant_id: "tenant-a".into(),
            workspace_id: Some("repo".into()),
            project_id: Some("all".into()),
            agent_id: None,
            run_id: None,
        };

        let count = scheduler
            .ingest_source(store.as_ref(), "oxidizedrag", &scope)
            .await
            .expect("ingest");
        assert_eq!(count, 1);
        assert_eq!(store.items.lock().unwrap().len(), 1);

        let status = scheduler.status().await;
        let stats = status.sources.get("oxidizedrag").expect("stats");
        assert_eq!(stats.last_success_count, 1);
        assert_eq!(stats.total_ingested, 1);
        assert!(stats.last_error.is_none());
    }

    #[tokio::test]
    async fn ingest_source_failure_does_not_advance_since_watermark() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/analyze"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let source = Arc::new(crate::OxidizedRAGSource::new(server.uri()));
        let mut scheduler = IngestionScheduler::new(60);
        scheduler.register_source(source);

        let store = Arc::new(RecordingStore {
            items: Mutex::new(Vec::new()),
        });
        let scope = ScopeKey {
            tenant_id: "tenant-a".into(),
            workspace_id: Some("repo".into()),
            project_id: Some("all".into()),
            agent_id: None,
            run_id: None,
        };

        let result = scheduler
            .ingest_source(store.as_ref(), "oxidizedrag", &scope)
            .await;
        assert!(result.is_err());

        let status = scheduler.status().await;
        let stats = status.sources.get("oxidizedrag").expect("stats");
        assert!(stats.last_poll_at_ms.is_none());
        assert_eq!(stats.last_success_count, 0);
        assert!(stats.last_error.is_some());
    }
}
