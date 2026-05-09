//! MOM Sources - Multi-source memory ingestion connectors
//!
//! Integrates external systems (oxidizedRAG, oxidizedgraph, data-fabric)
//! to automatically ingest memories into MOM.

use mom_core::ScopeKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod connectors;

pub use connectors::{
    DataFabricConnector, OxidizedGraphConnector, OxidizedRAGConnector, SourceConnector,
};

/// Source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Unique identifier for this source
    pub source_id: String,
    /// Type of source (oxidizedrag, oxidizedgraph, data-fabric)
    pub source_type: String,
    /// Base URL or endpoint
    pub endpoint: String,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Scope to assign to memories ingested from this source
    pub scope: ScopeKey,
    /// Poll interval in seconds
    pub poll_interval_secs: u64,
    /// Enabled/disabled
    pub enabled: bool,
}

/// Result of ingesting a memory from a source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionResult {
    pub memory_id: String,
    pub source_id: String,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp_ms: i64,
}

/// Manager for coordinating multiple source connectors
pub struct IngestionManager {
    connectors: Arc<RwLock<HashMap<String, Arc<dyn SourceConnector>>>>,
    configs: Arc<RwLock<HashMap<String, SourceConfig>>>,
}

impl IngestionManager {
    pub fn new() -> Self {
        Self {
            connectors: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a source connector
    pub async fn register(
        &self,
        config: SourceConfig,
        connector: Arc<dyn SourceConnector>,
    ) -> anyhow::Result<()> {
        let mut connectors = self.connectors.write().await;
        let mut configs = self.configs.write().await;

        connectors.insert(config.source_id.clone(), connector);
        configs.insert(config.source_id.clone(), config.clone());

        info!("Registered source connector: {}", config.source_id);
        Ok(())
    }

    /// Get a registered connector
    pub async fn get(&self, source_id: &str) -> Option<Arc<dyn SourceConnector>> {
        self.connectors.read().await.get(source_id).cloned()
    }

    /// List all registered sources
    pub async fn list_sources(&self) -> Vec<SourceConfig> {
        self.configs.read().await.values().cloned().collect()
    }

    /// Poll a specific source for new memories
    pub async fn poll_source(&self, source_id: &str) -> anyhow::Result<Vec<IngestionResult>> {
        let connector = self
            .get(source_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Source not found: {}", source_id))?;

        let config = self
            .configs
            .read()
            .await
            .get(source_id)
            .ok_or_else(|| anyhow::anyhow!("Config not found: {}", source_id))?
            .clone();

        if !config.enabled {
            warn!("Source is disabled: {}", source_id);
            return Ok(Vec::new());
        }

        let memories = connector.fetch_memories(&config).await?;
        debug!(
            "Fetched {} memories from source: {}",
            memories.len(),
            source_id
        );

        Ok(memories
            .into_iter()
            .map(|m| IngestionResult {
                memory_id: m.id.0,
                source_id: source_id.to_string(),
                success: true,
                error: None,
                timestamp_ms: chrono::Utc::now().timestamp_millis(),
            })
            .collect())
    }

    /// Poll all enabled sources
    pub async fn poll_all(&self) -> anyhow::Result<Vec<IngestionResult>> {
        let sources = self.list_sources().await;
        let mut results = Vec::new();

        for source in sources {
            if !source.enabled {
                continue;
            }

            match self.poll_source(&source.source_id).await {
                Ok(source_results) => results.extend(source_results),
                Err(e) => {
                    warn!("Error polling source {}: {}", source.source_id, e);
                    results.push(IngestionResult {
                        memory_id: String::new(),
                        source_id: source.source_id,
                        success: false,
                        error: Some(e.to_string()),
                        timestamp_ms: chrono::Utc::now().timestamp_millis(),
                    });
                }
            }
        }

        Ok(results)
    }
}

impl Default for IngestionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ingestion_manager_creation() {
        let manager = IngestionManager::new();
        assert!(manager.list_sources().await.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources_empty() {
        let manager = IngestionManager::new();
        let sources = manager.list_sources().await;
        assert_eq!(sources.len(), 0);
    }

    #[tokio::test]
    async fn test_ingestion_manager_default() {
        let manager = IngestionManager::default();
        assert!(manager.list_sources().await.is_empty());
    }

    #[tokio::test]
    async fn test_poll_nonexistent_source() {
        let manager = IngestionManager::new();
        let result = manager.poll_source("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_nonexistent_source() {
        let manager = IngestionManager::new();
        let connector = manager.get("nonexistent").await;
        assert!(connector.is_none());
    }

    #[tokio::test]
    async fn test_poll_all_empty() {
        let manager = IngestionManager::new();
        let results = manager.poll_all().await;
        assert!(results.is_ok());
        assert!(results.unwrap().is_empty());
    }

    #[test]
    fn test_source_config_creation() {
        let config = SourceConfig {
            source_id: "test-source".to_string(),
            source_type: "oxidizedrag".to_string(),
            endpoint: "http://localhost:8000".to_string(),
            api_key: Some("test-key".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: Some("repo".to_string()),
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            poll_interval_secs: 60,
            enabled: true,
        };

        assert_eq!(config.source_id, "test-source");
        assert_eq!(config.source_type, "oxidizedrag");
        assert_eq!(config.scope.tenant_id, "acme");
        assert!(config.enabled);
    }

    #[test]
    fn test_ingestion_result_success() {
        let result = IngestionResult {
            memory_id: "mem-123".to_string(),
            source_id: "source-1".to_string(),
            success: true,
            error: None,
            timestamp_ms: 1000,
        };

        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_ingestion_result_failure() {
        let result = IngestionResult {
            memory_id: "mem-123".to_string(),
            source_id: "source-1".to_string(),
            success: false,
            error: Some("Connection failed".to_string()),
            timestamp_ms: 1000,
        };

        assert!(!result.success);
        assert!(result.error.is_some());
    }
}
