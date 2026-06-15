//! MOM Multi-Source Ingestion - Unified memory layer connectors
//!
//! Integrates oxidizedRAG, oxidizedgraph, and data-fabric as memory sources.
//! Provides pluggable MemorySource trait for ingesting external memories into MOM.

use anyhow::Result;
use async_trait::async_trait;
use mom_core::{MemoryItem, ScopeKey};

pub mod datafabric;
pub mod http;
pub mod oxidizedgraph;
pub mod oxidizedrag;
pub mod scheduler;

pub use datafabric::DataFabricSource;
pub use oxidizedgraph::OxidizedGraphSource;
pub use oxidizedrag::OxidizedRAGSource;
pub use scheduler::{IngestionScheduler, IngestionStatusReport, SourceStats};

/// Error types for ingestion operations
#[derive(Debug, thiserror::Error)]
pub enum IngestionError {
    #[error("Source {0} unavailable: {1}")]
    SourceUnavailable(String, String),

    #[error("Invalid memory format: {0}")]
    InvalidMemory(String),

    #[error("Scope mismatch: {0}")]
    ScopeMismatch(String),

    #[error("Storage error: {0}")]
    StorageError(#[from] anyhow::Error),
}

/// Trait for external memory sources (Phase 2c - Issue #29)
#[async_trait]
pub trait MemorySource: Send + Sync {
    fn source_id(&self) -> &str;
    fn description(&self) -> &str;

    async fn fetch_memories(&self, scope: &ScopeKey, since: Option<i64>)
        -> Result<Vec<MemoryItem>>;

    async fn subscribe_updates(
        &self,
        _scope: &ScopeKey,
        _callback: Box<dyn Fn(MemoryItem) + Send + Sync>,
    ) -> Result<()> {
        Err(anyhow::anyhow!(
            "Real-time subscriptions not supported for {}",
            self.source_id()
        ))
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}
