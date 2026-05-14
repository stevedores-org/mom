//! MOM Multi-Source Ingestion - Unified memory layer connectors
//!
//! Integrates oxidizedRAG, oxidizedgraph, and data-fabric as memory sources.
//! Provides pluggable MemorySource trait for ingesting external memories into MOM.

use anyhow::Result;
use async_trait::async_trait;
use mom_core::{MemoryItem, ScopeKey};

pub mod datafabric;
pub mod oxidizedgraph;
pub mod oxidizedrag;

pub use datafabric::DataFabricSource;
pub use oxidizedgraph::OxidizedGraphSource;
pub use oxidizedrag::OxidizedRAGSource;

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
///
/// Implementations provide a unified interface for fetching memories from
/// external systems (oxidizedRAG, oxidizedgraph, data-fabric) and ingesting
/// them into MOM.
#[async_trait]
pub trait MemorySource: Send + Sync {
    /// Unique identifier for this source
    fn source_id(&self) -> &str;

    /// Human-readable description of what this source provides
    fn description(&self) -> &str;

    /// Fetch memories from this source for the given scope
    ///
    /// # Arguments
    /// * `scope` - The memory scope (tenant, workspace, project, agent, run)
    /// * `since` - Optional: only return items modified since this timestamp (ms)
    ///
    /// # Returns
    /// Vector of MemoryItems ready to be stored via MemoryStore::put()
    async fn fetch_memories(&self, scope: &ScopeKey, since: Option<i64>)
        -> Result<Vec<MemoryItem>>;

    /// Optional: Subscribe to real-time updates from this source
    ///
    /// Default implementation returns NotImplemented error.
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

    /// Optional: Check if this source is healthy/available
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// Ingestion scheduler for managing multi-source memory ingestion
pub struct IngestionScheduler {
    sources: Vec<Box<dyn MemorySource>>,
    poll_interval_secs: u64,
}

impl IngestionScheduler {
    /// Create a new ingestion scheduler
    pub fn new(poll_interval_secs: u64) -> Self {
        Self {
            sources: Vec::new(),
            poll_interval_secs,
        }
    }

    /// Register a memory source
    pub fn register_source(&mut self, source: Box<dyn MemorySource>) {
        self.sources.push(source);
    }

    /// Get the number of registered sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get polling interval in seconds
    pub fn poll_interval(&self) -> u64 {
        self.poll_interval_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingestion_scheduler_creation() {
        let scheduler = IngestionScheduler::new(60);
        assert_eq!(scheduler.source_count(), 0);
        assert_eq!(scheduler.poll_interval(), 60);
    }
}
