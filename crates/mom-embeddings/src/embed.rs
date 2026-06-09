//! Auto-embedding helpers for memory writes and ingestion.

use anyhow::Result;
use mom_core::{content_embed_text, Embedder, MemoryItem};
use tracing::warn;

/// Populate `embedding` and `embedding_model` when an embedder is available.
///
/// Skips items that already have an embedding or lack embeddable text.
/// Embedding failures are logged and do not fail the caller.
pub async fn maybe_embed_item(item: &mut MemoryItem, embedder: &dyn Embedder) -> Result<()> {
    if item.embedding.is_some() {
        return Ok(());
    }

    let Some(text) = content_embed_text(&item.content, item.kind) else {
        return Ok(());
    };

    match embedder.embed(&text).await {
        Ok(vector) => {
            item.embedding = Some(vector);
            item.embedding_model = Some(embedder.model_id().to_string());
        }
        Err(e) => {
            warn!(
                memory_id = %item.id.0,
                "auto-embed failed; storing without embedding: {}",
                e
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mom_core::{Content, MemoryId, MemoryKind, ScopeKey};
    use std::collections::BTreeMap;

    struct StubEmbedder;

    #[async_trait]
    impl Embedder for StubEmbedder {
        async fn embed(&self, _input: &str) -> Result<Vec<f32>> {
            Ok(vec![0.1, 0.2, 0.3])
        }

        fn dims(&self) -> usize {
            3
        }

        fn model_id(&self) -> &str {
            "stub-model"
        }
    }

    fn sample_item(kind: MemoryKind, content: Content) -> MemoryItem {
        MemoryItem {
            id: MemoryId("m1".to_string()),
            scope: ScopeKey {
                tenant_id: "acme".to_string(),
                workspace_id: None,
                project_id: None,
                agent_id: None,
                run_id: None,
            },
            kind,
            created_at_ms: 1,
            content,
            tags: vec![],
            importance: 0.5,
            confidence: 0.9,
            source: "agent".to_string(),
            ttl_ms: None,
            meta: BTreeMap::new(),
            embedding: None,
            embedding_model: None,
        }
    }

    #[tokio::test]
    async fn embeds_text_memory() {
        let embedder = StubEmbedder;
        let mut item = sample_item(
            MemoryKind::Event,
            Content::Text("deploy kubernetes".to_string()),
        );
        maybe_embed_item(&mut item, &embedder).await.unwrap();
        assert!(item.embedding.is_some());
        assert_eq!(item.embedding_model.as_deref(), Some("stub-model"));
    }

    #[tokio::test]
    async fn skips_checkpoint_kind() {
        let embedder = StubEmbedder;
        let mut item = sample_item(
            MemoryKind::Checkpoint,
            Content::Text("state blob".to_string()),
        );
        maybe_embed_item(&mut item, &embedder).await.unwrap();
        assert!(item.embedding.is_none());
    }

    #[tokio::test]
    async fn preserves_existing_embedding() {
        let embedder = StubEmbedder;
        let mut item = sample_item(MemoryKind::Fact, Content::Text("fact".to_string()));
        item.embedding = Some(vec![9.0]);
        item.embedding_model = Some("existing".to_string());
        maybe_embed_item(&mut item, &embedder).await.unwrap();
        assert_eq!(item.embedding, Some(vec![9.0]));
        assert_eq!(item.embedding_model.as_deref(), Some("existing"));
    }
}
