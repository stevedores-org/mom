//! Memory graph link persistence and traversal for SurrealDB.

use mom_core::{
    require_tenant_id, MemoryId, MemoryLink, MemoryLinkId, MemoryLinkStore, MemoryStore,
    RelationshipType, ScopeKey, TraversalStep,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use surrealdb::engine::local::Db;
use tracing::debug;

use crate::SurrealDBStore;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StoredLink {
    link_id: String,
    tenant_id: String,
    src_memory_id: String,
    dst_memory_id: String,
    rel: String,
    weight: f32,
    confidence: f32,
    created_at_ms: i64,
}

impl StoredLink {
    fn into_link(self) -> Option<MemoryLink> {
        Some(MemoryLink {
            id: MemoryLinkId(self.link_id),
            tenant_id: self.tenant_id,
            src: MemoryId(self.src_memory_id),
            dst: MemoryId(self.dst_memory_id),
            rel: RelationshipType::parse(&self.rel)?,
            weight: self.weight,
            confidence: self.confidence,
            created_at_ms: self.created_at_ms,
        })
    }
}

impl SurrealDBStore {
    async fn memory_exists_in_tenant(
        &self,
        tenant_id: &str,
        memory_id: &MemoryId,
    ) -> anyhow::Result<bool> {
        let scope = ScopeKey {
            tenant_id: tenant_id.to_string(),
            workspace_id: None,
            project_id: None,
            agent_id: None,
            run_id: None,
        };
        Ok(self.get_scoped(memory_id, &scope).await?.is_some())
    }

    fn validate_link_metadata(weight: f32, confidence: f32) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&weight) || !(0.0..=1.0).contains(&confidence) {
            anyhow::bail!("weight and confidence must be in 0..=1");
        }
        Ok(())
    }

    fn stored_from_link(link: &MemoryLink) -> StoredLink {
        StoredLink {
            link_id: link.id.0.clone(),
            tenant_id: link.tenant_id.clone(),
            src_memory_id: link.src.0.clone(),
            dst_memory_id: link.dst.0.clone(),
            rel: link.rel.as_str().to_string(),
            weight: link.weight,
            confidence: link.confidence,
            created_at_ms: link.created_at_ms,
        }
    }

    async fn fetch_links(
        &self,
        query: &str,
        tenant_id: &str,
        extra: impl Fn(surrealdb::method::Query<'_, Db>) -> surrealdb::method::Query<'_, Db>,
    ) -> anyhow::Result<Vec<MemoryLink>> {
        let builder = self.db.query(query).bind(("tenant", tenant_id.to_string()));
        let rows: Vec<StoredLink> = extra(builder).await?.take(0)?;
        Ok(rows.into_iter().filter_map(StoredLink::into_link).collect())
    }
}

#[async_trait::async_trait]
impl MemoryLinkStore for SurrealDBStore {
    async fn put_link(&self, link: MemoryLink) -> anyhow::Result<()> {
        require_tenant_id(&link.tenant_id)?;
        Self::validate_link_metadata(link.weight, link.confidence)?;

        if !self
            .memory_exists_in_tenant(&link.tenant_id, &link.src)
            .await?
        {
            anyhow::bail!("source memory not found in tenant");
        }
        if !self
            .memory_exists_in_tenant(&link.tenant_id, &link.dst)
            .await?
        {
            anyhow::bail!("destination memory not found in tenant");
        }

        let stored = Self::stored_from_link(&link);
        let _: Vec<StoredLink> = self
            .db
            .query("UPSERT type::thing('memory_links', $id) MERGE $data")
            .bind(("id", stored.link_id.clone()))
            .bind(("data", stored))
            .await?
            .take(0)?;

        debug!(link_id = %link.id.0, rel = link.rel.as_str(), "stored memory link");
        Ok(())
    }

    async fn update_link(&self, link: MemoryLink) -> anyhow::Result<()> {
        require_tenant_id(&link.tenant_id)?;
        Self::validate_link_metadata(link.weight, link.confidence)?;

        let existing = self.get_link(&link.tenant_id, &link.id).await?;
        let Some(existing) = existing else {
            anyhow::bail!("link not found");
        };

        let updated = MemoryLink {
            created_at_ms: existing.created_at_ms,
            ..link
        };
        self.put_link(updated).await
    }

    async fn delete_link(&self, tenant_id: &str, link_id: &MemoryLinkId) -> anyhow::Result<()> {
        require_tenant_id(tenant_id)?;
        let _: Vec<StoredLink> = self
            .db
            .query("DELETE memory_links WHERE link_id = $id AND tenant_id = $tenant")
            .bind(("id", link_id.0.clone()))
            .bind(("tenant", tenant_id.to_string()))
            .await?
            .take(0)?;
        Ok(())
    }

    async fn get_link(
        &self,
        tenant_id: &str,
        link_id: &MemoryLinkId,
    ) -> anyhow::Result<Option<MemoryLink>> {
        require_tenant_id(tenant_id)?;
        let rows: Vec<StoredLink> = self
            .db
            .query("SELECT * FROM memory_links WHERE link_id = $id AND tenant_id = $tenant LIMIT 1")
            .bind(("id", link_id.0.clone()))
            .bind(("tenant", tenant_id.to_string()))
            .await?
            .take(0)?;
        Ok(rows.into_iter().next().and_then(StoredLink::into_link))
    }

    async fn list_links_from(
        &self,
        tenant_id: &str,
        src: &MemoryId,
        rel: Option<RelationshipType>,
    ) -> anyhow::Result<Vec<MemoryLink>> {
        require_tenant_id(tenant_id)?;
        match rel {
            Some(rel) => {
                self.fetch_links(
                    "SELECT * FROM memory_links WHERE tenant_id = $tenant AND src_memory_id = $src AND rel = $rel",
                    tenant_id,
                    |b| {
                        b.bind(("src", src.0.clone()))
                            .bind(("rel", rel.as_str().to_string()))
                    },
                )
                .await
            }
            None => {
                self.fetch_links(
                    "SELECT * FROM memory_links WHERE tenant_id = $tenant AND src_memory_id = $src",
                    tenant_id,
                    |b| b.bind(("src", src.0.clone())),
                )
                .await
            }
        }
    }

    async fn traverse(
        &self,
        tenant_id: &str,
        from: &MemoryId,
        rel: Option<RelationshipType>,
        max_depth: usize,
    ) -> anyhow::Result<Vec<TraversalStep>> {
        require_tenant_id(tenant_id)?;
        let max_depth = max_depth.clamp(1, 32);

        let mut visited = HashSet::new();
        visited.insert(from.0.clone());
        let mut queue = VecDeque::from([(from.clone(), 0usize)]);
        let mut steps = Vec::new();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let outgoing = self.list_links_from(tenant_id, &current, rel).await?;
            for link in outgoing {
                let next_id = link.dst.clone();
                if visited.insert(next_id.0.clone()) {
                    steps.push(TraversalStep {
                        memory_id: next_id.clone(),
                        depth: depth + 1,
                        link: link.clone(),
                    });
                    queue.push_back((next_id, depth + 1));
                }
            }
        }

        Ok(steps)
    }

    async fn find_contradictions(
        &self,
        tenant_id: &str,
        memory_id: Option<&MemoryId>,
    ) -> anyhow::Result<Vec<MemoryLink>> {
        require_tenant_id(tenant_id)?;
        match memory_id {
            Some(id) => {
                self.fetch_links(
                    "SELECT * FROM memory_links WHERE tenant_id = $tenant AND rel = 'contradicts' AND (src_memory_id = $id OR dst_memory_id = $id)",
                    tenant_id,
                    |b| b.bind(("id", id.0.clone())),
                )
                .await
            }
            None => {
                self.fetch_links(
                    "SELECT * FROM memory_links WHERE tenant_id = $tenant AND rel = 'contradicts'",
                    tenant_id,
                    |b| b,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SurrealDBStore;
    use mom_core::{Content, MemoryItem, MemoryKind, MemoryStore, ScopeKey};

    fn tenant_scope(tenant: &str) -> ScopeKey {
        ScopeKey {
            tenant_id: tenant.into(),
            workspace_id: None,
            project_id: None,
            agent_id: None,
            run_id: None,
        }
    }

    fn sample_memory(tenant: &str, id: &str, text: &str) -> MemoryItem {
        MemoryItem {
            id: MemoryId(id.into()),
            scope: tenant_scope(tenant),
            kind: MemoryKind::Event,
            created_at_ms: 1_700_000_000_000,
            content: Content::Text(text.into()),
            tags: vec![],
            importance: 0.5,
            confidence: 1.0,
            source: "test".into(),
            ttl_ms: None,
            meta: Default::default(),
            embedding: None,
            embedding_model: None,
        }
    }

    #[tokio::test]
    async fn link_create_and_traverse_causal_chain() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        store
            .put(sample_memory("tenant-a", "event-a", "root cause"))
            .await
            .expect("put a");
        store
            .put(sample_memory("tenant-a", "event-b", "effect"))
            .await
            .expect("put b");

        let link = MemoryLink {
            id: MemoryLinkId("link-ab".into()),
            tenant_id: "tenant-a".into(),
            src: MemoryId("event-a".into()),
            dst: MemoryId("event-b".into()),
            rel: RelationshipType::Causal,
            weight: 0.9,
            confidence: 0.95,
            created_at_ms: 1_700_000_000_001,
        };
        store.put_link(link).await.expect("put link");

        let steps = store
            .traverse(
                "tenant-a",
                &MemoryId("event-a".into()),
                Some(RelationshipType::Causal),
                3,
            )
            .await
            .expect("traverse");

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].memory_id.0, "event-b");
        assert_eq!(steps[0].link.rel, RelationshipType::Causal);
    }

    #[tokio::test]
    async fn find_contradictions_returns_conflict_edges() {
        let store = SurrealDBStore::new("mem://").await.expect("store");
        store
            .put(sample_memory("tenant-a", "fact-1", "value is 10"))
            .await
            .expect("put 1");
        store
            .put(sample_memory("tenant-a", "fact-2", "value is 20"))
            .await
            .expect("put 2");

        store
            .put_link(MemoryLink {
                id: MemoryLinkId("link-cx".into()),
                tenant_id: "tenant-a".into(),
                src: MemoryId("fact-1".into()),
                dst: MemoryId("fact-2".into()),
                rel: RelationshipType::Contradicts,
                weight: 1.0,
                confidence: 0.8,
                created_at_ms: 1,
            })
            .await
            .expect("put contradicts");

        let conflicts = store
            .find_contradictions("tenant-a", Some(&MemoryId("fact-1".into())))
            .await
            .expect("conflicts");

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].rel, RelationshipType::Contradicts);
    }
}
