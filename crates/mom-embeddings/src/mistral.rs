//! Mistral embedding provider (API-based)

use anyhow::Result;
use async_trait::async_trait;
use mom_core::Embedder;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct MistralEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct MistralEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct MistralEmbedResponse {
    data: Vec<MistralEmbedding>,
}

/// Mistral embedding provider
///
/// Uses Mistral's API for embeddings.
/// Model: mistral-embed (1024 dimensions)
pub struct MistralEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl MistralEmbedder {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl Embedder for MistralEmbedder {
    async fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let url = "https://api.mistral.ai/v1/embeddings";
        let request = MistralEmbedRequest {
            model: self.model.clone(),
            input: vec![input.to_string()],
        };

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let body = response.json::<MistralEmbedResponse>().await?;
        Ok(body
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embeddings returned"))?
            .embedding
            .clone())
    }

    fn dims(&self) -> usize {
        // Mistral embed is 1024 dimensions
        1024
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_embedder_creation() {
        let embedder = MistralEmbedder::new("test-key".to_string(), "mistral-embed".to_string());
        assert_eq!(embedder.dims(), 1024);
        assert_eq!(embedder.model_id(), "mistral-embed");
    }
}
