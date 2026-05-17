//! OpenAI embedding provider (API-based)

use anyhow::Result;
use async_trait::async_trait;
use mom_core::Embedder;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbeddingData>,
}

/// OpenAI embedding provider
///
/// Uses OpenAI's API for embeddings.
/// Supported models:
/// - text-embedding-3-small (1536 dimensions)
/// - text-embedding-3-large (3072 dimensions)
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let url = "https://api.openai.com/v1/embeddings";
        let request = OpenAIEmbedRequest {
            model: self.model.clone(),
            input: input.to_string(),
        };

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let body = response.json::<OpenAIEmbedResponse>().await?;
        Ok(body
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embeddings returned"))?
            .embedding
            .clone())
    }

    fn dims(&self) -> usize {
        // OpenAI embedding dimensions
        match self.model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 3072, // Default to large
        }
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embedder_creation() {
        let embedder =
            OpenAIEmbedder::new("test-key".to_string(), "text-embedding-3-large".to_string());
        assert_eq!(embedder.dims(), 3072);
        assert_eq!(embedder.model_id(), "text-embedding-3-large");
    }
}
