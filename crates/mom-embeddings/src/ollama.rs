//! Ollama embedding provider (local, self-hosted)

use anyhow::Result;
use async_trait::async_trait;
use mom_core::Embedder;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

/// Ollama embedding provider
///
/// Requires Ollama running locally on the configured base URL.
/// Supports models like:
/// - mxbai-embed-large (1024 dims)
/// - nomic-embed-text (768 dims)
/// - mistral-embed (1024 dims via Ollama)
pub struct OllamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            prompt: input.to_string(),
        };

        let response = self.client.post(&url).json(&request).send().await?;

        let body = response.json::<OllamaEmbedResponse>().await?;
        Ok(body.embedding)
    }

    fn dims(&self) -> usize {
        // Default for mxbai-embed-large and mistral-embed
        match self.model.as_str() {
            "mxbai-embed-large" => 1024,
            "nomic-embed-text" => 768,
            "mistral-embed" => 1024,
            _ => 1024, // Default guess
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
    fn test_ollama_embedder_creation() {
        let embedder = OllamaEmbedder::new(
            "http://localhost:11434".to_string(),
            "mxbai-embed-large".to_string(),
        );
        assert_eq!(embedder.dims(), 1024);
        assert_eq!(embedder.model_id(), "mxbai-embed-large");
    }
}
