//! MOM Embeddings - Pluggable embedding providers for semantic search
//!
//! Supports multiple embedding backends:
//! - Ollama (local, self-hosted)
//! - Mistral (API-based)
//! - OpenAI (API-based)

use anyhow::Result;
use mom_core::Embedder;

pub mod mistral;
pub mod ollama;
pub mod openai;

pub use mistral::MistralEmbedder;
pub use ollama::OllamaEmbedder;
pub use openai::OpenAIEmbedder;

/// Create an embedder based on environment configuration
pub async fn create_embedder() -> Result<Box<dyn Embedder>> {
    let provider = std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "ollama".to_string());

    match provider.as_str() {
        "ollama" => {
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let model =
                std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "mxbai-embed-large".to_string());
            Ok(Box::new(OllamaEmbedder::new(base_url, model)))
        }
        "mistral" => {
            let api_key = std::env::var("MISTRAL_API_KEY")?;
            let model =
                std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-embed".to_string());
            Ok(Box::new(MistralEmbedder::new(api_key, model)))
        }
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")?;
            let model = std::env::var("OPENAI_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-large".to_string());
            Ok(Box::new(OpenAIEmbedder::new(api_key, model)))
        }
        _ => Err(anyhow::anyhow!("Unknown embedding provider: {}", provider)),
    }
}
