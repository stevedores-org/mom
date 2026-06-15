//! Shared HTTP client configuration for memory source connectors.

use reqwest::{Client, RequestBuilder, Response};
use std::time::Duration;
use tracing::warn;

pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_RETRIES: u32 = 3;

/// Build an HTTP client with a bounded request timeout so slow upstreams
/// cannot hang ingestion indefinitely.
pub fn build_http_client() -> Result<Client, reqwest::Error> {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
}

/// Attach bearer auth when an API key is configured.
pub fn apply_api_key(builder: RequestBuilder, api_key: &Option<String>) -> RequestBuilder {
    if let Some(key) = api_key {
        builder.header("Authorization", format!("Bearer {key}"))
    } else {
        builder
    }
}

/// Send a request with small exponential backoff on transient failures.
pub async fn send_with_retry(build: impl Fn() -> RequestBuilder) -> Result<Response, reqwest::Error> {
    let mut attempt = 0u32;
    loop {
        match build().send().await {
            Ok(resp) if resp.status().is_server_error() && attempt + 1 < MAX_RETRIES => {
                attempt += 1;
                warn!(
                    status = resp.status().as_u16(),
                    attempt,
                    "retrying upstream after server error"
                );
                tokio::time::sleep(Duration::from_millis(100 * 2u64.pow(attempt - 1))).await;
            }
            other => return other,
        }
    }
}
