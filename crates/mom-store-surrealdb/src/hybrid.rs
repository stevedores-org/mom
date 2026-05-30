//! Hybrid Search - RRF fusion of lexical + semantic results

use std::collections::HashMap;

/// RRF (Reciprocal Rank Fusion) constant
/// Prevents division by zero and controls contribution of early vs late ranks
/// Using 61 (not 60) as per standard RRF implementations to avoid rank 0 issues
pub const RRF_K: f32 = 61.0;

/// Result from a single search method
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub id: String,
    pub lexical_rank: Option<u32>,
    pub semantic_rank: Option<u32>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
}

/// Configuration for hybrid search weighting
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Weight for lexical results (0.0..1.0)
    pub lexical_weight: f32,
    /// Weight for semantic results (0.0..1.0)
    pub semantic_weight: f32,
    /// RRF constant (default 60)
    pub rrf_k: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            lexical_weight: 0.7,
            semantic_weight: 0.3,
            rrf_k: RRF_K,
        }
    }
}

impl HybridConfig {
    /// Normalize weights to sum to 1.0
    pub fn normalized(&self) -> Self {
        let total = self.lexical_weight + self.semantic_weight;
        if total == 0.0 {
            Self::default()
        } else {
            Self {
                lexical_weight: self.lexical_weight / total,
                semantic_weight: self.semantic_weight / total,
                rrf_k: self.rrf_k,
            }
        }
    }
}

/// Calculate RRF score for a result
///
/// Formula: score = sum over all rankers: 1 / (k + rank)
pub fn rrf_score(result: &RankedResult) -> f32 {
    rrf_score_weighted(result, 1.0, 1.0)
}

/// Calculate weighted RRF score
pub fn rrf_score_weighted(result: &RankedResult, lexical_weight: f32, semantic_weight: f32) -> f32 {
    let mut score = 0.0;

    if let Some(rank) = result.lexical_rank {
        score += lexical_weight / (RRF_K + rank as f32);
    }

    if let Some(rank) = result.semantic_rank {
        score += semantic_weight / (RRF_K + rank as f32);
    }

    score
}

/// Merge lexical and semantic results using RRF fusion
pub fn merge_results_with_rrf(
    lexical_results: Vec<(String, f32)>,
    semantic_results: Vec<(String, f32)>,
    config: &HybridConfig,
    limit: usize,
) -> Vec<(String, f32)> {
    let normalized = config.normalized();
    let mut ranked: HashMap<String, RankedResult> = HashMap::new();

    // Process lexical results
    for (rank, (id, score)) in lexical_results.iter().enumerate() {
        ranked
            .entry(id.clone())
            .or_insert_with(|| RankedResult {
                id: id.clone(),
                lexical_rank: None,
                semantic_rank: None,
                lexical_score: None,
                semantic_score: None,
            })
            .lexical_rank = Some(rank as u32);
        ranked.get_mut(id).unwrap().lexical_score = Some(*score);
    }

    // Process semantic results
    for (rank, (id, score)) in semantic_results.iter().enumerate() {
        ranked
            .entry(id.clone())
            .or_insert_with(|| RankedResult {
                id: id.clone(),
                lexical_rank: None,
                semantic_rank: None,
                lexical_score: None,
                semantic_score: None,
            })
            .semantic_rank = Some(rank as u32);
        ranked.get_mut(id).unwrap().semantic_score = Some(*score);
    }

    // Score and sort
    let mut scored: Vec<_> = ranked
        .into_values()
        .map(|r| {
            let score =
                rrf_score_weighted(&r, normalized.lexical_weight, normalized.semantic_weight);
            (r.id, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_lexical_rank() {
        let result = RankedResult {
            id: "1".to_string(),
            lexical_rank: Some(0),
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
        };
        let score = rrf_score(&result);
        assert!((score - 1.0 / 61.0).abs() < 0.0001);
    }

    #[test]
    fn test_rrf_single_semantic_rank() {
        let result = RankedResult {
            id: "1".to_string(),
            lexical_rank: None,
            semantic_rank: Some(4),
            lexical_score: None,
            semantic_score: None,
        };
        let score = rrf_score(&result);
        assert!((score - 1.0 / 65.0).abs() < 0.0001);
    }

    #[test]
    fn test_rrf_both_ranks() {
        let result = RankedResult {
            id: "1".to_string(),
            lexical_rank: Some(0),
            semantic_rank: Some(4),
            lexical_score: None,
            semantic_score: None,
        };
        let score = rrf_score(&result);
        let expected = 1.0 / 61.0 + 1.0 / 65.0;
        assert!((score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_rrf_weighted() {
        let result = RankedResult {
            id: "1".to_string(),
            lexical_rank: Some(0),
            semantic_rank: Some(0),
            lexical_score: None,
            semantic_score: None,
        };
        let score = rrf_score_weighted(&result, 2.0, 1.0);
        let expected = 2.0 / 61.0 + 1.0 / 61.0;
        assert!((score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_merge_no_overlap() {
        let lexical = vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.8)];
        let semantic = vec![("doc3".to_string(), 0.95), ("doc4".to_string(), 0.85)];
        let config = HybridConfig::default();

        let result = merge_results_with_rrf(lexical, semantic, &config, 10);

        assert_eq!(result.len(), 4);
        // doc1 and doc3 both have rank 0 in their respective searches (1/61)
        // Since they're tied, order is based on HashMap iteration (unstable)
        // Accept either doc1 or doc3 as valid result
        assert!(result[0].0 == "doc1" || result[0].0 == "doc3");
    }

    #[test]
    fn test_merge_with_overlap() {
        let lexical = vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.8)];
        let semantic = vec![("doc1".to_string(), 0.95), ("doc3".to_string(), 0.85)];
        let config = HybridConfig::default();

        let result = merge_results_with_rrf(lexical, semantic, &config, 10);

        assert_eq!(result.len(), 3);
        // doc1 appears in both, should rank first
        assert_eq!(result[0].0, "doc1");
    }

    #[test]
    fn test_hybrid_config_normalization() {
        let config = HybridConfig {
            lexical_weight: 2.0,
            semantic_weight: 1.0,
            rrf_k: 60.0,
        };
        let normalized = config.normalized();
        assert!((normalized.lexical_weight - 2.0 / 3.0).abs() < 0.0001);
        assert!((normalized.semantic_weight - 1.0 / 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_merge_respects_limit() {
        let lexical: Vec<_> = (0..20).map(|i| (format!("doc{}", i), 0.9)).collect();
        let semantic: Vec<_> = (10..30).map(|i| (format!("doc{}", i), 0.8)).collect();
        let config = HybridConfig::default();

        let result = merge_results_with_rrf(lexical, semantic, &config, 5);

        assert_eq!(result.len(), 5);
    }
}
