//! Typed payloads for `Fact` and `Preference` memory kinds (US-10).
//!
//! Structured fields live under well-known keys in `MemoryItem.meta` rather
//! than as first-class struct fields, so the kernel storage contract
//! (`BTreeMap<String, Value>`) stays backward-compatible with pre-US-10
//! items and existing rows in the SurrealDB store. The helpers in this
//! module codify the convention so consumers don't have to grep around
//! `meta` themselves.
//!
//! Meta layout for `MemoryKind::Fact`:
//!
//! ```jsonc
//! {
//!   "fact": { "subject": "...", "predicate": "...", "object": "..." },
//!   "provenance_ids": ["mem-1", "mem-2"],
//!   "version": 1,
//!   "superseded_by": "mem-99",          // present only on retired versions
//!   "semantic_conflicts": ["mem-7"]     // optional, written by the semantic-conflict pass
//! }
//! ```
//!
//! Meta layout for `MemoryKind::Preference`:
//!
//! ```jsonc
//! {
//!   "preference": {
//!     "rule": "retry-failed-calls",
//!     "decision": "retry up to 3 times",
//!     "priority": 100,
//!     "conditions": [{ "tool": "http_fetch" }]
//!   },
//!   "provenance_ids": [...]
//! }
//! ```

use crate::{MemoryId, MemoryKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Meta key under which a Fact item stores its structured `(subject,
/// predicate, object)` triple.
pub const META_FACT: &str = "fact";
/// Meta key under which a Preference item stores its structured payload.
pub const META_PREFERENCE: &str = "preference";
/// Meta key holding the list of upstream memory ids this item is derived
/// from. Empty array means no provenance tracked.
pub const META_PROVENANCE_IDS: &str = "provenance_ids";
/// Meta key for the integer version counter; absent ⇒ implicit version 1.
pub const META_VERSION: &str = "version";
/// Meta key set on a fact whose triple has been retired by a newer fact.
/// Holds the id of the newer fact.
pub const META_SUPERSEDED_BY: &str = "superseded_by";
/// Meta key holding ids of facts that the semantic-conflict pass flagged
/// as overlapping with this one (cosine sim above threshold, different
/// object). Advisory only — does NOT trigger supersession.
pub const META_SEMANTIC_CONFLICTS: &str = "semantic_conflicts";

/// Structured `(subject, predicate, object)` payload for a `Fact`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FactPayload {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Structured rule/decision/priority/conditions payload for a `Preference`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferencePayload {
    pub rule: String,
    pub decision: String,
    pub priority: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Value>,
}

/// Validation failure when reading a Fact payload out of `meta`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FactValidationError {
    #[error("MemoryItem.kind is {actual:?}, expected MemoryKind::Fact")]
    WrongKind { actual: MemoryKind },
    #[error("Fact meta is missing the `{META_FACT}` key")]
    MissingPayload,
    #[error("Fact meta `{META_FACT}` is not a JSON object")]
    PayloadNotObject,
    #[error("Fact meta `{META_FACT}.{field}` is missing")]
    MissingField { field: &'static str },
    #[error("Fact meta `{META_FACT}.{field}` is not a string")]
    NotString { field: &'static str },
    #[error("Fact meta `{META_FACT}.{field}` is empty after trim — must be non-empty")]
    Empty { field: &'static str },
}

/// Validation failure when reading a Preference payload out of `meta`.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PreferenceValidationError {
    #[error("MemoryItem.kind is {actual:?}, expected MemoryKind::Preference")]
    WrongKind { actual: MemoryKind },
    #[error("Preference meta is missing the `{META_PREFERENCE}` key")]
    MissingPayload,
    #[error("Preference meta `{META_PREFERENCE}` is not a JSON object")]
    PayloadNotObject,
    #[error("Preference meta `{META_PREFERENCE}.{field}` is missing")]
    MissingField { field: &'static str },
    #[error("Preference meta `{META_PREFERENCE}.{field}` is not a string")]
    NotString { field: &'static str },
    #[error("Preference meta `{META_PREFERENCE}.{field}` is empty after trim — must be non-empty")]
    Empty { field: &'static str },
    #[error(
        "Preference meta `{META_PREFERENCE}.priority` ({0}) is not a u32; must be 0..=u32::MAX"
    )]
    PriorityOutOfRange(i64),
    #[error("Preference meta `{META_PREFERENCE}.conditions` is not an array")]
    ConditionsNotArray,
}

impl FactPayload {
    /// Read the structured payload from a `MemoryItem.meta` map. Caller is
    /// responsible for checking `MemoryItem.kind == Fact` first if they
    /// want that guarantee.
    pub fn try_from_meta(meta: &BTreeMap<String, Value>) -> Result<Self, FactValidationError> {
        let payload = meta
            .get(META_FACT)
            .ok_or(FactValidationError::MissingPayload)?
            .as_object()
            .ok_or(FactValidationError::PayloadNotObject)?;
        let read = |field: &'static str| -> Result<String, FactValidationError> {
            let raw = payload
                .get(field)
                .ok_or(FactValidationError::MissingField { field })?
                .as_str()
                .ok_or(FactValidationError::NotString { field })?;
            if raw.trim().is_empty() {
                Err(FactValidationError::Empty { field })
            } else {
                Ok(raw.to_string())
            }
        };
        Ok(Self {
            subject: read("subject")?,
            predicate: read("predicate")?,
            object: read("object")?,
        })
    }

    /// Write this payload into a `meta` map, overwriting any existing
    /// `META_FACT` entry. Does NOT touch provenance/version/superseded_by.
    pub fn write_to_meta(&self, meta: &mut BTreeMap<String, Value>) {
        meta.insert(
            META_FACT.into(),
            serde_json::json!({
                "subject": self.subject,
                "predicate": self.predicate,
                "object": self.object,
            }),
        );
    }
}

impl PreferencePayload {
    pub fn try_from_meta(
        meta: &BTreeMap<String, Value>,
    ) -> Result<Self, PreferenceValidationError> {
        let payload = meta
            .get(META_PREFERENCE)
            .ok_or(PreferenceValidationError::MissingPayload)?
            .as_object()
            .ok_or(PreferenceValidationError::PayloadNotObject)?;
        let read_str = |field: &'static str| -> Result<String, PreferenceValidationError> {
            let raw = payload
                .get(field)
                .ok_or(PreferenceValidationError::MissingField { field })?
                .as_str()
                .ok_or(PreferenceValidationError::NotString { field })?;
            if raw.trim().is_empty() {
                Err(PreferenceValidationError::Empty { field })
            } else {
                Ok(raw.to_string())
            }
        };
        let priority_raw = payload
            .get("priority")
            .ok_or(PreferenceValidationError::MissingField { field: "priority" })?
            .as_i64()
            .ok_or(PreferenceValidationError::MissingField { field: "priority" })?;
        let priority = u32::try_from(priority_raw)
            .map_err(|_| PreferenceValidationError::PriorityOutOfRange(priority_raw))?;
        let conditions = match payload.get("conditions") {
            None => Vec::new(),
            Some(Value::Array(items)) => items.clone(),
            Some(_) => return Err(PreferenceValidationError::ConditionsNotArray),
        };
        Ok(Self {
            rule: read_str("rule")?,
            decision: read_str("decision")?,
            priority,
            conditions,
        })
    }

    pub fn write_to_meta(&self, meta: &mut BTreeMap<String, Value>) {
        meta.insert(
            META_PREFERENCE.into(),
            serde_json::json!({
                "rule": self.rule,
                "decision": self.decision,
                "priority": self.priority,
                "conditions": self.conditions,
            }),
        );
    }
}

/// Read the provenance chain from a meta map. Returns an empty Vec if the
/// key is absent or malformed — provenance is informational; the read path
/// shouldn't fail on legacy items missing the field.
pub fn read_provenance_ids(meta: &BTreeMap<String, Value>) -> Vec<MemoryId> {
    meta.get(META_PROVENANCE_IDS)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| MemoryId(s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Replace the provenance chain in a meta map.
pub fn write_provenance_ids(meta: &mut BTreeMap<String, Value>, ids: &[MemoryId]) {
    let arr: Vec<Value> = ids.iter().map(|id| Value::String(id.0.clone())).collect();
    meta.insert(META_PROVENANCE_IDS.into(), Value::Array(arr));
}

/// Read the version counter; absent means implicit version 1.
pub fn read_version(meta: &BTreeMap<String, Value>) -> u32 {
    meta.get(META_VERSION)
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(1)
}

pub fn write_version(meta: &mut BTreeMap<String, Value>, version: u32) {
    meta.insert(META_VERSION.into(), Value::Number(version.into()));
}

/// Returns the id of the fact that superseded this one, if any.
pub fn read_superseded_by(meta: &BTreeMap<String, Value>) -> Option<MemoryId> {
    meta.get(META_SUPERSEDED_BY)
        .and_then(|v| v.as_str())
        .map(|s| MemoryId(s.to_string()))
}

pub fn write_superseded_by(meta: &mut BTreeMap<String, Value>, id: &MemoryId) {
    meta.insert(META_SUPERSEDED_BY.into(), Value::String(id.0.clone()));
}

/// Append `id` to the semantic-conflict advisory list. Idempotent — won't
/// double-add the same id.
pub fn record_semantic_conflict(meta: &mut BTreeMap<String, Value>, id: &MemoryId) {
    let mut existing: Vec<Value> = meta
        .get(META_SEMANTIC_CONFLICTS)
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    let id_str = Value::String(id.0.clone());
    if !existing.iter().any(|v| v == &id_str) {
        existing.push(id_str);
    }
    meta.insert(META_SEMANTIC_CONFLICTS.into(), Value::Array(existing));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fact_meta() -> BTreeMap<String, Value> {
        let mut meta = BTreeMap::new();
        meta.insert(
            META_FACT.into(),
            json!({
                "subject": "users",
                "predicate": "prefer",
                "object": "Cloudflare for deployment",
            }),
        );
        meta
    }

    #[test]
    fn fact_payload_roundtrips_through_meta() {
        let original = FactPayload {
            subject: "api".into(),
            predicate: "rate_limit_per_min".into(),
            object: "1000".into(),
        };
        let mut meta = BTreeMap::new();
        original.write_to_meta(&mut meta);
        let recovered = FactPayload::try_from_meta(&meta).expect("roundtrip");
        assert_eq!(recovered, original);
    }

    #[test]
    fn fact_payload_rejects_missing_predicate() {
        let mut meta = fact_meta();
        if let Value::Object(ref mut obj) = meta.get_mut(META_FACT).unwrap() {
            obj.remove("predicate");
        }
        let err = FactPayload::try_from_meta(&meta).unwrap_err();
        assert_eq!(
            err,
            FactValidationError::MissingField { field: "predicate" }
        );
    }

    #[test]
    fn fact_payload_rejects_whitespace_only_object() {
        let mut meta = fact_meta();
        if let Value::Object(ref mut obj) = meta.get_mut(META_FACT).unwrap() {
            obj.insert("object".into(), json!("   "));
        }
        let err = FactPayload::try_from_meta(&meta).unwrap_err();
        // Empty-after-trim is treated the same as missing for the
        // purposes of the conflict-detection key; both would silently
        // collapse to "" which would let any new fact supersede every
        // prior one in the same scope.
        assert_eq!(err, FactValidationError::Empty { field: "object" });
    }

    #[test]
    fn fact_payload_rejects_non_string_subject() {
        let mut meta = fact_meta();
        if let Value::Object(ref mut obj) = meta.get_mut(META_FACT).unwrap() {
            obj.insert("subject".into(), json!(42));
        }
        let err = FactPayload::try_from_meta(&meta).unwrap_err();
        assert_eq!(err, FactValidationError::NotString { field: "subject" });
    }

    #[test]
    fn preference_payload_roundtrips_through_meta() {
        let original = PreferencePayload {
            rule: "retry-policy".into(),
            decision: "retry up to 3 times".into(),
            priority: 100,
            conditions: vec![json!({ "tool": "http_fetch" })],
        };
        let mut meta = BTreeMap::new();
        original.write_to_meta(&mut meta);
        let recovered = PreferencePayload::try_from_meta(&meta).expect("roundtrip");
        assert_eq!(recovered, original);
    }

    #[test]
    fn preference_payload_defaults_empty_conditions() {
        let mut meta = BTreeMap::new();
        meta.insert(
            META_PREFERENCE.into(),
            json!({
                "rule": "always-https",
                "decision": "deny http origins",
                "priority": 10,
            }),
        );
        let parsed = PreferencePayload::try_from_meta(&meta).expect("missing conditions ok");
        assert!(parsed.conditions.is_empty());
    }

    #[test]
    fn preference_payload_rejects_negative_priority() {
        let mut meta = BTreeMap::new();
        meta.insert(
            META_PREFERENCE.into(),
            json!({
                "rule": "r", "decision": "d", "priority": -1,
            }),
        );
        let err = PreferencePayload::try_from_meta(&meta).unwrap_err();
        assert_eq!(err, PreferenceValidationError::PriorityOutOfRange(-1));
    }

    #[test]
    fn provenance_ids_roundtrip() {
        let mut meta = BTreeMap::new();
        let ids = vec![MemoryId("a".into()), MemoryId("b".into())];
        write_provenance_ids(&mut meta, &ids);
        assert_eq!(read_provenance_ids(&meta), ids);
    }

    #[test]
    fn provenance_ids_default_to_empty_on_legacy_items() {
        // Legacy item from before US-10 has no `provenance_ids` key. Read
        // path must not panic and must return Vec::new() so a Fact item
        // without provenance is still usable.
        let meta = BTreeMap::new();
        assert!(read_provenance_ids(&meta).is_empty());
    }

    #[test]
    fn version_defaults_to_one_when_absent() {
        let meta = BTreeMap::new();
        assert_eq!(read_version(&meta), 1);
    }

    #[test]
    fn version_round_trips() {
        let mut meta = BTreeMap::new();
        write_version(&mut meta, 7);
        assert_eq!(read_version(&meta), 7);
    }

    #[test]
    fn superseded_by_roundtrips() {
        let mut meta = BTreeMap::new();
        let id = MemoryId("v2".into());
        write_superseded_by(&mut meta, &id);
        assert_eq!(read_superseded_by(&meta), Some(id));
    }

    #[test]
    fn semantic_conflict_advisory_is_idempotent() {
        let mut meta = BTreeMap::new();
        let id = MemoryId("conflicting".into());
        record_semantic_conflict(&mut meta, &id);
        record_semantic_conflict(&mut meta, &id);
        let stored = meta
            .get(META_SEMANTIC_CONFLICTS)
            .and_then(|v| v.as_array())
            .expect("should be an array");
        assert_eq!(stored.len(), 1, "duplicate semantic_conflict ids");
    }
}
