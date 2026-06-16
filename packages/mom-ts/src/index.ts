/**
 * MOM TypeScript Client
 * Works in Bun and Node.js using standard fetch API
 */

export type ScopeKey = {
  tenant_id: string;
  workspace_id?: string | null;
  project_id?: string | null;
  agent_id?: string | null;
  run_id?: string | null;
};

export type MemoryKind = "Event" | "Summary" | "Fact" | "Preference";

export type Content =
  | { Text: string }
  | { Json: Record<string, any> }
  | { TextJson: { text: string; json: Record<string, any> } };

export interface MemoryItem {
  id: string;
  scope: ScopeKey;
  kind: MemoryKind;
  created_at_ms: number;
  content: string | Record<string, any>;
  tags: string[];
  importance: number;
  confidence: number;
  source: string;
  ttl_ms?: number | null;
  meta: Record<string, any>;
}

export interface Query {
  scope: ScopeKey;
  text: string;
  kinds?: MemoryKind[] | null;
  tags_any?: string[] | null;
  limit: number;
  since_ms?: number | null;
  until_ms?: number | null;
}

export interface Scored<T> {
  score: number;
  item: T;
}

export interface Citation {
  memory_id: string;
  source: string;
  kind: MemoryKind;
  created_at_ms: number;
  score: number;
  preview: string;
}

export interface ContextPack {
  highlights: Scored<MemoryItem>[];
  summaries: Scored<MemoryItem>[];
  facts: Scored<MemoryItem>[];
  citations: Citation[];
  estimated_tokens: number;
  budget_tokens: number;
}

export interface ContextPackRequest {
  query: Query;
  budget_tokens?: number;
}

// US-10: structured payloads for Fact / Preference items. These live under
// well-known keys in `MemoryItem.meta` rather than as first-class fields, so
// the wire format stays backward-compatible with pre-US-10 clients. Use
// `MomClient.writeFact(...)` / `writePreference(...)` to construct items
// without having to remember the meta-key convention by hand.

/** Structured (subject, predicate, object) triple stored under `meta.fact`. */
export interface FactPayload {
  subject: string;
  predicate: string;
  object: string;
}

/** Structured rule/decision/priority/conditions payload stored under `meta.preference`. */
export interface PreferencePayload {
  rule: string;
  decision: string;
  /** 0..u32::MAX. Higher = stronger preference. */
  priority: number;
  /** Free-form per-rule predicates; serialized as a JSON array under meta.preference.conditions. */
  conditions?: unknown[];
}

/** Optional fields a caller can attach when writing a Fact / Preference. */
export interface KnowledgeWriteOptions {
  /** Sub-tenant identifier scope (defaults: workspace=undefined, etc.). */
  scope: ScopeKey;
  /** Override the auto-generated id. */
  id?: string;
  /** 0..1, defaults to 1.0 on the server when omitted. */
  confidence?: number;
  /** 0..1, defaults to 0.5 on the server when omitted. */
  importance?: number;
  /** Free-form classification tags. */
  tags?: string[];
  /** TTL in milliseconds; absent ⇒ no expiry. */
  ttl_ms?: number;
  /** Upstream memory ids this Fact / Preference is derived from. */
  provenance_ids?: string[];
  /** Provenance source label (matches the Rust `MemoryItem.source` field). */
  source?: string;
}

export interface MomClientOptions {
  baseUrl: string;
  headers?: Record<string, string>;
}

export class MomClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(options: MomClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.headers = {
      "content-type": "application/json",
      ...options.headers,
    };
  }

  async write(item: MemoryItem): Promise<MemoryItem> {
    const res = await fetch(`${this.baseUrl}/v1/memory`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(item),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  /**
   * US-10: write a structured Fact. The payload lives under `meta.fact`;
   * the server validates the triple at write-time and applies exact-key +
   * semantic conflict detection against existing active facts in scope.
   *
   * The returned item carries server-side `meta.version`, `meta.provenance_ids`,
   * and (on contradiction) `meta.semantic_conflicts`.
   */
  async writeFact(
    payload: FactPayload,
    options: KnowledgeWriteOptions,
  ): Promise<MemoryItem> {
    const meta: Record<string, unknown> = {
      fact: payload,
      provenance_ids: options.provenance_ids ?? [],
    };
    return this.write(buildKnowledgeItem("Fact", meta, options));
  }

  /**
   * US-10: write a structured Preference. The payload lives under
   * `meta.preference`.
   */
  async writePreference(
    payload: PreferencePayload,
    options: KnowledgeWriteOptions,
  ): Promise<MemoryItem> {
    const meta: Record<string, unknown> = {
      preference: {
        rule: payload.rule,
        decision: payload.decision,
        priority: payload.priority,
        conditions: payload.conditions ?? [],
      },
      provenance_ids: options.provenance_ids ?? [],
    };
    return this.write(buildKnowledgeItem("Preference", meta, options));
  }

  async get(id: string): Promise<MemoryItem | null> {
    const res = await fetch(`${this.baseUrl}/v1/memory/${id}`, {
      method: "GET",
      headers: this.headers,
    });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async list(params?: {
    tenant_id?: string;
    workspace_id?: string;
    project_id?: string;
    agent_id?: string;
    limit?: number;
    cursor?: string;
  }): Promise<{ items: MemoryItem[]; next_cursor: string | null }> {
    const url = new URL(`${this.baseUrl}/v1/memory`);
    if (params?.tenant_id) url.searchParams.append("tenant_id", params.tenant_id);
    if (params?.workspace_id)
      url.searchParams.append("workspace_id", params.workspace_id);
    if (params?.project_id)
      url.searchParams.append("project_id", params.project_id);
    if (params?.agent_id) url.searchParams.append("agent_id", params.agent_id);
    if (params?.limit) url.searchParams.append("limit", String(params.limit));
    if (params?.cursor) url.searchParams.append("cursor", params.cursor);

    const res = await fetch(url.toString(), {
      method: "GET",
      headers: this.headers,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async delete(id: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/v1/memory/${id}`, {
      method: "DELETE",
      headers: this.headers,
    });
    if (!res.ok) throw new Error(await res.text());
  }

  async recall(q: Query): Promise<Scored<MemoryItem>[]> {
    const res = await fetch(`${this.baseUrl}/v1/recall`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(q),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  /** @deprecated Use contextPack() for structured highlights/summaries/facts bundles. */
  async recallContext(query: string, scope: ScopeKey, budgetTokens?: number): Promise<Scored<MemoryItem>[]> {
    return this.recall({
      scope,
      text: query,
      limit: budgetTokens ? Math.ceil(budgetTokens / 150) : 10,
    });
  }

  async contextPack(req: ContextPackRequest): Promise<ContextPack> {
    const res = await fetch(`${this.baseUrl}/v1/context-pack`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
}

/**
 * Build a kind-tagged `MemoryItem` from the shared `KnowledgeWriteOptions`.
 * Internal — exported only because the test suite likes to assert structure
 * without exercising the network.
 */
export function buildKnowledgeItem(
  kind: "Fact" | "Preference",
  meta: Record<string, unknown>,
  options: KnowledgeWriteOptions,
): MemoryItem {
  return {
    id: options.id ?? "",
    scope: options.scope,
    kind,
    created_at_ms: 0, // server stamps current time when 0
    content: { Text: "" } as unknown as MemoryItem["content"],
    tags: options.tags ?? [],
    importance: options.importance ?? 0.5,
    confidence: options.confidence ?? 1.0,
    source: options.source ?? "agent",
    ttl_ms: options.ttl_ms ?? null,
    meta,
  };
}

// Helper to create a default tenant scope
export function defaultScope(tenantId: string = "default"): ScopeKey {
  return {
    tenant_id: tenantId,
    workspace_id: null,
    project_id: null,
    agent_id: null,
    run_id: null,
  };
}

// Helper to create agent-scoped memories
export function agentScope(
  tenantId: string,
  agentId: string,
  projectId?: string,
  runId?: string
): ScopeKey {
  return {
    tenant_id: tenantId,
    workspace_id: null,
    project_id: projectId || null,
    agent_id: agentId,
    run_id: runId || null,
  };
}
