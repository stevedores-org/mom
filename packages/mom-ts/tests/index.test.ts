import { describe, expect, test } from "bun:test";
import { agentScope, defaultScope } from "../src/index";

describe("scope helpers", () => {
  test("defaultScope creates a tenant-only scope", () => {
    expect(defaultScope("tenant-a")).toEqual({
      tenant_id: "tenant-a",
      workspace_id: null,
      project_id: null,
      agent_id: null,
      run_id: null,
    });
  });

  test("agentScope preserves optional project and run scope", () => {
    expect(agentScope("tenant-a", "agent-1", "project-1", "run-1")).toEqual({
      tenant_id: "tenant-a",
      workspace_id: null,
      project_id: "project-1",
      agent_id: "agent-1",
      run_id: "run-1",
    });
  });
});
