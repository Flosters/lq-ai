/**
 * Tests for `src/taskpane/playbookController.ts` — the playbooks tab's
 * state machine (subiendo → ingiriendo → ejecutando → aplicando →
 * done/error). All effects are injected; phase transitions are captured
 * via the subscriber.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createPlaybookController,
  type PlaybookDeps,
} from "../playbookController";
import type { ExecutionResults } from "../lqApi";

const RESULTS: ExecutionResults = {
  schema_version: "m3-a2-v1",
  summary: { deviates: 2, missing: 1 },
  positions: [
    {
      position_id: "p1",
      verdict: "deviates",
      matched_text: "El plazo es de 10 días.",
      justification: "Plazo menor al estándar.",
      redline: {
        old_text: "10 días",
        new_text: "El plazo es de 30 días.",
        justification: "Estándar: 30 días.",
      },
    },
    {
      position_id: "p2",
      verdict: "deviates",
      matched_text: "Sin límite de responsabilidad.",
      justification: "Falta tope de responsabilidad.",
      redline: null,
    },
    {
      position_id: "p3",
      verdict: "missing",
      matched_text: "",
      justification: "No hay cláusula de confidencialidad.",
      redline: null,
    },
    {
      position_id: "p4",
      verdict: "matches_standard",
      matched_text: "La ley aplicable es la argentina.",
      justification: "Coincide.",
      redline: null,
    },
  ],
};

function makeDeps(overrides: Partial<PlaybookDeps> = {}): PlaybookDeps {
  return {
    listPlaybooks: vi
      .fn()
      .mockResolvedValue([{ id: "pb-1", name: "NDA mutuo", contract_type: "nda" }]),
    getDocxBase64: vi.fn().mockResolvedValue(btoa("PK-fake")),
    uploadDocx: vi.fn().mockResolvedValue("file-1"),
    waitIngested: vi.fn().mockResolvedValue("doc-9"),
    executePlaybook: vi.fn().mockResolvedValue("exec-1"),
    pollExecution: vi.fn().mockResolvedValue(RESULTS),
    applyRedlines: vi
      .fn()
      .mockResolvedValue({ applied: 1, commented: 2, missed: 0 }),
    documentName: "nda.docx",
    ...overrides,
  };
}

describe("createPlaybookController", () => {
  let deps: PlaybookDeps;

  beforeEach(() => {
    deps = makeDeps();
  });

  it("loads the playbook list", async () => {
    const controller = createPlaybookController(deps);
    await controller.loadPlaybooks();
    expect(controller.getState().playbooks).toEqual([
      { id: "pb-1", name: "NDA mutuo", contract_type: "nda" },
    ]);
  });

  it("walks the happy-path phases in order and lands on done with a summary", async () => {
    const controller = createPlaybookController(deps);
    const phases: string[] = [];
    controller.subscribe(() => phases.push(controller.getState().phase));

    await controller.run("pb-1");

    expect(phases).toEqual(["subiendo", "ingiriendo", "ejecutando", "aplicando", "done"]);
    expect(deps.uploadDocx).toHaveBeenCalledWith("nda.docx", btoa("PK-fake"));
    expect(deps.waitIngested).toHaveBeenCalledWith("file-1");
    expect(deps.executePlaybook).toHaveBeenCalledWith("pb-1", "doc-9");
    expect(deps.pollExecution).toHaveBeenCalledWith("exec-1");
    expect(controller.getState().summary).toEqual({
      applied: 1,
      commented: 2,
      missed: 0,
      missing: 1,
    });
  });

  it("maps only anchorable deviating positions to redline items", async () => {
    const controller = createPlaybookController(deps);
    await controller.run("pb-1");

    expect(deps.applyRedlines).toHaveBeenCalledWith([
      {
        matchedText: "El plazo es de 10 días.",
        replacement: "El plazo es de 30 días.",
        comment: "Estándar: 30 días.",
      },
      {
        matchedText: "Sin límite de responsabilidad.",
        replacement: null,
        comment: "Falta tope de responsabilidad.",
      },
    ]);
  });

  it("lands on error when the ingest fails, with the phase context", async () => {
    deps = makeDeps({
      waitIngested: vi.fn().mockRejectedValue(new Error("decode_error")),
    });
    const controller = createPlaybookController(deps);

    await controller.run("pb-1");

    const state = controller.getState();
    expect(state.phase).toBe("error");
    expect(state.error).toMatch(/decode_error/);
    expect(deps.executePlaybook).not.toHaveBeenCalled();
  });

  it("lands on error when the execution fails", async () => {
    deps = makeDeps({
      pollExecution: vi.fn().mockRejectedValue(new Error("gateway down")),
    });
    const controller = createPlaybookController(deps);

    await controller.run("pb-1");

    expect(controller.getState().phase).toBe("error");
    expect(controller.getState().error).toMatch(/gateway down/);
    expect(deps.applyRedlines).not.toHaveBeenCalled();
  });

  it("refuses to start a second run while one is in flight", async () => {
    let resolveUpload: (v: string) => void = () => {};
    deps = makeDeps({
      uploadDocx: vi.fn(
        () => new Promise<string>((resolve) => (resolveUpload = resolve))
      ),
    });
    const controller = createPlaybookController(deps);

    const first = controller.run("pb-1");
    await controller.run("pb-1"); // no-op: already running
    resolveUpload("file-1");
    await first;

    expect(deps.uploadDocx).toHaveBeenCalledTimes(1);
  });
});
