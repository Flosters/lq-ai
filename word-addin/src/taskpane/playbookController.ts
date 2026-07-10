/**
 * Playbooks-tab state machine (DE-287), extracted from React so it
 * unit-tests hermetically. Phases:
 *
 *   idle → subiendo → ingiriendo → ejecutando → aplicando → done
 *                                                        ↘ error
 *
 * The run snapshots the open document, waits for the ingest pipeline
 * (which needs the DOCX branch — ADR 0017 — live in the deployment),
 * executes the playbook against the ingested Document, maps deviating
 * positions to redline items, and applies them as tracked changes.
 * "missing" positions have no clause to anchor to — they're surfaced
 * in the summary, not silently dropped.
 */
import type { ExecutionResults, PlaybookSummary } from "./lqApi";
import type { ApplyResult, RedlineItem } from "./redline";

export interface PlaybookDeps {
  listPlaybooks: () => Promise<PlaybookSummary[]>;
  getDocxBase64: () => Promise<string>;
  uploadDocx: (name: string, base64: string) => Promise<string>;
  waitIngested: (fileId: string) => Promise<string>;
  executePlaybook: (playbookId: string, documentId: string) => Promise<string>;
  pollExecution: (executionId: string) => Promise<ExecutionResults>;
  applyRedlines: (items: RedlineItem[]) => Promise<ApplyResult>;
  documentName: string;
}

export type PlaybookPhase =
  | "idle"
  | "subiendo"
  | "ingiriendo"
  | "ejecutando"
  | "aplicando"
  | "done"
  | "error";

export interface RunSummary extends ApplyResult {
  /** Posiciones con veredicto "missing" — sin cláusula que anclar. */
  missing: number;
}

export interface PlaybookState {
  playbooks: PlaybookSummary[];
  loadingPlaybooks: boolean;
  phase: PlaybookPhase;
  summary: RunSummary | null;
  error: string | null;
}

export interface PlaybookController {
  getState: () => PlaybookState;
  subscribe: (listener: () => void) => () => void;
  loadPlaybooks: () => Promise<void>;
  run: (playbookId: string) => Promise<void>;
}

/** Deviating positions with a matched clause become redline items;
 *  a missing redline proposal degrades to comment-only. */
export function toRedlineItems(results: ExecutionResults): RedlineItem[] {
  return results.positions
    .filter((p) => p.verdict === "deviates" && p.matched_text.trim() !== "")
    .map((p) => ({
      matchedText: p.matched_text,
      replacement: p.redline ? p.redline.new_text : null,
      comment: p.redline?.justification || p.justification,
    }));
}

export function createPlaybookController(deps: PlaybookDeps): PlaybookController {
  let state: PlaybookState = {
    playbooks: [],
    loadingPlaybooks: false,
    phase: "idle",
    summary: null,
    error: null,
  };
  const listeners = new Set<() => void>();

  function setState(patch: Partial<PlaybookState>): void {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  }

  const RUNNING_PHASES: PlaybookPhase[] = [
    "subiendo",
    "ingiriendo",
    "ejecutando",
    "aplicando",
  ];

  return {
    getState: () => state,

    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    async loadPlaybooks(): Promise<void> {
      setState({ loadingPlaybooks: true });
      try {
        const playbooks = await deps.listPlaybooks();
        setState({ playbooks, loadingPlaybooks: false });
      } catch (error) {
        setState({
          loadingPlaybooks: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },

    async run(playbookId: string): Promise<void> {
      if (RUNNING_PHASES.includes(state.phase)) return;
      setState({ phase: "subiendo", summary: null, error: null });
      try {
        const base64 = await deps.getDocxBase64();
        const fileId = await deps.uploadDocx(deps.documentName, base64);

        setState({ phase: "ingiriendo" });
        const documentId = await deps.waitIngested(fileId);

        setState({ phase: "ejecutando" });
        const executionId = await deps.executePlaybook(playbookId, documentId);
        const results = await deps.pollExecution(executionId);

        setState({ phase: "aplicando" });
        const applyResult = await deps.applyRedlines(toRedlineItems(results));

        const missing = results.positions.filter(
          (p) => p.verdict === "missing"
        ).length;
        setState({ phase: "done", summary: { ...applyResult, missing } });
      } catch (error) {
        setState({
          phase: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
  };
}
