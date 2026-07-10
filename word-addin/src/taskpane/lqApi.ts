/**
 * Client for the shipped lq-ai APIs the in-Word surfaces consume
 * (DE-287): file upload + ingest polling, chats with the ephemeral
 * `file_ids` attach channel, and playbook execution.
 *
 * Every call rides `authenticatedFetch` (bearer + 401-refresh-retry).
 * Paths and body shapes are pinned by docs/api/backend-openapi.yaml;
 * the tests in __tests__/lqApi.test.ts assert them exactly.
 */
import { authenticatedFetch } from "./auth";

export const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export interface PollOptions {
  /** Milliseconds between polls. Default 2000. */
  intervalMs?: number;
  /** Overall budget before giving up. Default 90000. */
  timeoutMs?: number;
}

export interface ChatMessage {
  id: string;
  role: string;
  content: string;
}

export interface PlaybookSummary {
  id: string;
  name: string;
  contract_type: string;
  description?: string;
}

export interface RedlineProposal {
  old_text: string;
  new_text: string;
  justification: string;
}

export interface PositionResult {
  position_id: string;
  issue?: string;
  verdict: "matches_standard" | "matches_fallback" | "deviates" | "missing";
  matched_text: string;
  justification: string;
  redline: RedlineProposal | null;
}

export interface ExecutionResults {
  schema_version: string;
  positions: PositionResult[];
  summary: Record<string, number>;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await authenticatedFetch(path, init);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body && body.detail) detail = `${res.status}: ${JSON.stringify(body.detail)}`;
    } catch {
      // non-JSON error body — keep the bare status
    }
    throw new Error(`lq-ai request failed (${detail})`);
  }
  return (await res.json()) as T;
}

function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(base64);
  const out = new Uint8Array(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Upload a base64 .docx snapshot; returns the new file id. Ingestion
 *  is async — pair with `waitIngested` when a Document is needed. */
export async function uploadDocx(name: string, base64: string): Promise<string> {
  const form = new FormData();
  form.append("file", new File([base64ToBytes(base64)], name, { type: DOCX_MIME }));
  const file = await requestJson<{ id: string }>("/files", {
    method: "POST",
    body: form,
  });
  return file.id;
}

interface FileStatusWire {
  id: string;
  ingestion_status: "pending" | "processing" | "ready" | "failed";
  ingestion_error?: string | null;
  document_id?: string | null;
}

/** Poll the file until the ingest pipeline flips it to `ready`; returns
 *  the parsed Document's id. Throws on `failed` or when the budget is
 *  exhausted (a stuck `processing` row should surface, not spin). */
export async function waitIngested(
  fileId: string,
  options: PollOptions = {}
): Promise<string> {
  const intervalMs = options.intervalMs ?? 2000;
  const timeoutMs = options.timeoutMs ?? 90000;
  const deadline = Date.now() + timeoutMs;

  for (;;) {
    const status = await requestJson<FileStatusWire>(`/files/${fileId}`);
    if (status.ingestion_status === "ready") {
      if (!status.document_id) {
        throw new Error("el archivo está ready pero no tiene documento asociado");
      }
      return status.document_id;
    }
    if (status.ingestion_status === "failed") {
      throw new Error(
        `la ingesta del documento falló (${status.ingestion_error ?? "sin detalle"})`
      );
    }
    if (Date.now() + intervalMs > deadline) {
      throw new Error("la ingesta del documento tardó demasiado (timed out)");
    }
    await sleep(intervalMs);
  }
}

/** Create a chat; returns its id. */
export async function createChat(title?: string): Promise<string> {
  const chat = await requestJson<{ id: string }>("/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(title ? { title } : {}),
  });
  return chat.id;
}

/** Post a user message (non-streaming) and return the assistant
 *  message. `fileIds` rides the ephemeral per-turn attach channel. */
export async function sendMessage(
  chatId: string,
  content: string,
  fileIds?: string[]
): Promise<ChatMessage> {
  const body: Record<string, unknown> = { content, stream: false };
  if (fileIds && fileIds.length > 0) body.file_ids = fileIds;
  const response = await requestJson<{ message: ChatMessage }>(
    `/chats/${chatId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  return response.message;
}

/** Playbooks visible to the caller (positions not inlined). */
export async function listPlaybooks(): Promise<PlaybookSummary[]> {
  return requestJson<PlaybookSummary[]>("/playbooks");
}

/** Kick off an execution against an ingested Document; returns the
 *  execution id to poll. */
export async function executePlaybook(
  playbookId: string,
  targetDocumentId: string
): Promise<string> {
  const execution = await requestJson<{ id: string }>(
    `/playbooks/${playbookId}/execute`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_document_id: targetDocumentId }),
    }
  );
  return execution.id;
}

interface ExecutionWire {
  id: string;
  status: "pending" | "running" | "completed" | "error";
  results: ExecutionResults | null;
  error?: string | null;
}

/** Poll an execution to its terminal state; returns the results on
 *  `completed`, throws on `error` or when the budget is exhausted. */
export async function pollExecution(
  executionId: string,
  options: PollOptions = {}
): Promise<ExecutionResults> {
  const intervalMs = options.intervalMs ?? 2000;
  const timeoutMs = options.timeoutMs ?? 180000;
  const deadline = Date.now() + timeoutMs;

  for (;;) {
    const execution = await requestJson<ExecutionWire>(
      `/playbook-executions/${executionId}`
    );
    if (execution.status === "completed") {
      if (!execution.results) {
        throw new Error("la ejecución terminó sin resultados");
      }
      return execution.results;
    }
    if (execution.status === "error") {
      throw new Error(
        `la ejecución del playbook falló (${execution.error ?? "sin detalle"})`
      );
    }
    if (Date.now() + intervalMs > deadline) {
      throw new Error("la ejecución del playbook tardó demasiado (timed out)");
    }
    await sleep(intervalMs);
  }
}
