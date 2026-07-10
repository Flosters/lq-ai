/**
 * Tests for `src/taskpane/lqApi.ts` — the add-in's client for the
 * already-shipped lq-ai APIs (files, chats + file_ids, playbooks).
 * `authenticatedFetch` is mocked; assertions pin the exact paths,
 * methods, and body shapes the backend documents in
 * docs/api/backend-openapi.yaml.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../auth", () => ({
  authenticatedFetch: vi.fn(),
}));

import { authenticatedFetch } from "../auth";
import {
  createChat,
  executePlaybook,
  listPlaybooks,
  pollExecution,
  sendMessage,
  uploadDocx,
  waitIngested,
} from "../lqApi";

const mockedFetch = vi.mocked(authenticatedFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  mockedFetch.mockReset();
});

describe("uploadDocx", () => {
  it("POSTs multipart to /files and returns the file id", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({ id: "file-1", ingestion_status: "pending" }, 201)
    );

    const fileId = await uploadDocx("contrato.docx", btoa("PK\x03\x04fake"));

    expect(fileId).toBe("file-1");
    const [path, init] = mockedFetch.mock.calls[0];
    expect(path).toBe("/files");
    expect(init?.method).toBe("POST");
    const form = init?.body as FormData;
    const part = form.get("file") as File;
    expect(part.name).toBe("contrato.docx");
    expect(part.type).toBe(
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    );
    // jsdom's File doesn't implement arrayBuffer(); the size pins the
    // decoded byte count ("PK\x03\x04fake" = 8 bytes).
    expect(part.size).toBe(8);
  });

  it("throws with the backend detail on failure", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({ detail: "payload_too_large" }, 413)
    );
    await expect(uploadDocx("x.docx", btoa("y"))).rejects.toThrow(
      /payload_too_large/
    );
  });
});

describe("waitIngested", () => {
  it("polls GET /files/{id} until ready and returns document_id", async () => {
    mockedFetch
      .mockResolvedValueOnce(
        jsonResponse({ id: "f", ingestion_status: "processing", document_id: null })
      )
      .mockResolvedValueOnce(
        jsonResponse({ id: "f", ingestion_status: "ready", document_id: "doc-9" })
      );

    const documentId = await waitIngested("f", { intervalMs: 1, timeoutMs: 5000 });

    expect(documentId).toBe("doc-9");
    expect(mockedFetch).toHaveBeenCalledTimes(2);
    expect(mockedFetch.mock.calls[0][0]).toBe("/files/f");
  });

  it("throws on ingestion_status=failed with the ingestion_error", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({
        id: "f",
        ingestion_status: "failed",
        ingestion_error: "unsupported_type",
        document_id: null,
      })
    );

    await expect(
      waitIngested("f", { intervalMs: 1, timeoutMs: 5000 })
    ).rejects.toThrow(/unsupported_type/);
    expect(mockedFetch).toHaveBeenCalledTimes(1);
  });

  it("gives up after the time budget", async () => {
    // Fresh Response per poll — a Response body is single-use.
    mockedFetch.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({ id: "f", ingestion_status: "processing", document_id: null })
      )
    );

    await expect(
      waitIngested("f", { intervalMs: 1, timeoutMs: 5 })
    ).rejects.toThrow(/tardó demasiado|timed out/i);
  });
});

describe("createChat / sendMessage", () => {
  it("creates a chat and returns its id", async () => {
    mockedFetch.mockResolvedValue(jsonResponse({ id: "chat-1" }, 201));

    const chatId = await createChat("Word: contrato.docx");

    expect(chatId).toBe("chat-1");
    const [path, init] = mockedFetch.mock.calls[0];
    expect(path).toBe("/chats");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      title: "Word: contrato.docx",
    });
  });

  it("sends a message with the ephemeral file_ids channel", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({
        message: { id: "m2", role: "assistant", content: "Resumen…" },
        citations: [],
      })
    );

    const reply = await sendMessage("chat-1", "resumí", ["file-1"]);

    expect(reply.content).toBe("Resumen…");
    const [path, init] = mockedFetch.mock.calls[0];
    expect(path).toBe("/chats/chat-1/messages");
    expect(JSON.parse(init?.body as string)).toEqual({
      content: "resumí",
      stream: false,
      file_ids: ["file-1"],
    });
  });

  it("omits file_ids entirely when not attaching", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({ message: { id: "m", role: "assistant", content: "ok" }, citations: [] })
    );

    await sendMessage("chat-1", "hola");

    expect(JSON.parse(mockedFetch.mock.calls[0][1]?.body as string)).toEqual({
      content: "hola",
      stream: false,
    });
  });
});

describe("playbooks", () => {
  it("lists playbooks", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse([{ id: "pb-1", name: "NDA mutuo", contract_type: "nda" }])
    );

    const playbooks = await listPlaybooks();

    expect(playbooks).toHaveLength(1);
    expect(playbooks[0].name).toBe("NDA mutuo");
    expect(mockedFetch.mock.calls[0][0]).toBe("/playbooks");
  });

  it("executes a playbook against a document and returns the execution id", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({ id: "exec-1", status: "pending" }, 202)
    );

    const executionId = await executePlaybook("pb-1", "doc-9");

    expect(executionId).toBe("exec-1");
    const [path, init] = mockedFetch.mock.calls[0];
    expect(path).toBe("/playbooks/pb-1/execute");
    expect(JSON.parse(init?.body as string)).toEqual({
      target_document_id: "doc-9",
    });
  });

  it("polls the execution until completed and returns the results", async () => {
    mockedFetch
      .mockResolvedValueOnce(jsonResponse({ id: "e", status: "running", results: null }))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "e",
          status: "completed",
          results: {
            schema_version: "m3-a2-v1",
            positions: [
              {
                position_id: "p1",
                verdict: "deviates",
                matched_text: "El plazo es de 10 días.",
                justification: "Plazo menor al estándar.",
                redline: {
                  old_text: "10 días",
                  new_text: "30 días",
                  justification: "Estándar del playbook.",
                },
              },
            ],
            summary: { deviates: 1 },
          },
        })
      );

    const results = await pollExecution("e", { intervalMs: 1, timeoutMs: 5000 });

    expect(results.positions).toHaveLength(1);
    expect(results.positions[0].redline?.new_text).toBe("30 días");
    expect(mockedFetch.mock.calls[0][0]).toBe("/playbook-executions/e");
  });

  it("throws when the execution errors", async () => {
    mockedFetch.mockResolvedValue(
      jsonResponse({ id: "e", status: "error", error: "document unreadable", results: null })
    );

    await expect(
      pollExecution("e", { intervalMs: 1, timeoutMs: 5000 })
    ).rejects.toThrow(/document unreadable/);
  });
});
