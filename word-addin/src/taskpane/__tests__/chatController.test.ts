/**
 * Tests for `src/taskpane/chatController.ts` — the chat surface's logic,
 * extracted from React so it tests hermetically (no RTL in the devDeps;
 * the component is a thin subscriber). All effects are injected.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createChatController, type ChatDeps } from "../chatController";

function makeDeps(overrides: Partial<ChatDeps> = {}): ChatDeps {
  return {
    getBodyText: vi.fn().mockResolvedValue("Cláusula 1. Objeto."),
    getDocxBase64: vi.fn().mockResolvedValue(btoa("PK-fake")),
    uploadDocx: vi.fn().mockResolvedValue("file-1"),
    createChat: vi.fn().mockResolvedValue("chat-1"),
    sendMessage: vi
      .fn()
      .mockResolvedValue({ id: "m", role: "assistant", content: "Respuesta." }),
    documentName: "contrato.docx",
    ...overrides,
  };
}

describe("createChatController", () => {
  let deps: ChatDeps;

  beforeEach(() => {
    deps = makeDeps();
  });

  it("starts with attach ON, no messages, not sending", () => {
    const chat = createChatController(deps);
    expect(chat.getState()).toMatchObject({
      messages: [],
      sending: false,
      attachDocument: true,
      error: null,
    });
  });

  it("lazily creates the chat on first send only", async () => {
    const chat = createChatController(deps);
    await chat.send("hola");
    await chat.send("otra");
    expect(deps.createChat).toHaveBeenCalledTimes(1);
    expect(deps.createChat).toHaveBeenCalledWith("Word: contrato.docx");
  });

  it("snapshots + uploads the document and attaches file_ids on first send", async () => {
    const chat = createChatController(deps);
    await chat.send("resumí las obligaciones");

    expect(deps.getDocxBase64).toHaveBeenCalledTimes(1);
    expect(deps.uploadDocx).toHaveBeenCalledTimes(1);
    expect(deps.sendMessage).toHaveBeenCalledWith(
      "chat-1",
      "resumí las obligaciones",
      ["file-1"]
    );
    const state = chat.getState();
    expect(state.messages).toEqual([
      { role: "user", content: "resumí las obligaciones" },
      { role: "assistant", content: "Respuesta." },
    ]);
    expect(state.sending).toBe(false);
  });

  it("skips re-upload when the body text hash is unchanged, still attaching the cached file", async () => {
    const chat = createChatController(deps);
    await chat.send("uno");
    await chat.send("dos");

    expect(deps.uploadDocx).toHaveBeenCalledTimes(1);
    expect(deps.sendMessage).toHaveBeenLastCalledWith("chat-1", "dos", ["file-1"]);
  });

  it("re-uploads when the document changed between sends", async () => {
    const bodies = ["v1", "v2"];
    deps = makeDeps({
      getBodyText: vi.fn(() => Promise.resolve(bodies.shift() ?? "v2")),
      uploadDocx: vi
        .fn()
        .mockResolvedValueOnce("file-1")
        .mockResolvedValueOnce("file-2"),
    });
    const chat = createChatController(deps);
    await chat.send("uno");
    await chat.send("dos");

    expect(deps.uploadDocx).toHaveBeenCalledTimes(2);
    expect(deps.sendMessage).toHaveBeenLastCalledWith("chat-1", "dos", ["file-2"]);
  });

  it("attaches nothing when the toggle is off", async () => {
    const chat = createChatController(deps);
    chat.setAttachDocument(false);
    await chat.send("hola");

    expect(deps.getDocxBase64).not.toHaveBeenCalled();
    expect(deps.uploadDocx).not.toHaveBeenCalled();
    expect(deps.sendMessage).toHaveBeenCalledWith("chat-1", "hola", undefined);
  });

  it("surfaces errors inline and keeps the user's message", async () => {
    deps = makeDeps({
      sendMessage: vi.fn().mockRejectedValue(new Error("gateway unreachable")),
    });
    const chat = createChatController(deps);
    await chat.send("hola");

    const state = chat.getState();
    expect(state.error).toMatch(/gateway unreachable/);
    expect(state.messages).toEqual([{ role: "user", content: "hola" }]);
    expect(state.sending).toBe(false);
  });

  it("notifies subscribers on every state change", async () => {
    const chat = createChatController(deps);
    const listener = vi.fn();
    chat.subscribe(listener);
    await chat.send("hola");
    expect(listener).toHaveBeenCalled();
  });

  it("ignores blank input", async () => {
    const chat = createChatController(deps);
    await chat.send("   ");
    expect(deps.createChat).not.toHaveBeenCalled();
    expect(chat.getState().messages).toEqual([]);
  });
});
