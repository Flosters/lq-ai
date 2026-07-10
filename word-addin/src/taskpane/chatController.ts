/**
 * Chat-surface logic for the in-Word chat tab (DE-287), extracted from
 * React so it unit-tests hermetically. The component subscribes via
 * `subscribe`/`getState` (useSyncExternalStore-compatible).
 *
 * Behavior:
 *  - The lq-ai chat is created lazily on the first send.
 *  - With the attach toggle ON (default), each send snapshots the open
 *    document (.docx → upload → `file_ids` ephemeral attach). A cheap
 *    body-text hash skips the re-upload when the document hasn't
 *    changed since the last snapshot — the cached file id is reused.
 *  - Errors surface inline on the state; the user's message stays.
 */
import type { ChatMessage } from "./lqApi";

export interface ChatDeps {
  getBodyText: () => Promise<string>;
  getDocxBase64: () => Promise<string>;
  uploadDocx: (name: string, base64: string) => Promise<string>;
  createChat: (title?: string) => Promise<string>;
  sendMessage: (
    chatId: string,
    content: string,
    fileIds?: string[]
  ) => Promise<ChatMessage>;
  /** Shown in the auto-generated chat title. */
  documentName: string;
}

export interface ChatBubble {
  role: "user" | "assistant";
  content: string;
}

export interface ChatState {
  messages: ChatBubble[];
  sending: boolean;
  attachDocument: boolean;
  error: string | null;
}

export interface ChatController {
  getState: () => ChatState;
  subscribe: (listener: () => void) => () => void;
  setAttachDocument: (on: boolean) => void;
  send: (text: string) => Promise<void>;
}

/** FNV-1a over the body text — collision-resistance needs are trivial
 *  (only "did the document change since the last snapshot"). */
function hashText(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16);
}

export function createChatController(deps: ChatDeps): ChatController {
  let state: ChatState = {
    messages: [],
    sending: false,
    attachDocument: true,
    error: null,
  };
  const listeners = new Set<() => void>();
  let chatId: string | null = null;
  let uploadedFileId: string | null = null;
  let uploadedHash: string | null = null;

  function setState(patch: Partial<ChatState>): void {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  }

  async function ensureSnapshotAttached(): Promise<string[]> {
    const bodyHash = hashText(await deps.getBodyText());
    if (uploadedFileId === null || bodyHash !== uploadedHash) {
      const base64 = await deps.getDocxBase64();
      uploadedFileId = await deps.uploadDocx(deps.documentName, base64);
      uploadedHash = bodyHash;
    }
    return [uploadedFileId];
  }

  return {
    getState: () => state,

    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    setAttachDocument(on: boolean): void {
      setState({ attachDocument: on });
    },

    async send(text: string): Promise<void> {
      const content = text.trim();
      if (!content || state.sending) return;

      setState({
        messages: [...state.messages, { role: "user", content }],
        sending: true,
        error: null,
      });
      try {
        if (chatId === null) {
          chatId = await deps.createChat(`Word: ${deps.documentName}`);
        }
        const fileIds = state.attachDocument
          ? await ensureSnapshotAttached()
          : undefined;
        const reply = await deps.sendMessage(chatId, content, fileIds);
        setState({
          messages: [
            ...state.messages,
            { role: "assistant", content: reply.content },
          ],
          sending: false,
        });
      } catch (error) {
        setState({
          sending: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
  };
}
