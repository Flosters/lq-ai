/**
 * In-Word chat surface (DE-287). Thin React subscriber over
 * `chatController` — the logic (lazy chat creation, snapshot attach,
 * hash-skip re-upload, inline errors) lives there and is unit-tested.
 */
import React, { useMemo, useRef, useState, useSyncExternalStore } from "react";
import { createChatController } from "../chatController";
import { getBodyText, getDocxBase64 } from "../office";
import { createChat, sendMessage, uploadDocx } from "../lqApi";

function documentName(): string {
  try {
    const url = Office.context.document?.url ?? "";
    const base = url.split(/[\\/]/).pop();
    return base && base.length > 0 ? base : "documento.docx";
  } catch {
    return "documento.docx";
  }
}

export const ChatPane: React.FC<{ deploymentOrigin: string }> = ({
  deploymentOrigin,
}) => {
  const controller = useMemo(
    () =>
      createChatController({
        getBodyText,
        getDocxBase64,
        uploadDocx,
        createChat,
        sendMessage,
        documentName: documentName(),
      }),
    []
  );
  const state = useSyncExternalStore(controller.subscribe, controller.getState);
  const [draft, setDraft] = useState("");
  const logRef = useRef<HTMLDivElement>(null);

  async function handleSend(): Promise<void> {
    const text = draft;
    setDraft("");
    await controller.send(text);
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }

  return (
    <section className="lq-chat" aria-label="Chat with the open document">
      <div className="lq-chat-log" ref={logRef} role="log" aria-live="polite">
        {state.messages.length === 0 && (
          <p className="lq-chat-empty">
            Preguntale a LQ.AI sobre el documento abierto — con el adjunto
            activado, cada consulta viaja con una copia del documento y la
            respuesta puede citarlo.
          </p>
        )}
        {state.messages.map((message, index) => (
          <div
            key={index}
            className={`lq-chat-bubble lq-chat-bubble-${message.role}`}
          >
            {message.content}
          </div>
        ))}
        {state.sending && (
          <div className="lq-chat-bubble lq-chat-bubble-assistant lq-chat-pending">
            …
          </div>
        )}
      </div>

      {state.error && (
        <p className="lq-chat-error" role="alert">
          {state.error}
        </p>
      )}

      <label className="lq-chat-attach">
        <input
          type="checkbox"
          checked={state.attachDocument}
          onChange={(event) => controller.setAttachDocument(event.target.checked)}
        />
        📎 Incluir documento abierto
      </label>

      <div className="lq-chat-composer">
        <textarea
          className="lq-chat-input"
          value={draft}
          rows={2}
          placeholder="Escribí tu consulta…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void handleSend();
            }
          }}
        />
        <button
          type="button"
          className="lq-chat-send"
          disabled={state.sending || draft.trim() === ""}
          onClick={() => void handleSend()}
        >
          Enviar
        </button>
      </div>

      <p className="lq-chat-footer">
        <a
          href={`${deploymentOrigin}/lq-ai`}
          target="_blank"
          rel="noopener noreferrer"
        >
          Abrir esta conversación en la web app
        </a>
      </p>
    </section>
  );
};
