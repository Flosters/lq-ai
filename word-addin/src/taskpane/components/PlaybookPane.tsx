/**
 * In-Word playbook execution surface (DE-287). Thin React subscriber
 * over `playbookController`; the pipeline (snapshot → upload → ingest →
 * execute → apply tracked changes) and its state machine live there.
 */
import React, { useEffect, useMemo, useSyncExternalStore } from "react";
import {
  createPlaybookController,
  type PlaybookPhase,
} from "../playbookController";
import { getDocxBase64 } from "../office";
import {
  executePlaybook,
  listPlaybooks,
  pollExecution,
  uploadDocx,
  waitIngested,
} from "../lqApi";
import { applyRedlines } from "../redline";

const PHASE_LABEL: Record<PlaybookPhase, string | null> = {
  idle: null,
  subiendo: "Subiendo copia del documento…",
  ingiriendo: "Ingiriendo el documento (extracción y análisis)…",
  ejecutando: "Ejecutando el playbook contra el documento…",
  aplicando: "Aplicando redlines como cambios rastreados…",
  done: null,
  error: null,
};

function documentName(): string {
  try {
    const url = Office.context.document?.url ?? "";
    const base = url.split(/[\\/]/).pop();
    return base && base.length > 0 ? base : "documento.docx";
  } catch {
    return "documento.docx";
  }
}

export const PlaybookPane: React.FC<{ deploymentOrigin: string }> = ({
  deploymentOrigin,
}) => {
  const controller = useMemo(
    () =>
      createPlaybookController({
        listPlaybooks,
        getDocxBase64,
        uploadDocx,
        waitIngested,
        executePlaybook,
        pollExecution,
        applyRedlines,
        documentName: documentName(),
      }),
    []
  );
  const state = useSyncExternalStore(controller.subscribe, controller.getState);

  useEffect(() => {
    void controller.loadPlaybooks();
  }, [controller]);

  const running = PHASE_LABEL[state.phase] !== null;

  return (
    <section className="lq-playbooks" aria-label="Run a playbook in Word">
      <p className="lq-playbooks-intro">
        Ejecutá un playbook contra el documento abierto: las desviaciones se
        aplican como <strong>cambios rastreados</strong> con el racional como
        comentario — los aceptás o rechazás desde la pestaña Revisar de Word.
      </p>

      {state.loadingPlaybooks && <p className="lq-playbooks-status">Cargando playbooks…</p>}

      <ul className="lq-playbooks-list">
        {state.playbooks.map((playbook) => (
          <li key={playbook.id} className="lq-playbooks-item">
            <div>
              <p className="lq-playbooks-name">{playbook.name}</p>
              <p className="lq-playbooks-type">{playbook.contract_type}</p>
            </div>
            <button
              type="button"
              className="lq-playbooks-run"
              disabled={running}
              onClick={() => void controller.run(playbook.id)}
            >
              Ejecutar contra este documento
            </button>
          </li>
        ))}
        {!state.loadingPlaybooks && state.playbooks.length === 0 && (
          <li className="lq-playbooks-empty">
            No hay playbooks visibles para tu usuario todavía.
          </li>
        )}
      </ul>

      {running && (
        <p className="lq-playbooks-status" role="status">
          {PHASE_LABEL[state.phase]}
        </p>
      )}

      {state.phase === "done" && state.summary && (
        <div className="lq-playbooks-summary" role="status">
          <p>
            {state.summary.applied} redline(s) aplicados como cambios
            rastreados, {state.summary.commented} comentario(s),{" "}
            {state.summary.missed} cláusula(s) no encontrada(s)
            {state.summary.missing > 0 &&
              `, ${state.summary.missing} cláusula(s) faltante(s) en el contrato`}
            .
          </p>
          <p className="lq-playbooks-hint">
            Los cambios quedan en control de cambios de Word: aceptalos o
            rechazalos desde la pestaña Revisar.
          </p>
        </div>
      )}

      {state.phase === "error" && state.error && (
        <p className="lq-chat-error" role="alert">
          {state.error}
        </p>
      )}

      <p className="lq-chat-footer">
        <a
          href={`${deploymentOrigin}/lq-ai/playbooks`}
          target="_blank"
          rel="noopener noreferrer"
        >
          Administrar playbooks en la web app
        </a>
      </p>
    </section>
  );
};
