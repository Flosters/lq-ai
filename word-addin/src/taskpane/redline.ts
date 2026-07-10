/**
 * Tracked-changes redline applier (DE-287).
 *
 * Maps a playbook execution's positions onto the open document: each
 * matched clause is located via `body.search`, the proposed language is
 * inserted as a tracked change (Word's Review pane accepts/rejects it),
 * and the position's rationale lands as a Word comment on the range.
 *
 * Honesty over pretending: Word's search caps needles at 255 chars and
 * OOXML quirks make some clause matches miss — a missed clause is
 * tallied and reported, never silently dropped. Requires WordApi 1.4
 * (`insertComment`, `changeTrackingMode`) — declared in manifest.xml.
 */

export interface RedlineItem {
  /** Texto de la cláusula detectada (matched_text del playbook). */
  matchedText: string;
  /** Redacción propuesta; null → sólo comentario. */
  replacement: string | null;
  /** Racional de la posición del playbook. */
  comment: string;
}

export interface ApplyResult {
  applied: number;
  commented: number;
  missed: number;
}

export async function applyRedlines(items: RedlineItem[]): Promise<ApplyResult> {
  return Word.run(async (ctx) => {
    ctx.document.changeTrackingMode = Word.ChangeTrackingMode.trackAll;
    const result: ApplyResult = { applied: 0, commented: 0, missed: 0 };

    for (const item of items) {
      // Word search caps at 255 chars — search the head of long clauses.
      const needle = item.matchedText.slice(0, 250);
      const ranges = ctx.document.body.search(needle, { matchCase: false });
      ranges.load("items");
      await ctx.sync();

      const range = ranges.items[0];
      if (!range) {
        result.missed += 1;
        continue;
      }
      if (item.replacement) {
        range.insertText(item.replacement, Word.InsertLocation.replace);
        result.applied += 1;
      }
      range.insertComment(item.comment);
      result.commented += 1;
      await ctx.sync();
    }

    return result;
  });
}
