/**
 * Tests for `src/taskpane/redline.ts` — the tracked-changes applier.
 * `Word.run` is stubbed with a scriptable context whose `body.search`
 * returns 0/1/n ranges per needle; assertions cover the insertText /
 * insertComment calls, the tracking-mode flip, and the returned tally.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { applyRedlines, type RedlineItem } from "../redline";

const g = globalThis as unknown as Record<string, unknown>;

afterEach(() => {
  Reflect.deleteProperty(g, "Word");
  vi.restoreAllMocks();
});

interface FakeRange {
  insertText: ReturnType<typeof vi.fn>;
  insertComment: ReturnType<typeof vi.fn>;
}

function makeRange(): FakeRange {
  return { insertText: vi.fn(), insertComment: vi.fn() };
}

/** Stub Word.run with a context whose search returns the given ranges
 *  keyed by needle text (missing key → no matches). */
function stubWord(matches: Record<string, FakeRange[]>): {
  document: { changeTrackingMode?: string; body: { search: ReturnType<typeof vi.fn> } };
} {
  const searchFn = vi.fn((needle: string) => {
    const items = matches[needle] ?? [];
    return { items, load: vi.fn() };
  });
  const ctx = {
    document: {
      changeTrackingMode: undefined as string | undefined,
      body: { search: searchFn },
    },
    sync: vi.fn().mockResolvedValue(undefined),
  };
  g.Word = {
    run: (fn: (c: typeof ctx) => Promise<unknown>) => fn(ctx),
    ChangeTrackingMode: { trackAll: "TrackAll" },
    InsertLocation: { replace: "Replace" },
  };
  return ctx;
}

describe("applyRedlines", () => {
  it("turns on tracked changes and applies replacement + comment on a match", async () => {
    const range = makeRange();
    const ctx = stubWord({ "El plazo es de 10 días.": [range] });

    const items: RedlineItem[] = [
      {
        matchedText: "El plazo es de 10 días.",
        replacement: "El plazo es de 30 días.",
        comment: "Estándar del playbook: 30 días.",
      },
    ];
    const result = await applyRedlines(items);

    expect(ctx.document.changeTrackingMode).toBe("TrackAll");
    expect(range.insertText).toHaveBeenCalledWith(
      "El plazo es de 30 días.",
      "Replace"
    );
    expect(range.insertComment).toHaveBeenCalledWith(
      "Estándar del playbook: 30 días."
    );
    expect(result).toEqual({ applied: 1, commented: 1, missed: 0 });
  });

  it("comment-only when replacement is null", async () => {
    const range = makeRange();
    stubWord({ "Cláusula X": [range] });

    const result = await applyRedlines([
      { matchedText: "Cláusula X", replacement: null, comment: "Revisar." },
    ]);

    expect(range.insertText).not.toHaveBeenCalled();
    expect(range.insertComment).toHaveBeenCalledWith("Revisar.");
    expect(result).toEqual({ applied: 0, commented: 1, missed: 0 });
  });

  it("tallies a miss when the clause is not found, without throwing", async () => {
    stubWord({});

    const result = await applyRedlines([
      { matchedText: "No existe", replacement: "X", comment: "c" },
    ]);

    expect(result).toEqual({ applied: 0, commented: 0, missed: 1 });
  });

  it("uses only the first match when the needle appears multiple times", async () => {
    const first = makeRange();
    const second = makeRange();
    stubWord({ dup: [first, second] });

    await applyRedlines([{ matchedText: "dup", replacement: "r", comment: "c" }]);

    expect(first.insertText).toHaveBeenCalled();
    expect(second.insertText).not.toHaveBeenCalled();
  });

  it("truncates long clauses to Word's 255-char search cap", async () => {
    const long = "x".repeat(300);
    const range = makeRange();
    const ctx = stubWord({ [long.slice(0, 250)]: [range] });

    const result = await applyRedlines([
      { matchedText: long, replacement: "r", comment: "c" },
    ]);

    expect(ctx.document.body.search).toHaveBeenCalledWith(
      long.slice(0, 250),
      { matchCase: false }
    );
    expect(result.applied).toBe(1);
  });

  it("processes every item and aggregates the tally", async () => {
    const a = makeRange();
    stubWord({ A: [a] });

    const result = await applyRedlines([
      { matchedText: "A", replacement: "A2", comment: "ca" },
      { matchedText: "B", replacement: null, comment: "cb" },
    ]);

    expect(result).toEqual({ applied: 1, commented: 1, missed: 1 });
  });
});
