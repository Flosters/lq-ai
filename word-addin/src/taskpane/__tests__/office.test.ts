/**
 * Tests for `src/taskpane/office.ts` — the bridge to the open Word
 * document. Office.js is not loaded in tests: `Word` and `Office` are
 * stubbed as globals per test, mirroring how auth.test.ts stubs fetch.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { getBodyText, getDocxBase64 } from "../office";

// office-js ambient types declare `Word`/`Office` as non-optional
// globals, so stubbing goes through an untyped view of globalThis.
const g = globalThis as unknown as Record<string, unknown>;

afterEach(() => {
  Reflect.deleteProperty(g, "Word");
  Reflect.deleteProperty(g, "Office");
  vi.restoreAllMocks();
});

describe("getBodyText", () => {
  it("loads and returns the document body text via Word.run", async () => {
    const body = { text: "Cláusula 1. Objeto.", load: vi.fn() };
    const ctx = { document: { body }, sync: vi.fn().mockResolvedValue(undefined) };
    g.Word = { run: (fn: (c: typeof ctx) => Promise<string>) => fn(ctx) };

    const text = await getBodyText();

    expect(text).toBe("Cláusula 1. Objeto.");
    expect(body.load).toHaveBeenCalledWith("text");
    expect(ctx.sync).toHaveBeenCalled();
  });
});

describe("getDocxBase64", () => {
  const SUCCEEDED = "succeeded";
  const FAILED = "failed";

  function stubOffice(opts: {
    slices: number[][];
    failSlice?: number;
    failFile?: boolean;
  }): { closeAsync: ReturnType<typeof vi.fn> } {
    const closeAsync = vi.fn((cb?: () => void) => cb?.());
    const file = {
      sliceCount: opts.slices.length,
      getSliceAsync: vi.fn((index: number, cb: (r: unknown) => void) => {
        if (index === opts.failSlice) {
          cb({ status: FAILED, error: { message: "slice boom" } });
          return;
        }
        cb({
          status: SUCCEEDED,
          value: { index, data: opts.slices[index] },
        });
      }),
      closeAsync,
    };
    g.Office = {
      FileType: { Compressed: "compressed" },
      AsyncResultStatus: { Succeeded: SUCCEEDED, Failed: FAILED },
      context: {
        document: {
          getFileAsync: vi.fn(
            (
              _type: string,
              _opts: unknown,
              cb: (r: unknown) => void
            ) => {
              if (opts.failFile) {
                cb({ status: FAILED, error: { message: "file boom" } });
                return;
              }
              cb({ status: SUCCEEDED, value: file });
            }
          ),
        },
      },
    };
    return { closeAsync };
  }

  it("concatenates slices in order and base64-encodes them", async () => {
    // "PK\x03\x04" split across two slices — order matters.
    const { closeAsync } = stubOffice({ slices: [[0x50, 0x4b], [0x03, 0x04]] });

    const b64 = await getDocxBase64();

    expect(b64).toBe(btoa("PK\x03\x04"));
    expect(closeAsync).toHaveBeenCalled();
  });

  it("closes the file and rejects when a slice read fails", async () => {
    const { closeAsync } = stubOffice({
      slices: [[0x50], [0x4b]],
      failSlice: 1,
    });

    await expect(getDocxBase64()).rejects.toThrow(/slice/i);
    expect(closeAsync).toHaveBeenCalled();
  });

  it("rejects when getFileAsync itself fails", async () => {
    stubOffice({ slices: [[0x50]], failFile: true });

    await expect(getDocxBase64()).rejects.toThrow("file boom");
  });
});
