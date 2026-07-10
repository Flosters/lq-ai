/**
 * Bridges to the open Word document (DE-287).
 *
 * Two capabilities the in-Word surfaces need:
 *
 *   - `getBodyText()` — the document's plain body text, used to detect
 *     "has the document changed since the last snapshot" cheaply.
 *   - `getDocxBase64()` — a full `.docx` snapshot of the open document
 *     (Office compressed-file slices reassembled and base64-encoded),
 *     uploaded to `POST /api/v1/files` so chat can attach it and
 *     playbooks can execute against a real ingested Document.
 *
 * Office.js is loaded by the task pane HTML; these helpers assume the
 * `Word` / `Office` globals exist (tests stub them).
 */

export async function getBodyText(): Promise<string> {
  return Word.run(async (ctx) => {
    const body = ctx.document.body;
    body.load("text");
    await ctx.sync();
    return body.text;
  });
}

/** Snapshot the open document as base64 .docx (Office compressed slices). */
export function getDocxBase64(): Promise<string> {
  return new Promise((resolve, reject) => {
    Office.context.document.getFileAsync(
      Office.FileType.Compressed,
      { sliceSize: 65536 },
      (result) => {
        if (result.status !== Office.AsyncResultStatus.Succeeded) {
          reject(new Error(result.error?.message ?? "getFileAsync failed"));
          return;
        }
        const file = result.value;
        const slices: Uint8Array[] = new Array<Uint8Array>(file.sliceCount);
        let received = 0;
        const readSlice = (i: number): void => {
          file.getSliceAsync(i, (sr) => {
            if (sr.status !== Office.AsyncResultStatus.Succeeded) {
              file.closeAsync(() =>
                reject(new Error("slice read failed while snapshotting document"))
              );
              return;
            }
            const slice = sr.value as { index: number; data: number[] };
            slices[slice.index] = new Uint8Array(slice.data);
            received += 1;
            if (received === file.sliceCount) {
              file.closeAsync(() => resolve(bytesToBase64(concat(slices))));
            } else {
              readSlice(slice.index + 1);
            }
          });
        };
        readSlice(0);
      }
    );
  });
}

function concat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

/** btoa over arbitrary bytes, chunked so the spread doesn't overflow
 *  the argument limit on large documents. */
function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}
