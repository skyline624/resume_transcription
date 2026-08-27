import { describe, expect, it, vi } from "vitest";

import { ApiFailure, HttpClient } from "./http";

function response(body: BodyInit | null, init: ResponseInit): Response {
  return new Response(body, init);
}

describe("HttpClient", () => {
  it("invokes fetch with the browser global as its receiver", async () => {
    const receiverAwareFetch = function (this: unknown): Promise<Response> {
      if (this !== globalThis) throw new TypeError("Illegal invocation");
      return Promise.resolve(
        response(JSON.stringify({ text: "bonjour" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    } as typeof fetch;
    const client = new HttpClient("", receiverAwareFetch);

    await expect(client.getJson("/health")).resolves.toEqual({ text: "bonjour" });
  });

  it("normalizes an OpenAI error envelope", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      response(
        JSON.stringify({
          error: {
            message: "The requested voice is unavailable",
            code: "voice_not_found",
            param: "voice",
          },
        }),
        {
          status: 404,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    const client = new HttpClient("", fetcher);

    const request = client.getJson("/v1/voices");

    await expect(request).rejects.toBeInstanceOf(ApiFailure);
    await expect(request).rejects.toMatchObject({
      message: "The requested voice is unavailable",
      status: 404,
      code: "voice_not_found",
      param: "voice",
    });
  });

  it("normalizes a FastAPI detail response", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      response(JSON.stringify({ detail: "Audio file is required" }), {
        status: 422,
        headers: { "content-type": "application/json" },
      }),
    );
    const client = new HttpClient("", fetcher);

    await expect(client.postJson("/summarize", {})).rejects.toMatchObject({
      message: "Audio file is required",
      status: 422,
    });
  });

  it("lets the browser set the multipart content type", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      response(JSON.stringify({ text: "bonjour" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    const client = new HttpClient("", fetcher);
    const form = new FormData();
    form.set("language", "fr");

    await client.postForm("/transcribe", form);

    const [, init] = fetcher.mock.calls[0] ?? [];
    expect(init?.body).toBe(form);
    expect(new Headers(init?.headers).has("content-type")).toBe(false);
  });

  it("returns an audio blob and its download metadata", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: {
          "content-type": "audio/wav",
          "content-disposition": "attachment; filename=\"speech.wav\"",
        },
      }),
    );
    const client = new HttpClient("", fetcher);

    const result = await client.postBlob("/v1/audio/speech", { input: "Bonjour" });

    expect(result.contentType).toBe("audio/wav");
    expect(result.filename).toBe("speech.wav");
    expect((await result.blob.arrayBuffer()).byteLength).toBe(3);
  });
});
