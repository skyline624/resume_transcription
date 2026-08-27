import type { AudioResult } from "./contracts";

type Fetcher = typeof fetch;

interface OpenAiErrorEnvelope {
  error?: {
    message?: unknown;
    code?: unknown;
    param?: unknown;
  };
}

interface FastApiErrorEnvelope {
  detail?: unknown;
}

export class ApiFailure extends Error {
  readonly status: number;
  readonly code?: string;
  readonly param?: string;

  constructor(message: string, status: number, code?: string, param?: string) {
    super(message);
    this.name = "ApiFailure";
    this.status = status;
    this.code = code;
    this.param = param;
  }
}

export class HttpClient {
  constructor(
    private readonly baseUrl = "",
    private readonly fetcher: Fetcher = fetch,
  ) {}

  getJson<T>(path: string, init?: RequestInit): Promise<T> {
    return this.requestJson<T>(path, { ...init, method: "GET" });
  }

  postJson<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
    const headers = new Headers(init?.headers);
    headers.set("content-type", "application/json");
    return this.requestJson<T>(path, {
      ...init,
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  }

  postForm<T>(path: string, body: FormData, init?: RequestInit): Promise<T> {
    const headers = new Headers(init?.headers);
    headers.delete("content-type");
    return this.requestJson<T>(path, {
      ...init,
      method: "POST",
      headers,
      body,
    });
  }

  async postBlob(
    path: string,
    body: unknown,
    init?: RequestInit,
  ): Promise<AudioResult> {
    const isForm = body instanceof FormData;
    const headers = new Headers(init?.headers);
    if (isForm) {
      headers.delete("content-type");
    } else {
      headers.set("content-type", "application/json");
    }
    const response = await this.fetcher.call(globalThis, this.url(path), {
      ...init,
      method: "POST",
      headers,
      body: isForm ? body : JSON.stringify(body),
    });
    await ensureSuccess(response);
    const contentType = response.headers.get("content-type") ?? "application/octet-stream";
    const filename = parseFilename(response.headers.get("content-disposition"));
    const result: AudioResult = {
      blob: await response.blob(),
      contentType,
    };
    if (filename) result.filename = filename;
    return result;
  }

  async delete(path: string, init?: RequestInit): Promise<void> {
    const response = await this.fetcher.call(globalThis, this.url(path), {
      ...init,
      method: "DELETE",
    });
    await ensureSuccess(response);
  }

  private async requestJson<T>(path: string, init: RequestInit): Promise<T> {
    const response = await this.fetcher.call(globalThis, this.url(path), init);
    await ensureSuccess(response);
    return (await response.json()) as T;
  }

  private url(path: string): string {
    if (!this.baseUrl) return path;
    return `${this.baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
  }
}

async function ensureSuccess(response: Response): Promise<void> {
  if (response.ok) return;

  let payload: OpenAiErrorEnvelope & FastApiErrorEnvelope = {};
  try {
    payload = (await response.clone().json()) as OpenAiErrorEnvelope & FastApiErrorEnvelope;
  } catch {
    // Some proxy errors are plain text or empty.
  }

  const openAi = payload.error;
  const message =
    textValue(openAi?.message) ??
    detailMessage(payload.detail) ??
    response.statusText ??
    `HTTP ${response.status}`;

  throw new ApiFailure(
    message || `HTTP ${response.status}`,
    response.status,
    textValue(openAi?.code),
    textValue(openAi?.param),
  );
}

function textValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function detailMessage(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return undefined;

  const messages = detail
    .map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object" && "msg" in item) {
        return textValue((item as { msg: unknown }).msg);
      }
      return undefined;
    })
    .filter((item): item is string => Boolean(item));
  return messages.length > 0 ? messages.join(" · ") : undefined;
}

function parseFilename(contentDisposition: string | null): string | undefined {
  if (!contentDisposition) return undefined;

  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(contentDisposition)?.[1];
  if (encoded) {
    try {
      return decodeURIComponent(encoded);
    } catch {
      return encoded;
    }
  }

  return /filename="?([^";]+)"?/i.exec(contentDisposition)?.[1];
}
