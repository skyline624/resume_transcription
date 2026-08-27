import { createContext, createElement, type ComponentChildren } from "preact";
import { useContext } from "preact/hooks";

import type { AudioResult } from "../api/contracts";
import type { RecorderPort } from "../media/recorder";
import type { HistoryRepository } from "../storage/history";

export interface HttpPort {
  getJson<T>(path: string, init?: RequestInit): Promise<T>;
  postJson<T>(path: string, body: unknown, init?: RequestInit): Promise<T>;
  postForm<T>(path: string, body: FormData, init?: RequestInit): Promise<T>;
  postBlob(path: string, body: unknown, init?: RequestInit): Promise<AudioResult>;
  delete(path: string, init?: RequestInit): Promise<void>;
}

export interface AppServices {
  http: HttpPort;
  history: HistoryRepository;
  recorderFactory: () => RecorderPort;
}

const ServicesContext = createContext<AppServices | null>(null);

export function ServicesProvider({
  services,
  children,
}: {
  services: AppServices;
  children: ComponentChildren;
}) {
  return createElement(ServicesContext.Provider, { value: services }, children);
}

export function useServices(): AppServices {
  const services = useContext(ServicesContext);
  if (!services) throw new Error("ServicesProvider est absent de l’application.");
  return services;
}
