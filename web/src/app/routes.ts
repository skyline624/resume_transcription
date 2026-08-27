export type RouteName =
  | "transcribe"
  | "summarize"
  | "speech"
  | "voices"
  | "history";

const paths: Record<RouteName, string> = {
  transcribe: "#/transcribe",
  summarize: "#/summarize",
  speech: "#/speech",
  voices: "#/voices",
  history: "#/history",
};

export function readRoute(hash: string): RouteName {
  const path = hash.split("?", 1)[0];
  const found = Object.entries(paths).find(([, value]) => value === path);
  return (found?.[0] as RouteName | undefined) ?? "transcribe";
}

export const routeHref = (route: RouteName): string => paths[route];
