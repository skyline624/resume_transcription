import type { Health } from "../../api/contracts";
import type { HttpPort } from "../../app/services";

export function getHealth(http: HttpPort): Promise<Health> {
  return http.getJson<Health>("/health");
}
