import { act, render } from "@testing-library/preact";
import { afterEach, describe, expect, it, vi } from "vitest";

import { fakeServices } from "../../test/fakes";
import { useHealth } from "./use-health";

afterEach(() => vi.useRealTimers());

describe("useHealth", () => {
  it("polls every ten seconds while the page is visible", async () => {
    vi.useFakeTimers();
    const services = fakeServices();
    function Harness() {
      const { health } = useHealth();
      return <span>{health?.status ?? "loading"}</span>;
    }

    render(<Harness />, { wrapper: services.wrapper });
    await act(async () => Promise.resolve());
    expect(services.http.getJson).toHaveBeenCalledTimes(1);

    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(services.http.getJson).toHaveBeenCalledTimes(2);
  });
});
