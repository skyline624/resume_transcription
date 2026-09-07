import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import type { AudioResult } from "../api/contracts";
import { PcmPlayer } from "./pcm-player";

export function useLivePlayback() {
  const current = useRef<{ controller: AbortController; player: PcmPlayer } | null>(null);
  const [active, setActive] = useState(false);
  const stop = useCallback(() => {
    current.current?.controller.abort();
    current.current?.player.stop();
  }, []);
  useEffect(() => stop, [stop]);

  const play = async (
    request: (onChunk: (chunk: Uint8Array) => void, signal: AbortSignal) => Promise<AudioResult>,
  ): Promise<AudioResult> => {
    const controller = new AbortController();
    const player = new PcmPlayer();
    current.current = { controller, player };
    setActive(true);
    try {
      await player.start();
      controller.signal.throwIfAborted();
      const result = await request(player.push, controller.signal);
      await player.finish();
      controller.signal.throwIfAborted();
      return result;
    } finally {
      player.stop();
      current.current = null;
      setActive(false);
    }
  };
  return { play, stop, active };
}
