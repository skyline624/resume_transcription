import { useEffect, useRef } from "preact/hooks";

interface WaveformProps {
  stream: MediaStream | null;
}

export function Waveform({ stream }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!stream || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let frame = 0;
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    const draw = () => {
      analyser.getByteTimeDomainData(samples);
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = "#176B75";
      context.lineWidth = 2;
      context.beginPath();
      samples.forEach((sample, index) => {
        const x = (index / (samples.length - 1)) * canvas.width;
        const y = (sample / 255) * canvas.height;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
      if (!reducedMotion) frame = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      if (frame) cancelAnimationFrame(frame);
      source.disconnect();
      void audioContext.close();
    };
  }, [stream]);

  return (
    <canvas
      aria-label="Niveau du microphone"
      class="waveform"
      height={72}
      ref={canvasRef}
      role="img"
      width={560}
    />
  );
}
