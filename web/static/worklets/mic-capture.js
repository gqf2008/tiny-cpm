// @ts-check
/**
 * AudioWorkletProcessor that captures mic input on the audio rendering thread
 * and forwards raw Float32 mono frames to the main thread. Running capture here
 * (instead of ScriptProcessorNode.onaudioprocess on the main thread) is what
 * keeps the mic live while the main thread is busy decoding base64/JSON and
 * pushing TTS chunks — true full-duplex.
 *
 *   main -> worklet:
 *     { kind: "start" }   begin forwarding frames
 *     { kind: "stop" }    stop forwarding (frames dropped)
 *
 *   worklet -> main:
 *     { kind: "frame", samples: Float32Array }   (transferable) per render block
 *
 * The frame is the native render quantum (typically 128 frames at the context
 * sampleRate). The main thread owns RMS gating, resampling to 16 kHz and the
 * WebSocket append — this worklet is deliberately dumb so the audio thread
 * never does anything but copy.
 */
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._active = true;
    this.port.onmessage = (e) => {
      const data = e.data;
      if (!data || typeof data !== "object") return;
      if (data.kind === "start") this._active = true;
      if (data.kind === "stop") this._active = false;
    };
  }

  process(inputs) {
    if (this._active) {
      const input = inputs[0];
      if (input && input.length > 0 && input[0] && input[0].length > 0) {
        // Copy out of the shared ring buffer before transferring to main.
        const frame = new Float32Array(input[0]);
        this.port.postMessage({ kind: "frame", samples: frame }, [frame.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
