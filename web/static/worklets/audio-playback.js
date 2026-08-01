// @ts-check
/**
 * AudioWorkletProcessor that plays Float32 mono samples pushed from the main
 * thread, upsampling the server's rate (24 kHz PCM16) to the AudioContext rate
 * (typically 48 kHz) with linear interpolation. Runs on the audio rendering
 * thread, so playback is never blocked by main-thread JSON/base64 work — this
 * is what makes capture + playback truly full-duplex.
 *
 *   main -> worklet:
 *     { kind: "config", inputRate: 24000 }       one-shot at startup
 *     { kind: "audio", samples: Float32Array }   (transferable) per chunk
 *     { kind: "clear" }                          wipe queue (barge-in)
 *
 *   worklet -> main:
 *     { kind: "stats", queuedMs, played }        every ~250 ms
 *     { kind: "underrun" }                       when the queue runs dry
 *
 * Anti-click strategy (mirrors the audio.cpp reference): fade-in on the first
 * chunk of a turn, fade-out on clear/underrun; on underrun we ramp to silence
 * instead of holding the last sample (holding produces audible buzz).
 */

const STATS_INTERVAL_FRAMES = 12000;
const FADE_FRAMES = 32;

class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._inputRate = 24000;
    this._stepRatio = this._inputRate / sampleRate;
    this._queue = [];
    this._readIdx = 0;
    this._fracPos = 0;
    this._playing = false;
    this._framesSinceStats = 0;
    this._totalPlayed = 0;
    this._fadeIn = 0;
    this._fadeOut = 0;
    this._lastSample = 0;
    this._underruns = 0;
    this._buffering = true;
    this._prebufferFrames = Math.round(0.5 * this._inputRate); // default 500ms cushion

    this.port.onmessage = (e) => {
      const data = e.data;
      if (!data || typeof data !== "object") return;
      switch (data.kind) {
        case "config":
          if (typeof data.inputRate === "number" && data.inputRate > 0) {
            this._inputRate = data.inputRate;
            this._stepRatio = this._inputRate / sampleRate;
          }
          if (typeof data.prebufferMs === "number" && data.prebufferMs >= 0) {
            this._prebufferFrames = Math.round((data.prebufferMs / 1000) * this._inputRate);
          }
          break;
        case "audio":
          if (data.samples instanceof Float32Array && data.samples.length > 0) {
            this._queue.push(data.samples);
            // Buffering: only start (or restart after a drain) once we've
            // accumulated `prebuffer` frames of cushion. The TTS producer runs
            // at ~0.7-1.1x realtime (marginally under), so playing immediately
            // starves the buffer at every sentence boundary — the "choppy"
            // symptom. Holding a target backlog hides the producer's bursts.
            if (!this._playing && !this._buffering) {
              this._buffering = true;
              this.port.postMessage({ kind: "buffering" });
            }
            if (this._buffering && this._queuedSamples() >= this._prebufferFrames) {
              this._buffering = false;
              this._playing = true;
              // Playback start after a stop/drain: ramp in to avoid a click.
              this._fadeIn = FADE_FRAMES;
              this._fadeOut = 0;
            }
          }
          break;
        case "clear":
          this._queue.length = 0;
          this._readIdx = 0;
          this._fracPos = 0;
          this._buffering = false;
          this._fadeOut = FADE_FRAMES;
          break;
        case "flush":
          // End of turn: play out whatever remains immediately (don't wait for
          // the prebuffer threshold), so short tails aren't held back.
          if (this._buffering && this._queue.length > 0) {
            this._buffering = false;
            this._playing = true;
            // Flush starts playback immediately: ramp in to avoid a click.
            this._fadeIn = FADE_FRAMES;
            this._fadeOut = 0;
          }
          break;
      }
    };
  }

  _queuedSamples() {
    let total = -this._readIdx;
    for (const buf of this._queue) total += buf.length;
    return Math.max(0, total);
  }

  /** Linear-interp read at the current fractional position. */
  _readInterpolated() {
    if (this._queue.length === 0) return null;
    const head = this._queue[0];
    const idx = this._readIdx;
    const frac = this._fracPos;
    const a = head[idx];
    let b;
    if (idx + 1 < head.length) {
      b = head[idx + 1];
    } else if (this._queue.length > 1) {
      b = this._queue[1][0];
    } else {
      b = a;
    }
    return a + (b - a) * frac;
  }

  /** Advance the read position by `stepRatio`; pop consumed buffers. */
  _advance() {
    this._fracPos += this._stepRatio;
    while (this._fracPos >= 1) {
      this._fracPos -= 1;
      this._readIdx += 1;
    }
    while (this._queue.length > 0 && this._readIdx >= this._queue[0].length) {
      this._readIdx -= this._queue[0].length;
      this._queue.shift();
    }
  }

  process(_, outputs) {
    const channels = outputs[0];
    if (!channels || channels.length === 0) return true;
    const out = channels[0];
    const stereo = channels.length > 1 ? channels[1] : null;

    for (let i = 0; i < out.length; i++) {
      let sample = 0;
      if (this._playing) {
        const v = this._readInterpolated();
        if (v === null) {
          // Drained: ramp to silence to avoid a click, then go back to
          // buffering so the next burst re-accumulates a cushion before
          // resuming (instead of stuttering on every chunk).
          sample = this._lastSample * Math.max(0, 1 - 1 / FADE_FRAMES);
          this._lastSample = sample;
          if (Math.abs(sample) < 1e-4) {
            this._playing = false;
            this._buffering = true;
            this._lastSample = 0;
            this._underruns += 1;
            this.port.postMessage({ kind: "underrun", t: currentTime, played: this._totalPlayed });
          }
        } else {
          sample = v;
          this._lastSample = v;
          this._advance();
        }
        if (this._fadeIn > 0) {
          sample *= 1 - this._fadeIn / FADE_FRAMES;
          this._fadeIn -= 1;
        }
        if (this._fadeOut > 0) {
          sample *= this._fadeOut / FADE_FRAMES;
          this._fadeOut -= 1;
          if (this._fadeOut === 0) {
            this._playing = false;
            this._lastSample = 0;
          }
        }
        this._totalPlayed += 1;
      }
      out[i] = sample;
      if (stereo) stereo[i] = sample;
    }

    this._framesSinceStats += out.length;
    if (this._framesSinceStats >= STATS_INTERVAL_FRAMES) {
      this._framesSinceStats = 0;
      const queuedMs = (this._queuedSamples() / this._inputRate) * 1000;
      this.port.postMessage({ kind: "stats", queuedMs, played: this._totalPlayed });
    }
    return true;
  }
}

registerProcessor("audio-playback", AudioPlaybackProcessor);
