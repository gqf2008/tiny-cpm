// End-to-end audio pipeline probe: read a 16 kHz mono wav, stream it as
// input_audio_buffer.append frames (server_vad), then measure wall-time to
// each milestone — first ASR transcript event, first audio delta, response
// done. This reproduces the real voice path WITHOUT clicking the mic button,
// so we can time ASR / LLM / TTS end to end.
//
//     cargo run --release --example audio_probe -p tiny-cpm-web -- \
//       <wav> <ws://addr/ws>
use anyhow::Result;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use std::time::Instant;
use tokio_tungstenite::connect_async;
use tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let wav = std::env::args().nth(1).expect("usage: audio_probe <wav> <ws>");
    let addr = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "ws://127.0.0.1:8080/ws".into());

    // Read 16-bit mono wav → f32 samples @ its native rate, then we assume the
    // file is already 16 kHz (server's VAD_SAMPLE_RATE). Resample if needed.
    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    eprintln!("wav: {} Hz, {} ch, {} bit", spec.sample_rate, spec.channels, spec.bits_per_sample);
    let raw: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    // downmix to mono if stereo
    let mono: Vec<i16> = if spec.channels == 2 {
        raw.chunks_exact(2).map(|c| ((c[0] as i32 + c[1] as i32) / 2) as i16).collect()
    } else {
        raw
    };
    // naive resample to 16 kHz (linear) if not already 16k
    let sr = spec.sample_rate as usize;
    let samples: Vec<i16> = if sr == 16000 {
        mono
    } else {
        let ratio = 16000.0 / sr as f64;
        let out_len = (mono.len() as f64 * ratio) as usize;
        (0..out_len)
            .map(|i| {
                let src = i as f64 / ratio;
                let idx = src as usize;
                mono.get(idx).copied().unwrap_or(0)
            })
            .collect()
    };
    let dur = samples.len() as f32 / 16000.0;
    eprintln!("sending {dur:.2}s of 16 kHz audio");

    let (mut ws, _) = connect_async(&addr).await?;
    let b64 = base64::engine::general_purpose::STANDARD;

    // session.update → server_vad (default), then append audio in ~400 ms chunks.
    ws.send(Message::Text(
        serde_json::json!({"type":"session.update","session":{"turn_detection":{"type":"server_vad","threshold":0.5,"silence_duration_ms":500}}}).to_string(),
    ))
    .await?;

    let t_start = Instant::now();
    let mut first_transcript: Option<f64> = None;
    let mut first_asr_delta: Option<f64> = None;
    let mut first_audio: Option<f64> = None;
    let mut llm_text = String::new();
    // Accumulate the streamed response.audio.delta PCM (24 kHz mono pcm16) so we can
    // save a wav and listen / inspect it for babble — the multi-model corruption check.
    let mut streamed_pcm: Vec<i16> = Vec::new();

    // Stream the audio in real-time-ish chunks (VAD needs frames to accumulate).
    let chunk = 6400; // 400 ms @ 16 kHz
    for seg in samples.chunks(chunk) {
        let bytes: Vec<u8> = seg.iter().flat_map(|s| s.to_le_bytes()).collect();
        ws.send(Message::Text(
            serde_json::json!({"type":"input_audio_buffer.append","audio":b64.encode(&bytes)}).to_string(),
        ))
        .await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Drain any events that arrived while streaming.
        while let Ok(Some(Ok(Message::Text(s)))) =
            tokio::time::timeout(std::time::Duration::from_millis(1), ws.next()).await
        {
            handle(&s, t_start, &mut first_transcript, &mut first_asr_delta, &mut first_audio, &mut llm_text, &mut streamed_pcm);
        }
    }
    eprintln!("audio sent in {:.2}s", t_start.elapsed().as_secs_f64());

    // Send ~1.5s of silence so the server-side VAD can endpoint (it needs a run
    // of non-speech frames after the utterance). Without this the segment is
    // never finalized and no turn fires.
    let sil = vec![0i16; 4000]; // 250 ms @ 16 kHz
    for _ in 0..6 {
        let bytes: Vec<u8> = sil.iter().flat_map(|s| s.to_le_bytes()).collect();
        ws.send(Message::Text(
            serde_json::json!({"type":"input_audio_buffer.append","audio":b64.encode(&bytes)}).to_string(),
        ))
        .await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Now read events until response.done.
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(60), ws.next()).await {
            Ok(Some(Ok(Message::Text(s)))) => {
                let v: serde_json::Value = serde_json::from_str(&s)?;
                let ty = v["type"].as_str().unwrap_or("");
                if ty == "response.done" {
                    println!("[t+{:.2}s] response.done", t_start.elapsed().as_secs_f64());
                    break;
                }
                handle(&s, t_start, &mut first_transcript, &mut first_asr_delta, &mut first_audio, &mut llm_text, &mut streamed_pcm);
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(e))) => { eprintln!("ws err: {e}"); break; }
            Ok(None) => { eprintln!("ws closed"); break; }
            Err(_) => { eprintln!("timeout waiting for response.done"); break; }
        }
    }

    println!("\n=== milestones (from audio start) ===");
    if let Some(t) = first_asr_delta { println!("first ASR transcript DELTA:   {t:.2}s (streaming ✓)"); }
    if let Some(t) = first_transcript { println!("first ASR transcript completed: {t:.2}s"); }
    if let Some(t) = first_audio { println!("first TTS audio delta:      {t:.2}s"); }
    if !llm_text.trim().is_empty() {
        println!("LLM reply text: {llm_text:?}");
    }

    // Save the streamed response audio (24 kHz mono) for listening / babble inspection.
    if !streamed_pcm.is_empty() {
        let out = std::env::args().nth(3).unwrap_or_else(|| "/tmp/probe_out.wav".into());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&out, spec)?;
        for &s in &streamed_pcm { w.write_sample(s)?; }
        w.finalize()?;
        let dur = streamed_pcm.len() as f32 / 24_000.0;
        // Simple speech-vs-noise heuristic: RMS and peak, and how many near-silent frames.
        let peak = streamed_pcm.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
        let rms = (streamed_pcm.iter().map(|&s| (s as i64 * s as i64) as f64).sum::<f64>()
            / streamed_pcm.len().max(1) as f64).sqrt();
        println!(
            "saved streamed audio → {out} ({dur:.2}s, peak={peak}, rms={rms:.0}, samples={})",
            streamed_pcm.len()
        );
    }
    Ok(())
}

fn handle(s: &str, t0: Instant, ft: &mut Option<f64>, fd: &mut Option<f64>, fa: &mut Option<f64>, llm: &mut String, pcm: &mut Vec<i16>) {
    let v: serde_json::Value = match serde_json::from_str(s) { Ok(v) => v, Err(_) => return };
    let t = t0.elapsed().as_secs_f64();
    let b64 = base64::engine::general_purpose::STANDARD;
    let ty = v["type"].as_str().unwrap_or("");
    match ty {
        "input_audio_buffer.speech_started" => println!("[t+{t:.2}s] speech_started"),
        "input_audio_buffer.speech_stopped" => println!("[t+{t:.2}s] speech_stopped"),
        "conversation.item.input_audio_transcription.delta" => {
            if fd.is_none() { *fd = Some(t); println!("[t+{t:.2}s] ASR delta {:?}", v["delta"].as_str().unwrap_or("")); }
        }
        "conversation.item.input_audio_transcription.completed" => {
            if ft.is_none() { *ft = Some(t); }
            println!("[t+{t:.2}s] ASR → {:?}", v["transcript"].as_str().unwrap_or(""));
        }
        "response.output_audio_transcript.delta" => {
            // LLM text streaming — accumulate to print the reply at the end.
            if let Some(d) = v["delta"].as_str() { llm.push_str(d); }
        }
        "response.output_audio.delta" => {
            if fa.is_none() {
                *fa = Some(t);
                println!("[t+{t:.2}s] first TTS audio");
            }
            // Decode base64 pcm16 24 kHz mono → accumulate.
            if let Some(delta) = v["delta"].as_str() {
                if let Ok(bytes) = b64.decode(delta) {
                    for c in bytes.chunks_exact(2) {
                        pcm.push(i16::from_le_bytes([c[0], c[1]]));
                    }
                }
            }
        }
        "response.created" => println!("[t+{t:.2}s] response.created"),
        // Log the Realtime lifecycle events added in Phase 4 (others are noisy deltas).
        "input_audio_buffer.committed"
        | "conversation.item.added"
        | "conversation.item.deleted"
        | "response.output_item.added"
        | "response.output_item.done"
        | "response.content_part.added"
        | "response.content_part.done"
        | "response.output_audio.done"
        | "response.output_audio_transcript.done"
        | "session.updated" => println!("[t+{t:.2}s] {ty}"),
        _ => {}
    }
}
