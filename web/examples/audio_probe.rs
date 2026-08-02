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
    let mut first_audio: Option<f64> = None;

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
            handle(&s, t_start, &mut first_transcript, &mut first_audio);
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
                handle(&s, t_start, &mut first_transcript, &mut first_audio);
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(e))) => { eprintln!("ws err: {e}"); break; }
            Ok(None) => { eprintln!("ws closed"); break; }
            Err(_) => { eprintln!("timeout waiting for response.done"); break; }
        }
    }

    println!("\n=== milestones (from audio start) ===");
    if let Some(t) = first_transcript { println!("first ASR transcript event: {t:.2}s"); }
    if let Some(t) = first_audio { println!("first TTS audio delta:      {t:.2}s"); }
    Ok(())
}

fn handle(s: &str, t0: Instant, ft: &mut Option<f64>, fa: &mut Option<f64>) {
    let v: serde_json::Value = match serde_json::from_str(s) { Ok(v) => v, Err(_) => return };
    let t = t0.elapsed().as_secs_f64();
    match v["type"].as_str().unwrap_or("") {
        "input_audio_buffer.speech_started" => println!("[t+{t:.2}s] speech_started"),
        "input_audio_buffer.speech_stopped" => println!("[t+{t:.2}s] speech_stopped"),
        "conversation.item.input_audio_transcription.completed" => {
            if ft.is_none() { *ft = Some(t); }
            println!("[t+{t:.2}s] ASR → {:?}", v["transcript"].as_str().unwrap_or(""));
        }
        "response.audio_transcript.delta" => {
            // LLM text streaming (verbose) — skip
        }
        "response.audio.delta" => {
            if fa.is_none() {
                *fa = Some(t);
                println!("[t+{t:.2}s] first TTS audio");
            }
        }
        "response.created" => println!("[t+{t:.2}s] response.created"),
        _ => {}
    }
}
