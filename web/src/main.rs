//! tiny-cpm-web: a Qwen-Audio / OpenAI-Realtime-style WebSocket voice server.
//!
//! Implements the core realtime event protocol: `session.update`,
//! `input_audio_buffer.append/commit`, `response.create/cancel`,
//! `conversation.item.create/delete`, and emits `session.created/updated`,
//! `input_audio_buffer.speech_started/stopped`,
//! `conversation.item.input_audio_transcription.completed`,
//! `response.created/audio_transcript.delta/audio.delta/...done`, `error`.
//!
//! Audio: input 16 kHz 16-bit mono PCM (base64 in JSON); output 24 kHz 16-bit
//! mono PCM (MOSS 48 k stereo downmixed). Engines (FireRedVAD → Qwen3-ASR →
//! MiniCPM5 → MOSS-TTS) reused from tiny-cpm. The browser owns AEC/NS (getUser-
//! Media), RMS metering, and history display/control (conversation items are
//! a transient per-session store the client drives via create/delete).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Html,
    routing::get,
    Router,
};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use candle_core::{Device, Tensor};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use tiny_cpm::{
    exec::{chat, moss_tts::MossEngine, qwen3_asr::Qwen3AsrEngine},
    models::fire_red_vad::vad::{FireRedVad, VadOverrides},
    quantized_minicpm5::ModelWeights,
};

const ASR_MAX_TOKENS: usize = 512;
const DEFAULT_MAX_TOKENS: usize = 1024;
const MOSS_MAX_FRAMES: usize = 300;
const TTS_CHUNK_FRAMES: usize = 5; // small chunks → smaller base64 deltas → less main-thread blocking in the browser
const VAD_FRAME_SAMPLES: usize = 400;
const VAD_SAMPLE_RATE: usize = 16_000;
const MIN_SEGMENT_SAMPLES: usize = 8000;
const OUT_SR: usize = 24_000;     // output sample rate (per doc)
const DEFAULT_PERSONA: &str =
    "你是一个语音助手。先简短思考用户的问题，然后用一两句简短的口语回答。不要输出 markdown 格式。";

struct Engines {
    device: Device,
    asr: Qwen3AsrEngine,
    llm: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    tts: MossEngine,
    ref_codes: Option<Tensor>,
}

/// Per-session conversation item (transient store; client drives via create/delete).
#[derive(Clone)]
struct ConvItem {
    id: String,
    role: String,  // user | assistant | system
    text: String,
}

#[derive(Clone, Debug)]
enum TurnMode {
    ServerVad { threshold: f32, silence_ms: u64 },
    Manual,
}

#[derive(Clone)]
struct Session {
    instructions: Option<String>,
    turn_mode: TurnMode,
    max_history_turns: usize,
    items: Vec<ConvItem>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            instructions: None,
            turn_mode: TurnMode::ServerVad { threshold: 0.5, silence_ms: 800 },
            max_history_turns: 20,
            items: Vec::new(),
        }
    }
}

/// One outbound WS text frame (a JSON event). All server→client is JSON (audio
/// is base64 inside `response.audio.delta`, per the doc).
struct Out(String);

/// A turn's input: voiced audio (→ ASR) or typed text (client already pushed
/// a user conversation item; use its text directly).
enum TurnInput {
    Audio(Vec<f32>),
    Text(String),
}

struct ServerState {
    eng: Arc<Mutex<Engines>>,
    vad_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .try_init();

    let usage = "usage: tiny-cpm-web --vad-dir <d> --asr-dir <d> --minicpm-dir <gguf|bf16-dir> --tokenizer <json> --moss-dir <d> --codec-dir <d> [--ref <wav>] [--addr 127.0.0.1:8080]";
    let mut vad_dir = None;
    let mut asr_dir = None;
    let mut minicpm_dir = None;
    let mut tok_path = None;
    let mut moss_dir = None;
    let mut codec_dir = None;
    let mut ref_wav = None;
    let mut addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().ok_or_else(|| anyhow!("{a} requires a value. {usage}"));
        match a.as_str() {
            "--vad-dir" => vad_dir = Some(v()?),
            "--asr-dir" => asr_dir = Some(v()?),
            "--minicpm-dir" => minicpm_dir = Some(v()?),
            "--tokenizer" => tok_path = Some(v()?),
            "--moss-dir" => moss_dir = Some(v()?),
            "--codec-dir" => codec_dir = Some(v()?),
            "--ref" => ref_wav = Some(v()?),
            "--addr" => addr = v()?.parse().map_err(|_| anyhow!("bad --addr. {usage}"))?,
            other => bail!("unknown option {other}. {usage}"),
        }
    }
    let (vad_dir, asr_dir, minicpm_dir, tok_path, moss_dir, codec_dir) = (
        vad_dir.ok_or_else(|| anyhow!("--vad-dir required. {usage}"))?,
        asr_dir.ok_or_else(|| anyhow!("--asr-dir required. {usage}"))?,
        minicpm_dir.ok_or_else(|| anyhow!("--minicpm-dir required. {usage}"))?,
        tok_path.ok_or_else(|| anyhow!("--tokenizer required. {usage}"))?,
        moss_dir.ok_or_else(|| anyhow!("--moss-dir required. {usage}"))?,
        codec_dir.ok_or_else(|| anyhow!("--codec-dir required. {usage}"))?,
    );

    let device = Device::new_metal(0)?;
    let t = Instant::now();
    let asr = Qwen3AsrEngine::load(&asr_dir, &device)?;
    eprintln!("loaded Qwen3-ASR in {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let llm = chat::load_model(&minicpm_dir, &device)?;
    eprintln!("loaded MiniCPM5 in {:.2}s", t.elapsed().as_secs_f64());
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let t = Instant::now();
    let tts = MossEngine::load(&moss_dir, &codec_dir, &device)?;
    eprintln!("loaded MOSS-TTS in {:.2}s", t.elapsed().as_secs_f64());
    let ref_codes = if let Some(ref_path) = &ref_wav {
        let t = Instant::now();
        let codes = tts.encode_ref(ref_path)?;
        eprintln!(
            "encoded voice ref {} ({} frames) in {:.2}s",
            ref_path,
            codes.dim(0)?,
            t.elapsed().as_secs_f64()
        );
        Some(codes)
    } else {
        None
    };
    eprintln!("output: {OUT_SR} Hz mono 16-bit (downmixed from MOSS)");

    let eng = Arc::new(Mutex::new(Engines {
        device: device.clone(),
        asr,
        llm,
        tokenizer,
        tts,
        ref_codes,
    }));
    let st = Arc::new(ServerState { eng, vad_dir });
    let app = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/ws", get(ws_handler))
        .with_state(st);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("listening on http://{addr} (open in Chrome)");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(ws: WebSocketUpgrade, State(st): State<Arc<ServerState>>) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_conn(socket, st))
}

async fn handle_conn(socket: WebSocket, st: Arc<ServerState>) {
    eprintln!("ws connected");
    let (mut sink, mut stream) = socket.split();
    let (out_tx, mut out_rx) = mpsc::channel::<Out>(64);
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<f32>>(256);
    let (seg_tx, seg_rx) = mpsc::channel::<TurnInput>(16);
    let seg_tx_listener = seg_tx.clone();
    let (param_tx, param_rx) = mpsc::channel::<VadOverrides>(64);

    let session = Arc::new(Mutex::new(Session::default()));
    let speaking = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let barge = Arc::new(AtomicBool::new(false));

    // session.created
    let _ = out_tx
        .send(Out(json!({
            "type": "session.created",
            "session": {
                "output_audio_format": { "sample_rate": OUT_SR, "channels": 1, "encoding": "pcm16" },
                "input_audio_format": { "sample_rate": VAD_SAMPLE_RATE, "channels": 1, "encoding": "pcm16" },
            }
        }).to_string()))
        .await;

    let session_r = session.clone();
    let out_tx_r = out_tx.clone();
    let barge_r = barge.clone();
    let read_task = tokio::spawn(async move {
        let mut manual_buf: Vec<f32> = Vec::new();
        while let Some(Ok(msg)) = stream.next().await {
            let txt = match msg {
                Message::Text(t) => t,
                Message::Binary(_) => continue,
                Message::Close(_) => break,
                _ => continue,
            };
            let ev: Value = match serde_json::from_str(&txt) {
                Ok(v) => v,
                Err(e) => {
                    let _ = out_tx_r.send(Out(json!({"type":"error","error":{"message":format!("bad json: {e}")}}).to_string())).await;
                    continue;
                }
            };
            let t = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match t {
                "session.update" => {
                    let s = ev.get("session").cloned().unwrap_or(json!({}));
                    apply_session_update(&session_r, &s, &param_tx).await;
                    let _ = out_tx_r.send(Out(json!({"type":"session.updated"}).to_string())).await;
                }
                "input_audio_buffer.append" => {
                    if let Some(a) = ev.get("audio").and_then(|v| v.as_str()) {
                        if let Ok(bytes) = B64.decode(a) {
                            let f: Vec<f32> = bytes
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                                .collect();
                            // server_vad → VAD listener; manual → buffer for commit
                            let is_manual = matches!(
                                session_r.lock().unwrap().turn_mode,
                                TurnMode::Manual
                            );
                            if is_manual {
                                manual_buf.extend_from_slice(&f);
                            } else {
                                let _ = mic_tx.try_send(f);
                            }
                        }
                    }
                }
                "input_audio_buffer.commit" => {
                    // manual: nothing special (the append already buffered); client
                    // follows with response.create to run a turn.
                }
                "response.create" => {
                    // Manual mode: if audio buffered, ASR it; else use the latest
                    // user conversation item (text the client pushed via
                    // conversation.item.create) as the prompt.
                    let buf = std::mem::take(&mut manual_buf);
                    if !buf.is_empty() {
                        let _ = seg_tx.try_send(TurnInput::Audio(buf));
                    } else {
                        let t = session_r
                            .lock()
                            .unwrap()
                            .items
                            .iter()
                            .rev()
                            .find(|i| i.role == "user")
                            .map(|i| i.text.clone());
                        if let Some(t) = t {
                            let _ = seg_tx.try_send(TurnInput::Text(t));
                        }
                    }
                }
                "response.cancel" => {
                    barge_r.store(true, Ordering::Relaxed);
                }
                "conversation.item.create" => {
                    if let Some(item) = ev.get("item") {
                        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                        // content[].text
                        let text = item
                            .get("content")
                            .and_then(|c| c.as_array())
                            .and_then(|arr| {
                                arr.iter().find_map(|p| {
                                    p.get("text")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                })
                            })
                            .unwrap_or_default();
                        if !text.is_empty() {
                            let mut s = session_r.lock().unwrap();
                            let id = format!("item_{}", s.items.len());
                            s.items.push(ConvItem { id, role, text });
                        }
                    }
                }
                "conversation.item.delete" => {
                    if let Some(id) = ev.get("item_id").and_then(|v| v.as_str()) {
                        let mut s = session_r.lock().unwrap();
                        s.items.retain(|i| i.id != id);
                    }
                }
                _ => {
                    let _ = out_tx_r.send(Out(json!({
                        "type":"error","error":{"message":format!("unsupported event: {t}")}
                    }).to_string())).await;
                }
            }
        }
    });

    // WS write task.
    let write_task = tokio::spawn(async move {
        while let Some(Out(json)) = out_rx.recv().await {
            if sink.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    // Listener thread (server_vad): FireRedVAD + onset/barg.
    let speaking_l = speaking.clone();
    let streaming_l = streaming.clone();
    let barge_l = barge.clone();
    let out_tx_l = out_tx.clone();
    let vad_dir = st.vad_dir.clone();
    let listener_thread = std::thread::spawn(move || {
        listener_loop(
            vad_dir,
            mic_rx,
            seg_tx_listener,
            param_rx,
            speaking_l,
            streaming_l,
            barge_l,
            out_tx_l,
        )
    });

    // Main turn thread.
    let eng = st.eng.clone();
    let session_m = session.clone();
    let streaming_m = streaming.clone();
    let barge_m = barge.clone();
    let out_tx_m = out_tx.clone();
    let speaking_m = speaking.clone();
    let main_thread = std::thread::spawn(move || {
        main_loop(eng, seg_rx, out_tx_m, session_m, barge_m, streaming_m, speaking_m)
    });

    let _ = read_task.await;
    write_task.abort();
    let _ = listener_thread.join();
    let _ = main_thread.join();
    eprintln!("ws disconnected");
}

async fn apply_session_update(
    session: &Arc<Mutex<Session>>,
    s: &Value,
    param_tx: &mpsc::Sender<VadOverrides>,
) {
    let mut sess = session.lock().unwrap();
    if let Some(ins) = s.get("instructions").and_then(|v| v.as_str()) {
        sess.instructions = Some(ins.to_string());
    }
    if let Some(m) = s.get("max_history_turns").and_then(|v| v.as_u64()) {
        sess.max_history_turns = m.clamp(1, 50) as usize;
    }
    if let Some(td) = s.get("turn_detection") {
        if td.is_null() {
            sess.turn_mode = TurnMode::Manual;
        } else if let Some(ty) = td.get("type").and_then(|v| v.as_str()) {
            match ty {
                "server_vad" => {
                    let threshold = td.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                    let silence_ms = td.get("silence_duration_ms").and_then(|v| v.as_u64()).unwrap_or(800);
                    sess.turn_mode = TurnMode::ServerVad { threshold, silence_ms };
                    // Map to FireRedVAD params: threshold→speech_threshold,
                    // silence_duration_ms→min_silence_frame (frame_shift ~10ms).
                    let _ = param_tx.try_send(VadOverrides {
                        speech_threshold: Some(threshold),
                        min_silence_frame: Some((silence_ms / 10).max(1) as usize),
                        ..Default::default()
                    });
                }
                "smart_turn" => {
                    // Not supported (no semantic turn model) — fall back to server_vad.
                    eprintln!("warning: smart_turn not supported, using server_vad");
                    sess.turn_mode = TurnMode::ServerVad { threshold: 0.5, silence_ms: 800 };
                }
                _ => {}
            }
        }
    }
}

fn listener_loop(
    vad_dir: String,
    mut mic_rx: mpsc::Receiver<Vec<f32>>,
    seg_tx: mpsc::Sender<TurnInput>,
    mut param_rx: mpsc::Receiver<VadOverrides>,
    _speaking: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
    barge: Arc<AtomicBool>,
    out_tx: mpsc::Sender<Out>,
) {
    let mut vad = match FireRedVad::init(&vad_dir, Some(&Device::Cpu), None, None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("listener: FireRedVAD init failed: {e}");
            return;
        }
    };
    let mut onset_count = 0usize;
    let mut mic_buf: Vec<f32> = Vec::new();
    let mut speech_started_emitted = false;
    loop {
        while let Ok(o) = param_rx.try_recv() {
            vad.update_params(&o);
        }
        let chunk = match mic_rx.blocking_recv() {
            Some(c) => c,
            None => break,
        };
        mic_buf.extend_from_slice(&chunk);
        while mic_buf.len() >= VAD_FRAME_SAMPLES {
            let frame: Vec<f32> = mic_buf.drain(..VAD_FRAME_SAMPLES).collect();
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            let vad_result = match vad.detect_frame_f32(frame, 1, Some(VAD_SAMPLE_RATE)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("vad frame error: {e}");
                    continue;
                }
            };
            let is_speech = vad.last_frame_speech();
            // speech_started/stopped events
            if is_speech && !speech_started_emitted {
                speech_started_emitted = true;
                let _ = out_tx.blocking_send(Out(
                    json!({"type":"input_audio_buffer.speech_started"}).to_string(),
                ));
                // barge: a new speech onset during a streaming response cancels it.
                // The browser clears its playback on speech_started; the main loop
                // sees `barge` and aborts LLM+TTS, then emits response.done{cancelled}.
                if streaming.load(Ordering::Relaxed) {
                    barge.store(true, Ordering::Relaxed);
                }
            }
            // barge onset (neural + rms floor) during streaming
            if streaming.load(Ordering::Relaxed) && is_speech && rms >= 0.015 {
                onset_count += 1;
                if onset_count >= 3 {
                    barge.store(true, Ordering::Relaxed);
                }
            } else {
                onset_count = 0;
            }
            let Some(result) = vad_result else { continue };
            let Some(segment) = result.orig_audio else { continue };
            let samples = match segment.to_vec1::<f32>() {
                Ok(s) => s,
                Err(_) => continue,
            };
            // endpoint → speech_stopped + segment
            if speech_started_emitted {
                speech_started_emitted = false;
                let _ = out_tx.blocking_send(Out(
                    json!({"type":"input_audio_buffer.speech_stopped"}).to_string(),
                ));
            }
            if samples.len() >= MIN_SEGMENT_SAMPLES {
                if seg_tx.blocking_send(TurnInput::Audio(samples)).is_err() {
                    break;
                }
            }
        }
    }
}

fn main_loop(
    eng: Arc<Mutex<Engines>>,
    mut seg_rx: mpsc::Receiver<TurnInput>,
    out_tx: mpsc::Sender<Out>,
    session: Arc<Mutex<Session>>,
    barge: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
) {
    let mut turn = 0usize;
    loop {
        let input = match seg_rx.blocking_recv() {
            Some(s) => s,
            None => break,
        };
        barge.store(false, Ordering::Relaxed);
        speaking.store(true, Ordering::Relaxed);
        streaming.store(true, Ordering::Relaxed);
        turn += 1;
        let resp_id = format!("resp_{turn}");
        let user_item_id = format!("item_u{turn}");
        let asst_item_id = format!("item_a{turn}");

        let _ = out_tx.blocking_send(Out(json!({
            "type":"response.created","response":{"id":resp_id}
        }).to_string()));

        // Voice → ASR + push a user item; Text → item already exists (client pushed it).
        let transcript = match input {
            TurnInput::Audio(samples) => {
                eprintln!("=== turn {turn}: utterance {:.2}s ===", samples.len() as f32 / VAD_SAMPLE_RATE as f32);
                let t = {
                    let mut e = eng.lock().unwrap();
                    match e.asr.transcribe_samples(&samples, ASR_MAX_TOKENS) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("turn {turn}: asr error: {e}");
                            let _ = out_tx.blocking_send(Out(json!({
                                "type":"response.done","response":{"id":resp_id,"status":"failed"}
                            }).to_string()));
                            speaking.store(false, Ordering::Relaxed);
                            streaming.store(false, Ordering::Relaxed);
                            continue;
                        }
                    }
                };
                if t.trim().is_empty() {
                    let _ = out_tx.blocking_send(Out(json!({
                        "type":"response.done","response":{"id":resp_id,"status":"completed"}
                    }).to_string()));
                    speaking.store(false, Ordering::Relaxed);
                    streaming.store(false, Ordering::Relaxed);
                    continue;
                }
                let _ = out_tx.blocking_send(Out(json!({
                    "type":"conversation.item.input_audio_transcription.completed",
                    "item_id": user_item_id, "transcript": t,
                }).to_string()));
                session.lock().unwrap().items.push(ConvItem {
                    id: user_item_id, role: "user".into(), text: t.clone(),
                });
                t
            }
            TurnInput::Text(t) => {
                eprintln!("=== turn {turn}: text input ===");
                t
            }
        };

        // Build history for the LLM.
        let (instructions, history) = {
            let s = session.lock().unwrap();
            let hist = build_history(&s);
            (s.instructions.clone(), hist)
        };
        let persona = instructions.unwrap_or_else(|| DEFAULT_PERSONA.to_string());
        eprintln!(
            "turn {turn}: history={} items, prompt=\"{}\", persona=\"{}\"",
            history.len(),
            transcript.chars().take(80).collect::<String>(),
            persona.chars().take(40).collect::<String>(),
        );
        for (i, (is_user, text)) in history.iter().enumerate() {
            eprintln!("  hist[{i}] {}: \"{}\"", if *is_user { "user" } else { "asst" }, text.chars().take(60).collect::<String>());
        }

        let _ = out_tx.blocking_send(Out(json!({
            "type":"response.audio_transcript.delta","response_id":resp_id,"delta":""
        }).to_string()));

        let mut e = eng.lock().unwrap();
        let eng_ref: &mut Engines = &mut *e;
        let device = eng_ref.device.clone();
        let tokenizer = &eng_ref.tokenizer;
        let llm = &mut eng_ref.llm;
        let tts = &mut eng_ref.tts;
        let ref_codes = eng_ref.ref_codes.as_ref();
        let out_ref = &out_tx;
        let barge_ref = &barge;
        let mut full_reply = String::new();
        let mut splitter = SentenceSplitter::new();
        let mut sink = |delta: &str| {
            if barge_ref.load(Ordering::Relaxed) {
                return;
            }
            full_reply.push_str(delta);
            let _ = out_ref.blocking_send(Out(json!({
                "type":"response.audio_transcript.delta","response_id":resp_id,"delta":delta
            }).to_string()));
            for sentence in splitter.push(delta) {
                // TTS this sentence → response.audio.delta (24k mono i16 b64)
                synth_sentence(tts, ref_codes, &sentence, barge_ref, out_ref, &resp_id);
            }
        };
        let _ = chat::generate_reply_with_history(
            llm,
            tokenizer,
            &device,
            Some(&persona),
            &history,
            &transcript,
            false,
            DEFAULT_MAX_TOKENS,
            &mut sink,
            Some(&barge),
        );
        drop(sink);
        // flush remainder
        if !barge.load(Ordering::Relaxed) {
            if let Some(rest) = splitter.flush() {
                full_reply.push_str(&rest);
                let _ = out_tx.blocking_send(Out(json!({
                    "type":"response.audio_transcript.delta","response_id":resp_id,"delta":rest
                }).to_string()));
                synth_sentence(tts, ref_codes, &rest, &barge, &out_tx, &resp_id);
            }
        }
        // transcript done
        let _ = out_tx.blocking_send(Out(json!({
            "type":"response.audio_transcript.done","response_id":resp_id,"transcript":full_reply
        }).to_string()));
        // assistant item
        {
            let mut s = session.lock().unwrap();
            s.items.push(ConvItem { id: asst_item_id, role: "assistant".into(), text: full_reply });
        }
        let status = if barge.load(Ordering::Relaxed) { "cancelled" } else { "completed" };
        let _ = out_tx.blocking_send(Out(json!({
            "type":"response.done","response":{"id":resp_id,"status":status}
        }).to_string()));
        speaking.store(false, Ordering::Relaxed);
        streaming.store(false, Ordering::Relaxed);
        eprintln!("turn {turn} {status}");
    }
}

/// Build the history slice for the LLM prompt: all conversation items EXCEPT
/// the last (the last is the current user prompt, passed separately as
/// `prompt`). Trimmed to the last `max_history_turns` turns.
fn build_history(s: &Session) -> Vec<(bool, String)> {
    let max_items = s.max_history_turns * 2;
    let n = s.items.len();
    if n <= 1 {
        return Vec::new();
    }
    let start = (n - 1).saturating_sub(max_items); // exclude the last (current prompt)
    s.items[start..n - 1]
        .iter()
        .map(|i| (i.role == "user", i.text.clone()))
        .collect()
}

fn synth_sentence(
    tts: &mut MossEngine,
    ref_codes: Option<&Tensor>,
    text: &str,
    barge: &AtomicBool,
    out_tx: &mpsc::Sender<Out>,
    resp_id: &str,
) {
    let resp_id = resp_id.to_string();
    let barge = barge;
    let mut on_chunk = |pcm: Vec<f32>| -> bool {
        if barge.load(Ordering::Relaxed) {
            return false;
        }
        let b64 = downmix_24k_mono_i16_b64(&pcm);
        let _ = out_tx.blocking_send(Out(json!({
            "type":"response.audio.delta","response_id":resp_id,"delta":b64
        }).to_string()));
        true
    };
    if let Err(e) = tts.synthesize_pcm_stream_with_codes(
        text,
        MOSS_MAX_FRAMES,
        TTS_CHUNK_FRAMES,
        ref_codes,
        &mut on_chunk,
    ) {
        eprintln!("tts error: {e}");
    }
}

/// MOSS 48 k stereo interleaved f32 → 24 k mono 16-bit → base64.
fn downmix_24k_mono_i16_b64(pcm: &[f32]) -> String {
    // The MOSS codec is stereo (2ch) at 48 k. Downmix to mono + decimate by 2.
    let ch = 2usize;
    let nframes = pcm.len() / ch;
    let mut out: Vec<u8> = Vec::with_capacity(nframes); // approx (nframes/2 *2 bytes)
    for f in (0..nframes).step_by(2) {
        let mut s = 0.0f32;
        for c in 0..ch {
            s += pcm[f * ch + c];
        }
        s = (s / ch as f32).clamp(-1.0, 1.0);
        let v = (s * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    B64.encode(&out)
}

struct SentenceSplitter {
    buf: String,
}
impl SentenceSplitter {
    const TERMINATORS: [char; 8] = ['。', '！', '？', '；', '.', '!', '?', ';'];
    fn new() -> Self {
        Self { buf: String::new() }
    }
    fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        let mut boundaries = Vec::new();
        let chars: Vec<(usize, char)> = self.buf.char_indices().collect();
        for (idx, &(i, c)) in chars.iter().enumerate() {
            if !Self::TERMINATORS.contains(&c) {
                continue;
            }
            let next = chars.get(idx + 1).map(|&(_, nc)| nc);
            if c.is_ascii() && next.is_some_and(|nc| !nc.is_whitespace()) {
                continue;
            }
            boundaries.push(i + c.len_utf8());
        }
        if let Some(&last) = boundaries.last() {
            let completed: String = self.buf.drain(..last).collect();
            let mut start = 0usize;
            for end in boundaries {
                let sentence = completed[start..end].trim();
                if sentence.chars().any(char::is_alphanumeric) {
                    out.push(sentence.to_string());
                }
                start = end;
            }
        }
        out
    }
    fn flush(&mut self) -> Option<String> {
        let rest = self.buf.trim();
        if rest.is_empty() || !rest.chars().any(char::is_alphanumeric) {
            None
        } else {
            Some(rest.to_string())
        }
    }
}

const INDEX_HTML: &str = include_str!("../static/index.html");
