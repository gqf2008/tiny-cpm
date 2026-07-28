//! tiny-cpm-web: WebSocket voice-dialogue server reusing the tiny-cpm engines
//! (FireRedVAD → Qwen3-ASR → MiniCPM5 → MOSS-TTS). The browser does mic
//! capture (getUserMedia AEC/NS/AGC — so speakers work without headphones),
//! TTS playback, live VAD-param sliders, and a persona field.

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
use candle_core::{Device, Tensor};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;

use tiny_cpm::{
    exec::{chat, moss_tts::MossEngine, qwen3_asr::Qwen3AsrEngine},
    models::fire_red_vad::vad::{FireRedVad, VadOverrides},
    quantized_minicpm5::ModelWeights,
};

const ASR_MAX_TOKENS: usize = 512;
const DEFAULT_MAX_TOKENS: usize = 256;
const MOSS_MAX_FRAMES: usize = 300;
const TTS_CHUNK_FRAMES: usize = 25;
const VAD_FRAME_SAMPLES: usize = 400;
const VAD_SAMPLE_RATE: usize = 16_000;
const MIN_SEGMENT_SAMPLES: usize = 8000;
const DEFAULT_PERSONA: &str =
    "你是一个语音助手的回复模块。直接用一两句简短的口语回答用户，禁止输出思考过程、分析、复述用户问题或任何 markdown 格式。";

struct Engines {
    device: Device,
    asr: Qwen3AsrEngine,
    llm: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    tts: MossEngine,
    ref_codes: Option<Tensor>,
    tts_sr: usize,
}

enum OutMsg {
    Audio(Vec<f32>),
    Text(String),
    StopAudio,
}

/// A turn's input: either captured audio (VAD → ASR) or typed text (skip ASR).
enum TurnInput {
    Audio(Vec<f32>),
    Text(String),
}

#[derive(Deserialize, Default, Clone)]
#[serde(default)]
struct VadMsg {
    speech_threshold: Option<f32>,
    min_speech_frame: Option<usize>,
    min_silence_frame: Option<usize>,
    min_speach_ratio: Option<f32>,
    end_silence_ratio: Option<f32>,
    min_speach_frames: Option<usize>,
    look_back_frames: Option<usize>,
}

#[derive(Deserialize)]
struct PersonaMsg {
    text: Option<String>,
}

#[derive(Deserialize)]
struct TextMsg {
    text: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ClientMsg {
    #[serde(rename = "vad")]
    Vad(VadMsg),
    #[serde(rename = "persona")]
    Persona(PersonaMsg),
    #[serde(rename = "text")]
    Text(TextMsg),
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
    let tts_sr = tts.sample_rate();
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

    let eng = Arc::new(Mutex::new(Engines {
        device: device.clone(),
        asr,
        llm,
        tokenizer,
        tts,
        ref_codes,
        tts_sr,
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

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(st): State<Arc<ServerState>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_conn(socket, st))
}

async fn handle_conn(socket: WebSocket, st: Arc<ServerState>) {
    eprintln!("ws connected");
    let (sink, stream) = socket.split();
    let (out_tx, out_rx) = mpsc::channel::<OutMsg>(64);
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<f32>>(256);
    let (seg_tx, seg_rx) = mpsc::channel::<TurnInput>(16);
    let seg_tx_listener = seg_tx.clone(); // listener + read-task both send
    let (param_tx, param_rx) = mpsc::channel::<VadOverrides>(64);

    let speaking = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let barge = Arc::new(AtomicBool::new(false));
    let persona = Arc::new(Mutex::new(None::<String>));

    // WS write task: drain out_rx → sink.
    let write_task = tokio::spawn(async move {
        let mut out_rx = out_rx;
        let mut sink = sink;
        while let Some(msg) = out_rx.recv().await {
            let m = match msg {
                OutMsg::Audio(pcm) => {
                    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                    Message::Binary(bytes)
                }
                OutMsg::Text(t) => Message::Text(t),
                OutMsg::StopAudio => Message::Text(r#"{"type":"stop_audio"}"#.to_string()),
            };
            if sink.send(m).await.is_err() {
                break;
            }
        }
    });

    // WS read task: stream → mic_tx / param_tx / persona.
    let persona_r = persona.clone();
    let read_task = tokio::spawn(async move {
        let mut stream = stream;
        while let Some(Ok(msg)) = stream.next().await {
            match msg {
                Message::Binary(buf) => {
                    if buf.len() % 4 != 0 {
                        continue;
                    }
                    let samples: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let _ = mic_tx.try_send(samples);
                }
                Message::Text(t) => match serde_json::from_str::<ClientMsg>(&t) {
                    Ok(ClientMsg::Vad(v)) => {
                        let ov = VadOverrides {
                            speech_threshold: v.speech_threshold,
                            min_speech_frame: v.min_speech_frame,
                            min_silence_frame: v.min_silence_frame,
                            min_speach_ratio: v.min_speach_ratio,
                            end_silence_ratio: v.end_silence_ratio,
                            min_speach_frames: v.min_speach_frames,
                            look_back_frames: v.look_back_frames,
                        };
                        let _ = param_tx.try_send(ov);
                    }
                    Ok(ClientMsg::Persona(p)) => {
                        *persona_r.lock().unwrap() =
                            p.text.filter(|s| !s.trim().is_empty());
                    }
                    Ok(ClientMsg::Text(tx)) => {
                        let s = tx.text.trim().to_string();
                        if !s.is_empty() {
                            let _ = seg_tx.try_send(TurnInput::Text(s));
                        }
                    }
                    Err(e) => eprintln!("bad client json: {e}: {t}"),
                },
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Listener thread (blocking): FireRedVAD + onset/barg.
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

    // Main turn thread (blocking): locks Engines.
    let eng = st.eng.clone();
    let streaming_m = streaming.clone();
    let barge_m = barge.clone();
    let persona_m = persona.clone();
    let out_tx_m = out_tx.clone();
    let speaking_m = speaking.clone();
    let main_thread = std::thread::spawn(move || {
        main_loop(eng, seg_rx, out_tx_m, persona_m, barge_m, streaming_m, speaking_m)
    });

    let _ = read_task.await;
    write_task.abort();
    let _ = listener_thread.join();
    let _ = main_thread.join();
    eprintln!("ws disconnected");
}

fn listener_loop(
    vad_dir: String,
    mut mic_rx: mpsc::Receiver<Vec<f32>>,
    seg_tx: mpsc::Sender<TurnInput>,
    mut param_rx: mpsc::Receiver<VadOverrides>,
    _speaking: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
    barge: Arc<AtomicBool>,
    out_tx: mpsc::Sender<OutMsg>,
) {
    let mut vad = match FireRedVad::init(&vad_dir, Some(&Device::Cpu), None, None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("listener: FireRedVAD init failed: {e}");
            return;
        }
    };
    let mut onset_count = 0usize;
    const ONSET_FRAMES: usize = 3;
    const MIN_BARGE_RMS: f32 = 0.015;
    loop {
        while let Ok(o) = param_rx.try_recv() {
            vad.update_params(&o);
            eprintln!("vad params updated");
        }
        let frame = match mic_rx.blocking_recv() {
            Some(f) => f,
            None => break,
        };
        if frame.len() < VAD_FRAME_SAMPLES {
            continue;
        }
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        let vad_result = match vad.detect_frame_f32(frame, 1, Some(VAD_SAMPLE_RATE)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("vad frame error: {e}");
                continue;
            }
        };
        let is_speech = vad.last_frame_speech();
        if streaming.load(Ordering::Relaxed) && is_speech && rms >= MIN_BARGE_RMS {
            onset_count += 1;
            if onset_count >= ONSET_FRAMES && !barge.swap(true, Ordering::Relaxed) {
                eprintln!("barge-in: neural speech onset (rms {rms:.3})");
                let _ = out_tx.blocking_send(OutMsg::StopAudio);
            }
        } else {
            onset_count = 0;
        }
        let Some(result) = vad_result else {
            continue;
        };
        let Some(segment) = result.orig_audio else {
            continue;
        };
        let samples = match segment.to_vec1::<f32>() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if samples.len() < MIN_SEGMENT_SAMPLES {
            continue;
        }
        if seg_tx.blocking_send(TurnInput::Audio(samples)).is_err() {
            break;
        }
    }
}

fn main_loop(
    eng: Arc<Mutex<Engines>>,
    mut seg_rx: mpsc::Receiver<TurnInput>,
    out_tx: mpsc::Sender<OutMsg>,
    persona: Arc<Mutex<Option<String>>>,
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
        turn += 1;
        let transcript = match input {
            TurnInput::Audio(samples) => {
                eprintln!(
                    "=== turn {turn}: utterance {:.2}s ===",
                    samples.len() as f32 / VAD_SAMPLE_RATE as f32
                );
                let mut eng = eng.lock().unwrap();
                match eng.asr.transcribe_samples(&samples, ASR_MAX_TOKENS) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("turn {turn}: asr error: {e}");
                        speaking.store(false, Ordering::Relaxed);
                        continue;
                    }
                }
            }
            TurnInput::Text(t) => {
                eprintln!("=== turn {turn}: text input ===");
                t
            }
        };
        let mut eng = eng.lock().unwrap();
        if transcript.trim().is_empty() {
            eprintln!("turn {turn}: empty transcript, skipping");
            speaking.store(false, Ordering::Relaxed);
            continue;
        }
        let _ = out_tx.blocking_send(OutMsg::Text(format!(
            r#"{{"type":"you","text":{}}}"#,
            serde_json::to_string(&transcript).unwrap_or_default()
        )));
        let persona_text = persona
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| DEFAULT_PERSONA.to_string());

        streaming.store(true, Ordering::Relaxed);
        let _ = out_tx.blocking_send(OutMsg::Text(
            r#"{"type":"state","value":"thinking"}"#.to_string(),
        ));
        let t_llm = Instant::now();
        // Disjoint field borrows need a plain `&mut Engines` (a MutexGuard's
        // Deref doesn't allow simultaneous &mut llm + &tokenizer + &mut tts).
        let eng_ref: &mut Engines = &mut *eng;
        let device = eng_ref.device.clone();
        let tokenizer = &eng_ref.tokenizer;
        let llm = &mut eng_ref.llm;
        let tts = &mut eng_ref.tts;
        let ref_codes = eng_ref.ref_codes.as_ref();
        let mut splitter = SentenceSplitter::new();
        let mut n = 0usize;
        let mut first = true;
        let barge_ref = &barge;
        let out_ref = &out_tx;
        let mut sink = |delta: &str| {
            if barge_ref.load(Ordering::Relaxed) {
                return;
            }
            for sentence in splitter.push(delta) {
                n += 1;
                let _ = out_ref.blocking_send(OutMsg::Text(format!(
                    r#"{{"type":"tiny","text":{}}}"#,
                    serde_json::to_string(&sentence).unwrap_or_default()
                )));
                if first {
                    eprintln!(
                        "turn {turn}: llm first sentence {:.2}s",
                        t_llm.elapsed().as_secs_f64()
                    );
                    first = false;
                    let _ = out_ref.blocking_send(OutMsg::Text(
                        r#"{"type":"state","value":"speaking"}"#.to_string(),
                    ));
                }
                let _ = synth_sentence(tts, ref_codes, &sentence, barge_ref, out_ref);
            }
        };
        let _ = chat::generate_reply_with_system(
            llm,
            tokenizer,
            &device,
            Some(&persona_text),
            &transcript,
            true,
            DEFAULT_MAX_TOKENS,
            &mut sink,
            Some(&barge),
        );
        drop(sink);
        if !barge.load(Ordering::Relaxed) {
            if let Some(rest) = splitter.flush() {
                n += 1;
                let _ = out_tx.blocking_send(OutMsg::Text(format!(
                    r#"{{"type":"tiny","text":{}}}"#,
                    serde_json::to_string(&rest).unwrap_or_default()
                )));
                let _ = synth_sentence(tts, ref_codes, &rest, &barge, &out_tx);
            }
        }
        streaming.store(false, Ordering::Relaxed);
        if barge.load(Ordering::Relaxed) {
            let _ = out_tx.blocking_send(OutMsg::StopAudio);
        }
        speaking.store(false, Ordering::Relaxed);
        eprintln!("turn {turn}: {n} sentence(s)");
    }
}

fn synth_sentence(
    tts: &mut MossEngine,
    ref_codes: Option<&Tensor>,
    text: &str,
    barge: &AtomicBool,
    out_tx: &mpsc::Sender<OutMsg>,
) -> Result<()> {
    let barge_clone = barge;
    let mut on_chunk = |pcm: Vec<f32>| -> bool {
        if barge_clone.load(Ordering::Relaxed) {
            return false;
        }
        out_tx.blocking_send(OutMsg::Audio(pcm)).is_ok()
    };
    tts.synthesize_pcm_stream_with_codes(
        text,
        MOSS_MAX_FRAMES,
        TTS_CHUNK_FRAMES,
        ref_codes,
        &mut on_chunk,
    )
    .map(|_| ())
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
