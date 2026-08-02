//! tiny-cpm-web: a Qwen-Audio / OpenAI-Realtime-style WebSocket voice server.
//!
//! Implements the core realtime event protocol: `session.update`,
//! `input_audio_buffer.append/commit`, `response.create/cancel`,
//! `conversation.item.create/delete`, and emits `session.created/updated`,
//! `input_audio_buffer.speech_started/stopped`,
//! `conversation.item.input_audio_transcription.completed`,
//! `response.created/output_audio_transcript.delta/output_audio.delta/...done`, `error`.
//!
//! Audio: input 16 kHz 16-bit mono PCM (base64 in JSON); output 24 kHz 16-bit
//! mono PCM (MOSS 48 k stereo downmixed, or Qwen3-TTS 24 k mono native — both
//! match the browser's pcm16 contract, so no client change for either engine).
//! Engines (FireRedVAD → Qwen3-ASR → MiniCPM5 → TTS[moss|qwen3]) reused from
//! tiny-cpm. The browser owns AEC/NS (getUser-Media), RMS metering, and history
//! display/control (conversation items are a transient per-session store the
//! client drives via create/delete).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    routing::get,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use candle_core::{Device, Tensor};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use tiny_cpm::{
    exec::{
        chat,
        moss_tts::MossEngine,
        qwen3_asr::Qwen3AsrEngine,
        qwen3_tts::{
            CodecDevice, Qwen3TtsEngine, STREAM_CHUNK_FRAMES, STREAM_FIRST_FRAMES,
            STREAM_LEFT_CONTEXT, TalkerQuant,
        },
    },
    models::{
        fire_red_vad::vad::{FireRedVad, VadOverrides},
        qwen3_tts::talker::RefVoice,
    },
    quantized_minicpm5::ModelWeights,
};

const ASR_MAX_TOKENS: usize = 512;
const DEFAULT_MAX_TOKENS: usize = 1024;
const MOSS_MAX_FRAMES: usize = 300;
const QWEN3_MAX_FRAMES: usize = 300; // ~24 s @ 12.5 fps codec
// Streaming granularity in codec frames (12.5 fps). Each chunk triggers a
// full codec re-decode + GPU→CPU sync, so smaller = more overhead (worse RTF,
// more playback underruns) but lower time-to-first-audio. With the browser
// AudioWorklet doing gapless queued playback, main-thread base64 size no
// longer matters — so we favor throughput. Overridable via TTS_CHUNK_FRAMES.
const DEFAULT_TTS_CHUNK_FRAMES: usize = 25;
const VAD_FRAME_SAMPLES: usize = 400;
const VAD_SAMPLE_RATE: usize = 16_000;
const MIN_SEGMENT_SAMPLES: usize = 8000;
const OUT_SR: usize = 24_000; // output sample rate (per doc)
// Persona split by mode:
// - FAST (no_think, default): a SHORT simple persona. The long complex persona
//   (with date/time + many prohibitions) is what made no_think repeat the
//   previous assistant turn — a 1B model treats the verbose prompt as a template
//   to copy. Measured: this short persona + no_think gives 6-31 tokens/turn with
//   ZERO repeats across 5 turns.
// - THINK mode (TINY_CPM_CHAT_THINK=1): the verbose persona, where the extra
//   guidance helps reasoning. Think is slower (~150 thinking tokens) but kept as
//   an A/B + rollback path.
const FAST_PERSONA: &str = "你是语音助手。每次针对用户当前这句话给出简短新回答，不要重复历史，不要加客套话或表情。";
const THINK_PERSONA: &str = "你是一个语音助手。直接用一两句简短的口语回答用户，禁止输出思考过程、分析、复述用户问题或任何 markdown 格式。不知道时间日期就直接回答不知道，不要编造；不要重复上一句话，不要每句都加客套话或表情。";
/// `true` = think mode (no_think=false, verbose persona). Env: TINY_CPM_CHAT_THINK.
fn think_mode() -> bool {
    matches!(
        std::env::var("TINY_CPM_CHAT_THINK").ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// Read a positive usize env knob with a default (for the Qwen3-TTS streaming tunables).
fn env_usize_or(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Selectable TTS engine, mirroring `live`'s `LiveTts`. The browser audio
/// contract is 24 kHz mono pcm16 (see `OUT_SR`), so both engines emit that:
/// MOSS is 48 k stereo downmixed to 24 k mono (`downmix_24k_mono_i16_b64`);
/// Qwen3 is 24 k mono natively (`mono_24k_i16_b64`).
enum WebTts {
    Moss(MossEngine),
    Qwen3(Qwen3TtsEngine),
}

/// Encoded reference voice, engine-specific (MOSS = codec code tensor; Qwen3 =
/// ECAPA speaker embedding + ref codec codes). Encoded once at startup and
/// reused per reply sentence (mirrors `live`'s `LiveRef`).
enum WebRef {
    Moss(Tensor),
    Qwen3(RefVoice),
}

#[derive(Clone, Copy, PartialEq)]
enum TtsChoice {
    Moss,
    Qwen3,
}
impl TtsChoice {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "moss" => Ok(Self::Moss),
            "qwen3" => Ok(Self::Qwen3),
            other => Err(anyhow!("unknown --tts `{other}` (expected moss | qwen3)")),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Moss => "MOSS",
            Self::Qwen3 => "Qwen3",
        }
    }
}

impl WebTts {
    fn encode_ref(&self, ref_wav: &str, ref_text: Option<&str>) -> Result<WebRef> {
        match self {
            WebTts::Moss(e) => Ok(WebRef::Moss(e.encode_ref(ref_wav)?)),
            WebTts::Qwen3(e) => {
                // `--ref-text` is validated at CLI parse time; this is a backstop.
                let rt = ref_text.ok_or_else(|| {
                    anyhow!("--tts qwen3 --ref requires --ref-text \"<full ref transcript>\"")
                })?;
                Ok(WebRef::Qwen3(e.encode_ref(ref_wav, rt)?))
            }
        }
    }
}

struct Engines {
    device: Device,
    asr: Qwen3AsrEngine,
    llm: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    tts: WebTts,
    ref_voice: Option<WebRef>,
}

/// Per-session conversation item (transient store; client drives via create/delete).
#[derive(Clone)]
struct ConvItem {
    id: String,
    role: String, // user | assistant | system
    text: String,
}

#[derive(Clone, Debug)]
enum TurnMode {
    ServerVad { threshold: f32, silence_ms: u64 },
    Manual,
}

#[derive(Clone)]
struct Session {
    id: String,
    conv_id: String,
    instructions: Option<String>,
    turn_mode: TurnMode,
    max_history_turns: usize,
    items: Vec<ConvItem>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            id: "sess_default".into(),
            conv_id: "conv_default".into(),
            instructions: None,
            turn_mode: TurnMode::ServerVad {
                threshold: 0.5,
                silence_ms: 800,
            },
            max_history_turns: 20,
            items: Vec::new(),
        }
    }
}

/// One outbound WS text frame (a JSON event). All server→client is JSON (audio
/// is base64 inside `response.output_audio.delta`, per the doc).
struct Out(String);

/// Per-connection monotonic id generator. One `Arc<AtomicU64>` shared (cloned)
/// across the read task, the VAD listener thread, and the main turn thread, so
/// every emitted event gets a globally-unique `event_id` — the Realtime contract
/// (clients key on it for ack/replay/dedup). Previously each turn reset a local
/// counter, so `event_id` repeated across turns.
#[derive(Clone)]
struct IdGen(Arc<AtomicU64>);
impl IdGen {
    fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }
    fn next(&self, prefix: &str) -> String {
        format!("{prefix}_{}", self.0.fetch_add(1, Ordering::Relaxed))
    }
    /// Next server `event_id` ("event_<n>").
    fn ev(&self) -> String {
        self.next("event")
    }
    /// Next conversation `item_id` ("item_<n>").
    fn item(&self) -> String {
        self.next("item")
    }
}

/// A turn's input: voiced audio (→ ASR) or typed text (client already pushed
/// a user conversation item; use its text directly). For audio, `item_id` is
/// pre-allocated in the VAD listener at speech_started so the speech_started /
/// speech_stopped / committed events and the resulting user conversation item all
/// share one id (Realtime contract).
enum TurnInput {
    Audio(Vec<f32>, String),
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

    let usage = "usage: tiny-cpm-web --vad-dir <d> --asr-dir <d> --minicpm-dir <gguf|bf16-dir> --tokenizer <json> --tts-dir <d> [--codec-dir <d: MOSS only>] [--tts moss|qwen3 (default qwen3)] [--talker-quant q4_k|q8_0|none (default q4_k)] [--ref <wav> [--ref-text \"<text>\"]] [--addr 127.0.0.1:8080] [--quant q8_0|q4_k_m|q5_k_m|q6_k|...]";
    let mut vad_dir = None;
    let mut asr_dir = None;
    let mut minicpm_dir = None;
    let mut tok_path = None;
    let mut tts_dir = None;
    let mut codec_dir = None;
    let mut tts_choice = TtsChoice::Qwen3;
    let mut talker_quant_cli: Option<TalkerQuant> = None;
    let mut ref_wav: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut quant = None;
    let mut addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || {
            args.next()
                .ok_or_else(|| anyhow!("{a} requires a value. {usage}"))
        };
        match a.as_str() {
            "--vad-dir" => vad_dir = Some(v()?),
            "--asr-dir" => asr_dir = Some(v()?),
            "--minicpm-dir" => minicpm_dir = Some(v()?),
            "--tokenizer" => tok_path = Some(v()?),
            "--tts-dir" => tts_dir = Some(v()?),
            "--codec-dir" => codec_dir = Some(v()?),
            "--tts" => tts_choice = TtsChoice::parse(&v()?)?,
            "--talker-quant" => {
                talker_quant_cli =
                    Some(TalkerQuant::parse(&v()?).map_err(|e| anyhow!("{e}. {usage}"))?);
            }
            "--ref" => ref_wav = Some(v()?),
            "--ref-text" => ref_text = Some(v()?),
            "--quant" => quant = Some(v()?),
            "--addr" => addr = v()?.parse().map_err(|_| anyhow!("bad --addr. {usage}"))?,
            other => bail!("unknown option {other}. {usage}"),
        }
    }
    let (vad_dir, asr_dir, minicpm_dir, tok_path, tts_dir) = (
        vad_dir.ok_or_else(|| anyhow!("--vad-dir required. {usage}"))?,
        asr_dir.ok_or_else(|| anyhow!("--asr-dir required. {usage}"))?,
        minicpm_dir.ok_or_else(|| anyhow!("--minicpm-dir required. {usage}"))?,
        tok_path.ok_or_else(|| anyhow!("--tokenizer required. {usage}"))?,
        tts_dir.ok_or_else(|| anyhow!("--tts-dir required. {usage}"))?,
    );
    // Engine-specific validation. `codec_dir` ends up `Some` only for MOSS.
    let codec_dir: Option<String> = match tts_choice {
        TtsChoice::Moss => {
            Some(codec_dir.ok_or_else(|| anyhow!("--tts moss needs --codec-dir. {usage}"))?)
        }
        TtsChoice::Qwen3 => {
            if codec_dir.is_some() {
                bail!("--codec-dir is MOSS-only; qwen3-tts bundles its codec. {usage}");
            }
            if ref_wav.is_some() && ref_text.is_none() {
                bail!("--tts qwen3 --ref requires --ref-text \"<full ref transcript>\". {usage}");
            }
            if ref_wav.is_none() && ref_text.is_some() {
                bail!("--ref-text requires --ref. {usage}");
            }
            None
        }
    };
    if matches!(tts_choice, TtsChoice::Moss) && ref_text.is_some() {
        eprintln!(
            "warning: --ref-text is ignored for --tts moss (it clones from --ref audio alone)"
        );
    }
    if matches!(tts_choice, TtsChoice::Moss) && talker_quant_cli.is_some() {
        eprintln!("warning: --talker-quant is ignored for --tts moss");
    }

    let device = Device::new_metal(0)?;
    let t = Instant::now();
    let asr = Qwen3AsrEngine::load(&asr_dir, &device)?;
    eprintln!("loaded Qwen3-ASR in {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let llm = chat::load_model_with_quant(&minicpm_dir, quant.as_deref(), &device)?;
    eprintln!("loaded MiniCPM5 in {:.2}s", t.elapsed().as_secs_f64());
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let t = Instant::now();
    // MOSS-only: snappier default sampling temperature. MOSS's upstream default
    // (1.7) drags each phoneme out and inserts more pauses — the reply sounds
    // slow. A/B on the same sentence: 1.7 → 3.84s / 27% silence, 1.2 → 3.44s /
    // 22% silence (and it never runs away like ≤0.8, where the end token stops
    // firing). Still overridable via MOSS_TEMPERATURE. Irrelevant for Qwen3
    // (its own sampling config drives codebook 0 + the code predictor).
    if matches!(tts_choice, TtsChoice::Moss) && std::env::var("MOSS_TEMPERATURE").is_err() {
        unsafe { std::env::set_var("MOSS_TEMPERATURE", "1.2") };
    }
    // Qwen3 talker quant: CLI flag > env TINY_CPM_QWEN3_TTS_TALKER > q4_k (the
    // live/web default, unlike the tts-CLI default bf16 — sub-realtime RTF).
    let talker_quant = match tts_choice {
        TtsChoice::Moss => None,
        TtsChoice::Qwen3 => Some(
            talker_quant_cli
                .or_else(|| {
                    std::env::var("TINY_CPM_QWEN3_TTS_TALKER")
                        .ok()
                        .and_then(|s| TalkerQuant::parse(&s).ok())
                })
                .unwrap_or(TalkerQuant::Q4K),
        ),
    };
    let tts = match tts_choice {
        TtsChoice::Moss => WebTts::Moss(MossEngine::load(&tts_dir, &codec_dir.unwrap(), &device)?),
        TtsChoice::Qwen3 => WebTts::Qwen3(Qwen3TtsEngine::load_with_quant_and_codec_device(
            &tts_dir,
            &device,
            talker_quant.unwrap(),
            // Dedicated Metal device for the codec decoder so its streaming
            // sliding-window decode is isolated from the talker's GPU pool — the
            // multi-model babble fix (see CLAUDE.md "streaming decode corrupts in a
            // multi-model Metal process"). Env-overridable: `cpu` = zero-code
            // fallback (guaranteed isolation), `shared` = single-device (batched-only).
            CodecDevice::from_env_default(CodecDevice::Dedicated),
        )?),
    };
    eprintln!(
        "loaded {} TTS{} in {:.2}s",
        tts_choice.label(),
        match tts_choice {
            TtsChoice::Moss => String::new(),
            TtsChoice::Qwen3 => format!(" (talker {})", talker_quant.unwrap().label()),
        },
        t.elapsed().as_secs_f64()
    );
    let ref_voice = if let Some(ref_path) = &ref_wav {
        let t = Instant::now();
        let r = tts.encode_ref(ref_path, ref_text.as_deref())?;
        let detail = match &r {
            WebRef::Moss(codes) => format!("{} frames", codes.dim(0)?),
            WebRef::Qwen3(_) => "spk emb + ref codes".to_string(),
        };
        eprintln!(
            "encoded voice ref {} ({detail}) in {:.2}s",
            ref_path,
            t.elapsed().as_secs_f64()
        );
        Some(r)
    } else {
        None
    };
    eprintln!(
        "output: {OUT_SR} Hz mono 16-bit{}",
        match tts_choice {
            TtsChoice::Moss => " (downmixed from MOSS 48 k stereo)",
            TtsChoice::Qwen3 => " (Qwen3 native)",
        }
    );

    let eng = Arc::new(Mutex::new(Engines {
        device: device.clone(),
        asr,
        llm,
        tokenizer,
        tts,
        ref_voice,
    }));
    let st = Arc::new(ServerState { eng, vad_dir });
    // Serve the UI from disk (not baked in) so index.html / worklets/*.js edits
    // don't need a rebuild. STATIC_DIR is resolved at compile time to web/static.
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .nest_service(
            "/",
            tower_http::services::ServeDir::new(STATIC_DIR).append_index_html_on_directories(true),
        )
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
    // Per-connection id generator (event_id / item_id) shared by the read task,
    // VAD listener, and main turn thread — globally-unique ids across the session.
    let idg = IdGen::new();
    // `input_audio_buffer.clear` request: the read task sets this, the VAD listener
    // drains its accumulated mic buffer on the next frame.
    let want_clear = Arc::new(AtomicBool::new(false));
    // Assign stable session / conversation ids for this connection (Realtime echoes
    // them in session.created and every response.created / response.done).
    {
        let mut s = session.lock().unwrap();
        s.id = idg.next("sess");
        s.conv_id = idg.next("conv");
    }

    // session.created — echo the full Realtime session object. Build the payload under
    // the lock, then drop the guard BEFORE the .await (MutexGuard is not Send).
    let payload = session_payload(&session.lock().unwrap());
    let _ = out_tx
        .send(Out(json!({
            "type": "session.created",
            "event_id": idg.ev(),
            "session": payload,
        }).to_string()))
        .await;

    let session_r = session.clone();
    let out_tx_r = out_tx.clone();
    let barge_r = barge.clone();
    let idg_r = idg.clone();
    let want_clear_r = want_clear.clone();
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
                    let _ = out_tx_r
                        .send(error_event(
                            &idg_r,
                            "invalid_request_error",
                            "invalid_event",
                            &format!("bad json: {e}"),
                            None,
                        ))
                        .await;
                    continue;
                }
            };
            let t = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match t {
                "session.update" => {
                    let s = ev.get("session").cloned().unwrap_or(json!({}));
                    apply_session_update(&session_r, &s, &param_tx);
                    let payload = session_payload(&session_r.lock().unwrap());
                    let _ = out_tx_r
                        .send(Out(json!({
                            "type": "session.updated",
                            "event_id": idg_r.ev(),
                            "session": payload,
                        }).to_string()))
                        .await;
                }
                "input_audio_buffer.append" => {
                    if let Some(a) = ev.get("audio").and_then(|v| v.as_str()) {
                        if let Ok(bytes) = B64.decode(a) {
                            let f: Vec<f32> = bytes
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                                .collect();
                            // server_vad → VAD listener; manual → buffer for commit
                            let is_manual =
                                matches!(session_r.lock().unwrap().turn_mode, TurnMode::Manual);
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
                "input_audio_buffer.clear" => {
                    // Drop any buffered manual audio and tell the VAD listener to drain
                    // its accumulated mic buffer on the next frame. Spec: respond with
                    // input_audio_buffer.cleared.
                    manual_buf.clear();
                    want_clear_r.store(true, Ordering::Relaxed);
                    let _ = out_tx_r
                        .send(Out(json!({
                            "event_id": idg_r.ev(),
                            "type": "input_audio_buffer.cleared"
                        }).to_string()))
                        .await;
                }
                "response.create" => {
                    // Manual mode: if audio buffered, ASR it; else use the latest
                    // user conversation item (text the client pushed via
                    // conversation.item.create) as the prompt.
                    let buf = std::mem::take(&mut manual_buf);
                    if !buf.is_empty() {
                        let _ = seg_tx.try_send(TurnInput::Audio(buf, idg_r.item()));
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
                        let role = item
                            .get("role")
                            .and_then(|v| v.as_str())
                            .unwrap_or("user")
                            .to_string();
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
                            let (prev, id) = {
                                let mut s = session_r.lock().unwrap();
                                let prev = s.items.last().map(|i| i.id.clone());
                                let id = idg_r.item();
                                s.items.push(ConvItem {
                                    id: id.clone(),
                                    role: role.clone(),
                                    text: text.clone(),
                                });
                                (prev, id)
                            };
                            // Spec: respond with conversation.item.added. A user message
                            // content part is `input_text`; an assistant text item is `text`.
                            let part_type = if role == "assistant" { "text" } else { "input_text" };
                            let _ = out_tx_r
                                .send(Out(json!({
                                    "event_id": idg_r.ev(),
                                    "type": "conversation.item.added",
                                    "previous_item_id": prev,
                                    "item": {
                                        "id": id, "object": "realtime.item", "type": "message",
                                        "role": role, "status": "completed",
                                        "content": [{ "type": part_type, "text": text }]
                                    }
                                }).to_string()))
                                .await;
                        }
                    }
                }
                "conversation.item.delete" => {
                    if let Some(id) = ev.get("item_id").and_then(|v| v.as_str()) {
                        {
                            let mut s = session_r.lock().unwrap();
                            s.items.retain(|i| i.id != id);
                        }
                        // Spec: respond with conversation.item.deleted.
                        let _ = out_tx_r
                            .send(Out(json!({
                                "event_id": idg_r.ev(),
                                "type": "conversation.item.deleted",
                                "item_id": id
                            }).to_string()))
                            .await;
                    }
                }
                _ => {
                    let client_eid = ev.get("event_id").and_then(|v| v.as_str());
                    let _ = out_tx_r
                        .send(error_event(
                            &idg_r,
                            "invalid_request_error",
                            "unsupported_event",
                            &format!("unsupported event: {t}"),
                            client_eid,
                        ))
                        .await;
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
    let idg_l = idg.clone();
    let want_clear_l = want_clear.clone();
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
            idg_l,
            want_clear_l,
        )
    });

    // Main turn thread.
    let eng = st.eng.clone();
    let session_m = session.clone();
    let streaming_m = streaming.clone();
    let barge_m = barge.clone();
    let out_tx_m = out_tx.clone();
    let speaking_m = speaking.clone();
    let idg_m = idg.clone();
    let main_thread = std::thread::spawn(move || {
        main_loop(
            eng,
            seg_rx,
            out_tx_m,
            session_m,
            barge_m,
            streaming_m,
            speaking_m,
            idg_m,
        )
    });

    let _ = read_task.await;
    write_task.abort();
    let _ = listener_thread.join();
    let _ = main_thread.join();
    eprintln!("ws disconnected");
}

fn apply_session_update(
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
    // turn_detection lives under `audio.input` in the 2025-08 session shape.
    let td = s
        .get("audio")
        .and_then(|a| a.get("input"))
        .and_then(|i| i.get("turn_detection"));
    if let Some(td) = td {
        if td.is_null() {
            sess.turn_mode = TurnMode::Manual;
        } else if let Some(ty) = td.get("type").and_then(|v| v.as_str()) {
            match ty {
                "server_vad" => {
                    let threshold =
                        td.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                    let silence_ms = td
                        .get("silence_duration_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(800);
                    sess.turn_mode = TurnMode::ServerVad {
                        threshold,
                        silence_ms,
                    };
                    // Map to FireRedVAD params: threshold→speech_threshold,
                    // silence_duration_ms→min_silence_frame (frame_shift ~25ms).
                    let _ = param_tx.try_send(VadOverrides {
                        speech_threshold: Some(threshold),
                        min_silence_frame: Some((silence_ms / 25).max(1) as usize),
                        ..Default::default()
                    });
                }
                "smart_turn" => {
                    // Not supported (no semantic turn model) — fall back to server_vad.
                    eprintln!("warning: smart_turn not supported, using server_vad");
                    sess.turn_mode = TurnMode::ServerVad {
                        threshold: 0.5,
                        silence_ms: 800,
                    };
                }
                _ => {}
            }
        }
    }
}

/// Serialize the session into the Realtime `session` object (echoed in
/// `session.created` / `session.updated`), in the 2025-08 shape: nested
/// `audio.input` / `audio.output`, `output_modalities`, `max_output_tokens`.
/// `max_history_turns` is a non-standard extension the local browser client uses.
fn session_payload(sess: &Session) -> Value {
    let turn_detection = match &sess.turn_mode {
        TurnMode::ServerVad { threshold, silence_ms } => json!({
            "type": "server_vad",
            "threshold": threshold,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_ms,
            "create_response": true,
            "interrupt_response": true,
            "eagerness": "auto",
        }),
        TurnMode::Manual => Value::Null,
    };
    json!({
        "id": sess.id,
        "object": "realtime.session",
        "type": "realtime",
        "model": "tiny-cpm",
        "output_modalities": ["audio"],
        "instructions": sess.instructions.clone().unwrap_or_default(),
        "tools": [],
        "tool_choice": "auto",
        "max_output_tokens": "inf",
        "tracing": null,
        "expires_at": null,
        "audio": {
            "input": {
                "format": { "type": "audio/pcm", "rate": VAD_SAMPLE_RATE },
                "transcription": { "model": "qwen3-asr-0.6b" },
                "noise_reduction": null,
                "turn_detection": turn_detection,
            },
            "output": {
                "format": { "type": "audio/pcm", "rate": OUT_SR },
                "voice": "alloy",
                "speed": 1,
            }
        },
        // non-standard: how many past turns the local LLM keeps in prompt history.
        "max_history_turns": sess.max_history_turns,
    })
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
    idg: IdGen,
    want_clear: Arc<AtomicBool>,
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
    // Total samples written to the input buffer this session — drives the
    // audio_start_ms / audio_end_ms offsets in speech_started / speech_stopped.
    let mut total_samples: usize = 0;
    // item_id pre-allocated at speech_started; reused for speech_stopped and the
    // user conversation item created from the committed buffer (Realtime contract).
    let mut pending_item_id: Option<String> = None;
    loop {
        while let Ok(o) = param_rx.try_recv() {
            vad.update_params(&o);
        }
        // input_audio_buffer.clear: drain the accumulated mic buffer.
        if want_clear.swap(false, Ordering::Relaxed) {
            mic_buf.clear();
        }
        let chunk = match mic_rx.blocking_recv() {
            Some(c) => c,
            None => break,
        };
        total_samples += chunk.len();
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
            // Diagnostic (opt-in via VAD_DEBUG=1): log every ~1s of frames to see
            // whether VAD detects speech/endpoints on the incoming mic audio.
            if std::env::var("VAD_DEBUG").is_ok() {
                static VAD_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let n = VAD_LOG.fetch_add(1, Ordering::Relaxed);
                if n % 40 == 0 {
                    eprintln!("vad: frame#{n} rms={rms:.4} speech={is_speech} emitted={speech_started_emitted}");
                }
            }
            // speech_started/stopped events
            if is_speech && !speech_started_emitted {
                speech_started_emitted = true;
                let item_id = idg.item();
                pending_item_id = Some(item_id.clone());
                let audio_start_ms = total_samples * 1000 / VAD_SAMPLE_RATE;
                let _ = out_tx.blocking_send(Out(
                    json!({"event_id": idg.ev(),
                           "type":"input_audio_buffer.speech_started",
                           "audio_start_ms": audio_start_ms,
                           "item_id": item_id}).to_string(),
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
            let Some(segment) = result.orig_audio else {
                continue;
            };
            let samples = match segment.to_vec1::<f32>() {
                Ok(s) => s,
                Err(_) => continue,
            };
            if std::env::var("VAD_DEBUG").is_ok() {
                eprintln!("vad: endpoint → segment {:.2}s ({} samples)", samples.len() as f32 / VAD_SAMPLE_RATE as f32, samples.len());
            }
            // endpoint → speech_stopped + segment
            if speech_started_emitted {
                speech_started_emitted = false;
                let audio_end_ms = total_samples * 1000 / VAD_SAMPLE_RATE;
                let item_id = pending_item_id.clone().unwrap_or_else(|| idg.item());
                let _ = out_tx.blocking_send(Out(
                    json!({"event_id": idg.ev(),
                           "type":"input_audio_buffer.speech_stopped",
                           "audio_end_ms": audio_end_ms,
                           "item_id": item_id}).to_string(),
                ));
            }
            if samples.len() >= MIN_SEGMENT_SAMPLES {
                let item_id = pending_item_id.take().unwrap_or_else(|| idg.item());
                if seg_tx.blocking_send(TurnInput::Audio(samples, item_id)).is_err() {
                    break;
                }
            }
        }
    }
}

/// Emit a Realtime lifecycle event with a globally-unique `event_id` (from the
/// per-connection `IdGen`). `$idg` is an `IdGen` (or `&IdGen`) in scope at the call
/// site — typically `main_loop`'s clone, but usable from any thread holding one.
/// High-frequency audio/transcript deltas go through the same id generator.
macro_rules! emit {
    ($tx:expr, $idg:expr, $($k:literal : $v:tt),+ $(,)?) => {
        let _ = $tx.blocking_send(Out(json!({
            "event_id": $idg.ev(),
            $( $k: $v ),+
        }).to_string()));
    };
}

/// Build a Realtime `error` event with the full error object the spec requires
/// (`type` / `code` / `message` / `param` / `event_id`). `client_event_id` is the id
/// of the client event that triggered the error (None when it had none, e.g.
/// unparseable JSON). `etype` is `"invalid_request_error"` | `"server_error"`.
fn error_event(
    idg: &IdGen,
    etype: &str,
    code: &str,
    message: &str,
    client_event_id: Option<&str>,
) -> Out {
    Out(json!({
        "event_id": idg.ev(),
        "type": "error",
        "error": {
            "type": etype,
            "code": code,
            "message": message,
            "param": null,
            "event_id": client_event_id,
        }
    }).to_string())
}

fn main_loop(
    eng: Arc<Mutex<Engines>>,
    mut seg_rx: mpsc::Receiver<TurnInput>,
    out_tx: mpsc::Sender<Out>,
    session: Arc<Mutex<Session>>,
    barge: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
    idg: IdGen,
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
        let conv_id = session.lock().unwrap().conv_id.clone();
        // For voiced input the user item id was pre-allocated at speech_started and
        // shared across speech_started / committed / conversation.item. Text input has
        // none — user_item_id is only consumed on the Audio path below.
        let user_item_id = match &input {
            TurnInput::Audio(_, id) => id.clone(),
            TurnInput::Text(_) => format!("item_u{turn}"),
        };
        let asst_item_id = format!("item_a{turn}");
        let prev_item_id = session
            .lock()
            .unwrap()
            .items
            .last()
            .map(|i| i.id.clone());

        if matches!(input, TurnInput::Audio(..)) {
            // server_vad committed the audio buffer at the VAD endpoint.
            emit!(&out_tx, &idg, "type": "input_audio_buffer.committed",
                  "previous_item_id": prev_item_id,
                  "item_id": user_item_id);
        }
        emit!(&out_tx, &idg, "type": "response.created",
              "response": {
                  "id": resp_id, "object": "realtime.response",
                  "status": "in_progress", "status_details": null,
                  "output": [], "conversation_id": conv_id,
                  "output_modalities": ["audio"], "max_output_tokens": "inf",
                  "audio": { "output": { "format": { "type": "audio/pcm", "rate": OUT_SR }, "voice": "alloy" } },
                  "usage": null, "metadata": null
              });

        // Voice → ASR + push a user item; Text → item already exists (client pushed it).
        let transcript = match input {
            TurnInput::Audio(samples, _) => {
                eprintln!(
                    "=== turn {turn}: utterance {:.2}s ===",
                    samples.len() as f32 / VAD_SAMPLE_RATE as f32
                );
                let t_asr = std::time::Instant::now();
                let t = {
                    let mut e = eng.lock().unwrap();
                    // Streaming ASR: emit input_audio_transcription.delta (the new suffix)
                    // as the transcript grows; .completed fires with the final text below.
                    // The decoded prefix is monotonic (generate only grows, BPE decode is a
                    // left-to-right concat), so the suffix = full[len(last)..].
                    let mut last = String::new();
                    let mut on_partial = |full: &str| {
                        let delta = if full.starts_with(last.as_str()) {
                            &full[last.len()..]
                        } else {
                            // Prefix refined (rare) — resend the whole transcript.
                            full
                        };
                        if !delta.is_empty() {
                            let _ = out_tx.blocking_send(Out(
                                json!({
                                    "event_id": idg.ev(),
                                    "type":"conversation.item.input_audio_transcription.delta",
                                    "item_id": user_item_id,
                                    "content_index": 0,
                                    "delta": delta
                                })
                                .to_string(),
                            ));
                            last = full.to_string();
                        }
                    };
                    match e
                        .asr
                        .transcribe_samples_streaming(&samples, ASR_MAX_TOKENS, &mut on_partial)
                    {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("turn {turn}: asr error: {e}");
                            emit!(&out_tx, &idg, "type": "response.done",
                                  "response": {
                                      "id": resp_id, "object": "realtime.response",
                                      "status": "failed",
                                      "status_details": { "type": "asr_failed", "reason": format!("{e}") },
                                      "output": [], "conversation_id": conv_id,
                                      "output_modalities": ["audio"], "max_output_tokens": "inf",
                                      "usage": null, "metadata": null
                                  });
                            speaking.store(false, Ordering::Relaxed);
                            streaming.store(false, Ordering::Relaxed);
                            continue;
                        }
                    }
                };
                eprintln!(
                    "turn {turn}: asr {:.2}s → \"{}\"",
                    t_asr.elapsed().as_secs_f64(),
                    t.trim().chars().take(60).collect::<String>()
                );
                if t.trim().is_empty() {
                    emit!(&out_tx, &idg, "type": "response.done",
                          "response": {
                              "id": resp_id, "object": "realtime.response",
                              "status": "completed", "status_details": null,
                              "output": [], "conversation_id": conv_id,
                              "output_modalities": ["audio"], "max_output_tokens": "inf",
                              "usage": null, "metadata": null
                          });
                    speaking.store(false, Ordering::Relaxed);
                    streaming.store(false, Ordering::Relaxed);
                    continue;
                }
                emit!(&out_tx, &idg, "type": "conversation.item.input_audio_transcription.completed",
                      "item_id": user_item_id, "content_index": 0, "transcript": t);
                session.lock().unwrap().items.push(ConvItem {
                    id: user_item_id.clone(),
                    role: "user".into(),
                    text: t.clone(),
                });
                emit!(&out_tx, &idg, "type": "conversation.item.added",
                      "previous_item_id": prev_item_id,
                      "item": {
                          "id": user_item_id, "object": "realtime.item", "type": "message",
                          "role": "user", "status": "completed",
                          "content": [{ "type": "input_text", "text": t }]
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
        // Persona + think mode: fast path uses a short simple persona (no date
        // injection — the verbose prompt was what caused no_think to repeat). Think
        // mode keeps the verbose persona + injected clock (helps reasoning; "几点"
        // would otherwise hallucinate).
        let think = think_mode();
        let default_persona = if think { THINK_PERSONA } else { FAST_PERSONA };
        let base_persona = instructions.unwrap_or_else(|| default_persona.to_string());
        let persona = if think {
            format!("{base_persona}\n现在是：{}。", now_string())
        } else {
            base_persona
        };
        eprintln!(
            "turn {turn}: history={} items, prompt=\"{}\", persona=\"{}\"",
            history.len(),
            transcript.chars().take(80).collect::<String>(),
            persona.chars().take(40).collect::<String>(),
        );
        for (i, (is_user, text)) in history.iter().enumerate() {
            eprintln!(
                "  hist[{i}] {}: \"{}\"",
                if *is_user { "user" } else { "asst" },
                text.chars().take(60).collect::<String>()
            );
        }

        // Realtime response-item lifecycle: one assistant message output (output_index 0)
        // carrying one audio content part (content_index 0). The transcript + audio deltas
        // below stream into this part; audio.done / content_part.done / output_item.done
        // close it after the TTS worker drains.
        let out_index = 0u32;
        let content_index = 0u32;
        emit!(&out_tx, &idg, "type": "response.output_item.added",
              "response_id": resp_id, "output_index": out_index,
              "item": { "id": asst_item_id, "object": "realtime.item", "type": "message",
                        "role": "assistant", "status": "in_progress", "content": [] });
        emit!(&out_tx, &idg, "type": "response.content_part.added",
              "response_id": resp_id, "item_id": asst_item_id,
              "output_index": out_index, "content_index": content_index,
              "part": { "type": "audio", "audio": "" });

        let mut e = eng.lock().unwrap();
        let eng_ref: &mut Engines = &mut *e;
        let device = eng_ref.device.clone();
        let tokenizer = &eng_ref.tokenizer;
        let llm = &mut eng_ref.llm;
        let tts = &mut eng_ref.tts;
        let ref_voice = eng_ref.ref_voice.as_ref();
        let out_ref = &out_tx;
        let barge_ref = &barge;
        let mut full_reply = String::new();
        let mut splitter = SentenceSplitter::new();

        // Pipeline LLM → TTS: the LLM generates at full speed and pushes whole
        // sentences onto a channel; a scoped worker thread runs the (blocking,
        // Metal-heavy) TTS synthesis. Without this decoupling the sink does TTS
        // inline per sentence, serializing the two models on the same Metal
        // device and dragging the LLM from ~40 tok/s down to ~3 tok/s.
        let (sen_tx, mut sen_rx) = mpsc::channel::<String>(8);
        let out_tts = out_tx.clone();
        let resp_tts = resp_id.to_string();
        let item_tts = asst_item_id.clone();
        let barge_tts = Arc::clone(&barge);
        let idg_tts = idg.clone();
        let tts_worker = move || {
            while let Some(sentence) = sen_rx.blocking_recv() {
                if barge_tts.load(Ordering::Relaxed) {
                    continue;
                }
                synth_sentence(
                    tts, ref_voice, &sentence, &barge_tts, &out_tts, &resp_tts,
                    &item_tts, out_index, content_index, &idg_tts,
                );
            }
        };

        std::thread::scope(|scope| {
            let handle = scope.spawn(tts_worker);

            let mut sink = |delta: &str| {
                if barge_ref.load(Ordering::Relaxed) {
                    return;
                }
                full_reply.push_str(delta);
                let _ = out_ref.blocking_send(Out(json!({
                    "event_id": idg.ev(),
                    "type":"response.output_audio_transcript.delta","response_id":resp_id,
                    "item_id": asst_item_id, "output_index": out_index,
                    "content_index": content_index, "delta": delta
                })
                .to_string()));
                for sentence in splitter.push(delta) {
                    let _ = sen_tx.blocking_send(sentence);
                }
            };
            // no_think (fast, default): MiniCPM5-1B answers directly in 6-31
            // tokens with a SHORT simple persona (FAST_PERSONA). The earlier
            // "no_think repeats the previous assistant turn" bug was caused by
            // the long complex persona (a 1B model copies the verbose prompt as a
            // template), NOT by no_think itself — verified: short persona + no_think
            // across 5 turns = zero repeats. stream_clean in no_think mode emits the
            // raw generated text (no think tags in the output). TINY_CPM_CHAT_THINK=1
            // switches to think mode (verbose persona + <think>, ~150 tokens slower)
            // for A/B comparison or rollback.
            let no_think = !think;
            let _ = chat::generate_reply_with_history(
                llm,
                tokenizer,
                &device,
                Some(&persona),
                &history,
                &transcript,
                no_think,
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
                        "event_id": idg.ev(),
                        "type":"response.output_audio_transcript.delta","response_id":resp_id,
                        "item_id": asst_item_id, "output_index": out_index,
                        "content_index": content_index, "delta": rest
                    })
                    .to_string()));
                    let _ = sen_tx.blocking_send(rest);
                }
            }
            // Close the sentence channel; the worker drains queued sentences
            // then exits, and we wait for it so `tts` outlives the scope.
            drop(sen_tx);
            let _ = handle.join();
        });
        // Realtime close lifecycle for the assistant audio part. The spec requires
        // output_audio.done / output_audio_transcript.done / content_part.done /
        // output_item.done on BOTH completion and cancellation (interrupted/cancelled
        // still close the in-progress part). The final item carries the transcript as
        // an `output_audio` content part. status: completed → "completed", cancelled →
        // the item is "incomplete" while the response itself is "cancelled".
        let status = if barge.load(Ordering::Relaxed) {
            "cancelled"
        } else {
            "completed"
        };
        let item_status = if status == "completed" { "completed" } else { "incomplete" };
        emit!(&out_tx, &idg, "type": "response.output_audio.done",
              "response_id": resp_id, "item_id": asst_item_id,
              "output_index": out_index, "content_index": content_index);
        emit!(&out_tx, &idg, "type": "response.output_audio_transcript.done",
              "response_id": resp_id, "item_id": asst_item_id,
              "output_index": out_index, "content_index": content_index,
              "transcript": full_reply);
        emit!(&out_tx, &idg, "type": "response.content_part.done",
              "response_id": resp_id, "item_id": asst_item_id,
              "output_index": out_index, "content_index": content_index,
              "part": { "type": "output_audio", "transcript": full_reply });
        emit!(&out_tx, &idg, "type": "response.output_item.done",
              "response_id": resp_id, "output_index": out_index,
              "item": { "id": asst_item_id, "object": "realtime.item", "type": "message",
                        "role": "assistant", "status": item_status,
                        "content": [{ "type": "output_audio", "transcript": full_reply }] });
        // Push the assistant conversation item (next turn's history includes it), then
        // emit conversation.item.created with the previous item id.
        let prev_for_asst = {
            let mut s = session.lock().unwrap();
            let prev = s.items.last().map(|i| i.id.clone());
            s.items.push(ConvItem {
                id: asst_item_id.clone(),
                role: "assistant".into(),
                text: full_reply.clone(),
            });
            prev
        };
        if status == "completed" {
            emit!(&out_tx, &idg, "type": "conversation.item.added",
                  "previous_item_id": prev_for_asst,
                  "item": {
                      "id": asst_item_id, "object": "realtime.item", "type": "message",
                      "role": "assistant", "status": "completed",
                      "content": [{ "type": "output_audio", "transcript": full_reply }]
                  });
        }
        emit!(&out_tx, &idg, "type": "response.done",
              "response": {
                  "id": resp_id, "object": "realtime.response",
                  "status": status, "status_details": null,
                  "output": [{
                      "id": asst_item_id, "type": "message", "status": item_status,
                      "role": "assistant",
                      "content": [{ "type": "output_audio", "transcript": full_reply }]
                  }],
                  "conversation_id": conv_id,
                  "output_modalities": ["audio"], "max_output_tokens": "inf",
                  "audio": { "output": { "format": { "type": "audio/pcm", "rate": OUT_SR }, "voice": "alloy" } },
                  "usage": null, "metadata": null
              });
        speaking.store(false, Ordering::Relaxed);
        streaming.store(false, Ordering::Relaxed);
        eprintln!("turn {turn} {status}");
    }
}

/// Local UTC offset in seconds, detected once via `date +%z` (cached). Falls
/// back to UTC+8 if detection fails. Overridable via SYSTEM_LOCAL_UTC_OFFSET.
fn local_utc_offset() -> i64 {
    use std::sync::OnceLock;
    static OFF: OnceLock<i64> = OnceLock::new();
    *OFF.get_or_init(|| {
        if let Ok(v) = std::env::var("SYSTEM_LOCAL_UTC_OFFSET")
            && let Ok(n) = v.parse::<i64>()
        {
            return n;
        }
        let from_date = std::process::Command::new("date")
            .arg("+%z")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                let s = s.trim();
                let (sign, rest) = s.split_at(1);
                if rest.len() == 4 && rest.chars().all(|c| c.is_ascii_digit()) {
                    let h: i64 = rest[..2].parse().ok()?;
                    let m: i64 = rest[2..].parse().ok()?;
                    let secs = h * 3600 + m * 60;
                    Some(if sign == "-" { -secs } else { secs })
                } else {
                    None
                }
            });
        from_date.unwrap_or(8 * 3600)
    })
}

/// Format the current local date/weekday/time for injection into the system
/// prompt. No chrono dependency — derived from SystemTime + the local UTC
/// offset detected once at startup (see `local_utc_offset`).
fn now_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let offset = local_utc_offset();
    let local = secs + offset;
    let days = local.div_euclid(86400);
    let tod = local.rem_euclid(86400);
    let (hh, mm) = (tod / 3600, (tod % 3600) / 60);
    // Civil date from days since epoch (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    // Weekday: 1970-01-01 was Thursday.
    let wd = (days + 4).rem_euclid(7);
    let wd_cn = ["日", "一", "二", "三", "四", "五", "六"][wd as usize];
    format!("{y}年{m}月{d}日 星期{wd_cn} {hh:02}:{mm:02}")
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
    tts: &mut WebTts,
    ref_voice: Option<&WebRef>,
    text: &str,
    barge: &AtomicBool,
    out_tx: &mpsc::Sender<Out>,
    resp_id: &str,
    item_id: &str,
    output_index: u32,
    content_index: u32,
    idg: &IdGen,
) {
    let resp_id = resp_id.to_string();
    let item_id = item_id.to_string();
    match tts {
        WebTts::Moss(e) => {
            let ref_codes = match ref_voice {
                Some(WebRef::Moss(c)) => Some(c),
                _ => None,
            };
            let mut on_chunk = |pcm: Vec<f32>| -> bool {
                if barge.load(Ordering::Relaxed) {
                    return false;
                }
                let b64 = downmix_24k_mono_i16_b64(&pcm);
                let _ = out_tx.blocking_send(Out(
                    json!({"event_id": idg.ev(),
                           "type":"response.output_audio.delta","response_id":resp_id,
                           "item_id":item_id,"output_index":output_index,
                           "content_index":content_index,"delta":b64})
                        .to_string(),
                ));
                true
            };
            let chunk_frames: usize = std::env::var("TTS_CHUNK_FRAMES")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n| n > 0)
                .unwrap_or(DEFAULT_TTS_CHUNK_FRAMES);
            if let Err(e) = e.synthesize_pcm_stream_with_codes(
                text,
                MOSS_MAX_FRAMES,
                chunk_frames,
                ref_codes,
                &mut on_chunk,
            ) {
                eprintln!("tts error: {e}");
            }
        }
        WebTts::Qwen3(e) => {
            let rv = match ref_voice {
                Some(WebRef::Qwen3(rv)) => Some(rv),
                _ => None,
            };
            // Streaming GENERATION + streaming DECODE: the talker emits frames one at
            // a time (abortable via the per-frame `should_abort` predicate — barge-in
            // still halts mid-sentence), and each window is codec-decoded on the
            // DEDICATED Metal device (its own buffer pool, isolated from the talker's
            // GPU workspace — the multi-model babble fix). First audio arrives after
            // `first_frames` (~0.96 s @ 12 frames) instead of after the whole sentence.
            // Tune via QWEN3_TTS_STREAM_FIRST/_CHUNK/_CTX; correctness self-check with
            // QWEN3_TTS_STREAM_CHECK=1 (the engine re-decodes a batch reference and the
            // CLI diffs them — the web path emits the streamed tail directly).
            let first_frames = env_usize_or("QWEN3_TTS_STREAM_FIRST", STREAM_FIRST_FRAMES);
            let chunk_frames = env_usize_or("QWEN3_TTS_STREAM_CHUNK", STREAM_CHUNK_FRAMES);
            let left_ctx = env_usize_or("QWEN3_TTS_STREAM_CTX", STREAM_LEFT_CONTEXT);
            let mut on_audio = |pcm: &[f32]| {
                if barge.load(Ordering::Relaxed) {
                    return;
                }
                let b64 = mono_24k_i16_b64(pcm);
                let _ = out_tx.blocking_send(Out(
                    json!({"event_id": idg.ev(),
                           "type":"response.output_audio.delta","response_id":resp_id,
                           "item_id":item_id,"output_index":output_index,
                           "content_index":content_index,"delta":b64})
                        .to_string(),
                ));
            };
            let abort = || barge.load(Ordering::Relaxed);
            if let Err(e) = e.synthesize_pcm_streaming_with_abort(
                text,
                "auto",
                rv,
                // Cap (300 frames ≈ 24 s). The Base talker rarely emits codec_eos for
                // Chinese, so this is also the hard stop. Lower for snappier replies via
                // QWEN3_TTS_MAX_FRAMES (env).
                env_usize_or("QWEN3_TTS_MAX_FRAMES", QWEN3_MAX_FRAMES),
                first_frames,
                chunk_frames,
                left_ctx,
                &mut on_audio,
                Some(&abort),
            ) {
                eprintln!("tts error: {e}");
            }
        }
    }
}

/// Qwen3-TTS 24 kHz mono f32 → 16-bit PCM → base64. The browser's pcm16
/// contract is already 24 kHz mono (`OUT_SR`), so no downmix/resample is
/// needed — unlike `downmix_24k_mono_i16_b64`, which is for MOSS 48 k stereo.
fn mono_24k_i16_b64(pcm: &[f32]) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    B64.encode(&out)
}

/// MOSS 48 k stereo interleaved f32 → 24 k mono 16-bit → base64.
///
/// Resamples properly instead of naive `step_by(2)` decimation: decimating
/// without an anti-aliasing filter folds energy above 12 kHz back into the
/// audible band (the "weird/metallic" TTS sound). We downmix to mono, then do a
/// linear-interp resample 48 k → 24 k with a fractional read position — the
/// same fix audio.cpp applies via soxr (`resample_mono_soxr_or_linear`). For an
/// exact 2:1 ratio the linear interp also averages each sample pair, which is a
/// rudimentary low-pass that removes most aliasing.
fn downmix_24k_mono_i16_b64(pcm: &[f32]) -> String {
    // Deinterleave to mono (average L/R).
    let ch = 2usize;
    let nframes = pcm.len() / ch;
    if nframes == 0 {
        return String::new();
    }
    let mut mono = Vec::with_capacity(nframes);
    for f in 0..nframes {
        let mut s = 0.0f32;
        for c in 0..ch {
            s += pcm[f * ch + c];
        }
        mono.push(s / ch as f32);
    }
    // Linear-interp resample 48 k → 24 k (step = 2.0 in source frames).
    const SRC_RATE: f64 = 48000.0;
    const DST_RATE: f64 = 24000.0;
    let step = SRC_RATE / DST_RATE; // source frames per output frame
    let out_len = (nframes as f64 / step).floor() as usize;
    let mut out: Vec<u8> = Vec::with_capacity(out_len * 2);
    let mut pos = 0.0f64;
    for _ in 0..out_len {
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = mono.get(i0).copied().unwrap_or(0.0);
        let b = mono.get(i0 + 1).copied().unwrap_or(a);
        let s = (a + (b - a) * frac).clamp(-1.0, 1.0);
        let v = (s * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
        pos += step;
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

/// Absolute path to `web/static`, resolved at compile time. Serving from disk
/// lets UI edits (index.html, worklets/*.js) take effect on a plain reload.
const STATIC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/static");
