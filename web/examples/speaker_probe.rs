// Speaker-consistency probe: send a multi-sentence text, collect each
// `response.audio.delta` chunk (one per sentence for the batched Qwen3 path),
// embed each with the model's OWN ECAPA-TDNN speaker encoder, and report
// pairwise cosine similarity — the direct measure of whether every sentence
// is the same speaker (the "three voices in one reply" complaint).
//
//     cargo run --release --example speaker_probe -p tiny-cpm-web -- \
//       <model-dir> <ws://addr/ws> "<text>" [out-dir]
use anyhow::Result;
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use futures_util::{SinkExt, StreamExt};
use tiny_cpm::models::qwen3_tts::speaker_encoder::SpeakerEncoder;
use tiny_cpm::models::qwen3_tts::config::SpeakerEncoderParams;
use tokio_tungstenite::connect_async;
use tungstenite::Message;

fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb))
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-TTS-12Hz-1.7B-Base".into());
    let addr = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "ws://127.0.0.1:8080/ws".into());
    let text = std::env::args().nth(3).unwrap_or_else(|| {
        "你好！今天天气很好。我是MiniCPM系列模型，由面壁智能开发。很高兴见到你。你觉得怎么样？".into()
    });
    let out_dir = std::env::args().nth(4).unwrap_or_else(|| "/tmp/voice_probe".into());
    std::fs::create_dir_all(&out_dir)?;

    // Load just the speaker encoder (F32) from the model safetensors.
    let device = Device::new_metal(0)?;
    let tts_cfg: tiny_cpm::models::qwen3_tts::config::Qwen3TTSConfig =
        serde_json::from_slice(&std::fs::read(format!("{model_dir}/config.json"))?)?;
    let model_list: Vec<String> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let vb_f32 = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, DType::F32, &device)? };
    let spk_params = SpeakerEncoderParams {
        enc_dim: tts_cfg.speaker_encoder_config.enc_dim,
        sample_rate: tts_cfg.speaker_encoder_config.sample_rate,
        ..SpeakerEncoderParams::default()
    };
    let spk = SpeakerEncoder::new(vb_f32.pp("speaker_encoder"), spk_params, &device)?;
    eprintln!("loaded speaker encoder");

    // --- WS turn ---
    let (mut ws, _) = connect_async(&addr).await?;
    println!("connected → sending: {text}");
    ws.send(Message::Text(
        serde_json::json!({
            "type": "conversation.item.create",
            "item": {"role": "user", "content": [{"type": "input_text", "text": text}]}
        })
        .to_string(),
    ))
    .await?;
    ws.send(Message::Text(
        serde_json::json!({"type": "response.create", "response": {}}).to_string(),
    ))
    .await?;

    let b64 = base64::engine::general_purpose::STANDARD;
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    loop {
        match ws.next().await {
            Some(Ok(Message::Text(s))) => {
                let v: serde_json::Value = serde_json::from_str(&s)?;
                match v["type"].as_str().unwrap_or("") {
                    "response.audio.delta" => {
                        if let Some(d) = v["delta"].as_str() {
                            let bytes = b64.decode(d)?;
                            let pcm: Vec<f32> = bytes
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                                .collect();
                            chunks.push(pcm);
                        }
                    }
                    "response.audio_transcript.done" => {
                        println!("[transcript: {:?}]", v["transcript"]);
                    }
                    "response.done" => break,
                    _ => {}
                }
            }
            other => {
                println!("[ws] {other:?}");
                break;
            }
        }
    }

    println!("\n=== {} chunk(s) ===", chunks.len());
    let mut embeds = Vec::new();
    for (i, pcm) in chunks.iter().enumerate() {
        let secs = pcm.len() as f32 / 24000.0;
        let emb = spk.embed(pcm)?;
        embeds.push(emb);
        let path = format!("{out_dir}/chunk_{i}.wav");
        let t = Tensor::from_vec(pcm.clone(), (1, pcm.len()), &Device::Cpu)?;
        tiny_cpm::utils::audio_utils::save_wav_mono(&t, &path, 24000)?;
        println!("chunk {i}: {secs:.2}s → {path}");
    }
    println!("\n=== speaker-embedding cosine similarity ===");
    for i in 0..embeds.len() {
        for j in (i + 1)..embeds.len() {
            println!("  chunk {i} vs {j}: {:.4}", cosine(&embeds[i], &embeds[j])?);
        }
    }
    Ok(())
}
