//! burn spike 2 CLI: `burn-tts-spike <model-dir> "<text>" <out.wav> [--language <lang>] [--max-frames N]`
//!
//! Qwen3-TTS-12Hz-1.7B core path on burn 0.22.0-pre.1 Metal:
//! Qwen2 BPE → talker (codebook 0, 12.5 Hz) → code predictor (books 1-15) →
//! Mimi codec decoder → 24 kHz mono wav. No --ref / --stream / quantization.
//!
//! Verification env: BURN_TTS_GREEDY=1 (argmax everywhere), BURN_TTS_DUMP_CODES=1
//! (write the (n,16) code sequence to stderr for bit-compare vs candle).

#![recursion_limit = "256"]

mod audio;
mod codec;
mod config;
mod model;
mod speaker_encoder;
mod talker;

use std::time::Instant;

use anyhow::{Result, bail};
use burn::tensor::{DType, Device, DeviceKind, Int, Tensor, TensorData};
use model::Weights;
use tokenizers::Tokenizer;

/// Load all *.safetensors in a dir into a name → f16/f32 TensorData map.
fn load_weights(
    model_dir: &str,
    target: DType,
) -> Result<std::collections::HashMap<String, TensorData>> {
    let mut files: Vec<String> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file() && e.path().extension().is_some_and(|x| x == "safetensors"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no safetensors files found in {model_dir}");
    }
    let mut out = std::collections::HashMap::new();
    for f in &files {
        let bytes = std::fs::read(f)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("safetensors {f}: {e}"))?;
        for (name, t) in st.tensors() {
            let dt = match t.dtype() {
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I64 => DType::I64,
                safetensors::Dtype::I32 => DType::I32,
                other => bail!("unsupported safetensors dtype {other:?} in {f}"),
            };
            let td =
                TensorData::from_bytes_vec(t.data().to_vec(), t.shape(), dt).convert_dtype(target);
            out.insert(name, td);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(model_dir) = args.get(1) else {
        bail!(
            "usage: burn-tts-spike <model-dir> \"<text>\" <out.wav> [--language <lang>] [--max-frames N]"
        );
    };
    let Some(text) = args.get(2) else {
        bail!(
            "usage: burn-tts-spike <model-dir> \"<text>\" <out.wav> [--language <lang>] [--max-frames N]"
        );
    };
    let Some(out_wav) = args.get(3) else {
        bail!(
            "usage: burn-tts-spike <model-dir> \"<text>\" <out.wav> [--language <lang>] [--max-frames N]"
        );
    };
    let mut language = "auto".to_string();
    let mut max_frames = 2048usize;
    let mut codes_file: Option<String> = None;
    let mut encode_wav: Option<String> = None;
    let mut spk_embed_wav: Option<String> = None;
    let mut ref_wav: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--language" => {
                language = args.get(i + 1).cloned().unwrap_or_else(|| "auto".into());
                i += 2;
            }
            "--max-frames" => {
                max_frames = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2048);
                i += 2;
            }
            "--codes-file" => {
                codes_file = args.get(i + 1).cloned();
                i += 2;
            }
            "--encode-wav" => {
                encode_wav = args.get(i + 1).cloned();
                i += 2;
            }
            "--spk-embed" => {
                spk_embed_wav = args.get(i + 1).cloned();
                i += 2;
            }
            "--ref" => {
                ref_wav = args.get(i + 1).cloned();
                i += 2;
            }
            "--ref-text" => {
                ref_text = args.get(i + 1).cloned();
                i += 2;
            }
            other => bail!("unknown flag {other}"),
        }
    }

    let device = Device::metal(DeviceKind::DefaultDevice);
    let greedy = std::env::var("BURN_TTS_GREEDY").is_ok();
    eprintln!(
        "backend: burn {} (wgpu/cubecl, Metal); talker {}, codec f32{}",
        env!("CARGO_PKG_VERSION"),
        if std::env::var("BURN_TTS_F32").is_ok() {
            "f32"
        } else {
            "f16"
        },
        if greedy { ", GREEDY" } else { "" }
    );

    // --- configs ---
    let tts_cfg: config::Qwen3TTSConfig =
        serde_json::from_slice(&std::fs::read(format!("{model_dir}/config.json"))?)?;
    let gen_path = format!("{model_dir}/generation_config.json");
    let gen_cfg: config::Qwen3TTSGenerationConfig = if std::path::Path::new(&gen_path).exists() {
        serde_json::from_slice(&std::fs::read(gen_path)?)?
    } else {
        config::Qwen3TTSGenerationConfig::default()
    };
    let codec_dir = format!("{model_dir}/speech_tokenizer");
    let codec_cfg: config::SpeechTokenizerConfig =
        serde_json::from_slice(&std::fs::read(format!("{codec_dir}/config.json"))?)?;
    let sample_rate = codec_cfg.output_sample_rate;
    let tokenizer = load_tokenizer(model_dir)?;
    let newline_token = tokenizer
        .encode("\n", false)
        .map_err(|e| anyhow::anyhow!("tokenize newline: {e}"))?
        .get_ids()
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tokenizer: '\\n' encodes to nothing"))?;
    let text_ids = tokenizer
        .encode(text.clone(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    eprintln!(
        "text: {} chars → {} tokens, newline={newline_token}, lang={language}, max_frames={max_frames}",
        text.chars().count(),
        text_ids.len()
    );

    // --- weights + models ---
    let t0 = Instant::now();
    let talker_weights = load_weights(model_dir, model::dt())?;
    let codec_weights = load_weights(&codec_dir, model::CODEC_DT)?;
    let w = Weights::new(talker_weights, device.clone());
    let wc = Weights::new(codec_weights, device.clone());
    let decoder = codec::CodecDecoder::new(&wc, &codec_cfg.decoder_config)?;
    eprintln!(
        "loaded models in {:?} (talker f16 + codec f32)",
        t0.elapsed()
    );

    // --encode-wav: codec-encoder-only verification — encode a wav to (16, T)
    // codes and dump them (compare against candle's QWEN3_TTS_DUMP_REF=1).
    if let Some(ew) = encode_wav {
        let (samples, sr) = audio::decode_audio(&ew)?;
        let samples = if sr == sample_rate {
            samples
        } else {
            audio::resample_sinc(&samples, sr, sample_rate)
        };
        let n = samples.len();
        let wav_t: Tensor<3, burn::tensor::Float> =
            Tensor::<1, burn::tensor::Float>::from_floats(samples.as_slice(), &device)
                .reshape([1, 1, n]);
        let encoder = codec::CodecEncoder::new(
            &wc,
            &codec_cfg.encoder_config,
            codec_cfg.encoder_valid_num_quantizers,
        )?;
        let t1 = Instant::now();
        let codes = encoder.encode(wav_t)?; // (1, 16, T)
        let codes = codes.squeeze_dim::<2>(0); // (16, T)
        let v: Vec<i32> = codes
            .to_data()
            .to_vec::<i32>()
            .map_err(|e| anyhow::anyhow!("codes read: {e}"))?;
        let t = v.len() / 16;
        eprintln!(
            "encode-wav: {ew} {n} samples → {t} frames in {:.2?}",
            t1.elapsed()
        );
        eprintln!("BURN_REF_CODE {t} {:?}", v);
        return Ok(());
    }

    // --spk-embed: speaker-encoder-only verification — dump the raw embedding.
    if let Some(sw) = spk_embed_wav {
        let (samples, sr) = audio::decode_audio(&sw)?;
        let samples = if sr == sample_rate {
            samples
        } else {
            audio::resample_sinc(&samples, sr, sample_rate)
        };
        let spk_params = config::SpeakerEncoderParams {
            enc_dim: tts_cfg.speaker_encoder_config.enc_dim,
            sample_rate: tts_cfg.speaker_encoder_config.sample_rate,
            ..Default::default()
        };
        let spk = speaker_encoder::SpeakerEncoder::new(&w, spk_params.clone())?;
        let t1 = Instant::now();
        let emb = spk.embed(&samples, &device)?; // (enc_dim,)
        let v: Vec<f32> = emb
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("spk read: {e}"))?;
        eprintln!(
            "spk-embed: {sw} {} samples → dim {} in {:.2?}",
            samples.len(),
            v.len(),
            t1.elapsed()
        );
        eprintln!("BURN_REF_SPK {:?}", v);
        return Ok(());
    }

    // --codes-file: skip the talker, decode a (n,16) code sequence straight to
    // PCM (codec-only verification against candle's codes).
    if let Some(cf) = codes_file {
        let raw = std::fs::read_to_string(&cf)?;
        let nums: Vec<i64> = raw
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        let n = nums.len() / 16;
        // The flat dump is frame-major (frame×16 codebooks) — build (n, 16) then
        // transpose to the codec's (1, 16, n) layout. A plain reshape interleaves
        // codebooks with frames (element [cb][f] = flat[cb*n+f] ≠ flat[f*16+cb]).
        let codes_t: Tensor<3, Int> = Tensor::<1, Int>::from_ints(
            nums.iter()
                .map(|&v| v as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
            &device,
        )
        .reshape([n, 16])
        .swap_dims(0, 1)
        .unsqueeze(); // (1, 16, n)
        let t1 = Instant::now();
        let wav = decoder.chunked_decode(&codes_t, 300, 25)?;
        let pcm: Vec<f32> = wav
            .squeeze_dim::<2>(0)
            .squeeze_dim::<1>(0)
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("pcm read: {e}"))?;
        device
            .sync()
            .map_err(|e| anyhow::anyhow!("device sync: {e}"))?;
        if std::env::var("BURN_TTS_DUMP_PCM").is_ok() {
            eprintln!("BURN_PCM {:?}", pcm);
        }
        let total = t1.elapsed();
        eprintln!(
            "codec-only: {n} frames in {:.2?} ({:.2} s audio, RTF {:.3})",
            total,
            pcm.len() as f64 / sample_rate as f64,
            total.as_secs_f64() / (pcm.len() as f64 / sample_rate as f64)
        );
        if std::env::var("BURN_TTS_DUMP_PCM").is_ok() {
            eprintln!("BURN_PCM {:?}", pcm);
        }
        save_wav_mono(&pcm, out_wav, sample_rate as u32)?;
        eprintln!(
            "wrote {out_wav} ({} samples, {sample_rate} Hz mono)",
            pcm.len()
        );
        return Ok(());
    }

    let mut talker = talker::Talker::new(&w, &tts_cfg)?;

    // --- ref voice (ICL cloning): speaker embed + ref codes + ref text ids ---
    let ref_voice: Option<talker::RefVoice> = match (&ref_wav, &ref_text) {
        (Some(rw), Some(rt)) => {
            let (samples, sr) = audio::decode_audio(rw)?;
            let samples = if sr == sample_rate {
                samples
            } else {
                audio::resample_sinc(&samples, sr, sample_rate)
            };
            let spk_params = config::SpeakerEncoderParams {
                enc_dim: tts_cfg.speaker_encoder_config.enc_dim,
                sample_rate: tts_cfg.speaker_encoder_config.sample_rate,
                ..Default::default()
            };
            let spk = speaker_encoder::SpeakerEncoder::new(&w, spk_params)?;
            let emb = spk.embed(&samples, &device)?.cast(model::dt()); // (enc_dim,) talker dtype
            let n = samples.len();
            let wav_t: Tensor<3, burn::tensor::Float> =
                Tensor::<1, burn::tensor::Float>::from_floats(samples.as_slice(), &device)
                    .reshape([1, 1, n]);
            let encoder = codec::CodecEncoder::new(
                &wc,
                &codec_cfg.encoder_config,
                codec_cfg.encoder_valid_num_quantizers,
            )?;
            let ref_code = encoder.encode(wav_t)?.squeeze_dim::<2>(0); // (16, T)
            let ref_text_ids = tokenizer
                .encode(rt.clone(), false)
                .map_err(|e| anyhow::anyhow!("ref text tokenize: {e}"))?
                .get_ids()
                .to_vec();
            eprintln!(
                "ref voice: {rw} {} samples → {} frames, spk dim {}, ref-text {} tokens",
                samples.len(),
                ref_code.dims()[1],
                emb.dims()[0],
                ref_text_ids.len()
            );
            Some(talker::RefVoice {
                spk_embed: emb,
                ref_code,
                ref_text_ids,
            })
        }
        (None, None) => None,
        _ => bail!("--ref and --ref-text must be given together"),
    };

    // --- synthesize: warmup run then measured run ---
    let synth = |talker: &mut talker::Talker,
                 tag: &str|
     -> Result<(Vec<f32>, usize, std::time::Duration)> {
        let t1 = Instant::now();
        let codes = talker.generate(
            &text_ids,
            &language,
            newline_token,
            ref_voice.as_ref(),
            &gen_cfg,
            max_frames,
        )?; // (n, 16)
        let n_frames = codes.dims()[0];
        if std::env::var("BURN_TTS_DUMP_CODES").is_ok() {
            let v: Vec<i32> = codes
                .clone()
                .to_data()
                .to_vec::<i32>()
                .map_err(|e| anyhow::anyhow!("codes read: {e}"))?;
            eprintln!("BURN_CODES {tag} frames={n_frames} {:?}", v);
        }
        let t2 = Instant::now();
        let codes_t: Tensor<3, Int> = codes.clone().swap_dims(0, 1).unsqueeze(); // (1, 16, n)
        let wav = decoder.chunked_decode(&codes_t, 300, 25)?; // (1, 1, n*1920)
        let pcm: Vec<f32> = wav
            .squeeze_dim::<2>(0)
            .squeeze_dim::<1>(0)
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("pcm read: {e}"))?;
        device
            .sync()
            .map_err(|e| anyhow::anyhow!("device sync: {e}"))?;
        if std::env::var("BURN_TTS_DUMP_PCM").is_ok() {
            eprintln!("BURN_PCM {:?}", pcm);
        }
        let total = t1.elapsed();
        eprintln!(
            "{tag}: {n_frames} frames, gen {:.2?}, codec {:.2?}, total {:.2?} ({:.2} s audio, RTF {:.3})",
            t2 - t1,
            total - (t2 - t1),
            total,
            pcm.len() as f64 / sample_rate as f64,
            total.as_secs_f64() / (pcm.len() as f64 / sample_rate as f64)
        );
        Ok((pcm, n_frames, total))
    };

    let (_, n1, _) = synth(&mut talker, "warmup")?;
    let (pcm, n2, _) = synth(&mut talker, "decode")?;
    eprintln!("frames: warmup={n1} measured={n2}");

    save_wav_mono(&pcm, out_wav, sample_rate as u32)?;
    eprintln!(
        "wrote {out_wav} ({} samples, {sample_rate} Hz mono)",
        pcm.len()
    );
    Ok(())
}

/// Same i16 normalization as candle's save_wav_mono.
fn save_wav_mono(audio: &[f32], save_path: &str, sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let max = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let mut writer = hound::WavWriter::create(save_path, spec)?;
    for &x in audio {
        writer.write_sample((x * ratio).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// tokenizer.json if present, else BPE from vocab.json + merges.txt with added
/// tokens from tokenizer_config.json (same as the ASR spike / tiny-cpm).
fn load_tokenizer(model_dir: &str) -> Result<Tokenizer> {
    let tokenizer_file = format!("{model_dir}/tokenizer.json");
    if std::path::Path::new(&tokenizer_file).exists() {
        return Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("tokenizer from file: {e}"));
    }
    use tokenizers::AddedToken;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let vocab_file = format!("{model_dir}/vocab.json");
    let merges_file = format!("{model_dir}/merges.txt");
    if !std::path::Path::new(&vocab_file).exists() {
        bail!("neither tokenizer.json nor vocab.json found in {model_dir}");
    }
    let bpe = BPE::from_file(&vocab_file, &merges_file)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build BPE tokenizer: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let config_file = format!("{model_dir}/tokenizer_config.json");
    if std::path::Path::new(&config_file).exists() {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        if let Some(added_tokens_decoder) = config.get("added_tokens_decoder") {
            let mut special_tokens = Vec::new();
            if let serde_json::Value::Object(tokens_map) = added_tokens_decoder {
                for (_, token_info) in tokens_map {
                    if let serde_json::Value::Object(token_obj) = token_info
                        && let Some(content) = token_obj.get("content").and_then(|v| v.as_str())
                    {
                        let special = token_obj
                            .get("special")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        special_tokens.push(AddedToken::from(content.to_string(), special));
                    }
                }
            }
            tokenizer.add_tokens(&special_tokens);
        }
    }
    Ok(tokenizer)
}
