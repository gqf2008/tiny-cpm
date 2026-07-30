//! Ported from github.com/QwenLM/Qwen3-TTS qwen_tts/core/models/configuration_qwen3_tts.py
//! and qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py.
//!
//! Serde structs mirroring the HF `config.json` (top-level + `talker_config` +
//! `speaker_encoder_config`) and `speech_tokenizer/config.json` (`encoder_config` +
//! `decoder_config`) for Qwen3-TTS-12Hz-1.7B-Base. Unknown/extra fields are ignored
//! (the upstream configs carry many GenerationMixin / training fields we don't need).
use serde::Deserialize;
use std::collections::HashMap;

/// Top-level `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSConfig {
    pub assistant_token_id: u32,
    pub im_end_token_id: u32,
    pub im_start_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub tts_model_type: String,
    pub speaker_encoder_config: SpeakerEncoderConfig,
    pub talker_config: TalkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    pub enc_dim: usize,
    pub sample_rate: usize,
    // The remaining ECAPA-TDNN hyper-params are not in config.json; upstream
    // configuration_qwen3_tts.py defaults are used (see `SpeakerEncoderParams::default`).
}

/// `talker_config` — the LM. Architecturally a stock Qwen3 decoder (RMSNorm,
/// RoPE, GQA, per-head Q/K RMSNorm, SwiGLU) plus the TTS dual-vocab fields.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "default_silu")]
    pub hidden_act: String,
    /// Codec vocab (codes 0..2047 + specials 2048..=2157).
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub num_code_groups: usize,
    // Codec-track special ids (index into `codec_embedding`).
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, bool>,
    pub code_predictor_config: CodePredictorConfig,
}

fn default_silu() -> String {
    "silu".to_string()
}

/// `talker_config.code_predictor_config` — the 5-layer mini-Qwen that predicts
/// codebooks 1..=15 after the talker emits codebook 0.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "default_silu")]
    pub hidden_act: String,
    /// Single codec codebook (2048).
    pub vocab_size: usize,
    pub num_code_groups: usize,
}

/// `generation_config.json` sampling defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSGenerationConfig {
    #[serde(default = "default_true")]
    pub do_sample: bool,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_true")]
    pub subtalker_dosample: bool,
    #[serde(default = "default_temperature")]
    pub subtalker_temperature: f64,
    #[serde(default = "default_top_p")]
    pub subtalker_top_p: f32,
    #[serde(default = "default_top_k")]
    pub subtalker_top_k: usize,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

impl Default for Qwen3TTSGenerationConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            repetition_penalty: default_rep_penalty(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            subtalker_dosample: true,
            subtalker_temperature: default_temperature(),
            subtalker_top_p: default_top_p(),
            subtalker_top_k: default_top_k(),
            max_new_tokens: default_max_new_tokens(),
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_rep_penalty() -> f32 {
    1.05
}
fn default_temperature() -> f64 {
    0.9
}
fn default_top_p() -> f32 {
    1.0
}
fn default_top_k() -> usize {
    50
}
fn default_max_new_tokens() -> usize {
    2048
}

/// `speech_tokenizer/config.json` (the codec).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    #[serde(default)]
    pub model_type: String,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_config: CodecEncoderConfig,
    pub decoder_config: CodecDecoderConfig,
}

impl SpeechTokenizerConfig {
    /// Codec frame rate (12.5 Hz): output_sample_rate / decode_upsample_rate.
    pub fn frame_rate(&self) -> f64 {
        self.output_sample_rate as f64 / self.decode_upsample_rate as f64
    }
}

/// `speech_tokenizer/config.json.encoder_config` — a transformers MimiConfig
/// (encoder only; the decoder modules are absent). All convs causal.
#[derive(Debug, Clone, Deserialize)]
pub struct CodecEncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub num_residual_layers: usize,
    pub compress: usize,
    /// Encoder downsampling strides; applied in REVERSE order (e.g. [8,6,5,4] → strides 4,5,6,8).
    pub upsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_gelu")]
    pub hidden_act: String,
    pub norm_eps: f64,
    pub layer_scale_initial_scale: f32,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vector_quantization_hidden_dimension: usize,
    #[serde(default)]
    pub pad_mode: String,
    #[serde(default)]
    pub use_causal_conv: bool,
}

fn default_gelu() -> String {
    "gelu".to_string()
}

/// `speech_tokenizer/config.json.decoder_config` — the custom Qwen3TTSTokenizerV2 decoder.
#[derive(Debug, Clone, Deserialize)]
pub struct CodecDecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_silu")]
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_scale_initial_scale: f32,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    /// Waveform decoder transposed-conv strides (e.g. [8,5,4,3]).
    pub upsample_rates: Vec<usize>,
    /// Pre-waveform transposed-conv ratios (e.g. [2,2]).
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

impl CodecDecoderConfig {
    /// Samples of waveform per codec frame (1920): prod(upsample_rates) * prod(upsampling_ratios).
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}

/// ECAPA-TDNN hyper-params (upstream configuration_qwen3_tts.py defaults — not
/// all are present in config.json, so they're hardcoded to the reference values).
#[derive(Debug, Clone)]
pub struct SpeakerEncoderParams {
    pub mel_dim: usize,
    pub enc_dim: usize,
    pub channels: [usize; 5],
    pub kernel_sizes: [usize; 5],
    pub dilations: [usize; 5],
    pub attention_channels: usize,
    pub res2net_scale: usize,
    pub se_channels: usize,
    pub sample_rate: usize,
    // Mel frontend.
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl Default for SpeakerEncoderParams {
    fn default() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 2048,
            channels: [512, 512, 512, 512, 1536],
            kernel_sizes: [5, 3, 3, 3, 1],
            dilations: [1, 2, 3, 4, 1],
            attention_channels: 128,
            res2net_scale: 8,
            se_channels: 128,
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            fmin: 0.0,
            fmax: 12000.0,
        }
    }
}

impl Qwen3TTSConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

impl SpeechTokenizerConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_main_config() {
        let json = r#"{
            "assistant_token_id": 77091, "im_end_token_id": 151645, "im_start_token_id": 151644,
            "tts_bos_token_id": 151672, "tts_eos_token_id": 151673, "tts_pad_token_id": 151671,
            "model_type": "qwen3_tts", "tts_model_type": "base",
            "speaker_encoder_config": {"enc_dim": 2048, "sample_rate": 24000},
            "talker_config": {
                "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
                "vocab_size": 3072, "text_vocab_size": 151936, "text_hidden_size": 2048,
                "num_code_groups": 16, "codec_bos_id": 2149, "codec_eos_token_id": 2150,
                "codec_think_id": 2154, "codec_nothink_id": 2155, "codec_pad_id": 2148,
                "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
                "codec_language_id": {"chinese": 2055, "english": 2050},
                "code_predictor_config": {
                    "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 5,
                    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                    "max_position_embeddings": 65536, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
                    "vocab_size": 2048, "num_code_groups": 16
                }
            }
        }"#;
        let cfg: Qwen3TTSConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.talker_config.hidden_size, 2048);
        assert_eq!(cfg.talker_config.num_hidden_layers, 28);
        assert_eq!(cfg.talker_config.codec_eos_token_id, 2150);
        assert_eq!(cfg.talker_config.codec_language_id["chinese"], 2055);
        assert_eq!(cfg.talker_config.code_predictor_config.hidden_size, 1024);
        assert_eq!(cfg.talker_config.code_predictor_config.num_hidden_layers, 5);
    }

    #[test]
    fn parse_codec_config() {
        let json = r#"{
            "model_type": "qwen3_tts_tokenizer_12hz", "encoder_valid_num_quantizers": 16,
            "input_sample_rate": 24000, "output_sample_rate": 24000,
            "decode_upsample_rate": 1920, "encode_downsample_rate": 1920,
            "encoder_config": {
                "audio_channels": 1, "num_filters": 64, "kernel_size": 7, "residual_kernel_size": 3,
                "last_kernel_size": 3, "num_residual_layers": 1, "compress": 2,
                "upsampling_ratios": [8,6,5,4], "hidden_size": 512, "num_hidden_layers": 8,
                "num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 64,
                "intermediate_size": 2048, "hidden_act": "gelu", "norm_eps": 1e-5,
                "layer_scale_initial_scale": 0.01, "rope_theta": 10000, "max_position_embeddings": 8000,
                "sliding_window": 250, "num_quantizers": 32, "num_semantic_quantizers": 1,
                "codebook_size": 2048, "codebook_dim": 256, "vector_quantization_hidden_dimension": 256,
                "pad_mode": "constant", "use_causal_conv": true
            },
            "decoder_config": {
                "latent_dim": 1024, "codebook_dim": 512, "codebook_size": 2048, "decoder_dim": 1536,
                "hidden_size": 512, "intermediate_size": 1024, "hidden_act": "silu",
                "num_hidden_layers": 8, "num_attention_heads": 16, "num_key_value_heads": 16,
                "head_dim": 64, "rms_norm_eps": 1e-5, "layer_scale_initial_scale": 0.01,
                "rope_theta": 10000, "max_position_embeddings": 8000, "sliding_window": 72,
                "num_quantizers": 16, "num_semantic_quantizers": 1,
                "upsample_rates": [8,5,4,3], "upsampling_ratios": [2,2],
                "vector_quantization_hidden_dimension": 512
            }
        }"#;
        let cfg: SpeechTokenizerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.encoder_valid_num_quantizers, 16);
        assert!((cfg.frame_rate() - 12.5).abs() < 1e-9);
        assert_eq!(cfg.decoder_config.total_upsample(), 1920);
        assert_eq!(cfg.decoder_config.upsample_rates, vec![8, 5, 4, 3]);
        assert_eq!(cfg.encoder_config.upsampling_ratios, vec![8, 6, 5, 4]);
    }
}
