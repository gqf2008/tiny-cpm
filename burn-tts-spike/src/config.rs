//! Serde configs for Qwen3-TTS, reduced to the fields the spike uses.
//! Mirrors tiny-cpm src/models/qwen3_tts/config.rs (unknown fields ignored).

use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSConfig {
    pub assistant_token_id: u32,
    pub im_start_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub talker_config: TalkerConfig,
    pub speaker_encoder_config: SpeakerEncoderConfigJson,
}

/// config.json.speaker_encoder_config — the checkpoint only carries
/// enc_dim/sample_rate; the rest of the ECAPA-TDNN hyper-params are the
/// upstream defaults (SpeakerEncoderParams::default).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfigJson {
    pub enc_dim: usize,
    pub sample_rate: usize,
}

/// ECAPA-TDNN hyper-params (upstream configuration_qwen3_tts.py defaults, same
/// as candle's SpeakerEncoderParams).
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

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub num_code_groups: usize,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_language_id: HashMap<String, u32>,
    pub code_predictor_config: CodePredictorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub vocab_size: usize,
    pub num_code_groups: usize,
}

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

/// speech_tokenizer/config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub output_sample_rate: usize,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub encoder_config: CodecEncoderConfig,
    pub decoder_config: CodecDecoderConfig,
}

/// speech_tokenizer/config.json.encoder_config — a transformers MimiConfig
/// (encoder only). All convs causal.
#[derive(Debug, Clone, Deserialize)]
pub struct CodecEncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    /// Encoder downsampling strides; applied in REVERSE order ([8,6,5,4] → 4,5,6,8).
    pub upsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodecDecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub upsampling_ratios: Vec<usize>,
    pub upsample_rates: Vec<usize>,
}

impl CodecDecoderConfig {
    /// PCM samples per codec frame (1920 at 24 kHz).
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}
