//! Serde configs for Qwen3-ASR, reduced to the fields the spike actually uses.
//! Mirrors tiny-cpm src/models/qwen3_asr/config.rs (unknown JSON fields are
//! ignored by serde).

use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3ASRConfig {
    pub thinker_config: ThinkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioConfig,
    pub audio_token_id: u32,
    pub dtype: String,
    pub text_config: TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub d_model: usize,
    pub downsample_hidden_size: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
    pub n_window: usize,
    pub conv_chunksize: usize,
    pub activation_function: Activation,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub eos_token_id: Vec<u32>,
    pub temperature: f32,
}

/// whisper-style preprocessor (preprocessor_config.json).
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureExtractorConfig {
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub dither: f64,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
}

fn default_sampling_rate() -> usize {
    16000
}
