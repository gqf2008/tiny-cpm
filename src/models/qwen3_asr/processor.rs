//! Ported from aha (github.com/jhqxxx/aha) src/models/qwen3_asr/processor.rs
//!
//! Adaptation notes (tiny-cpm has no OpenAI-style chat params):
//! - aha's `extract_audio_vec` / `process_info` took `ChatCompletionParameters`
//!   (audio URLs + language metadata). Here audio comes from a local file path,
//!   so `process_info` becomes `process_audio_path(render, audio_path, tokenizer)`;
//!   the "language" metadata branch is dropped (the CLI does not pass one).
//! - Chunking still uses `split_audio_into_chunks` (already in
//!   `crate::utils::audio_utils`, max 1200s per chunk), same as aha.
use crate::common::modules::float_range_normalize;
use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};

use crate::{
    models::feature_extractor::{
        config::FeatureExtractor, feature_extraction_whisper::WhisperFeatureExtractor,
    },
    tokenizer::TokenizerModel,
    utils::audio_utils::{load_audio_with_resample, split_audio_into_chunks},
};

pub struct Qwen3AsrProcessor {
    device: Device,
    sample_rate: usize,
    support_language: Vec<String>,
    max_asr_input_seconds: f32,
    whisper_feature_extracor: WhisperFeatureExtractor,
    audio_token: String,
}

impl Qwen3AsrProcessor {
    pub fn new(device: &Device, config: &FeatureExtractor) -> Result<Self> {
        let support_language: Vec<String> = vec![
            "Chinese",
            "English",
            "Cantonese",
            "Arabic",
            "German",
            "French",
            "Spanish",
            "Portuguese",
            "Indonesian",
            "Italian",
            "Korean",
            "Russian",
            "Thai",
            "Vietnamese",
            "Japanese",
            "Turkish",
            "Hindi",
            "Malay",
            "Dutch",
            "Swedish",
            "Danish",
            "Finnish",
            "Polish",
            "Czech",
            "Filipino",
            "Persian",
            "Greek",
            "Romanian",
            "Hungarian",
            "Macedonian",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let whisper_feature_extracor = WhisperFeatureExtractor::new(
            config.feature_size,
            config.hop_length,
            // config.chunk_length,
            config.n_fft,
            config.dither,
            // config.padding_value,
            config.sampling_rate,
            device,
        )?;
        Ok(Self {
            device: device.clone(),
            sample_rate: 16000,
            support_language,
            max_asr_input_seconds: 1200.0,
            whisper_feature_extracor,
            audio_token: "<|audio_pad|>".to_string(),
        })
    }

    #[allow(dead_code)] // kept verbatim from aha; the CLI path uses process_audio_path
    pub fn validate_language(&self, lang: &String) -> bool {
        self.support_language.contains(lang)
    }

    fn replace_special_tokens(&self, text: &str, token_len: usize) -> String {
        let replace = "<|audio_placeholder|>".repeat(token_len);
        let text = text.replacen(&self.audio_token, &replace, 1);
        text.replace("<|audio_placeholder|>", &self.audio_token)
    }

    /// aha `process_audio_tensor`: single in-memory mono waveform (1-D, 16kHz).
    /// Used by `Qwen3AsrEngine::transcribe_samples` (the live loop); the CLI
    /// path uses process_audio_path.
    pub fn process_audio_tensor(
        &self,
        render: &str,
        audio: &Tensor,
        tokenizer: &TokenizerModel,
    ) -> Result<AudioData> {
        let audio_len = audio.dim(0)? as f32;
        if audio_len > self.sample_rate as f32 * self.max_asr_input_seconds {
            return Err(anyhow!("vad_res orig_audio is too long!"));
        }
        let mut audio = audio.unsqueeze(0)?;
        audio = float_range_normalize(&audio)?;
        let (input_features, _) =
            self.whisper_feature_extracor
                .call(&audio, self.sample_rate, false)?;
        let audio_len = input_features.dim(2)?;
        let output_len = get_feat_extract_output_lengths(audio_len);
        let text = self.replace_special_tokens(render, output_len);
        let input_ids = tokenizer.text_encode(text, &self.device)?;
        let input_features = input_features.squeeze(0)?;
        let audio_data = AudioData {
            input_features,
            input_ids,
        };
        Ok(audio_data)
    }

    /// aha `process_info`, with the local audio file path in place of
    /// `ChatCompletionParameters` (one audio per render for the CLI).
    pub fn process_audio_path(
        &self,
        render: &str,
        audio_path: &str,
        tokenizer: &TokenizerModel,
    ) -> Result<Vec<AudioData>> {
        let audio_count = render
            .matches("<|audio_start|><|audio_pad|><|audio_end|>")
            .count();
        let render = if audio_count > 1 {
            render.replace(
                &"<|audio_start|><|audio_pad|><|audio_end|>".repeat(audio_count),
                "<|audio_start|><|audio_pad|><|audio_end|>",
            )
        } else {
            render.to_string()
        };
        // aha extracted audio tensors from the chat message; tiny-cpm loads a local file.
        let wav =
            load_audio_with_resample(audio_path, &self.device, Some(self.sample_rate), Some(1))?;
        let wav = float_range_normalize(&wav)?;
        if audio_count != 1 {
            return Err(anyhow::anyhow!("audio_pad num != audio num"));
        }
        let split_wavs =
            split_audio_into_chunks(&wav, self.sample_rate, self.max_asr_input_seconds)?;
        let mut audio_datas = vec![];
        for wav in split_wavs.iter() {
            let (input_features, _) =
                self.whisper_feature_extracor
                    .call(wav, self.sample_rate, false)?;
            let audio_len = input_features.dim(2)?;
            let output_len = get_feat_extract_output_lengths(audio_len);
            let text = self.replace_special_tokens(&render, output_len);
            let input_ids = tokenizer.text_encode(text, &self.device)?;
            let input_features = input_features.squeeze(0)?;
            let audio = AudioData {
                input_features,
                input_ids,
            };
            audio_datas.push(audio);
        }
        Ok(audio_datas)
    }
}

pub struct AudioData {
    pub input_features: Tensor,
    pub input_ids: Tensor,
}

pub fn get_feat_extract_output_lengths(audio_len: usize) -> usize {
    let input_len_leave = audio_len % 100;
    if input_len_leave > 0 {
        let feat_lengths = (input_len_leave - 1) / 2 + 1;
        ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (audio_len / 100) * 13
    } else {
        (audio_len / 100) * 13
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The audio encoder processes features in n_window*2 = 100-frame windows;
    // each full 100-frame window survives the 3 stride-2 convs as 13 frames.
    #[test]
    fn feat_extract_output_lengths() {
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(100), 13);
        assert_eq!(get_feat_extract_output_lengths(200), 26);
        assert_eq!(get_feat_extract_output_lengths(3000), 390);
        // remainder windows go through the conv chain on their own
        assert_eq!(get_feat_extract_output_lengths(50), 7);
        assert_eq!(get_feat_extract_output_lengths(150), 20);
        assert_eq!(get_feat_extract_output_lengths(101), 14);
    }

    // cross-check the remainder branch against the raw conv-output formula
    // out = (in - 1) / 2 + 1 applied three times (kernel 3, stride 2, pad 1)
    #[test]
    fn feat_extract_remainder_matches_convs() {
        fn conv3(mut n: usize) -> usize {
            for _ in 0..3 {
                n = (n - 1) / 2 + 1;
            }
            n
        }
        for rem in 1..100usize {
            let audio_len = 500 + rem; // 5 full windows + remainder
            assert_eq!(
                get_feat_extract_output_lengths(audio_len),
                5 * 13 + conv3(rem)
            );
        }
    }
}
