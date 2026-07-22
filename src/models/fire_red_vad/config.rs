//! Ported from aha (github.com/jhqxxx/aha) src/models/fire_red_vad/config.rs
pub struct FireRedVadConfig {
    pub smooth_window_size: usize,
    pub speech_threshold: f32,
    pub singing_threshold: f32,
    pub music_threshold: f32,
    pub pad_start_frame: usize,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_event_frame: usize,
    pub max_event_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
    pub chunk_max_frame: usize,
}

impl FireRedVadConfig {
    pub fn default_vad() -> Self {
        Self {
            smooth_window_size: 5,
            speech_threshold: 0.4,
            singing_threshold: 0.5,
            music_threshold: 0.5,
            pad_start_frame: 5,
            min_speech_frame: 20,
            max_speech_frame: 2000,
            min_event_frame: 20,
            max_event_frame: 2000,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
            chunk_max_frame: 30000,
        }
    }

    pub fn default_stream_vad() -> Self {
        Self {
            smooth_window_size: 1,
            speech_threshold: 0.5,
            singing_threshold: 0.5,
            music_threshold: 0.5,
            pad_start_frame: 5,
            min_speech_frame: 8,
            max_speech_frame: 2000,
            min_event_frame: 20,
            max_event_frame: 2000,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
            chunk_max_frame: 30000,
        }
    }

    pub fn default_aed() -> Self {
        Self {
            smooth_window_size: 5,
            speech_threshold: 0.4,
            singing_threshold: 0.5,
            music_threshold: 0.5,
            pad_start_frame: 5,
            min_speech_frame: 8,
            max_speech_frame: 2000,
            min_event_frame: 20,
            max_event_frame: 2000,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
            chunk_max_frame: 30000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct CMVNData {
    pub cmvn: Vec<Vec<f32>>,
}

pub struct DetectModelConfig {
    pub idim: usize,
    pub r: usize,
    pub m: usize,
    pub h: usize,
    pub p: usize,
    pub n1: usize,
    pub s1: usize,
    pub n2: usize,
    pub s2: usize,
    pub odim: usize,
}

impl DetectModelConfig {
    pub fn default_vad() -> Self {
        Self {
            idim: 80,
            r: 8,
            m: 1,
            h: 256,
            p: 128,
            n1: 20,
            s1: 1,
            n2: 20,
            s2: 1,
            odim: 1,
        }
    }

    pub fn default_stream_vad() -> Self {
        Self {
            idim: 80,
            r: 8,
            m: 1,
            h: 256,
            p: 128,
            n1: 20,
            s1: 1,
            n2: 0,
            s2: 1,
            odim: 1,
        }
    }

    pub fn default_aed() -> Self {
        Self {
            idim: 80,
            r: 8,
            m: 1,
            h: 256,
            p: 128,
            n1: 20,
            s1: 1,
            n2: 20,
            s2: 1,
            odim: 3,
        }
    }
}
