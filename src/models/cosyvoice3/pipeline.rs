//! CosyVoice3 pipeline (see mod.rs header).
//!
//! End-to-end glue over the ported components, mirroring CrispASR
//! `cv3_synth_with_voice` (src/cosyvoice3_tts.cpp:5492-5671) for baked voices
//! and `cv3_extract_native_runtime_voice` (:5204-5245) for zero-shot cloning.
//!
//! Baked voices come from `voices.gguf` in the model dir (converter
//! convert-cosyvoice3-voices-to-gguf.py): a `voice.names` string-array
//! metadata entry, plus per voice a metadata STRING `<prefix><name>.prompt_text`
//! and tensors `.prompt_speech_tokens` (I32), `.spk_emb` (F32[192], raw
//! CAMPPlus) and `.ref_mel` (F32[T,80], 24 kHz matcha mel). `<prefix>` is
//! `cv3.voices.` or, as the reference converter writes it, `voice.`. The file
//! is read by a bespoke minimal GGUF parser below — candle's gguf reader
//! rejects the whole file on the I32 tensor dtype.
//!
//! Alignment trap (CrispASR §5505-5526): the prompt speech tokens and the
//! ref mel MUST be trimmed so that T_ref_mel == TOKEN_MEL_RATIO *
//! n_prompt_tokens, otherwise the flow's cond prefix is misaligned with mu
//! and the output comes out ~14 dB quiet. `flow::align_prompt_len` encodes
//! the trim rule; synthesize() applies it to baked and cloned voices alike.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::campplus::{CampPlus, compute_prompt_feat_24k};
use super::flow::{CosyVoice3Flow, TOKEN_MEL_RATIO, align_prompt_len};
use super::hift::Hift;
use super::lm::CosyVoice3LM;
use super::s3tok::{S3Tok, find_gguf};
use crate::utils::audio_utils::load_audio_with_resample;

/// HiFT output sample rate (mono).
pub const SAMPLE_RATE: u32 = 24000;

/// `<|endofprompt|>` id. CosyVoice-BlankEN vocab.json stops at 151642; the
/// CV3 specials 151643-151646 (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`,
/// `<|endofprompt|>`) are registered manually in the reference
/// (cosyvoice3_tts.cpp:932-956). Only `<|endofprompt|>` appears in prompt
/// texts; it is split out BEFORE BPE and its id inserted between chunks
/// (cv3_tokenise_prompt, :4948-4974).
pub const ENDOFPROMPT_ID: u32 = 151646;
const ENDOFPROMPT: &str = "<|endofprompt|>";

/// A synthesis voice: prompt text + aligned speech tokens / speaker
/// embedding / ref mel (either baked from voices.gguf or cloned at runtime).
#[derive(Clone)]
pub struct Voice {
    pub name: String,
    pub prompt_text: String,
    pub prompt_speech_tokens: Vec<u32>,
    /// (192,) raw CAMPPlus embedding (the flow L2-normalizes internally).
    pub spk_emb: Tensor,
    /// (T, 80) 24 kHz matcha mel; trimmed to 2*n_prompt_tokens in synthesize.
    pub ref_mel: Tensor,
}

/// Per-synthesis diagnostics (returned alongside the waveform).
#[derive(Debug, Default)]
pub struct SynthStats {
    pub n_text_ids: usize,
    pub n_prompt_tokens: usize,
    pub n_gen_tokens: usize,
    pub t_mel_out: usize,
    pub lm_secs: f64,
    pub flow_secs: f64,
    pub hift_secs: f64,
    pub audio_secs: f64,
}

/// Tokenize a CV3 prompt fragment: split on the literal `<|endofprompt|>`,
/// BPE each chunk, emit ENDOFPROMPT_ID between chunks (cv3_tokenise_prompt).
pub fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for (i, chunk) in text.split(ENDOFPROMPT).enumerate() {
        if i > 0 {
            ids.push(ENDOFPROMPT_ID);
        }
        if !chunk.is_empty() {
            let enc = tokenizer
                .encode(chunk, false)
                .map_err(|e| anyhow!("cosyvoice3 tokenizer: {e}"))?;
            ids.extend_from_slice(enc.get_ids());
        }
    }
    Ok(ids)
}

/// Qwen2 byte-level BPE from CosyVoice-BlankEN/vocab.json + merges.txt.
/// ByteLevel WITHOUT add_prefix_space (Qwen2 convention). Deliberate
/// divergence from CrispASR's `tokenize_simple` bytes-to-unicode
/// approximation: the real BPE matches the upstream Qwen2 tokenizer exactly
/// (e.g. "hello  world" -> [14990,220,1879] vs CrispASR [14990,1879];
/// "a\nb" -> [64,198,65] vs [64,Ġb]).
fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let tok_dir = dir.join("CosyVoice-BlankEN");
    let vocab = tok_dir.join("vocab.json");
    let merges = tok_dir.join("merges.txt");
    let bpe = BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
        .build()
        .map_err(|e| anyhow!("cosyvoice3 tokenizer from {}: {e}", tok_dir.display()))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    Ok(tok)
}

// ---------------------------------------------------------------------------
// Minimal GGUF reader for the voice bank
// ---------------------------------------------------------------------------
//
// candle's `gguf_file::Content::read` rejects the WHOLE file when any tensor
// has a dtype it doesn't know — voices.gguf stores prompt_speech_tokens as
// I32 (ggml type 26), which candle 0.11 doesn't recognize. This bespoke
// parser handles exactly the voice-bank layout (GGUF v3, little-endian, as
// written by convert-cosyvoice3-voices-to-gguf.py):
//   header:   magic "GGUF", version u32, tensor_count u64, metadata_kv_count u64
//   metadata: key string, value_type u32, value (scalars LE; string = u64 len
//             + bytes; array = elem type u32 + count u64 + elements)
//   tensors:  name string, n_dims u32, dims u64[n_dims] in GGML ORDER
//             (reversed vs the torch/numpy layout), dtype u32, offset u64
//             relative to the data section
//   data:     starts at align(end_of_infos, alignment); payloads are C-order
//             in the ORIGINAL (reversed) shape.
// Only F32 (0) and I32 (26) tensor payloads are supported; anything else
// fails with a clear "unsupported dtype" error.

const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_I32: u32 = 26;

// GGUF metadata value types.
const GGUF_VAL_UINT8: u32 = 0;
const GGUF_VAL_INT8: u32 = 1;
const GGUF_VAL_UINT16: u32 = 2;
const GGUF_VAL_INT16: u32 = 3;
const GGUF_VAL_UINT32: u32 = 4;
const GGUF_VAL_INT32: u32 = 5;
const GGUF_VAL_FLOAT32: u32 = 6;
const GGUF_VAL_BOOL: u32 = 7;
const GGUF_VAL_STRING: u32 = 8;
const GGUF_VAL_ARRAY: u32 = 9;
const GGUF_VAL_UINT64: u32 = 10;
const GGUF_VAL_INT64: u32 = 11;
const GGUF_VAL_FLOAT64: u32 = 12;

/// Byte size of a scalar metadata value type (string/array handled separately).
fn gguf_scalar_size(vtype: u32) -> Result<usize> {
    match vtype {
        GGUF_VAL_UINT8 | GGUF_VAL_INT8 | GGUF_VAL_BOOL => Ok(1),
        GGUF_VAL_UINT16 | GGUF_VAL_INT16 => Ok(2),
        GGUF_VAL_UINT32 | GGUF_VAL_INT32 | GGUF_VAL_FLOAT32 => Ok(4),
        GGUF_VAL_UINT64 | GGUF_VAL_INT64 | GGUF_VAL_FLOAT64 => Ok(8),
        t => bail!("voices.gguf: unsupported metadata value type {t}"),
    }
}

struct GgufCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> GgufCursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| anyhow!("voices.gguf: truncated at offset {}", self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).context("voices.gguf: invalid utf-8 string")
    }
}

struct VoiceTensorInfo {
    dims: Vec<usize>, // ggml order (reversed vs torch)
    dtype: u32,
    offset: u64, // relative to the data section
}

/// Parsed voices.gguf: string/string-array metadata + tensor directory +
/// the raw file bytes (tensors are sliced out of it lazily).
struct VoicesGguf {
    strings: HashMap<String, String>,
    string_arrays: HashMap<String, Vec<String>>,
    tensors: HashMap<String, VoiceTensorInfo>,
    buf: Vec<u8>,
    data_start: usize,
}

impl VoicesGguf {
    fn read(buf: Vec<u8>) -> Result<Self> {
        let mut cur = GgufCursor { buf: &buf, pos: 0 };
        if cur.take(4)? != b"GGUF" {
            bail!("voices.gguf: bad magic (not a GGUF file)");
        }
        let version = cur.u32()?;
        if version != 3 {
            bail!("voices.gguf: unsupported GGUF version {version} (expected 3)");
        }
        let n_tensors = cur.u64()?;
        let n_kv = cur.u64()?;

        let mut strings = HashMap::new();
        let mut string_arrays = HashMap::new();
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for _ in 0..n_kv {
            let key = cur.string()?;
            let vtype = cur.u32()?;
            match vtype {
                GGUF_VAL_STRING => {
                    strings.insert(key, cur.string()?);
                }
                GGUF_VAL_ARRAY => {
                    let etype = cur.u32()?;
                    let n = cur.u64()? as usize;
                    if etype == GGUF_VAL_STRING {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(cur.string()?);
                        }
                        string_arrays.insert(key, v);
                    } else {
                        cur.take(n * gguf_scalar_size(etype)?)?;
                    }
                }
                GGUF_VAL_UINT32 => {
                    let v = cur.u32()?;
                    if key == "general.alignment" {
                        alignment = v as usize;
                    }
                }
                t => {
                    cur.take(gguf_scalar_size(t)?)?;
                }
            }
        }

        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.u64()? as usize);
            }
            let dtype = cur.u32()?;
            let offset = cur.u64()?;
            tensors.insert(
                name,
                VoiceTensorInfo {
                    dims,
                    dtype,
                    offset,
                },
            );
        }
        let data_start = cur.pos.div_ceil(alignment) * alignment;
        Ok(Self {
            strings,
            string_arrays,
            tensors,
            buf,
            data_start,
        })
    }

    /// Tensor payload slice + directory entry, validating dtype and bounds.
    fn tensor_bytes(&self, name: &str) -> Result<(&VoiceTensorInfo, &[u8])> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("voices.gguf: missing tensor `{name}`"))?;
        if info.dtype != GGML_TYPE_F32 && info.dtype != GGML_TYPE_I32 {
            bail!(
                "unsupported dtype {} for tensor `{name}` in voices.gguf (only F32/I32 supported)",
                info.dtype
            );
        }
        let numel: usize = info.dims.iter().product();
        let start = self.data_start + info.offset as usize;
        let end = start + numel * 4;
        if end > self.buf.len() {
            bail!("voices.gguf: tensor `{name}` data out of bounds");
        }
        Ok((info, &self.buf[start..end]))
    }

    /// F32 tensor -> candle Tensor with ggml dims reversed back to torch layout.
    fn tensor_f32(&self, name: &str, device: &Device) -> Result<Tensor> {
        let (info, bytes) = self.tensor_bytes(name)?;
        if info.dtype != GGML_TYPE_F32 {
            bail!(
                "voices.gguf: tensor `{name}` is dtype {}, expected F32",
                info.dtype
            );
        }
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        Ok(Tensor::from_vec(v, shape.as_slice(), device)?)
    }

    /// I32 tensor -> token ids.
    fn tensor_i32(&self, name: &str) -> Result<Vec<u32>> {
        let (info, bytes) = self.tensor_bytes(name)?;
        if info.dtype != GGML_TYPE_I32 {
            bail!(
                "voices.gguf: tensor `{name}` is dtype {}, expected I32",
                info.dtype
            );
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect())
    }
}

/// Load the baked voice bank (`*voices*.gguf` in `dir`). Returns an empty
/// map when the file is absent (zero-shot cloning still works).
fn load_voice_bank(dir: &Path, device: &Device) -> Result<HashMap<String, Voice>> {
    let path = match find_gguf(dir, "voices") {
        Ok(p) => p,
        // 0 hits -> no voice bank (zero-shot cloning still works);
        // multiple hits is an error, not an empty bank.
        Err(e) if e.to_string().starts_with("no *") => return Ok(HashMap::new()),
        Err(e) => return Err(e),
    };
    let buf = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let gguf = VoicesGguf::read(buf).with_context(|| format!("parse {}", path.display()))?;

    // Key prefix: `cv3.voices.` per the planned scheme, `voice.` as written
    // by convert-cosyvoice3-voices-to-gguf.py.
    let prefix = if gguf.tensors.keys().any(|k| k.starts_with("cv3.voices.")) {
        "cv3.voices."
    } else {
        "voice."
    };
    // Voice names: the `<prefix>names` string-array metadata entry when
    // present (the converter writes `voice.names`), else enumerate the
    // per-voice tensor keys.
    let mut names: Vec<String> = gguf
        .string_arrays
        .get(&format!("{prefix}names"))
        .cloned()
        .unwrap_or_default();
    if names.is_empty() {
        let mut set = BTreeSet::new();
        for key in gguf.tensors.keys() {
            if let Some(rest) = key.strip_prefix(prefix)
                && let Some((name, _field)) = rest.split_once('.')
            {
                set.insert(name.to_string());
            }
        }
        names = set.into_iter().collect();
    }

    let mut voices = HashMap::new();
    for name in names {
        let base = format!("{prefix}{name}");
        let prompt_text = gguf
            .strings
            .get(&format!("{base}.prompt_text"))
            .cloned()
            .ok_or_else(|| anyhow!("{}: missing metadata `{base}.prompt_text`", path.display()))?;
        let prompt_speech_tokens = gguf.tensor_i32(&format!("{base}.prompt_speech_tokens"))?;
        let spk_emb = gguf
            .tensor_f32(&format!("{base}.spk_emb"), device)?
            .flatten_all()?;
        let ref_mel = gguf.tensor_f32(&format!("{base}.ref_mel"), device)?;
        // Shape validation (C++ cosyvoice3_tts.cpp:5177): spk_emb is a raw
        // 192-d CAMPPlus embedding, ref_mel rows are 80-bin mel frames.
        if spk_emb.elem_count() != 192 {
            bail!(
                "{}: voice '{name}': spk_emb has {} elements, expected 192",
                path.display(),
                spk_emb.elem_count()
            );
        }
        if ref_mel.elem_count() % 80 != 0 {
            bail!(
                "{}: voice '{name}': ref_mel has {} elements, not a multiple of 80",
                path.display(),
                ref_mel.elem_count()
            );
        }
        voices.insert(
            name.clone(),
            Voice {
                name,
                prompt_text,
                prompt_speech_tokens,
                spk_emb,
                ref_mel,
            },
        );
    }
    Ok(voices)
}

/// CosyVoice3 end-to-end pipeline: LM + flow + hift + BPE + voice bank,
/// with s3tok/CAMPPlus loaded lazily on the first zero-shot clone request.
pub struct CosyVoice3Pipeline {
    lm: CosyVoice3LM,
    flow: CosyVoice3Flow,
    hift: Hift,
    tokenizer: Tokenizer,
    voices: HashMap<String, Voice>,
    s3tok: Option<S3Tok>,
    campplus: Option<CampPlus>,
    model_dir: PathBuf,
    device: Device,
}

impl CosyVoice3Pipeline {
    /// Load all synthesis components from the Fun-CosyVoice3 model dir.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let t0 = Instant::now();
        let lm = CosyVoice3LM::load(dir, device)?;
        eprintln!("cosyvoice3: llm.pt loaded in {:.2?}", t0.elapsed());
        let t0 = Instant::now();
        let flow = CosyVoice3Flow::load(dir, device)?;
        eprintln!("cosyvoice3: flow.pt loaded in {:.2?}", t0.elapsed());
        let t0 = Instant::now();
        let hift = Hift::load(dir, device)?;
        eprintln!("cosyvoice3: hift.pt loaded in {:.2?}", t0.elapsed());
        let tokenizer = load_tokenizer(dir)?;
        let voices = load_voice_bank(dir, device)?;
        if voices.is_empty() {
            eprintln!("cosyvoice3: no voices.gguf in model dir — baked voices unavailable");
        } else {
            let mut names: Vec<&str> = voices.keys().map(String::as_str).collect();
            names.sort_unstable();
            eprintln!(
                "cosyvoice3: {} baked voices: {}",
                names.len(),
                names.join(", ")
            );
        }
        Ok(Self {
            lm,
            flow,
            hift,
            tokenizer,
            voices,
            s3tok: None,
            campplus: None,
            model_dir: dir.to_path_buf(),
            device: device.clone(),
        })
    }

    /// Names of the baked voices (empty without voices.gguf).
    pub fn voice_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.voices.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// A baked voice by name.
    pub fn get_voice(&self, name: &str) -> Result<Voice> {
        self.voices.get(name).cloned().ok_or_else(|| {
            let names = self.voice_names();
            anyhow!(
                "cosyvoice3: unknown voice '{name}' (available: {})",
                if names.is_empty() {
                    "none — voices.gguf missing".to_string()
                } else {
                    names.join(", ")
                }
            )
        })
    }

    /// Zero-shot voice cloning from a reference wav (cv3_extract_native_runtime_voice):
    /// resample to 16 kHz (s3tok tokens + CAMPPlus embedding) and 24 kHz
    /// (ref mel). `ref_text` is the transcript of the ref wav; plain text is
    /// fine, `<|endofprompt|>` is optional (tokenize_prompt handles both).
    pub fn clone_voice(&mut self, ref_wav: &str, ref_text: &str) -> Result<Voice> {
        if self.s3tok.is_none() {
            let t0 = Instant::now();
            self.s3tok = Some(
                S3Tok::load(&self.model_dir, &self.device)
                    .context("zero-shot cloning (--ref) needs the s3tok GGUF in the model dir")?,
            );
            eprintln!("cosyvoice3: s3tok loaded in {:.2?}", t0.elapsed());
        }
        if self.campplus.is_none() {
            let t0 = Instant::now();
            self.campplus =
                Some(CampPlus::load(&self.model_dir, &self.device).context(
                    "zero-shot cloning (--ref) needs the campplus GGUF in the model dir",
                )?);
            eprintln!("cosyvoice3: campplus loaded in {:.2?}", t0.elapsed());
        }
        let wav16 = load_audio_with_resample(ref_wav, &self.device, Some(16000), Some(1))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let wav24 = load_audio_with_resample(ref_wav, &self.device, Some(24000), Some(1))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let prompt_speech_tokens = self.s3tok.as_ref().unwrap().encode(&wav16)?;
        if prompt_speech_tokens.is_empty() {
            bail!("cosyvoice3: s3tok produced 0 tokens for '{ref_wav}'");
        }
        let spk_emb = self.campplus.as_ref().unwrap().embed(&wav16)?;
        let ref_mel = compute_prompt_feat_24k(&wav24, &self.device)?;
        Ok(Voice {
            name: "runtime".to_string(),
            prompt_text: ref_text.to_string(),
            prompt_speech_tokens,
            spk_emb,
            ref_mel,
        })
    }

    /// Synthesize `text` with `voice` -> 24 kHz mono f32 samples.
    /// `max_tokens == 0` -> 20 * n_text_ids (min 16); `seed == 0` -> 42.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &Voice,
        max_tokens: usize,
        n_steps: usize,
        cfg: f64,
        seed: u64,
    ) -> Result<(Vec<f32>, SynthStats)> {
        let mut stats = SynthStats::default();

        // 0. Align prompt tokens + ref_mel to TOKEN_MEL_RATIO (module header:
        // skipping this makes the output ~14 dB quiet).
        let n_prompt = align_prompt_len(voice.prompt_speech_tokens.len(), voice.ref_mel.dim(0)?);
        if n_prompt == 0 {
            bail!(
                "cosyvoice3: voice '{}' has an empty aligned prompt",
                voice.name
            );
        }
        let prompt_tokens = &voice.prompt_speech_tokens[..n_prompt];
        let t_ref_mel = n_prompt * TOKEN_MEL_RATIO;
        let ref_mel = voice.ref_mel.narrow(0, 0, t_ref_mel)?;

        // 1. text ids = BPE(voice.prompt_text) + BPE(user text), with
        // <|endofprompt|> split out of both (reference: prompt_ids ++ user_ids).
        let mut text_ids = tokenize_prompt(&self.tokenizer, &voice.prompt_text)?;
        text_ids.extend(tokenize_prompt(&self.tokenizer, text)?);
        if text_ids.is_empty() {
            bail!("cosyvoice3: empty text after tokenisation");
        }

        // 2. LM AR-decode speech tokens.
        let t0 = Instant::now();
        let gen_tokens =
            self.lm
                .generate_speech_tokens(&text_ids, prompt_tokens, max_tokens, seed)?;
        stats.lm_secs = t0.elapsed().as_secs_f64();
        if gen_tokens.is_empty() {
            bail!("cosyvoice3: LM produced 0 speech tokens");
        }

        // 3. Flow Euler -> mel INCLUDING the ref prefix frames.
        let mut full_tokens = Vec::with_capacity(prompt_tokens.len() + gen_tokens.len());
        full_tokens.extend_from_slice(prompt_tokens);
        full_tokens.extend_from_slice(&gen_tokens);
        let t0 = Instant::now();
        let mel_full =
            self.flow
                .synthesize_mel(&full_tokens, &voice.spk_emb, &ref_mel, n_steps, cfg, seed)?;
        stats.flow_secs = t0.elapsed().as_secs_f64();

        // 4. Slice off the prompt-mel prefix (T_ref_mel == n_prompt * ratio
        // by construction, so this never overruns).
        let mel = mel_full.narrow(0, t_ref_mel, mel_full.dim(0)? - t_ref_mel)?;

        // 5. HiFT -> 24 kHz audio.
        let t0 = Instant::now();
        let wav = self.hift.mel_to_waveform(&mel)?;
        stats.hift_secs = t0.elapsed().as_secs_f64();

        stats.n_text_ids = text_ids.len();
        stats.n_prompt_tokens = n_prompt;
        stats.n_gen_tokens = gen_tokens.len();
        stats.t_mel_out = mel.dim(0)?;
        stats.audio_secs = wav.len() as f64 / SAMPLE_RATE as f64;
        Ok((wav, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manual <|endofprompt|> split must emit the special id between
    /// chunks and never feed the delimiter string to BPE.
    #[test]
    fn tokenize_prompt_splits_endofprompt() {
        let tok_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN");
        if !tok_dir.join("vocab.json").exists() {
            eprintln!("tokenizer files not downloaded; skipping");
            return;
        }
        let tok = load_tokenizer(&tok_dir.parent().unwrap().to_path_buf()).unwrap();
        let plain = tokenize_prompt(&tok, "Hello world").unwrap();
        assert!(!plain.is_empty());
        assert!(!plain.contains(&ENDOFPROMPT_ID));
        let mixed =
            tokenize_prompt(&tok, "You are a helpful assistant.<|endofprompt|>Hello").unwrap();
        assert_eq!(mixed.iter().filter(|&&i| i == ENDOFPROMPT_ID).count(), 1);
        // Chunk boundaries: ids = BPE(a) + [EOP] + BPE(b).
        let a = tokenize_prompt(&tok, "You are a helpful assistant.").unwrap();
        let b = tokenize_prompt(&tok, "Hello").unwrap();
        let mut expect = a.clone();
        expect.push(ENDOFPROMPT_ID);
        expect.extend(b);
        assert_eq!(mixed, expect);
        // Empty prompt text -> no ids at all.
        assert!(tokenize_prompt(&tok, "").unwrap().is_empty());
    }

    /// Parse the real voices.gguf: names, key scheme, dtypes and shapes.
    #[test]
    fn voice_bank_parses() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/Fun-CosyVoice3-0.5B-2512");
        if !dir.join("cosyvoice3-voices.gguf").exists() {
            eprintln!("voices.gguf not downloaded; skipping");
            return;
        }
        let voices = load_voice_bank(&dir, &Device::Cpu).unwrap();
        assert!(!voices.is_empty());
        eprintln!("voices: {:?}", voices.keys().collect::<Vec<_>>());
        for v in voices.values() {
            assert!(!v.prompt_speech_tokens.is_empty(), "{}: no tokens", v.name);
            assert!(
                v.prompt_speech_tokens
                    .iter()
                    .all(|&t| (t as usize) < super::super::lm::SPEECH_CODEBOOK),
                "{}: token id out of codebook",
                v.name
            );
            assert_eq!(v.spk_emb.dims(), &[192], "{}: spk_emb shape", v.name);
            let (t, mel) = v.ref_mel.dims2().unwrap();
            assert_eq!(mel, 80, "{}: ref_mel dim", v.name);
            // Converter bakes the alignment: T_ref_mel == 2 * n_tokens.
            assert_eq!(
                t,
                v.prompt_speech_tokens.len() * TOKEN_MEL_RATIO,
                "{}: ref_mel/tokens misaligned",
                v.name
            );
        }
    }
}
