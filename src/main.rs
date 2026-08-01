use tiny_cpm::{
    common, exec, models, position_embed, quantized_minicpm5, token_output_stream, tokenizer, utils,
};

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        usage();
        std::process::exit(1);
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "chat" => exec::chat::run(rest),
        "dialogue" => exec::dialogue::run(rest),
        "asr" => match rest.first().map(String::as_str) {
            Some("funasr") => exec::fun_asr_nano::run(&rest[1..]),
            Some("qwen3") => exec::qwen3_asr::run(&rest[1..]),
            _ => bail!("usage: tiny-cpm asr <funasr|qwen3> <model-dir> <audio-file> [max_tokens]"),
        },
        "tts" => match rest.first().map(String::as_str) {
            Some("voxcpm") => exec::voxcpm::run(&rest[1..]),
            Some("moss") => exec::moss_tts::run(&rest[1..]),
            Some("cosyvoice3") => exec::cosyvoice3::run(&rest[1..]),
            Some("qwen3") => exec::qwen3_tts::run(&rest[1..]),
            _ => bail!(
                "usage: tiny-cpm tts <voxcpm|moss> <model-dir> \"<text>\" <out.wav> [--codec <codec-dir>] [--ref <ref.wav>] [--max-len N]\n       tiny-cpm tts cosyvoice3 <model-dir> \"<text>\" <out.wav> [--voice <name>] [--ref <ref.wav> --ref-text \"<text>\"] [--steps N] [--max-tokens N]\n       tiny-cpm tts qwen3 <model-dir> \"<text>\" <out.wav> [--ref <ref.wav> --ref-text \"<text>\"] [--language <lang>] [--max-frames N]"
            ),
        },
        "vad" => exec::vad::run(rest),
        "live" => exec::live::run(rest),
        "codec-rt" => exec::moss_tts::run_codec_rt(rest),
        _ => {
            usage();
            std::process::exit(1);
        }
    }
}

fn usage() {
    eprintln!("usage:");
    eprintln!("  tiny-cpm chat <bf16-dir> <tokenizer.json> \"<prompt>\" [max_tokens]");
    eprintln!("  tiny-cpm asr <funasr|qwen3> <model-dir> <audio-file> [max_tokens]");
    eprintln!(
        "  tiny-cpm tts <voxcpm|moss> <model-dir> \"<text>\" <out.wav> [--codec <codec-dir>] [--ref <ref.wav>] [--max-len N]"
    );
    eprintln!(
        "  tiny-cpm tts cosyvoice3 <model-dir> \"<text>\" <out.wav> [--voice <name>] [--ref <ref.wav> --ref-text \"<text>\"] [--steps N] [--max-tokens N]"
    );
    eprintln!("  tiny-cpm vad <model-dir> <audio-file>");
    eprintln!(
        "  tiny-cpm dialogue <funasr-dir> <bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]"
    );
    eprintln!(
        "  tiny-cpm live <vad-dir> <qwen3asr-dir> <bf16-dir> <tokenizer.json> <tts-model-dir> [<codec-dir: MOSS only>] [--tts moss|qwen3] [--ref <wav> [--ref-text \"<text>\"]] [--input <wav>] [--output <wav>] [--max-tokens N]"
    );
}
