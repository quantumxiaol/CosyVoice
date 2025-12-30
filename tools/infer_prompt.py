import argparse
from pathlib import Path

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel


def _save_chunks(generator, sample_rate, out_prefix):
    for idx, output in enumerate(generator):
        out_path = f"{out_prefix}_{idx}.wav"
        torchaudio.save(out_path, output["tts_speech"].cpu(), sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Run CosyVoice inference with a prompt audio file."
    )
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/CosyVoice-300M",
        help="Path or ModelScope id for the model.",
    )
    parser.add_argument(
        "--prompt-wav",
        required=True,
        help="Prompt audio path (wav/mp3/ogg supported by torchaudio).",
    )
    parser.add_argument(
        "--mode",
        choices=["cross_lingual", "zero_shot"],
        default="cross_lingual",
        help="Inference mode to use.",
    )
    parser.add_argument(
        "--prompt-text",
        default="",
        help="Prompt text for zero_shot mode.",
    )
    parser.add_argument(
        "--text-zh",
        required=True,
        help="Chinese text to synthesize.",
    )
    parser.add_argument(
        "--text-ja",
        required=True,
        help="Japanese text to synthesize.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/infer_prompt",
        help="Output directory for generated wav files.",
    )
    args = parser.parse_args()

    if args.mode == "zero_shot" and not args.prompt_text:
        parser.error("--prompt-text is required when --mode=zero_shot")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(model_dir=args.model_dir)

    if args.mode == "cross_lingual":
        zh_gen = cosyvoice.inference_cross_lingual(
            args.text_zh, args.prompt_wav, stream=False
        )
        ja_gen = cosyvoice.inference_cross_lingual(
            args.text_ja, args.prompt_wav, stream=False
        )
    else:
        zh_gen = cosyvoice.inference_zero_shot(
            args.text_zh, args.prompt_text, args.prompt_wav, stream=False
        )
        ja_gen = cosyvoice.inference_zero_shot(
            args.text_ja, args.prompt_text, args.prompt_wav, stream=False
        )

    _save_chunks(zh_gen, cosyvoice.sample_rate, str(out_dir / "zh"))
    _save_chunks(ja_gen, cosyvoice.sample_rate, str(out_dir / "ja"))


if __name__ == "__main__":
    main()
