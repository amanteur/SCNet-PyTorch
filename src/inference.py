import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple, List

import soundfile as sf
import torch.nn as nn
import torchaudio

from model.separator import Separator

SOURCES: List[str] = ["drums", "bass", "other", "vocals"]
SAVE_SAMPLE_RATE: int = 44100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Argument Parser for Separator")
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Input path to .wav audio file/directory containing audio files",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory to save separated audio files in .wav format",
    )
    parser.add_argument(
        "-c",
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "-w", "--window-size", type=int, default=11, help="Window size (default: 11)"
    )
    parser.add_argument(
        "-s", "--step-size", type=float, default=5.5, help="Step size (default: 5.5)"
    )
    parser.add_argument(
        "-p",
        "--use-progress-bar",
        action="store_true",
        help="Use progress bar (default: True)",
    )
    return parser.parse_args()


def load_paths(input_path: str, output_path: str) -> Iterable[Tuple[Path, Path]]:
    """
    Load input and output paths.

    Args:
        input_path (str): Input path to audio files.
        output_path (str): Output directory to save separated audio files.

    Yields:
        Tuple[Path, Path]: Tuple of input and output file paths.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    if input_path.is_file():
        if not (input_path.suffix == ".wav" or input_path.suffix == ".mp3"):
            raise ValueError("Input audio file should be in .wav or .mp3 formats.")
        fp_out = output_path / input_path.stem
        fp_out.mkdir(exist_ok=True, parents=True)
        yield input_path, fp_out
    elif input_path.is_dir():
        for fp_in in input_path.glob("*"):
            if fp_in.suffix in (".wav", ".mp3"):
                fp_out = output_path / fp_in.stem
                fp_out.mkdir(exist_ok=True, parents=True)
                yield fp_in, fp_out
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


def process_files(
    separator: nn.Module,
    input_path: str,
    output_path: str
) -> None:
    for fp_in, fp_out in load_paths(input_path, output_path):
        y, sr = torchaudio.load(fp_in)
        y = y.to(args.device)
        y_separated = separator.separate(y).cpu()
        for y_source, source in zip(y_separated, SOURCES):
            sf.write(f"{fp_out}/{source}.wav", y_source.T, 44100)


def main():
    args = parse_arguments()

    logger.info(
        f"Initializing Separator with following checkpoint {args.ckpt_path}..."
    )
    separator = Separator.load_from_checkpoint(
        path=args.ckpt_path,
        batch_size=args.batch_size,
        window_size=args.window_size,
        step_size=args.step_size,
        use_progress_bar=args.use_progress_bar,
    ).to(args.device)

    logger.info("Processing audio files...")
    process_files(separator, args.input_path, args.output_path)
    logger.info(f"Audio files processing completed.")


if __name__ == "__main__":
    main()
