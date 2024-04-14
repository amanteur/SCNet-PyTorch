import argparse
import os
import logging
from typing import Dict, Iterable, List, Tuple

import fast_bss_eval
import pandas as pd
import torchaudio
import torch.nn as nn
from tqdm import tqdm

from model.separator import Separator

SOURCES: List[str] = ["drums", "bass", "other", "vocals"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Argument Parser for Separator")
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
    return parser.parse_args()


def load_data(dataset_path: str) -> Iterable[Tuple[str, str, Dict[str, str]]]:
    """
    Load data from the dataset.

    Args:
        dataset_path (str): Path to the dataset.

    Yields:
        Tuple[str, str, Dict[str, str]]: Tuple containing track name, mixture path, and source paths.
    """
    df = pd.read_parquet(dataset_path)
    df = df[df["subset"].eq("test")]
    track_names = df["track_name"].unique()
    for track_name in tqdm(track_names):
        rows = df[df["track_name"].eq(track_name)]
        mixture_path = rows[rows["source_type"].eq("mixture")]["path"].values[0]
        source_paths = (
            rows[~rows["source_type"].eq("mixture")]
            .set_index("source_type")["path"]
            .to_dict()
        )
        yield track_name, mixture_path, source_paths


def compute_sdrs(separator: nn.Module, dataset_path: str, device: str) -> str:
    """
    Compute evaluation SDRs.

    Args:
        separator (nn.Module): Separator model.
        dataset_path (str): Path to the dataset.pqt.
        device (str): Device to send tensors on.

    Returns:
        str: Evaluation SDRs table.
    """
    sdrs = []
    for track_name, mixture_path, source_paths in load_data(dataset_path):
        y, sr = torchaudio.load(mixture_path)
        y_separated = separator.separate(y.to(device)).cpu()
        for y_source_est, source_type in zip(y_separated, SOURCES):
            y_source_ref, _ = torchaudio.load(source_paths[source_type])
            sdr, *_ = fast_bss_eval.bss_eval_sources(
                y_source_ref, y_source_est, compute_permutation=False, load_diag=1e-7
            )
            sdrs.append((track_name, source_type, sdr.mean().item()))
    sdrs_df = pd.DataFrame(sdrs, columns=["track_name", "source_type", "sdr"])

    return (
        sdrs_df.groupby("source_type")["sdr"].mean().reset_index(name="sdr").to_string()
    )


def main():
    args = parse_arguments()

    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path is None:
        raise ValueError("DATASET_PATH environment variable is not set.")

    log.info(f"Initializing Separator with following checkpoint {args.ckpt_path}...")
    separator = Separator.load_from_checkpoint(
        path=args.ckpt_path,
        batch_size=args.batch_size,
        window_size=args.window_size,
        step_size=args.step_size,
    ).to(args.device)

    log.info("Starting evaluation...")
    metrics = compute_sdrs(separator, dataset_path, args.device)
    log.info(f"Evaluation completed with following metrics:\n{metrics}")


if __name__ == "__main__":
    main()
