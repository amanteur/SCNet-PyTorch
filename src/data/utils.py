from typing import List
from pathlib import Path
from typing import Optional

import pandas as pd
import torchaudio


def construct_dataset(
    dataset_dirs: List[str], extension: str = "wav", save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Constructs a dataset DataFrame from audio files in the specified directories.

    Args:
    - dataset_dirs (List[str]): List of directories containing audio files.
    - extension (str): Extension of the audio files to consider. Defaults to "wav".
    - save_path (Optional[str]): Optional path to save the constructed DataFrame as a parquet file.

    Returns:
    - pd.DataFrame: DataFrame containing information about the audio files.
    """
    dataset = []
    for dataset_dir in dataset_dirs:
        for path in Path(dataset_dir).glob(f"**/*.{extension}"):
            abs_path = str(path.resolve())
            source_type = path.stem
            subset, track_name = path.relative_to(dataset_dir).parts[:2]
            audio_info = torchaudio.info(path)
            dataset.append(
                (
                    abs_path,
                    track_name,
                    source_type,
                    subset,
                    audio_info.num_frames,
                    audio_info.num_frames / audio_info.sample_rate,
                    audio_info.sample_rate,
                    audio_info.num_channels,
                )
            )
    columns = [
        "path",
        "track_name",
        "source_type",
        "subset",
        "total_frames",
        "total_seconds",
        "sample_rate",
        "num_channels",
    ]
    dataset = pd.DataFrame(dataset, columns=columns)
    if save_path is not None:
        assert Path(save_path).suffix in [
            ".pqt",
            ".parquet",
        ], "save_path should be in .parquet/.pqt format."
        dataset.to_parquet(save_path, index=False)
    return dataset
