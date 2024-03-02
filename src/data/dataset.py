from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, Subset, random_split

from data.utils import construct_dataset


class SourceSeparationDataset(Dataset):
    """
    Dataset class for source separation tasks.

    Args:
    - dataset_dirs (str): Comma-separated paths to directories where datasets are located.
    - subset (str): Subset of the dataset to load.
    - window_size (int): Size of the sliding window in seconds.
    - step_size (int): Step size of the sliding window in seconds.
    - sample_rate (int): Sample rate of the audio.
    - dataset_extension (str, optional): Extension of the dataset files. Defaults to 'wav'.
    - dataset_path (str, optional): Path to cache the dataset. Defaults to None.
        If specified, dataset metadata will be cached in parquet format for future retrieval.
        Defaults to None.
    - mixture_name (str, optional): Name of the mixture. Defaults to None.
    - sources (List[str], optional): List of source names. Defaults to ['drums', 'bass', 'other', 'vocals'].
    """

    MIXTURE_NAME: str = "mixture"
    SOURCE_NAMES: List[str] = ["drums", "bass", "other", "vocals"]

    def __init__(
        self,
        dataset_dirs: str,
        subset: str,
        window_size: int,
        step_size: int,
        sample_rate: int,
        dataset_extension: str = "wav",
        dataset_path: Optional[str] = None,
        mixture_name: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ):
        """
        Initializes the SourceSeparationDataset.
        """
        super().__init__()

        self.dataset_dirs: List[str] = dataset_dirs.split(",")
        self.subset = subset
        self.dataset_extension = dataset_extension
        self.dataset_path = dataset_path

        self.window_size = int(window_size * sample_rate)
        self.step_size = int(step_size * sample_rate)
        self.sample_rate = sample_rate

        self.mixture_name = mixture_name or self.MIXTURE_NAME
        self.sources = sources or self.SOURCE_NAMES

        self.df = self.load_df()
        self.segment_ids, self.track_ids = self.get_ids()

    def generate_offsets(
        self,
        total_frames: int,
    ) -> List[int]:
        """
        Generates the offsets based on total frames of track, window size and step size of segments.

        Args:
        - total_frames (int): Total number of frames of audio.

        Returns:
        - List[int]: List of offsets.
        """
        return [
            start
            for start in range(0, total_frames - self.window_size + 1, self.step_size)
        ]

    def load_df(self) -> pd.DataFrame:
        """
        Loads the DataFrame based on the train/test subset and populates data based on window/step sizes.

        Returns:
        - pd.DataFrame: Loaded DataFrame.
        """
        df = construct_dataset(
            self.dataset_dirs,
            extension=self.dataset_extension,
            save_path=self.dataset_path,
        )
        df = df[df["subset"].eq(self.subset)]
        df["offset"] = df["total_frames"].apply(self.generate_offsets)
        df = df.explode("offset")
        df["track_id"] = df["track_name"].factorize()[0]
        df["segment_id"] = df.set_index(["track_id", "offset"]).index.factorize()[0]
        return df

    def get_ids(self) -> Tuple[List[int], List[int]]:
        """
        Gets the segment and track IDs.

        Returns:
        - Tuple[List[int], List[int]]: Tuple containing segment and track IDs.
        """
        segment_ids = self.df["segment_id"].tolist()
        track_ids = self.df["track_id"].unique().tolist()
        return segment_ids, track_ids

    def load_audio(self, segment_info: Dict[str, Any]) -> torch.Tensor:
        """
        Loads the audio based on segment information.

        Args:
        - segment_info (Dict[str, Any]): Segment information.

        Returns:
        - torch.Tensor: Loaded audio tensor.
        """
        audio, sr = torchaudio.load(
            segment_info["path"],
            num_frames=self.window_size,
            frame_offset=segment_info["offset"],
        )
        assert (
            sr == self.sample_rate
        ), f"Sample rate of the audio should be {self.sample_rate}Hz instead of {sr}Hz."
        return audio

    def load_mixture(self, idx: int) -> torch.Tensor:
        """
        Loads the audio mixture based on the provided index.

        Args:
        - idx (int): Index of the mixture.

        Returns:
        - torch.Tensor: Loaded audio mixture tensor.
        """
        segment_info = (
            self.df[
                self.df["segment_id"].eq(idx)
                & self.df["source_type"].eq(self.mixture_name)
            ]
            .iloc[0]
            .to_dict()
        )
        audio = self.load_audio(segment_info)
        return audio

    def load_sources(self, idx: int) -> torch.Tensor:
        """
        Loads the separated sources based on the provided index.

        Args:
        - idx (int): Index of the source.

        Returns:
        - torch.Tensor: Loaded and stacked audio sources tensor.
        """
        audios = []
        for source in self.sources:
            segment_info = (
                self.df[
                    self.df["segment_id"].eq(idx) & self.df["source_type"].eq(source)
                ]
                .iloc[0]
                .to_dict()
            )
            audio = self.load_audio(segment_info)
            audios.append(audio)
        return torch.stack(audios)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an item from the dataset based on the index.

        Args:
        - idx (int): Index of the item.

        Returns:
        - Dict[str, torch.Tensor]: Dictionary containing mixture and sources.
        """
        segment_id = self.segment_ids[idx]

        mixture = self.load_mixture(segment_id)
        sources = self.load_sources(segment_id)

        return {
            "mixture": mixture,
            "sources": sources,
        }

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        return len(self.segment_ids)

    def get_train_val_split(
        self, lengths: List[float], seed: Optional[int] = None
    ) -> Tuple[Subset, Subset]:
        """
        Splits the dataset into training and validation subsets.

        Args:
        - lengths (List[float]): List containing the lengths of the training and validation subsets.
        - seed (Optional[int]): Random seed for reproducibility. Defaults to None.

        Returns:
        - Tuple[Subset, Subset]: Tuple containing the training and validation subsets.
        """
        assert (
            self.subset == "train"
        ), "Only train subset of the dataset can be split into train and val."
        assert len(lengths) == 2, "Dataset can be only split into two subset."
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        train_track_ids, val_track_ids = random_split(
            self.track_ids, lengths=lengths, generator=generator
        )
        train_segment_ids = self.df[
            self.df.track_id.isin(train_track_ids.indices)
        ].segment_id.to_list()
        val_segment_ids = self.df[
            self.df.track_id.isin(val_track_ids.indices)
        ].segment_id.to_list()

        train_subset = Subset(self, indices=train_segment_ids)
        val_subset = Subset(self, indices=val_segment_ids)

        return train_subset, val_subset
