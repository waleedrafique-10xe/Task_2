import os
import random

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset

from ....utils.dataset_utils import DatasetProvider





class BlockTrainDataset(Dataset):
    """
    Dataset used to store intermediate hidden states for block-wise quantization.

    This dataset is designed to collect and serve the input/output activations
    at specific layers (typically block-level inputs) during quantization-aware training (QAT).

    It supports two modes:
    1. In-memory mode (default): stores tensors in RAM as a pre-allocated torch.Tensor
    2. Offload-to-disk mode: saves and loads each sample tensor as a separate .pt file on disk

    Args:
        size (int): Total number of samples (should be divisible by batch_size).
        seqlen (int): Sequence length (tokens per sample).
        hidden_size (int): Feature size (e.g., model's hidden dimension).
        batch_size (int): Batch size for training.
        dtype (torch.dtype): Data type (e.g., torch.float16).
        cache_path (str): Directory to store cached .pt files (used if off_load_to_disk=True).
        off_load_to_disk (bool): Whether to use disk-based storage (reduces RAM usage).
    """

    def __init__(
        self,
        size: int,
        seqlen: int,
        hidden_size: int,
        batch_size: int,
        dtype: torch.dtype,
        cache_path: str = "./cache/block_training_data",
        off_load_to_disk: bool = False,
    ):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        # Ensure dataset can be cleanly divided into batches
        assert size % batch_size == 0

        self._capacity_batches = size // self.batch_size
        self._active_batches = self._capacity_batches

        if self.off_load_to_disk:
            # Disk-based: Create directory and initialize tensors
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            # Memory-based: Pre-allocate tensor storage in memory
            self.data = torch.zeros(
                (
                    self._capacity_batches,
                    self.batch_size,
                    self.seqlen,
                    self.hidden_size,
                ),
                dtype=self.dtype,
            )

    def set_active_batches(self, num_batches: int):
        """
        Limit how many batches are considered valid for iteration.

        Args:
            num_batches (int): Number of batches (<= capacity) to keep active.
        """
        if num_batches < 0 or num_batches > self._capacity_batches:
            raise ValueError("num_batches is out of range for BlockTrainDataset")
        self._active_batches = num_batches

    def _initialize_data_on_disk(self):
        """
        Initializes empty tensors on disk for each batch index.
        Creates one `.pt` file per batch.
        """
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros(
                (self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype
            )
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        """
        Returns the file path for storing the `idx`-th batch on disk.

        Args:
            idx (int): Batch index.

        Returns:
            str: Full path to corresponding .pt file.
        """
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        """
        Total number of batches in this dataset.
        """
        return self._active_batches

    def __getitem__(self, idx):
        """
        Loads a batch tensor by index.

        Args:
            idx (int): Batch index (0 <= idx < len(self))

        Returns:
            Tensor: Shape (batch_size, seqlen, hidden_size)
        """
        if idx >= self._active_batches:
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            # Load from disk
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            # Load from memory
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        """
        Updates the stored tensor at a given index with new data.

        Args:
            idx (int): Batch index to update.
            new_data (Tensor): New tensor to store (shape must match).
        """
        if idx >= self._capacity_batches:
            raise IndexError("Index out of range")
        if new_data.shape[-2] != self.seqlen:
            if new_data.shape[-2] > self.seqlen:
                new_data = new_data[..., : self.seqlen, :]
            else:
                pad_size = list(new_data.shape)
                pad_size[-2] = self.seqlen - new_data.shape[-2]
                pad_tensor = torch.zeros(
                    pad_size,
                    dtype=new_data.dtype,
                    device=new_data.device,
                )
                new_data = torch.cat([new_data, pad_tensor], dim=-2)

        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype).cpu(), filepath)
        else:
            self.data[idx] = new_data.to(self.dtype).cpu()
