"""
Dataset utilities for CHASE protein fitness benchmarks.

Supports the four benchmarks from Kirjner et al. (2023):
  AAV Medium, AAV Hard, GFP Medium, GFP Hard

The benchmarks are restricted subsets of the full AAV and GFP datasets,
simulating low-data / distribution-shift scenarios.

Expected CSV format: columns ['sequence', 'fitness'] (or similar names)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import esm


# ESM2 tokenizer constants
ESM2_ALPHABET = esm.data.Alphabet.from_architecture("ESM-1b")


def load_benchmark(
    data_dir: str,
    dataset_name: str,
    fitness_col: str = "fitness",
    seq_col: str = "sequence",
):
    """
    Load a benchmark CSV file.

    Args:
        data_dir:     directory containing CSV files
        dataset_name: e.g. 'aav_medium', 'gfp_hard'
        fitness_col:  column name for fitness values
        seq_col:      column name for sequences

    Returns:
        sequences: list of strings
        fitness:   numpy array of float32
    """
    path = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download from https://github.com/kirjner/GGS and place in {data_dir}/"
        )

    df = pd.read_csv(path)

    # Flexible column name handling
    seq_candidates = [seq_col, "sequence", "seq", "Sequence", "aa_seq"]
    fit_candidates = [fitness_col, "fitness", "score", "Fitness", "y"]

    for c in seq_candidates:
        if c in df.columns:
            sequences = df[c].tolist()
            break
    else:
        raise ValueError(f"Could not find sequence column. Available: {df.columns.tolist()}")

    for c in fit_candidates:
        if c in df.columns:
            fitness = df[c].values.astype(np.float32)
            break
    else:
        raise ValueError(f"Could not find fitness column. Available: {df.columns.tolist()}")

    return sequences, fitness


def normalize_fitness(fitness: np.ndarray, f_min: float = None, f_max: float = None):
    """Normalize fitness to [0, 1]."""
    if f_min is None:
        f_min = fitness.min()
    if f_max is None:
        f_max = fitness.max()
    return (fitness - f_min) / (f_max - f_min + 1e-8), f_min, f_max


class ProteinFitnessDataset(Dataset):
    """
    Dataset of (tokenized_sequence, fitness) pairs.

    Tokenization uses ESM2 batch converter.
    """

    def __init__(
        self,
        sequences: list,
        fitness: np.ndarray,
        batch_converter,
        max_seq_len: int = 512,
    ):
        assert len(sequences) == len(fitness)
        self.sequences = sequences
        self.fitness = torch.tensor(fitness, dtype=torch.float32)
        self.batch_converter = batch_converter
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        f = self.fitness[idx]
        return seq, f

    def collate_fn(self, batch):
        seqs, fitnesses = zip(*batch)
        labels = [("protein", s) for s in seqs]
        _, _, tokens = self.batch_converter(labels)
        fitnesses = torch.stack(list(fitnesses))
        return tokens, fitnesses


def get_dataloaders(
    data_dir: str,
    dataset_name: str,
    batch_size: int = 128,
    train_frac: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Build train/val DataLoaders for a benchmark dataset.

    Returns:
        train_loader, val_loader, (f_min, f_max)
    """
    sequences, fitness = load_benchmark(data_dir, dataset_name)
    fitness, f_min, f_max = normalize_fitness(fitness)

    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    dataset = ProteinFitnessDataset(sequences, fitness, batch_converter)

    n_train = int(len(dataset) * train_frac)
    n_val = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, (f_min, f_max)


class BootstrappedDataset(Dataset):
    """
    Combines the original dataset A with synthetic pairs A_syn.
    Used for the bootstrapping stage (Section 3.4).
    """

    def __init__(self, base_dataset: ProteinFitnessDataset, syn_sequences: list, syn_fitness: np.ndarray):
        self.base = base_dataset
        self.syn_sequences = syn_sequences
        self.syn_fitness = torch.tensor(syn_fitness, dtype=torch.float32)
        self.batch_converter = base_dataset.batch_converter

    def __len__(self):
        return len(self.base) + len(self.syn_sequences)

    def __getitem__(self, idx):
        if idx < len(self.base):
            return self.base[idx]
        else:
            i = idx - len(self.base)
            return self.syn_sequences[i], self.syn_fitness[i]

    def collate_fn(self, batch):
        return self.base.collate_fn(batch)
