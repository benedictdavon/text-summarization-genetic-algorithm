from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def print_file_size(filename: str | os.PathLike[str]) -> None:
    print(f"File Size: {os.path.getsize(filename) / 1e6:.2f} MB")


def load_custom_dataset(end_index: int | None = None):
    from datasets import load_dataset

    ds = load_dataset("antash420/text-summarization-alpaca-format")
    inputs = ds["train"]["input"][:end_index]
    references = ds["train"]["output"][:end_index]
    return inputs, references


def load_embeddings_in_chunks(filename: str | os.PathLike[str], chunk_size: int = 100000, end_index: int | None = None) -> np.ndarray:
    embeddings = np.load(filename, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {embeddings.shape}")
    limit = min(chunk_size, len(embeddings))
    if end_index is not None:
        limit = min(limit, end_index)
    subset = embeddings[:limit]
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Loading first {len(subset)} embeddings")
    print(f"Embedding dimension: {subset.shape[1]}")
    return subset


def load_embedding_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest.get("documents"), list):
        raise ValueError("Manifest must contain a documents list")
    return manifest


def aligned_embedding_slice(manifest: dict[str, Any], dataset_index: int, embedding_matrix: np.ndarray, sentence_count: int) -> np.ndarray:
    records = [record for record in manifest["documents"] if record.get("dataset_index") == dataset_index]
    if len(records) != 1:
        raise ValueError(f"Expected one embedding record for dataset index {dataset_index}, found {len(records)}")
    record = records[0]
    start, end = int(record["start"]), int(record["end"])
    if end - start != sentence_count or record.get("sentence_count") != sentence_count:
        raise ValueError(f"Embedding/sentence count mismatch for index {dataset_index}: slice {start}:{end}, sentences {sentence_count}")
    if start < 0 or end > len(embedding_matrix):
        raise ValueError(f"Embedding slice {start}:{end} is outside matrix with {len(embedding_matrix)} rows")
    return np.asarray(embedding_matrix[start:end])
