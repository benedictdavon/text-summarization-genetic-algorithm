import numpy as np
import pytest

from src.text_summarizer import TextSummarizerGA
from src.utils import aligned_embedding_slice


def test_alignment_uses_document_slice_not_global_prefix() -> None:
    matrix = np.arange(20, dtype=float).reshape(10, 2)
    manifest = {"documents": [{"dataset_index": 4, "start": 6, "end": 8, "sentence_count": 2}]}
    np.testing.assert_array_equal(aligned_embedding_slice(manifest, 4, matrix, 2), matrix[6:8])


def test_alignment_rejects_wrong_sentence_count() -> None:
    matrix = np.zeros((4, 3))
    manifest = {"documents": [{"dataset_index": 0, "start": 0, "end": 2, "sentence_count": 2}]}
    with pytest.raises(ValueError, match="mismatch"):
        aligned_embedding_slice(manifest, 0, matrix, 3)


def test_summarizer_rejects_misaligned_embeddings() -> None:
    with pytest.raises(ValueError, match="one row per"):
        TextSummarizerGA(np.zeros((1, 3)), ["one", "two"], ["one"])
