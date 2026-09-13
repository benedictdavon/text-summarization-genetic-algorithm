import numpy as np

from src.text_summarizer import TextSummarizerGA


def test_single_sentence_crossover_is_safe() -> None:
    summarizer = TextSummarizerGA(np.ones((1, 2)), ["A sentence."], ["A sentence."], population_size=3)
    first, second = summarizer.crossover(np.array([1]), np.array([0]))
    np.testing.assert_array_equal(first, [1])
    np.testing.assert_array_equal(second, [0])
