# Extractive Text Summarization with a Genetic Algorithm

Historical Evolutionary Computation coursework project that selects sentences with a binary genetic algorithm. The implementation combines ROUGE-L F fitness, embedding-based relevance/coherence, redundancy control, and a target summary-length penalty.

## What the implementation actually does

- **Selection:** tournament selection with tournament size 3.
- **Crossover:** single-point crossover.
- **Mutation:** bit-flip mutation with a linearly decayed rate.
- **Fitness:** reference-guided ROUGE-L F, cosine-based diversity and coherence, relevance, and length terms.
- **Data:** the runner targets `antash420/text-summarization-alpaca-format`; the checked-in historical CSV is an output artifact, not a fresh benchmark.

The reference summary is used by the fitness function during optimization. This makes the experiment reference-guided rather than an inference-time summarizer. The repository does not claim a deployment, a general-purpose summarization benchmark, or superiority over TextRank/Lead-3 without a reproducible baseline run.

## Methodology audit

See [`docs/methodology-audit.md`](docs/methodology-audit.md) for the verified mismatch findings and the corrections made. In particular, sentence embeddings must be aligned to each document's sentence slice; a single global prefix is invalid for a multi-document dataset.

## Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runner requires a separately generated, aligned embedding artifact and a manifest mapping each dataset row to its embedding slice. An example schema is provided at [`data/embedding_manifest.example.json`](data/embedding_manifest.example.json). The historical tracked `.npy` file was incomplete and has been removed; no new metric is reported by this cleanup.

```bash
python run_experiments.py --embedding-manifest data/embedding_manifest.json --output result/experiment_results.csv
```

The dataset and embedding model are external inputs. Do not commit credentials, generated caches, or unverified benchmark outputs.

## Scope and attribution

This is an individual course project at NYCU. It is an implementation and investigation of a reference-guided genetic-algorithm experiment, not a novel summarization architecture. The included report and historical results are retained as context but should be read with the audit qualification above.

## Tests

```bash
pytest -q
python -m compileall src run_experiments.py
```
