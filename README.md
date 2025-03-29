# 🧬 Text Summarization Using Genetic Algorithms

This project implements an **extractive text summarization system** that uses a **genetic algorithm (GA)** to optimize sentence selection. The approach combines traditional NLP techniques (e.g., ROUGE) with modern sentence embeddings (BERT), offering a flexible and bio-inspired way to generate concise, coherent summaries.

> 📘 Built as a final project for the *Evolutionary Computation* course @ NYCU

---

## 📌 Features

- Sentence selection modeled as a binary genetic representation
- ROUGE-based fitness scoring with penalties for redundancy and length
- Sentence embeddings from `all-MiniLM-L6-v2` (via `sentence-transformers`)
- Customizable GA parameters: population size, mutation rate, crossover, etc.
- Experimental logging and automatic CSV output
- Outperforms common baselines like **TextRank** and **Lead-3** on sample texts

---

## 🧠 How It Works

Each chromosome is a binary vector representing a subset of sentences in a document. A genetic algorithm is used to evolve the best summary by applying:

- **Selection**: Roulette Wheel Selection
- **Crossover**: Two-point crossover
- **Mutation**: Bit-flipping
- **Fitness Function**:
  - ROUGE-1 F1 (content coverage)
  - Sentence diversity via cosine similarity
  - Coherence & length penalties

---

## 📂 Project Structure

```
text-summarization-ga/
├── src/                  # Core modules
│   ├── text_summarizer.py
│   ├── utils.py
│   └── __init__.py
├── scripts/
│   └── log_file_extractor.py
├── notebooks/
│   ├── data_analysis.ipynb
│   └── summarizer_demo.ipynb
├── data/                 # Sentence embeddings, etc.
│   └── sentence_embeddings.npy
├── report/               # Full report
├── result/               # experiment_results.csv
├── img/                  # Visuals for reports or README
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📈 Dataset

We used the [AG News Corpus](https://huggingface.co/datasets/ag_news) for testing. Each input text is paired with a human-written reference summary.

---

## 🧪 How to Run

Make sure your virtual environment is active and the dependencies are installed:

```bash
pip install -r requirements.txt
```

Run batch experiments:

```bash
python scripts/run_experiments.py
```

---


## 📊 Sample Results

| Topic                    | Population / Gen  | Fitness Score  | Generated Summary Snippet |
|--------------------------|-------------------|----------------|----------------------------|
| Harry Potter             | 50 / 250          | 0.6442         | Harry Potter star Daniel Radcliffe gains access to a £20M fortune as he turns 18... |
| Mentally Ill Inmates     | 250 / 500         | 0.5795         | Soledad O'Brien reports on Miami's "forgotten floor," where mentally ill inmates are held... |
| Minneapolis Bridge Crash | 250 / 250         | 0.5952         | Survivors of the Minneapolis bridge collapse describe free falls and burning vehicles... |

> Text length and complexity directly influenced the GA’s ability to generate high-scoring summaries.  
> 📝 Reference summaries and additional metrics available in [result/experiment_results.csv](result/experiment_results.csv)


---

## 🚧 Limitations & Future Work

- **ROUGE** alone doesn't capture semantic quality (consider BERTScore, BLEU)
- Scaling to longer documents or real-time summarization needs optimization

Plans for:
- Multi-objective GA
- Human-in-the-loop evaluation
- Word-level summarization (instead of sentence-level)

---

## 📄 Full Report

For a comprehensive explanation of the methodology, experimental setup, results, and analysis, please refer to the full project report:

[**📘 Optimizing Text Summarization Using Genetic Algorithms for Sentence Selection**](report/Optimizing%20Text%20Summarization%20Using%20Genetic%20Algorithms%20for%20Sentence%20Selection.pdf)

---

## 👤 Author

**Benedict Davon Martono**  
📍 National Yang Ming Chiao Tung University (NYCU)  
🔗 [GitHub](https://github.com/benedictdavon) • [LinkedIn](https://www.linkedin.com/in/davon-martono-a0b5571b8/)

---

## 📄 License

This project is intended for educational and research use only.
You are welcome to use or modify the code with appropriate attribution.
