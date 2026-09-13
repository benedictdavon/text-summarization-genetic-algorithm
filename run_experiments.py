import pandas as pd
import numpy as np
import logging
from pathlib import Path

from nltk.tokenize import sent_tokenize

from src.text_summarizer import TextSummarizerGA
from src.utils import aligned_embedding_slice, load_custom_dataset, load_embedding_manifest, load_embeddings_in_chunks

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def experiment(population_sizes, max_generations_list, text_indices, inputs, references, sentence_embeddings, manifest, output_path):
    # Iterate through parameters and text indices
    results = []
    for population_size in population_sizes:
        for max_generations in max_generations_list:
            for text_index in text_indices:
                logger.info(f"Processing Text Index {text_index} | Population Size {population_size} | Max Generations {max_generations}")

                input_text = inputs[text_index]
                input_sentences = sent_tokenize(input_text)
                input_references = [references[text_index]]

                # Get embeddings for this specific input
                input_sentence_embeddings = aligned_embedding_slice(manifest, text_index, sentence_embeddings, len(input_sentences))

                # Initialize and run genetic algorithm
                ga_summarizer = TextSummarizerGA(
                    sentence_embeddings=input_sentence_embeddings,
                    original_sentences=input_sentences,
                    reference_summaries=input_references,
                    population_size=population_size,
                    max_generations=max_generations
                )

                # Evolve and generate summary
                best_chromosome = ga_summarizer.evolve()
                generated_summary = ga_summarizer.generate_summary(best_chromosome)

                # Log results
                logger.info(f"Generated Summary: {generated_summary}")
                logger.info(f"Reference Summary: {input_references[0]}")

                # Store results
                results.append({
                    "text_index": text_index,
                    "population_size": population_size,
                    "max_generations": max_generations,
                    "original_text": input_text,
                    "generated_summary": generated_summary,
                    "reference_summary": input_references[0],
                })
                df = pd.DataFrame(results)

                # Save to CSV
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the reference-guided summarization experiment.")
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--output", default="result/experiment_results.csv", type=Path)
    args = parser.parse_args()
    # # Define parameters
    # text_index = 1
    # population_size = 500
    # max_generations = 500

    # Fail closed before downloading the external dataset if the required
    # aligned artifact manifest is absent.
    manifest = load_embedding_manifest(args.embedding_manifest)
    end_index = 1000
    inputs, references = load_custom_dataset(end_index=end_index)
    
    # logger.info(f"Dataset is loaded with {len(inputs)} samples")
    
    # # Tokenize sentences (as you've already done)
    # tokenized_inputs = [sent_tokenize(text) for text in inputs]
    
    # print_file_size('sentence_embeddings.npy')
    # # Load pre-computed sentence embeddings
    sentence_embeddings = load_embeddings_in_chunks(manifest["embedding_file"], end_index=end_index)
    
    # # Select a specific input to summarize (first example)
    
    # input_text = inputs[text_index]
    # input_sentences = sent_tokenize(input_text)
    # input_references = [references[text_index]]
    
    # # Get embeddings for this specific input
    # input_sentence_embeddings = sentence_embeddings[:len(input_sentences)]
    
    # # Initialize and run genetic algorithm
    # ga_summarizer = TextSummarizerGA(
    #     sentence_embeddings=input_sentence_embeddings,
    #     original_sentences=input_sentences,
    #     reference_summaries=input_references,
    #     population_size=population_size,
    #     max_generations=max_generations
    # )
    
    # # Evolve and generate summary
    # best_chromosome = ga_summarizer.evolve()
    # generated_summary = ga_summarizer.generate_summary(best_chromosome)
    
    # print("Original Text:", input_text)
    # print("\nGenerated Summary:", generated_summary)
    # print("\nReference Summary:", input_references[0])

    # Define parameters to iterate
    population_sizes = [50, 100, 250]  # Example population sizes
    max_generations_list = [10, 50, 100]  # Example max generations
    text_indices = range(5)  # First ten texts (index 0-9)

    experiment(population_sizes, max_generations_list, text_indices, inputs, references, sentence_embeddings, manifest, args.output)

if __name__ == "__main__":
    main()
