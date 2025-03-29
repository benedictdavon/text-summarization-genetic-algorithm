import numpy as np
import random
from typing import List, Tuple
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TextSummarizerGA:
    def __init__(
        self,
        sentence_embeddings: np.ndarray,
        original_sentences: List[str],
        reference_summaries: List[str],
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        preferred_summary_length: int = 5
    ):
        """
        Initialize Genetic Algorithm for Text Summarization with enhanced logging
        """
        self.sentence_embeddings = sentence_embeddings
        self.original_sentences = original_sentences
        self.reference_summaries = reference_summaries
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.preferred_summary_length = preferred_summary_length
        self.num_sentences = len(original_sentences)

        # Initialize ROUGE evaluator
        self.rouge = Rouge()

        # Logging setup
        logger.info(f"Initializing Genetic Algorithm")
        logger.info(f"Number of Sentences: {self.num_sentences}")
        logger.info(f"Population Size: {self.population_size}")
        logger.info(f"Max Generations: {self.max_generations}")
        logger.info(f"Mutation Rate: {self.mutation_rate}")
        logger.info(f"Crossover Rate: {self.crossover_rate}")
        logger.info(f"Preferred Summary Length: {self.preferred_summary_length}")

    def initialize_population(self) -> List[np.ndarray]:
        """
        Create initial population of chromosomes with logging
        """
        logger.info("Generating Initial Population")
        population = [np.random.randint(2, size=self.num_sentences) for _ in range(self.population_size)]
        
        # Log initial population statistics
        selected_sentences_count = [np.sum(chrom) for chrom in population]
        logger.info(f"Initial Population Sentence Selection Stats:")
        logger.info(f"Min Selected Sentences: {min(selected_sentences_count)}")
        logger.info(f"Max Selected Sentences: {max(selected_sentences_count)}")
        logger.info(f"Avg Selected Sentences: {np.mean(selected_sentences_count):.2f}")
        
        return population
    

    def evaluate_coherence(self, selected_indices, sentence_embeddings):
        if len(selected_indices) < 2:
            return 1.0  # Single sentence is coherent by default

        embeddings = [sentence_embeddings[i] for i in selected_indices]
        similarities = cosine_similarity(embeddings)

        # Penalize abrupt changes (adjacent dissimilarity)
        avg_adjacent_similarity = np.mean([
            similarities[i, i + 1]
            for i in range(len(similarities) - 1)
        ])

        # Ensure global coherence (overall average similarity)
        avg_global_similarity = np.mean(similarities)

        # Weighted coherence
        coherence_score = 0.7 * avg_adjacent_similarity + 0.3 * avg_global_similarity
        return coherence_score

    

    def compute_contextual_relevance(self, sentences, sentence_embeddings):
        central_theme_vector = np.mean(sentence_embeddings, axis=0)
        relevance_scores = cosine_similarity(sentence_embeddings, [central_theme_vector]).flatten()
        return relevance_scores

    def fitness_function(self, chromosome: np.ndarray) -> float:
        # If no sentences are selected, return very low fitness
        if np.sum(chromosome) == 0:
            return 0.0

        selected_indices = [i for i, val in enumerate(chromosome) if val == 1]

        # Extract selected sentences
        selected_sentences = [
            self.original_sentences[i]
            for i in range(self.num_sentences)
            if chromosome[i]
        ]
        selected_embeddings = self.sentence_embeddings[chromosome == 1]

        # Content preservation (ROUGE)
        summary = " ".join(selected_sentences)
        rouge_scores = [
            self.rouge.get_scores(summary, ref)[0]["rouge-l"]["f"]
            for ref in self.reference_summaries
        ]
        content_score = np.mean(rouge_scores)

        # Diversity (minimize redundancy)
        if len(selected_embeddings) > 1:
            similarity_matrix = cosine_similarity(selected_embeddings)
            np.fill_diagonal(similarity_matrix, 0)
            diversity_score = 1 - np.mean(similarity_matrix)
        else:
            diversity_score = 1.0

        # Readability (length penalty)
        length_penalty = 1 / (
            1 + abs(len(selected_sentences) - self.preferred_summary_length)
        )

        # Contextual relevance
        relevance_scores = self.compute_contextual_relevance(self.original_sentences, self.sentence_embeddings)
        relevance_score = np.mean([relevance_scores[i] for i in selected_indices])

        # Coherence
        coherence_score = self.evaluate_coherence(selected_indices, self.sentence_embeddings)

        # Composite fitness
        fitness = (
            0.3 * content_score +
            0.2 * coherence_score +
            0.2 * diversity_score +
            0.2 * length_penalty +
            0.1 * relevance_score
        )
        return fitness

    def selection(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        Select chromosomes for next generation using tournament selection

        Args:
            population: Current generation of chromosomes

        Returns:
            Selected chromosomes
        """
        fitness_scores = [self.fitness_function(chrom) for chrom in population]

        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            # Select 3 random chromosomes and choose the best
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover

        Args:
            parent1, parent2: Parent chromosomes

        Returns:
            Two offspring chromosomes
        """
        crossover_point = random.randint(1, self.num_sentences - 1)

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def mutation(
        self, chromosome: np.ndarray, mutation_rate: float = 0.1
    ) -> np.ndarray:
        """
        Perform bit-flip mutation

        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation for each gene

        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        for i in range(self.num_sentences):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated

    def evolve(self) -> np.ndarray:
        """
        Main genetic algorithm evolution process with detailed logging

        Returns:
            Best chromosome found
        """
        # Initialize population
        population = self.initialize_population()

        # Track best chromosomes across generations
        best_chromosomes = []
        best_fitness_scores = []

        initial_mutation_rate = self.mutation_rate
        initial_crossover_rate = self.crossover_rate

        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = [self.fitness_function(chrom) for chrom in population]
            
            self.mutation_rate = initial_mutation_rate * (1 - generation / self.max_generations)
            self.crossover_rate = initial_crossover_rate * (1 - generation / self.max_generations)
            # Select best chromosome for tracking
            best_chromosome = population[np.argmax(fitness_scores)]
            best_fitness = max(fitness_scores)

            # Log detailed generation information
            logger.info(f"Generation {generation + 1} Progress:")
            logger.info(f"  Best Fitness: {best_fitness:.4f}")
            logger.info(f"  Avg Fitness: {np.mean(fitness_scores):.4f}")
            logger.info(f"  Fitness Std Dev: {np.std(fitness_scores):.4f}")

            # Track best chromosomes
            best_chromosomes.append(best_chromosome)
            best_fitness_scores.append(best_fitness)

            # Detailed chromosome logging
            selected_sentences = [
                self.original_sentences[i]
                for i in range(self.num_sentences)
                if best_chromosome[i]
            ]
            logger.info(f"  Selected Sentences ({len(selected_sentences)}):")
            for idx, sentence in enumerate(selected_sentences, 1):
                logger.info(
                    f"    {idx}. {sentence[:100]}..."
                )  # Truncate for readability

            # Selection
            selected_population = self.selection(population)

            # Create next generation
            next_population = []

            # Crossover and mutation
            while len(next_population) < self.population_size:
                # Select parents
                parent1, parent2 = random.sample(selected_population, 2)

                # Crossover
                if random.random() < self.crossover_rate:  # Configurable crossover rate
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self.mutation(child1, self.mutation_rate)  # Configurable mutation rate
                child2 = self.mutation(child2, self.mutation_rate)

                next_population.extend([child1, child2])

            # Truncate to population size
            population = next_population[: self.population_size]

        # Final logging of best overall result
        best_overall_index = np.argmax(best_fitness_scores)
        best_overall_chromosome = best_chromosomes[best_overall_index]
        best_overall_fitness = best_fitness_scores[best_overall_index]

        logger.info("\n--- Genetic Algorithm Completed ---")
        logger.info(f"Best Overall Fitness: {best_overall_fitness:.4f}")
        logger.info(f"Best Chromosome Generation: {best_overall_index + 1}")

        # Return best overall chromosome
        return best_overall_chromosome

    def generate_summary(self, best_chromosome: np.ndarray) -> str:
        """
        Generate summary from best chromosome

        Args:
            best_chromosome: Best chromosome found by genetic algorithm

        Returns:
            Generated summary
        """
        selected_sentences = [
            self.original_sentences[i]
            for i in range(self.num_sentences)
            if best_chromosome[i]
        ]
        return " ".join(selected_sentences)
