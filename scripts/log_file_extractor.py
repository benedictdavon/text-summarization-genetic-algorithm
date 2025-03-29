import re
import pandas as pd

# Read the log file
log_file_path = "8-Code.log"
with open(log_file_path, "r") as file:
    log_data = file.readlines()

# Define regex patterns for extracting relevant data
text_index_pattern = r"Processing Text Index (\d+)"
population_size_pattern = r"Population Size: (\d+)"
max_generations_pattern = r"Max Generations: (\d+)"
best_fitness_pattern = r"Best Fitness: ([0-9.]+)"
avg_fitness_pattern = r"Avg Fitness: ([0-9.]+)"
fitness_std_dev_pattern = r"Fitness Std Dev: ([0-9.]+)"
# Define new regex patterns for the additional required fields
best_overall_fitness_pattern = r"Best Overall Fitness: ([0-9.]+)"
best_chromosome_generation_pattern = r"Best Chromosome Generation: (\d+)"

# Initialize variables
data = []
current_text_index = None
current_population_size = None
current_max_generations = None
current_best_fitness = None
current_avg_fitness = None
current_fitness_std_dev = None
current_best_overall_fitness = None
current_best_chromosome_generation = None

# Parse the log file
for line in log_data:
    # Check for text index
    text_index_match = re.search(text_index_pattern, line)
    if text_index_match:
        if current_text_index is not None:
            # Append the previous set of data
            data.append(
                [
                    current_text_index,
                    current_population_size,
                    current_max_generations,
                    current_best_fitness or 0,
                    current_avg_fitness or 0,
                    current_fitness_std_dev or 0,
                    current_best_overall_fitness or 0,
                    current_best_chromosome_generation or 0,
                ]
            )
        # Reset variables
        current_text_index = int(text_index_match.group(1))
        current_population_size = None
        current_max_generations = None
        current_best_fitness = None
        current_avg_fitness = None
        current_fitness_std_dev = None

    # Check for population size
    population_size_match = re.search(population_size_pattern, line)
    if population_size_match:
        current_population_size = int(population_size_match.group(1))

    # Check for max generations
    max_generations_match = re.search(max_generations_pattern, line)
    if max_generations_match:
        current_max_generations = int(max_generations_match.group(1))

    # Check for best fitness
    best_fitness_match = re.search(best_fitness_pattern, line)
    if best_fitness_match:
        current_best_fitness = float(best_fitness_match.group(1))

    # Check for average fitness
    avg_fitness_match = re.search(avg_fitness_pattern, line)
    if avg_fitness_match:
        current_avg_fitness = float(avg_fitness_match.group(1))

    # Check for fitness standard deviation
    fitness_std_dev_match = re.search(fitness_std_dev_pattern, line)
    if fitness_std_dev_match:
        current_fitness_std_dev = float(fitness_std_dev_match.group(1))

    # Check for best overall fitness
    best_overall_fitness_match = re.search(best_overall_fitness_pattern, line)
    if best_overall_fitness_match:
        current_best_overall_fitness = float(best_overall_fitness_match.group(1))

    # Check for best chromosome generation
    best_chromosome_generation_match = re.search(
        best_chromosome_generation_pattern, line
    )
    if best_chromosome_generation_match:
        current_best_chromosome_generation = int(
            best_chromosome_generation_match.group(1)
        )

# Append the last set of data
if current_text_index is not None:
    data.append(
        [
            current_text_index,
            current_population_size,
            current_max_generations,
            current_best_fitness or 0,
            current_avg_fitness or 0,
            current_fitness_std_dev or 0,
            current_best_overall_fitness or 0,
            current_best_chromosome_generation or 0,
        ]
    )

# Create a DataFrame
columns = [
    "test_index",
    "population_size",
    "max_generations",
    "best_fitness",
    "avg_fitness",
    "fitness_std_dev",
    "best_overall_fitness",
    "best_chromosome_generation",
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_csv_path = "training_results_extracted.csv"
df.to_csv(output_csv_path, index=False)
output_csv_path
