import random
import numpy as np

# Define the problem function to maximize
def objective_function(x):
    return x * np.sin(x)

# GA Parameters
population_size = 15    # Number of individuals in the population
generations = 30       # Number of generations
mutation_rate = 0.01    # Probability of mutation
crossover_rate = 0.8    # Probability of crossover
search_range = (-10, 10) # Range for x values

# Initialize population with random values within the search range
def initialize_population():
    return [random.uniform(search_range[0], search_range[1]) for _ in range(population_size)]

# Evaluate fitness of each individual (maximize objective function)
def evaluate_fitness(population):
    return [objective_function(individual) for individual in population]

# Select parents based on fitness (roulette wheel selection)
def select_parents(population, fitness):
    # Ensure fitness values are non-negative by adding an offset if necessary
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]  # Shift fitness values to make them non-negative

    # Compute selection probabilities
    total_fitness = sum(fitness)
    selection_probs = [f / total_fitness for f in fitness]
    
    # Select population_size//2 pairs of parents using roulette wheel
    parents = np.random.choice(population, size=(population_size // 2, 2), p=selection_probs)
    return parents


# Perform crossover between two parents
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        alpha = random.random()
        return alpha * parent1 + (1 - alpha) * parent2
    return parent1 if random.random() < 0.5 else parent2

# Apply mutation to an individual
def mutate(individual):
    if random.random() < mutation_rate:
        mutation_value = random.uniform(-1, 1)
        individual = individual + mutation_value
        # Ensure individual remains within search range
        individual = max(min(individual, search_range[1]), search_range[0])
    return individual

# Main Genetic Algorithm loop
def genetic_algorithm():
    # Initialize population
    population = initialize_population()

    for generation in range(generations):
        # Evaluate fitness of the current population
        fitness = evaluate_fitness(population)
        
        # Selection process to form a mating pool
        parents = select_parents(population, fitness)
        
        # Create the next generation
        next_generation = []
        for i in range(parents.shape[0]):  # Number of parent pairs
            parent1, parent2 = parents[i]

            # Apply crossover
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
            
            # Apply mutation
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            next_generation.extend([offspring1, offspring2])

        # Update population with new generation
        population = next_generation

        # Print best solution in the current generation
        best_solution = max(population, key=objective_function)
        best_fitness = objective_function(best_solution)
        print(f"Generation {generation + 1}: Best Solution = {best_solution:.4f}, Best Fitness = {best_fitness:.4f}")
    
    # Output the best solution found
    final_solution = max(population, key=objective_function)
    print("\nBest Solution Found:")
    print(f"x = {final_solution:.4f}, f(x) = {objective_function(final_solution):.4f}")

# Run the Genetic Algorithm
genetic_algorithm()
