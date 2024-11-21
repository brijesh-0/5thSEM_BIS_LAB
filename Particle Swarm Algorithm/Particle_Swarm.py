import numpy as np
import random

# Define the grid size and obstacle map (0 = free space, 1 = obstacle)
grid_size = (40, 40)
obstacle_map = np.zeros(grid_size)

# Option 1: Add Random Obstacles
num_obstacles = 900  # Number of random obstacles to place
for _ in range(num_obstacles):
    x = random.randint(0, grid_size[0] - 1)
    y = random.randint(0, grid_size[1] - 1)
    obstacle_map[x, y] = 1

# Option 2: Add Multiple Obstacle Regions (e.g., 2 regions)
obstacle_map[5:10, 5:10] = 1  # Obstacle region 1 (5x5 grid)
obstacle_map[12:17, 12:17] = 1  # Obstacle region 2 (5x5 grid)

# Parameters for PSO
num_particles = 20
num_iterations = 30
inertia_weight = 0.9  # Increased inertia to allow more exploration
cognitive_coeff = 1.5
social_coeff = 1.5

# Starting and target positions
start_pos = np.array([0, 0])
target_pos = np.array([grid_size[0] - 1, grid_size[1] - 1])

# Define the fitness function (minimize distance to target while avoiding obstacles)
def fitness(position):
    x, y = int(position[0]), int(position[1])
    
    # Penalty for positions in obstacle cells
    if obstacle_map[x, y] == 1:
        return float('inf')  # High penalty for positions in obstacle cells
    
    # Euclidean distance to target
    return np.linalg.norm(position - target_pos)

# Initialize particles with random positions and velocities
particles = [np.array([random.uniform(0, grid_size[0]-1), random.uniform(0, grid_size[1]-1)]) for _ in range(num_particles)]
velocities = [np.random.uniform(-1, 1, 2) for _ in range(num_particles)]
personal_best_positions = particles[:]
personal_best_scores = [fitness(p) for p in particles]

# Global best position and score
global_best_position = min(personal_best_positions, key=fitness)
global_best_score = fitness(global_best_position)

# Main PSO Loop
for iteration in range(num_iterations):
    for i, particle in enumerate(particles):
        # Update the velocity
        inertia = inertia_weight * velocities[i]
        cognitive = cognitive_coeff * random.random() * (personal_best_positions[i] - particle)
        social = social_coeff * random.random() * (global_best_position - particle)
        velocities[i] = inertia + cognitive + social

        # Update the particle's position
        particles[i] = particles[i] + velocities[i]
        
        # Keep particles within grid bounds
        particles[i] = np.clip(particles[i], [0, 0], [grid_size[0] - 1, grid_size[1] - 1])

        # Evaluate fitness
        current_fitness = fitness(particles[i])
        
        # Update personal best
        if current_fitness < personal_best_scores[i]:
            personal_best_positions[i] = particles[i]
            personal_best_scores[i] = current_fitness
        
        # Update global best
        if current_fitness < global_best_score:
            global_best_position = particles[i]
            global_best_score = current_fitness

    print(f"Iteration {iteration+1}: Best Score = {global_best_score:.4f}, Best Position = {global_best_position}")

# Output the best solution found
print("\nBest Path Solution Found:")
print(f"Position = {global_best_position}, Distance to Target = {global_best_score}")
