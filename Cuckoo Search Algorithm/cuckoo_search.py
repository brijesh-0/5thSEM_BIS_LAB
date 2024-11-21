import numpy as np
from PIL import Image
import math
import os

# Cuckoo Search Algorithm (CSA)
def cucooku_search(objective_function, n_dim, lb, ub, n_pop=50, max_iter=100, pa=0.25, beta=1.5):
    nests = np.random.rand(n_pop, n_dim) * (ub - lb) + lb
    fitness = np.array([objective_function(ind) for ind in nests])
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for t in range(max_iter):
        for i in range(n_pop):
            step_size = levy_flight(beta, n_dim)
            new_nest = nests[i] + step_size * (nests[np.random.randint(0, n_pop)] - nests[i])
            new_nest = np.clip(new_nest, lb, ub)
            new_fitness = objective_function(new_nest)

            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
                if new_fitness < best_fitness:
                    best_nest = new_nest
                    best_fitness = new_fitness

        for i in range(n_pop):
            if np.random.rand() < pa:
                nests[i] = np.random.rand(n_dim) * (ub - lb) + lb
                fitness[i] = objective_function(nests[i])
                if fitness[i] < best_fitness:
                    best_nest = nests[i]
                    best_fitness = fitness[i]

    return best_nest, best_fitness


def levy_flight(beta, n_dim):
    sigma_u = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, n_dim)
    v = np.random.normal(0, 1, n_dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


# Image Processing Objective Function
def image_processing_objective_function(x, image_path):
    try:
        # Verify if the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return float('inf')  # Return infinity to penalize the solution

        # Load the image
        img = Image.open(image_path)

        # Print values of brightness and contrast to debug
        # print(f"Brightness (x[0]): {x[0]}, Contrast (x[1]): {x[1]}")

        # Apply brightness and contrast adjustments
        img = img.point(lambda p: p * x[0] + x[1])

        # Save the processed image for visualization
        img.save("processed_image.png")

        # Load the target image
        target_image_path = image_path  # Ensure this is the correct path to your target image
        target_image = Image.open(target_image_path)
        img = img.resize(target_image.size)

        mse = np.mean((np.array(img) - np.array(target_image))**2)
        return mse

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return float('inf')  # Return infinity to penalize the solution


# Example usage for image processing:

image_path = "image.jpg"  # Ensure this is the correct image path
n_dim = 2
lb = np.array([0.01, -50])  # Significantly lower bound for brightness
ub = np.array([1.0, 50])    # Adjust upper bound for brightness and contrast

best_solution, best_fitness = cucooku_search(lambda x: image_processing_objective_function(x, image_path),
                                              n_dim, lb, ub)

print("Best solution (for image processing):", best_solution)
print("Best fitness (for image processing):", best_fitness)

# Apply the best solution (brightness and contrast) to the image
img = Image.open(image_path)
img = img.point(lambda p: p * best_solution[0] + best_solution[1])

# Save the processed image
# img.save("ResultImage_from_cuckoo_search.jpg")
# print("Processed image saved as ResultImage_from_cuckoo_search.png")
