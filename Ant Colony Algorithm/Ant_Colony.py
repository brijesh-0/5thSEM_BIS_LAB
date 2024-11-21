import numpy as np
import random

# Define the distance between cities
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Define the total route length
def route_length(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]  # Return to start
    return total_distance

# ACO algorithm for TSP
class ACO_TSP:
    def __init__(self, cities, num_ants, alpha=1.0, beta=2.0, rho=0.5, iterations=100):
        self.cities = cities
        self.num_ants = num_ants
        self.alpha = alpha        # Importance of pheromone
        self.beta = beta          # Importance of heuristic (1/distance)
        self.rho = rho            # Pheromone evaporation rate
        self.iterations = iterations
        self.num_cities = len(cities)
        
        # Initialize distance matrix
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distance_matrix[i][j] = calculate_distance(self.cities[i], self.cities[j])
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))

    def run(self):
        best_route = None
        best_distance = float('inf')
        
        for _ in range(self.iterations):
            all_routes = []
            for _ in range(self.num_ants):
                route = self.construct_solution()
                distance = route_length(route, self.distance_matrix)
                all_routes.append((route, distance))
                
                # Update best route if found
                if distance < best_distance:
                    best_route = route
                    best_distance = distance
            
            # Update pheromones
            self.update_pheromones(all_routes)
        
        return best_route, best_distance

    def construct_solution(self):
        route = [random.randint(0, self.num_cities - 1)]
        while len(route) < self.num_cities:
            current_city = route[-1]
            probabilities = self.calculate_transition_probabilities(current_city, route)
            next_city = self.choose_next_city(probabilities)
            route.append(next_city)
        return route

    def calculate_transition_probabilities(self, current_city, route):
        probabilities = []
        for next_city in range(self.num_cities):
            if next_city not in route:
                pheromone = self.pheromone_matrix[current_city][next_city] ** self.alpha
                heuristic = (1 / self.distance_matrix[current_city][next_city]) ** self.beta
                probabilities.append((next_city, pheromone * heuristic))
        total = sum(prob for _, prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]
        return probabilities

    def choose_next_city(self, probabilities):
        rand = random.random()
        cumulative = 0
        for city, prob in probabilities:
            cumulative += prob
            if cumulative >= rand:
                return city
        return probabilities[-1][0]  # Fallback

    def update_pheromones(self, all_routes):
        # Evaporate pheromone
        self.pheromone_matrix *= (1 - self.rho)
        
        # Deposit pheromone for each route
        for route, distance in all_routes:
            pheromone_to_add = 1 / distance
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i]][route[i + 1]] += pheromone_to_add
            # Add pheromone for return to start
            self.pheromone_matrix[route[-1]][route[0]] += pheromone_to_add

# Define the cities as (x, y) coordinates
cities = [(0, 0), (1, 5), (2, 3), (5, 2), (4, 0)]
num_ants = 30
alpha = 1.5
beta = 3.0
rho = 0.7
iterations = 100

# Run the ACO algorithm
aco_tsp = ACO_TSP(cities, num_ants, alpha, beta, rho, iterations)
best_route, best_distance = aco_tsp.run()
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance}")