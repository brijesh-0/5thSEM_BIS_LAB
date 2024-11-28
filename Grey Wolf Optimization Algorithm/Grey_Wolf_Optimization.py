import numpy as np

def cost_function(x):
    latency = x[0]  
    throughput = x[1]  
    return latency - 0.5 * throughput  

class GreyWolfOptimizer:
    def __init__(self, cost_func, dim, population_size, max_iter, bounds):
        self.cost_func = cost_func
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.alpha, self.beta, self.delta = None, None, None
        self.alpha_score, self.beta_score, self.delta_score = float('inf'), float('inf'), float('inf')
        self.positions = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.population_size):
                fitness = self.cost_func(self.positions[i])
                
                if fitness < self.alpha_score:
                    self.delta_score, self.delta = self.beta_score, self.beta
                    self.beta_score, self.beta = self.alpha_score, self.alpha
                    self.alpha_score, self.alpha = fitness, self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score, self.delta = self.beta_score, self.beta
                    self.beta_score, self.beta = fitness, self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta = fitness, self.positions[i].copy()
            
            a = 2 - 2 * t / self.max_iter  
            for i in range(self.population_size):
                for d in range(self.dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * self.alpha[d] - self.positions[i][d])
                    X1 = self.alpha[d] - A1 * D_alpha
                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * self.beta[d] - self.positions[i][d])
                    X2 = self.beta[d] - A2 * D_beta
                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * self.delta[d] - self.positions[i][d])
                    X3 = self.delta[d] - A3 * D_delta
                    
                    self.positions[i][d] = (X1 + X2 + X3) / 3
                    
                    self.positions[i][d] = np.clip(self.positions[i][d], self.bounds[0][d], self.bounds[1][d])
        return self.alpha, self.alpha_score

bounds = [(5, 50),  
          (20, 80)]  

gwo = GreyWolfOptimizer(cost_function, dim=2, population_size=30, max_iter=100, bounds=bounds)

best_position, best_cost = gwo.optimize()

print("Optimized Latency and Throughput:")
print("Best Position (Latency, Throughput):", best_position)
print("Best Cost (Objective Value):", best_cost)