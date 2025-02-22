import numpy as np
import random

def evolutionary_algorithm(fitness_func, pop_size=10, x1_range=(-5, 5), x2_range=(-5, 5),
                     parent_selection='roulette', survival_selection='truncation',
                     crossover_type='arithmetic', mutation_type='gaussian',
                     generations=40, runs=10):
    
    def evaluate_fitness(x, y):
        try:
            return eval(fitness_func)
        except:
            return float('inf')
    
    def crossover(p1, p2, cross_type):
        if cross_type == 'arithmetic':
            alpha = random.random()
            return (alpha * p1[0] + (1-alpha) * p2[0],
                   alpha * p1[1] + (1-alpha) * p2[1])
        elif cross_type == 'swap':
            if random.random() < 0.5:
                return (p2[0], p1[1])
            else:
                return (p1[0], p2[1])
        elif cross_type == 'blend':
            alpha = 0.3
            c1_x = p1[0] + (random.random() * (1 + 2*alpha) - alpha) * (p2[0] - p1[0])
            c1_y = p1[1] + (random.random() * (1 + 2*alpha) - alpha) * (p2[1] - p1[1])
            return (c1_x, c1_y)
    
    def mutation(individual, mut_type):
        if mut_type == 'gaussian':
            return (individual[0] + random.gauss(0, 0.1),
                   individual[1] + random.gauss(0, 0.1))
        elif mut_type == 'uniform':
            delta = 0.1
            return (individual[0] + random.uniform(-delta, delta),
                   individual[1] + random.uniform(-delta, delta))
        elif mut_type == 'creep':
            step = 0.1 * (1 - gen/generations)
            return (individual[0] + random.uniform(-step, step),
                   individual[1] + random.uniform(-step, step))

    
    def initialize_population():
        return [(random.uniform(*x1_range), random.uniform(*x2_range)) 
                for _ in range(pop_size)]
    
    def selection_methods(pop, fits, method):
        if method == 'roulette':
            min_fit = min(fits)
            shifted_fits = [f - min_fit + 1.0 if min_fit <= 0 else f for f in fits]
            total_fit = sum(shifted_fits)
            if total_fit == 0:  # Handle case where all fitness values are equal
                return random.choice(pop)
                
            probs = [f/total_fit for f in shifted_fits]
            probs = [p/sum(probs) for p in probs]
            
            try:
                selected_idx = np.random.choice(len(pop), p=probs)
                return pop[selected_idx]
            except ValueError:  # Fallback if probabilities are invalid
                return random.choice(pop)
        elif method == 'tournament':
            tournament_size = 2
            contestants = random.sample(range(len(pop)), tournament_size)
            winner = max(contestants, key=lambda i: fits[i])
            return pop[winner]
        elif method == 'rank':
            sorted_indices = np.argsort(fits)[::-1]
            n = len(pop)
            total_rank = sum(range(1, n + 1))
            probs = [(n - i) / total_rank for i in range(n)]
            
            try:
                selected_idx = np.random.choice(n, p=probs)
                return pop[sorted_indices[selected_idx]]
            except ValueError:
                return random.choice(pop)
    
    run_data = []
    averages = []
    
    for run in range(runs):
        generation_bests = []
        population = initialize_population()
        
        for run in range(runs):
            generation_bests = []
            population = initialize_population()
            
            for gen in range(generations):
                # Evaluate fitness
                fitnesses = [evaluate_fitness(x, y) for x, y in population]
                best_fit = min(fitnesses)  # Always minimizing
                generation_bests.append(best_fit)
                
                new_pop = []
                while len(new_pop) < pop_size:
                    # Parent selection
                    p1 = selection_methods(population, fitnesses, parent_selection)
                    p2 = selection_methods(population, fitnesses, parent_selection)
                    
                    # Crossover
                    c1 = crossover(p1, p2, crossover_type)
                    
                    # Mutation
                    if random.random() < 0.1:
                        c1 = mutation(c1, mutation_type)
                    
                    new_pop.append(c1)
                
                population = new_pop
            
            run_data.append(generation_bests)
    
    # Calculate average across all runs
    average = np.mean(run_data, axis=0)
    result = {
        'run_data': run_data,
        'average': average.tolist()
    }
    
    # Clear averages
    averages.clear()
    
    return result