import numpy as np
import random
from evo_algo import get_results_csv_string

def differential_evolution(fitness_func, pop_size=30, bounds=None, strategy='rand1bin', 
                          mutation_factor=0.8, crossover_prob=0.7, generations=100, runs=1):
    """
    Implements differential evolution algorithm.
    
    Parameters:
    -----------
    fitness_func : str
        String representation of fitness function
    pop_size : int
        Population size
    bounds : list of tuples
        Bounds for each dimension [(x_min, x_max), (y_min, y_max), ...]
    strategy : str
        Mutation strategy ('rand1bin', 'best1bin', 'rand2bin', 'best2bin')
    mutation_factor : float
        Mutation factor F
    crossover_prob : float
        Crossover probability CR
    generations : int
        Number of generations
    runs : int
        Number of independent runs
        
    Returns:
    --------
    dict
        Results dictionary with run data, average, best solution, etc.
    """
    if bounds is None:
        bounds = [(-5, 5), (-5, 5)]
    
    dimensions = len(bounds)
    
    # Convert string fitness function to callable
    def eval_fitness(individual):
        x, y = individual[0], individual[1]
        return -eval(fitness_func)  # Negative for minimization
    
    all_runs_data = []
    all_best_solutions = []
    
    for run in range(runs):
        # Initialize population
        population = []
        for i in range(pop_size):
            individual = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensions)]
            population.append(individual)
        
        # Evaluate initial population
        fitness_values = [eval_fitness(ind) for ind in population]
        
        # Track best solution
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        # For tracking progress
        run_data = [best_fitness]
        
        # Evolution loop
        for generation in range(generations):
            for i in range(pop_size):
                # Select base vector
                if strategy.startswith('best'):
                    base_idx = np.argmin(fitness_values)
                else:  # rand
                    base_idx = random.randint(0, pop_size-1)
                
                # Select random indices different from i and base
                available_indices = list(range(pop_size))
                available_indices.remove(i)
                if base_idx in available_indices:
                    available_indices.remove(base_idx)
                
                # Select indices for mutation
                if strategy.endswith('1bin'):
                    r1, r2 = random.sample(available_indices, 2)
                    diff_vectors = [(r1, r2)]
                else:  # '2bin'
                    r1, r2, r3, r4 = random.sample(available_indices, 4)
                    diff_vectors = [(r1, r2), (r3, r4)]
                
                # Create trial vector with mutation
                trial = population[base_idx].copy()
                for r1, r2 in diff_vectors:
                    for j in range(dimensions):
                        trial[j] += mutation_factor * (population[r1][j] - population[r2][j])
                
                # Ensure trial is within bounds
                for j in range(dimensions):
                    if trial[j] < bounds[j][0]:
                        trial[j] = bounds[j][0]
                    elif trial[j] > bounds[j][1]:
                        trial[j] = bounds[j][1]
                
                # Crossover
                crossover_points = [random.random() < crossover_prob for _ in range(dimensions)]
                if not any(crossover_points):
                    # Ensure at least one dimension is crossed over
                    crossover_points[random.randint(0, dimensions-1)] = True
                
                for j in range(dimensions):
                    if not crossover_points[j]:
                        trial[j] = population[i][j]
                
                # Selection
                trial_fitness = eval_fitness(trial)
                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness
                    
                    # Update best if needed
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Record best fitness for this generation
            run_data.append(best_fitness)
        
        all_runs_data.append(run_data)
        all_best_solutions.append(best_solution)
    
    # Calculate average across runs
    max_len = max(len(run) for run in all_runs_data)
    padded_runs = [run + [run[-1]]*(max_len - len(run)) for run in all_runs_data]
    average = np.mean(padded_runs, axis=0).tolist()
    
    # Find overall best solution
    best_run_idx = np.argmin([run[-1] for run in all_runs_data])
    overall_best_solution = all_best_solutions[best_run_idx]
    
    # Return results in the same format as other algorithms
    return {
        'run_data': all_runs_data,
        'average': average,
        'best_solution': overall_best_solution,
        'best_fitness': all_runs_data[best_run_idx][-1]
    }