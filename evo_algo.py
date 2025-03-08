import numpy as np
import random
import csv
import json
import os
from io import StringIO

def evolutionary_algorithm(fitness_func, pop_size=10, x1_range=(-5, 5), x2_range=(-5, 5),
                     parent_selection='roulette', survival_selection='truncation',
                     crossover_type='arithmetic', mutation_type='gaussian',
                     generations=40, runs=10):
    
    def evaluate_fitness(x, y):
        try:
            return eval(fitness_func)
        except:
            return float('inf')
    
    def constrain_to_bounds(individual):
        """Ensure individual stays within the specified ranges"""
        x = max(min(individual[0], x1_range[1]), x1_range[0])
        y = max(min(individual[1], x2_range[1]), x2_range[0])
        return (x, y)
    
    def crossover(p1, p2, cross_type):
        if cross_type == 'arithmetic':
            alpha = random.random()
            offspring = (alpha * p1[0] + (1-alpha) * p2[0],
                        alpha * p1[1] + (1-alpha) * p2[1])
            return constrain_to_bounds(offspring)
        elif cross_type == 'swap':
            if random.random() < 0.5:
                return constrain_to_bounds((p2[0], p1[1]))
            else:
                return constrain_to_bounds((p1[0], p2[1]))
        elif cross_type == 'blend':
            alpha = 0.3
            c1_x = p1[0] + (random.random() * (1 + 2*alpha) - alpha) * (p2[0] - p1[0])
            c1_y = p1[1] + (random.random() * (1 + 2*alpha) - alpha) * (p2[1] - p1[1])
            return constrain_to_bounds((c1_x, c1_y))
    
    def mutation(individual, mut_type):
        if mut_type == 'gaussian':
            mutated = (individual[0] + random.gauss(0, 0.1),
                      individual[1] + random.gauss(0, 0.1))
            return constrain_to_bounds(mutated)
        elif mut_type == 'uniform':
            delta = 0.1
            mutated = (individual[0] + random.uniform(-delta, delta),
                      individual[1] + random.uniform(-delta, delta))
            return constrain_to_bounds(mutated)
        elif mut_type == 'creep':
            step = 0.1 * (1 - gen/generations)
            mutated = (individual[0] + random.uniform(-step, step),
                      individual[1] + random.uniform(-step, step))
            return constrain_to_bounds(mutated)

    
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
                
                # Survival selection
                if survival_selection == 'truncation':
                    combined_pop = population + new_pop
                    combined_fitnesses = [evaluate_fitness(x, y) for x, y in combined_pop]
                    sorted_indices = np.argsort(combined_fitnesses)[::-1]    
                    population = [combined_pop[i] for i in sorted_indices[:pop_size]]
                    # combined_fitnesses = [evaluate_fitness(x, y) for x, y in combined_pop]
                    # sorted_indices = np.argsort(combined_fitnesses)
                    # population = [combined_pop[i] for i in sorted_indices[:pop_size]]
                elif survival_selection == 'tournament':
                    combined_pop = population + new_pop
                    combined_fitnesses = [evaluate_fitness(x, y) for x, y in combined_pop]
                    new_population = []
                    while len(new_population) < pop_size:
                        contestants = random.sample(range(len(combined_pop)), 2)
                        winner = max(contestants, key=lambda i: combined_fitnesses[i])
                        new_population.append(combined_pop[winner])
                    population = new_population
                
                # population = new_pop
            
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

def export_results_to_csv(result_data, filename="evo_algo_results.csv"):
    """
    Export evolution algorithm results to a CSV file
    
    Args:
        result_data: Dictionary containing 'run_data' and 'average'
        filename: Name of the file to save
    
    Returns:
        Path to the saved file
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        header = ["Generation"] + [f"Run {i+1}" for i in range(len(result_data['run_data']))] + ["Average"]
        writer.writerow(header)
        
        # Write data rows
        for gen in range(len(result_data['average'])):
            row = [gen+1]  # Generation number (1-indexed)
            for run in result_data['run_data']:
                row.append(run[gen])
            row.append(result_data['average'][gen])
            writer.writerow(row)
    
    return output_path

def get_results_csv_string(result_data):
    """
    Get results as a CSV string for direct download in web applications
    
    Args:
        result_data: Dictionary containing 'run_data' and 'average'
    
    Returns:
        CSV string of the results
    """
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header row
    header = ["Generation"] + [f"Run {i+1}" for i in range(len(result_data['run_data']))] + ["Average"]
    writer.writerow(header)
    
    # Write data rows
    for gen in range(len(result_data['average'])):
        row = [gen+1]  # Generation number (1-indexed)
        for run in result_data['run_data']:
            row.append(run[gen])
        row.append(result_data['average'][gen])
        writer.writerow(row)
    
    return output.getvalue()

def export_results_to_json(result_data, filename="evo_algo_results.json"):
    """
    Export evolution algorithm results to a JSON file
    
    Args:
        result_data: Dictionary containing 'run_data' and 'average'
        filename: Name of the file to save
    
    Returns:
        Path to the saved file
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    # Prepare data in a more structured format
    generations = len(result_data['average'])
    structured_data = {
        "generations": generations,
        "runs": len(result_data['run_data']),
        "data_by_generation": [
            {
                "generation": gen+1,
                "run_results": [run[gen] for run in result_data['run_data']],
                "average": result_data['average'][gen]
            } for gen in range(generations)
        ],
        "raw_data": result_data
    }
    
    with open(output_path, 'w') as jsonfile:
        json.dump(structured_data, jsonfile, indent=2)
    
    return output_path