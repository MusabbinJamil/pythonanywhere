import numpy as np
import random
import math
import os
import json
import csv
import io

def genetic_algorithm(fitness_func, chromosome_length=20, pop_size=50, 
                     variable_ranges=[(-5,5), (-5,5)], n_variables=2,
                     selection_method='tournament', crossover_method='single_point',
                     mutation_method='bit_flip', mutation_rate=0.01,
                     elite_size=2, generations=100, runs=10,
                     save_to_file=None, file_format='csv'):
    """
    Genetic algorithm for function optimization using binary encoding.
    
    Parameters:
    - fitness_func: String representation of function to optimize
    - chromosome_length: Total bit length of chromosome
    - pop_size: Population size
    - variable_ranges: List of tuples with min/max for each variable
    - n_variables: Number of variables in the function
    - selection_method: 'tournament', 'roulette', or 'rank'
    - crossover_method: 'single_point', 'two_point', or 'uniform'
    - mutation_method: 'bit_flip', 'swap', or 'inversion'
    - mutation_rate: Probability of mutation per bit
    - elite_size: Number of best individuals to preserve
    - generations: Number of generations
    - runs: Number of independent runs
    """
    
    # Bits per variable
    bits_per_var = chromosome_length // n_variables
    
    def decode_chromosome(chromosome):
        """Convert binary chromosome to real variables"""
        variables = []
        for i in range(n_variables):
            # Extract bits for this variable
            start = i * bits_per_var
            end = start + bits_per_var
            segment = chromosome[start:end]
            
            # Convert binary to decimal
            decimal_value = int(''.join(map(str, segment)), 2)
            
            # Scale to range
            min_val, max_val = variable_ranges[i]
            scaled_value = min_val + decimal_value * (max_val - min_val) / (2**bits_per_var - 1)
            variables.append(scaled_value)
        
        return variables
    
    def evaluate_fitness(chromosome):
        """Evaluate chromosome fitness"""
        variables = decode_chromosome(chromosome)
        
        # Create local variables for function evaluation
        locals_dict = {f'x{i+1}': val for i, val in enumerate(variables)}
        
        try:
            return eval(fitness_func, {}, locals_dict)
        except Exception:
            return float('inf')
    
    def initialize_population():
        """Create initial random population"""
        return [[random.randint(0, 1) for _ in range(chromosome_length)] 
                for _ in range(pop_size)]
    
    def selection(population, fitnesses, method):
        """Select individual based on specified method"""
        if method == 'tournament':
            k = 3  # Tournament size
            contestants = random.sample(range(len(population)), k)
            winner = min(contestants, key=lambda i: fitnesses[i])
            return population[winner]
            
        elif method == 'roulette':
            # For minimization, invert fitness (higher is better)
            max_fit = max(fitnesses)
            adjusted_fits = [max_fit - f + 1e-10 for f in fitnesses]
            total_fit = sum(adjusted_fits)
            
            r = random.random() * total_fit
            cumulative = 0
            for i, fit in enumerate(adjusted_fits):
                cumulative += fit
                if cumulative > r:
                    return population[i]
            return population[-1]  # Fallback
            
        elif method == 'rank':
            # Sort indices by fitness (ascending for minimization)
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            ranks = [len(fitnesses) - sorted_indices.index(i) for i in range(len(fitnesses))]
            total_rank = sum(ranks)
            
            r = random.random() * total_rank
            cumulative = 0
            for i, rank in enumerate(ranks):
                cumulative += rank
                if cumulative > r:
                    return population[i]
            return population[-1]  # Fallback
    
    def crossover(parent1, parent2, method):
        """Perform crossover between parents"""
        if method == 'single_point':
            point = random.randint(1, chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            
        elif method == 'two_point':
            p1, p2 = sorted(random.sample(range(1, chromosome_length), 2))
            child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
            child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
            
        elif method == 'uniform':
            child1 = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(chromosome_length)]
            child2 = [parent2[i] if random.random() < 0.5 else parent1[i] for i in range(chromosome_length)]
            
        return child1, child2
    
    def mutate(chromosome, method):
        """Mutate chromosome based on specified method"""
        result = chromosome.copy()
        
        if method == 'bit_flip':
            for i in range(chromosome_length):
                if random.random() < mutation_rate:
                    result[i] = 1 - result[i]  # Flip bit
                    
        elif method == 'swap':
            if random.random() < mutation_rate:
                i, j = random.sample(range(chromosome_length), 2)
                result[i], result[j] = result[j], result[i]
                
        elif method == 'inversion':
            if random.random() < mutation_rate:
                i, j = sorted(random.sample(range(chromosome_length), 2))
                result[i:j+1] = reversed(result[i:j+1])
                
        return result
    
    run_data = []
    
    for run in range(runs):
        population = initialize_population()
        generation_bests = []
        
        for gen in range(generations):
            # Evaluate population
            fitnesses = [evaluate_fitness(chrom) for chrom in population]
            
            # Record best fitness for this generation
            best_fit = min(fitnesses)
            generation_bests.append(best_fit)
            
            # Elitism - keep best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
            elite = [population[i].copy() for i in elite_indices]
            
            # Create new population starting with elite
            new_pop = []
            new_pop.extend(elite)
            
            # Calculate how many children we need
            children_needed = pop_size - len(new_pop)
            
            # Fill rest of population with crossover and mutation
            while len(new_pop) < pop_size:
                # Selection (parents still selected from current generation)
                parent1 = selection(population, fitnesses, selection_method)
                parent2 = selection(population, fitnesses, selection_method)
                
                # Crossover
                child1, child2 = crossover(parent1, parent2, crossover_method)
                
                # Mutation
                child1 = mutate(child1, mutation_method)
                child2 = mutate(child2, mutation_method)
                
                # Add only as many children as needed
                if len(new_pop) < pop_size:
                    new_pop.append(child1)
                if len(new_pop) < pop_size:
                    new_pop.append(child2)
            
            # Replace old population with new one
            population = new_pop
        
        run_data.append(generation_bests)
    
    # Calculate average across all runs
    average = np.mean(run_data, axis=0)
    result = {
        'run_data': run_data,
        'average': average.tolist()
    }

    # Save results to file if requested
    if save_to_file is not None:
        filepath = save_results(result, save_to_file, file_format)
        if filepath:
            result['filepath'] = filepath
    
    return result

def save_results(result, filename=None, format='csv'):
    """
    Save genetic algorithm results to a file.
    
    Parameters:
    - result: Dictionary containing 'run_data' and 'average'
    - filename: Name of the file to save the results to (optional)
    - format: Format to save in ('csv' or 'json')
    
    Returns:
    - filepath: Full path to the saved file
    """
    from datetime import datetime
    
    # Create a timestamped filename if none provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ga_results_{timestamp}"
    
    # Ensure the filename has the correct extension
    if format.lower() == 'csv' and not filename.endswith('.csv'):
        filename += '.csv'
    elif format.lower() == 'json' and not filename.endswith('.json'):
        filename += '.json'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else '.', exist_ok=True)
    
    try:
        if format.lower() == 'csv':
            return save_to_csv(result, filename)
        elif format.lower() == 'json':
            return save_to_json(result, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

def save_to_csv(result, filename):
    """Save results to CSV format"""
    import csv
    
    # Get the data
    run_data = result['run_data']
    average = result['average']
    runs = len(run_data)
    generations = len(average)
    
    try:
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Create header row
            header = ['Generation', 'Average']
            for i in range(runs):
                header.append(f'Run {i+1}')
            writer.writerow(header)
            
            # Write data for each generation
            for gen in range(generations):
                row = [gen, average[gen]]
                for run in range(runs):
                    row.append(run_data[run][gen])
                writer.writerow(row)
        
        return os.path.abspath(filename)
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        return None

def save_to_json(result, filename):
    """Save results to JSON format"""
    try:
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        return os.path.abspath(filename)
    except Exception as e:
        print(f"Failed to save JSON: {e}")
        return None

def get_results_csv_string(result):
    """
    Convert genetic algorithm results to a CSV string for download.
    
    Parameters:
    - result: Dictionary containing 'run_data' and 'average'
    
    Returns:
    - csv_string: CSV formatted string of the results
    """
    
    # Get the data
    run_data = result['run_data']
    average = result['average']
    runs = len(run_data)
    generations = len(average)
    
    # Create a string buffer to write CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Create header row
    header = ['Generation', 'Average']
    for i in range(runs):
        header.append(f'Run {i+1}')
    writer.writerow(header)
    
    # Write data for each generation
    for gen in range(generations):
        row = [gen, average[gen]]
        for run in range(runs):
            row.append(run_data[run][gen])
        writer.writerow(row)
    
    # Get the CSV string
    csv_string = output.getvalue()
    output.close()
    
    return csv_string