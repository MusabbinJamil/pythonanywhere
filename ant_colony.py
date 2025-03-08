import numpy as np
import random
import math
import csv
import json
import os

def ant_colony_optimization(problem_type='tsp', problem_size=10, num_ants=20, 
                           iterations=100, alpha=1.0, beta=2.0, 
                           evaporation_rate=0.5, initial_pheromone=0.1,
                           runs=10, custom_coordinates=None, custom_binary_fitness=None):    
    """
    Ant Colony Optimization algorithm for solving optimization problems
    
    Args:
        problem_type: Type of problem to solve ('tsp' or 'binary')
        problem_size: Number of cities for TSP or bits for binary optimization
        num_ants: Number of ants in the colony
        iterations: Number of algorithm iterations
        alpha: Pheromone importance factor
        beta: Heuristic information importance factor
        evaporation_rate: Rate at which pheromone evaporates
        initial_pheromone: Initial pheromone level
        runs: Number of independent runs
    
    Returns:
        Dictionary with results of the algorithm execution
    """
    all_runs_data = []
    best_solutions = []
    best_quality = float('inf') if problem_type == 'tsp' else float('-inf')
    
    # Generate random coordinates for TSP
    if problem_type == 'tsp':
        if custom_coordinates is not None:
            coordinates = custom_coordinates
            problem_size = len(coordinates)  # Update problem size to match coordinates
        else:
            coordinates = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(problem_size)]

        print(coordinates)
        
        # Calculate distance matrix
        distance_matrix = np.zeros((problem_size, problem_size))
        for i in range(problem_size):
            for j in range(problem_size):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distance_matrix[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                else:
                    distance_matrix[i][j] = 1e-10  # Small non-zero value
    
    for run in range(runs):
        # Initialize pheromone matrix
        if problem_type == 'tsp':
            pheromone = np.ones((problem_size, problem_size)) * initial_pheromone
        else:  # binary
            pheromone = np.ones((problem_size, 2)) * initial_pheromone  # Binary choices
        
        # Track best solution for this run
        run_best_solution = None
        run_best_quality = float('inf') if problem_type == 'tsp' else float('-inf')
        
        # Track quality over iterations
        iteration_qualities = []
        
        # Run for specified iterations
        for iteration in range(iterations):
            # Solutions for all ants in this iteration
            all_solutions = []
            all_qualities = []
            
            # Let each ant construct a solution
            for ant in range(num_ants):
                if problem_type == 'tsp':
                    # Construct TSP tour
                    solution = construct_tsp_solution(
                        problem_size, distance_matrix, pheromone, alpha, beta
                    )
                    # Calculate tour length (smaller is better)
                    quality = calculate_tsp_tour_length(solution, distance_matrix)
                    all_qualities.append(quality)
                else:
                    # Construct binary solution
                    solution = construct_binary_solution(problem_size, pheromone, alpha, beta)
                    # Calculate quality (larger is better for this example)
                    quality = calculate_binary_quality(solution)
                    all_qualities.append(quality)
                
                all_solutions.append(solution)
            
            # Find best solution in this iteration
            if problem_type == 'tsp':
                iter_best_idx = np.argmin(all_qualities)
                iter_best_quality = all_qualities[iter_best_idx]
                if iter_best_quality < run_best_quality:
                    run_best_quality = iter_best_quality
                    run_best_solution = all_solutions[iter_best_idx]
            else:
                iter_best_idx = np.argmax(all_qualities)
                iter_best_quality = all_qualities[iter_best_idx]
                if iter_best_quality > run_best_quality:
                    run_best_quality = iter_best_quality
                    run_best_solution = all_solutions[iter_best_idx]
            
            # Record quality for this iteration
            iteration_qualities.append(run_best_quality)
            
            # Update pheromone trails
            pheromone = update_pheromones(
                pheromone, all_solutions, all_qualities, problem_type, evaporation_rate
            )
        
        # Record best solution from this run
        all_runs_data.append(iteration_qualities)
        best_solutions.append(run_best_solution)
        
        # Update overall best solution
        if ((problem_type == 'tsp' and run_best_quality < best_quality) or 
            (problem_type != 'tsp' and run_best_quality > best_quality)):
            best_quality = run_best_quality
            best_solution_overall = run_best_solution
    
    # Calculate average convergence across runs
    avg_convergence = np.mean(all_runs_data, axis=0)
    
    return {
        'run_data': all_runs_data,
        'average': avg_convergence.tolist(),
        'best_solution': best_solution_overall,
        'best_quality': best_quality,
        'coordinates': coordinates if problem_type == 'tsp' else None
    }

def construct_tsp_solution(problem_size, distance_matrix, pheromone, alpha, beta):
    """Construct a TSP solution using ACO"""
    # Start at a random city
    current_city = random.randint(0, problem_size - 1)
    unvisited = set(range(problem_size))
    unvisited.remove(current_city)
    
    tour = [current_city]
    
    # Construct solution
    while unvisited:
        # Calculate probabilities for next move
        probabilities = []
        for city in unvisited:
            # Pheromone and heuristic (inverse distance)
            p = (pheromone[current_city][city] ** alpha) * \
                ((1.0 / distance_matrix[current_city][city]) ** beta)
            probabilities.append((city, p))
        
        # Normalize probabilities
        total = sum(p for _, p in probabilities)
        if total == 0:
            # If all probabilities are zero, choose randomly
            next_city = random.choice(list(unvisited))
        else:
            # Choose next city using roulette wheel selection
            r = random.random() * total
            cumsum = 0
            next_city = None
            for city, p in probabilities:
                cumsum += p
                if cumsum >= r:
                    next_city = city
                    break
            
            # Fallback in case of numerical issues
            if next_city is None:
                next_city = probabilities[-1][0]
        
        # Move to next city
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return tour

def construct_binary_solution(problem_size, pheromone, alpha, beta):
    """Construct a binary solution using ACO"""
    solution = []
    
    for i in range(problem_size):
        # Calculate probabilities for 0 and 1
        p0 = pheromone[i][0] ** alpha
        p1 = pheromone[i][1] ** alpha
        
        # Choose bit value using roulette wheel selection
        if random.random() < (p1 / (p0 + p1)):
            solution.append(1)
        else:
            solution.append(0)
    
    return solution

def calculate_tsp_tour_length(tour, distance_matrix):
    """Calculate the total length of a TSP tour"""
    tour_length = 0
    for i in range(len(tour)):
        tour_length += distance_matrix[tour[i]][tour[(i+1) % len(tour)]]
    return tour_length

def calculate_binary_quality(solution):
    """Calculate the quality of a binary solution"""
    # Simple example: Count number of 1s
    return sum(solution)

def update_pheromones(pheromone, solutions, qualities, problem_type, evaporation_rate):
    """Update pheromone levels based on solutions quality"""
    if problem_type == 'tsp':
        # Evaporation
        pheromone = pheromone * (1 - evaporation_rate)
        
        # Deposit new pheromone
        for solution, quality in zip(solutions, qualities):
            deposit = 1.0 / quality  # Better solutions (shorter tours) deposit more
            for i in range(len(solution)):
                # Connect the last city to the first to complete the tour
                j, k = solution[i], solution[(i+1) % len(solution)]
                pheromone[j][k] += deposit
                pheromone[k][j] += deposit  # Symmetric problem
    else:
        # Binary problem
        # Evaporation
        pheromone = pheromone * (1 - evaporation_rate)
        
        for solution, quality in zip(solutions, qualities):
            deposit = quality  # Better solutions (higher quality) deposit more
            for i, bit in enumerate(solution):
                pheromone[i][bit] += deposit
    
    return pheromone