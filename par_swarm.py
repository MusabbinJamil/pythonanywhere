import numpy as np

def particle_swarm_optimization(fitness_func, num_particles=30, bounds=[(-5, 5), (-5, 5)], 
                               w=0.5, c1=1.5, c2=1.5, iterations=100, runs=1):
    """
    Particle Swarm Optimization algorithm.
    
    Parameters:
    - fitness_func: String representing the fitness function to be minimized (e.g., 'x**2 + y**2')
    - num_particles: Number of particles in the swarm
    - bounds: List of tuples representing the bounds for each dimension [(min_x, max_x), (min_y, max_y)]
    - w: Inertia weight
    - c1: Cognitive coefficient (personal best influence)
    - c2: Social coefficient (global best influence)
    - iterations: Maximum number of iterations
    - runs: Number of independent runs to perform
    
    Returns:
    - Dictionary containing the results of all runs and statistics
    """
    dimensions = len(bounds)
    
    # Compiling the fitness function for performance
    var_names = ['x', 'y', 'z'][:dimensions]
    vars_str = ', '.join(var_names[:dimensions])
    
    # For dimensions > 2, convert to x1, x2, etc.
    if dimensions > len(var_names):
        var_names = [f'x{i+1}' for i in range(dimensions)]
        vars_str = ', '.join(var_names)
    
    # Replace variable names in fitness function
    if dimensions == 1:
        func_str = f"lambda {vars_str}: {fitness_func}"
    else:
        # Handle multi-dimensional case
        compiled_func_str = fitness_func
        for i, var in enumerate(var_names):
            if f'x{i+1}' in fitness_func:
                # Function already uses x1, x2 format
                pass
            else:
                # Replace x, y, z with position array indices
                compiled_func_str = compiled_func_str.replace(var, f'position[{i}]')
        
        func_str = f"lambda position: {compiled_func_str}"
    
    # Compile the function
    try:
        compiled_func = eval(func_str)
    except Exception as e:
        raise ValueError(f"Error compiling fitness function: {e}")
    
    # Prepare results storage
    all_runs_data = []
    all_best_positions = []
    all_best_values = []
    
    # Run multiple times
    for run in range(runs):
        # Initialize particles
        positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(num_particles, dimensions)
        )
        
        # Initialize velocities
        velocity_range = [abs(b[1] - b[0]) for b in bounds]
        velocities = np.random.uniform(
            low=[-vr for vr in velocity_range],
            high=velocity_range,
            size=(num_particles, dimensions)
        )
        
        # Initialize personal best
        personal_best_positions = positions.copy()
        personal_best_values = np.array([compiled_func(p) for p in positions])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_value = personal_best_values[global_best_idx]
        
        # Storage for convergence history
        convergence_history = [global_best_value]
        
        # Main loop
        for i in range(iterations):
            # Update velocities and positions
            r1 = np.random.random((num_particles, dimensions))
            r2 = np.random.random((num_particles, dimensions))
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_positions - positions) +
                         c2 * r2 * (global_best_position - positions))
            
            positions = positions + velocities
            
            # Enforce bounds
            for j in range(dimensions):
                positions[:, j] = np.clip(positions[:, j], bounds[j][0], bounds[j][1])
            
            # Evaluate new positions
            values = np.array([compiled_func(p) for p in positions])
            
            # Update personal bests
            improved = values < personal_best_values
            personal_best_positions[improved] = positions[improved]
            personal_best_values[improved] = values[improved]
            
            # Update global best
            if np.min(personal_best_values) < global_best_value:
                global_best_idx = np.argmin(personal_best_values)
                global_best_position = personal_best_positions[global_best_idx].copy()
                global_best_value = personal_best_values[global_best_idx]
            
            # Record convergence
            convergence_history.append(global_best_value)
        
        # Store run results
        all_runs_data.append(convergence_history)
        all_best_positions.append(global_best_position.tolist())
        all_best_values.append(global_best_value)
    
    # Calculate average convergence
    iterations_count = len(all_runs_data[0])
    average_convergence = np.zeros(iterations_count)
    
    for i in range(iterations_count):
        values_at_iteration = [run_data[i] for run_data in all_runs_data]
        average_convergence[i] = np.mean(values_at_iteration)
    
    # Find best overall solution
    best_run_idx = np.argmin(all_best_values)
    best_position = all_best_positions[best_run_idx]
    best_value = all_best_values[best_run_idx]
    
    # Prepare results
    results = {
        'run_data': all_runs_data,
        'best_positions': all_best_positions,
        'best_values': all_best_values,
        'best_position': best_position,
        'best_value': best_value,
        'average': average_convergence.tolist()
    }
    
    return results