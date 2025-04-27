import random
from collections import defaultdict

class Node:
    def __init__(self, name, values=None):
        """
        Initialize a node in a Bayesian Network
        
        Parameters:
        - name: name of the node/variable
        - values: possible values this variable can take (default: [True, False] for boolean)
        """
        self.name = name
        self.values = values if values is not None else [True, False]
        self.parents = []
        self.cpt = {}  # Conditional probability table
        
    def add_parents(self, parents):
        """Add parent nodes to this node"""
        self.parents = parents
        
    def set_cpt(self, cpt):
        """
        Set the conditional probability table
        
        For example, if node has parents A and B, both boolean:
        {
            (True, True): {True: 0.9, False: 0.1},
            (True, False): {True: 0.8, False: 0.2},
            (False, True): {True: 0.7, False: 0.3},
            (False, False): {True: 0.1, False: 0.9}
        }
        """
        self.cpt = cpt
        
    def sample(self, parent_values):
        """Sample a value based on parent values"""
        if not self.parents:
            # No parents, use unconditional distribution
            probs = self.cpt
        else:
            # Use conditional distribution based on parent values
            parent_vals = tuple(parent_values[parent.name] for parent in self.parents)
            probs = self.cpt[parent_vals]
        
        # Sample from the distribution
        values = list(probs.keys())
        probabilities = list(probs.values())
        return random.choices(values, weights=probabilities, k=1)[0]
                
class BayesianNetwork:
    def __init__(self):
        """Initialize an empty Bayesian Network"""
        self.nodes = {}
        self.sorted_nodes = []
        
    def add_node(self, node):
        """Add a node to the network"""
        self.nodes[node.name] = node
        self._sort_nodes()  # Re-sort nodes in topological order
        
    def _sort_nodes(self):
        """Sort nodes in topological order (parents before children)"""
        visited = set()
        temp = set()
        order = []
        
        def visit(node):
            if node.name in visited:
                return
            if node.name in temp:
                raise ValueError("Cycle detected in Bayesian network")
            
            temp.add(node.name)
            
            for parent in node.parents:
                visit(parent)
                
            temp.remove(node.name)
            visited.add(node.name)
            order.append(node)
            
        for node in self.nodes.values():
            if node.name not in visited:
                visit(node)
                
        self.sorted_nodes = order
        
    def sample(self):
        """Generate a single sample from the network"""
        sample = {}
        
        for node in self.sorted_nodes:
            parent_values = {parent.name: sample[parent.name] for parent in node.parents}
            sample[node.name] = node.sample(parent_values)
            
        return sample
        
def logic_sampling(bn, n_samples):
    """
    Perform logic sampling on a Bayesian Network
    
    Parameters:
    - bn: BayesianNetwork instance
    - n_samples: number of samples to generate
    
    Returns:
    - A list of samples, where each sample is a dictionary mapping variable names to values
    """
    samples = []
    for _ in range(n_samples):
        samples.append(bn.sample())
    return samples

def query_probability(samples, query_var, query_val, evidence=None):
    """
    Estimate P(query_var=query_val | evidence) from samples
    
    Parameters:
    - samples: list of samples from logic_sampling
    - query_var: the variable to query
    - query_val: the value of the query variable
    - evidence: dictionary of evidence variables and their values
    
    Returns:
    - Estimated probability
    """
    if evidence is None:
        evidence = {}
        
    # Filter samples that match the evidence
    matching_samples = []
    for sample in samples:
        matches_evidence = True
        for var, val in evidence.items():
            if sample[var] != val:
                matches_evidence = False
                break
        if matches_evidence:
            matching_samples.append(sample)
    
    # Count occurrences where query variable equals query value
    count = sum(1 for sample in matching_samples if sample[query_var] == query_val)
    
    # Avoid division by zero
    if not matching_samples:
        return 0.0
        
    return count / len(matching_samples)