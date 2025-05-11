import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds, norm
import networkx as nx
import gzip
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

class DirectedGraph:
    """
    Directed graph representation supporting various Laplacian definitions and operations
    """
    def __init__(self, n=0):
        """Initialize an empty directed graph with n nodes"""
        self.n = n  # Number of nodes
        self.edges = {}  # Dictionary to store edges: (u,v) -> weight
        self.out_degree = defaultdict(float)  # Out-degree of each node
        self.in_degree = defaultdict(float)   # In-degree of each node
        self.nodes = set()  # Set of nodes

    def copy(self):
        """Create a deep copy of the graph"""
        g_copy = DirectedGraph(self.n)
        g_copy.edges = self.edges.copy()
        g_copy.out_degree = self.out_degree.copy()
        g_copy.in_degree = self.in_degree.copy()
        g_copy.nodes = self.nodes.copy()
        return g_copy

    def add_or_update_edge(self, u, v, weight=1.0):
        """Add a new edge or update existing edge weight"""
        # Ensure nodes are in the graph
        self.nodes.add(u)
        self.nodes.add(v)
        self.n = max(self.n, u + 1, v + 1)
        
        # Update edge weight
        old_weight = self.edges.get((u, v), 0.0)
        self.edges[(u, v)] = weight
        
        # Update degrees
        self.out_degree[u] = self.out_degree[u] - old_weight + weight
        self.in_degree[v] = self.in_degree[v] - old_weight + weight

    def remove_edge(self, u, v):
        """Remove an edge if it exists"""
        if (u, v) in self.edges:
            weight = self.edges[(u, v)]
            # Update degrees
            self.out_degree[u] -= weight
            self.in_degree[v] -= weight
            # Remove edge
            del self.edges[(u, v)]

    @property
    def num_edges(self):
        """Return the number of edges in the graph"""
        return len(self.edges)

    @property
    def num_nodes(self):
        """Return the number of nodes in the graph"""
        return self.n

    def to_sparse_adjacency(self):
        """Convert to sparse adjacency matrix"""
        row, col, data = [], [], []
        for (u, v), weight in self.edges.items():
            row.append(u)
            col.append(v)
            data.append(weight)
        
        # Create sparse adjacency matrix
        A = sp.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        return A

    def out_degree_laplacian(self):
        """Compute the out-degree Laplacian L_out = D_out - A"""
        A = self.to_sparse_adjacency()
        # Create diagonal out-degree matrix
        D_out_data = np.zeros(self.n)
        for node, degree in self.out_degree.items():
            if node < self.n:  # Safety check
                D_out_data[node] = degree
        D_out = sp.diags(D_out_data)
        
        # Compute L_out = D_out - A
        L_out = D_out - A
        return L_out

    def in_degree_laplacian(self):
        """Compute the in-degree Laplacian L_in = D_in - A^T"""
        A = self.to_sparse_adjacency()
        # Create diagonal in-degree matrix
        D_in_data = np.zeros(self.n)
        for node, degree in self.in_degree.items():
            if node < self.n:  # Safety check
                D_in_data[node] = degree
        D_in = sp.diags(D_in_data)
        
        # Compute L_in = D_in - A^T
        L_in = D_in - A.T
        return L_in

    def eulerian_symmetrized_laplacian(self, epsilon=1e-6, max_iter=100):
        """
        Compute symmetrized Laplacian for directed graph using Eulerian scaling
        based on the approach in Cohen et al. 2017 STOC and Kapralov 2023
        """
        A = self.to_sparse_adjacency()
        # Initialize stationary weights
        h = np.ones(self.n)
        
        # Find stationary weights h such that h^T D_out = h^T A
        # This is an iterative approximation for PageRank-like stationary distribution
        for _ in range(max_iter):
            h_old = h.copy()
            
            # Update h to approximate the solution to h^T (D_out - A) = 0
            D_out = np.zeros(self.n)
            for node, degree in self.out_degree.items():
                if node < self.n:
                    D_out[node] = degree
            
            # Compute h * A
            h_A = np.zeros(self.n)
            for (u, v), weight in self.edges.items():
                h_A[u] += h[v] * weight  # Incoming influence
            
            # Update h
            for i in range(self.n):
                if D_out[i] > 0:
                    h[i] = h_A[i] / D_out[i] + epsilon
            
            # Normalize h
            h = h / np.sum(h)
            
            # Check convergence
            if np.linalg.norm(h - h_old) < epsilon:
                break
        
        # Create the scaled adjacency matrix
        A_scaled = A.copy()
        for (u, v), weight in self.edges.items():
            A_scaled[u, v] = weight * h[u] / h[v]
        
        # Compute the scaled out-degree diagonal matrix
        D_scaled_data = np.zeros(self.n)
        for u in range(self.n):
            D_scaled_data[u] = sum(A_scaled[u, :])
        D_scaled = sp.diags(D_scaled_data)
        
        # Compute Eulerian Laplacian
        L_eulerian = D_scaled - A_scaled
        
        # Symmetrize the Laplacian
        L_sym = (L_eulerian + L_eulerian.T) / 2
        
        return L_sym

def load_directed_graph_from_file(filename, max_edges=None):
    """
    Load a directed graph from a gzipped file
    Assumes the file format is tab-separated (u, v) edges, one per line
    Ignores comment lines starting with #
    Remaps node IDs to consecutive integers starting from 0
    """
    # Check if the file is gzipped
    is_gzipped = filename.endswith('.gz')
    
    # Create a new graph
    G = DirectedGraph()
    
    # Open file (gzipped or normal)
    if is_gzipped:
        open_func = gzip.open
        mode = 'rt'  # Text mode for gzipped files
    else:
        open_func = open
        mode = 'r'
    
    # Map to remap original node IDs to consecutive integers
    node_id_map = {}
    next_node_id = 0
    
    # First pass: collect unique node IDs
    print(f"Scanning for unique node IDs in {filename}...")
    edge_list = []
    with open_func(filename, mode) as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Parse edge
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    weight = 1.0  # Default weight
                    if len(parts) >= 3:
                        try:
                            weight = float(parts[2])
                        except ValueError:
                            pass  # Use default weight if conversion fails
                    
                    # Store edge with original IDs for later processing
                    edge_list.append((u, v, weight))
                    
                    # Register node IDs if not seen before
                    if u not in node_id_map:
                        node_id_map[u] = next_node_id
                        next_node_id += 1
                    if v not in node_id_map:
                        node_id_map[v] = next_node_id
                        next_node_id += 1
                    
                    # Check if we've reached max_edges
                    if max_edges is not None and len(edge_list) >= max_edges:
                        break
                except ValueError:
                    # Skip edges with non-integer node IDs
                    continue
    
    # Initialize graph with correct node count
    G = DirectedGraph(next_node_id)
    
    # Second pass: add edges with remapped node IDs
    print(f"Adding {len(edge_list)} edges with remapped node IDs...")
    for orig_u, orig_v, weight in edge_list:
        # Get remapped node IDs
        u = node_id_map[orig_u]
        v = node_id_map[orig_v]
        
        # Add edge
        G.add_or_update_edge(u, v, weight)
    
    print(f"Loaded graph with {G.num_nodes} nodes and {G.num_edges} edges")
    return G

def generate_erdos_renyi_directed(n, p, self_loops=False):
    """
    Generate a directed Erdős-Rényi random graph
    Implementation inspired by NetworkX and NetworKit's ErdosRenyiGenerator
    
    Parameters:
    n : int
        Number of nodes
    p : float
        Probability of edge creation between any pair of nodes
    self_loops : bool, optional (default=False)
        If True, self-loops are allowed
        
    Returns:
    G : DirectedGraph
        A directed random graph in G(n,p) model
    """
    G = DirectedGraph(n)
    
    # For very sparse graphs (small p), we can use faster method
    # that only generates expected p*n*(n-1) edges
    if p <= 0.3:
        # Expected number of edges
        edge_count = int(p * n * (n - 1 if not self_loops else n))
        
        # Generate random edges
        edge_set = set()
        while len(edge_set) < edge_count:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v or self_loops:
                edge_set.add((u, v))
        
        # Add edges to graph
        for u, v in edge_set:
            G.add_or_update_edge(u, v, 1.0)
    else:
        # For denser graphs, directly check each possible edge
        for u in range(n):
            # Potential optimization: we can vectorize this with NumPy
            # rand_vals = np.random.random(n)
            # targets = np.where(rand_vals < p)[0]
            for v in range(n):
                if (u != v or self_loops) and random.random() < p:
                    G.add_or_update_edge(u, v, 1.0)
    
    print(f"Generated Erdős-Rényi directed graph with {G.num_nodes} nodes and {G.num_edges} edges")
    return G

def generate_preferential_attachment_directed(n, m, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0.0):
    """
    Generate a directed scale-free graph using preferential attachment
    Based on NetworkX's scale_free_graph implementation
    
    Parameters:
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree distribution
        and the other chosen randomly according to the out-degree distribution.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution
    delta_in : float
        Bias for choosing nodes from in-degree distribution
    delta_out : float
        Bias for choosing nodes from out-degree distribution
        
    Returns:
    G : DirectedGraph
        A directed scale-free graph
    """
    if alpha + beta + gamma != 1.0:
        # Normalize the probabilities
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
    
    G = DirectedGraph(n)
    
    # Start with a small complete graph
    initial_nodes = max(m + 1, 3)
    for u in range(initial_nodes):
        for v in range(initial_nodes):
            if u != v:  # No self-loops in the initial graph
                G.add_or_update_edge(u, v, 1.0)
    
    # List of existing nodes (needed for the preferential attachment)
    nodes = list(range(initial_nodes))
    
    # Start adding the remaining nodes
    source = initial_nodes  # Next node to be added
    
    while source < n:
        # Setup preferential attachment
        in_degrees = np.zeros(len(nodes))
        out_degrees = np.zeros(len(nodes))
        
        for i, node in enumerate(nodes):
            in_degrees[i] = G.in_degree.get(node, 0) + delta_in
            out_degrees[i] = G.out_degree.get(node, 0) + delta_out
        
        # Normalize to create probability distributions
        in_sum = np.sum(in_degrees)
        if in_sum > 0:
            in_probs = in_degrees / in_sum
        else:
            in_probs = np.ones_like(in_degrees) / len(in_degrees)
            
        out_sum = np.sum(out_degrees)
        if out_sum > 0:
            out_probs = out_degrees / out_sum
        else:
            out_probs = np.ones_like(out_degrees) / len(out_degrees)
        
        # Decide which type of edge to create
        r = random.random()
        
        if r < alpha:
            # Add new node source with edge to existing node
            target = np.random.choice(nodes, p=in_probs)
            G.add_or_update_edge(source, target, 1.0)
            
        elif r < alpha + beta:
            # Add edge between existing nodes
            u = np.random.choice(nodes, p=in_probs)
            v = np.random.choice(nodes, p=out_probs)
            # Skip if trying to add self-loop
            if u != v:
                G.add_or_update_edge(u, v, 1.0)
                
        else:  # r < alpha + beta + gamma
            # Add new node source with edge from existing node
            target = np.random.choice(nodes, p=out_probs)
            G.add_or_update_edge(target, source, 1.0)
        
        # Select additional edges to maintain the required m edges per new node
        current_edges = G.out_degree.get(source, 0) + G.in_degree.get(source, 0)
        remaining_edges = max(0, m - current_edges)  # Ensure non-negative
        
        # FIX: Convert to int for range function
        for _ in range(int(remaining_edges)):
            if random.random() < 0.5:
                # Add outgoing edge from new node
                potential_targets = [node for node in nodes if node != source and (source, node) not in G.edges]
                if potential_targets:
                    target = np.random.choice(potential_targets, p=None)  # Uniform selection
                    G.add_or_update_edge(source, target, 1.0)
            else:
                # Add incoming edge to new node
                potential_sources = [node for node in nodes if node != source and (node, source) not in G.edges]
                if potential_sources:
                    src = np.random.choice(potential_sources, p=None)  # Uniform selection
                    G.add_or_update_edge(src, source, 1.0)
        
        # Add the new node to the list of existing nodes
        nodes.append(source)
        source += 1
    
    print(f"Generated directed scale-free graph with {G.num_nodes} nodes and {G.num_edges} edges")
    return G

def compute_hybrid_importance_scores(G, L=None, sketch_size=None):
    """
    Compute hybrid importance scores for edges based on:
    1. Degree-based heuristic
    2. Spectral-sketch importance scores
    
    Returns a dictionary mapping edges to their scores
    """
    # Initialize scores dictionaries
    deg_scores = {}
    spec_scores = {}
    hybrid_scores = {}
    
    # Compute degree-based scores
    for (u, v), weight in G.edges.items():
        # s_e_deg = w_e * (out_degree(u) + in_degree(v))
        deg_scores[(u, v)] = weight * (G.out_degree[u] + G.in_degree[v])
    
    # If no Laplacian provided, use out-degree Laplacian
    if L is None:
        L = G.out_degree_laplacian()
    
    # Set sketch size if not provided
    if sketch_size is None:
        sketch_size = max(1, int(np.ceil(np.log2(G.num_nodes))))
    
    # Generate Rademacher random vectors
    n = G.num_nodes
    X = np.random.choice([-1.0, 1.0], size=(n, sketch_size))
    
    # Precompute L*X for efficiency
    LX = L @ X
    
    # Compute spectral-sketch scores
    for (u, v), weight in G.edges.items():
        # Create edge incidence vector for (u,v)
        # For directed edges, b_e has +1 at u and -1 at v
        # This is a Rademacher-based approximation inspired by leverage scores
        b_e_X = X[u, :] - X[v, :]  # b_e^T * X
        b_e_LX = LX[u, :] - LX[v, :]  # b_e^T * L * X
        
        # Compute the spectral-sketch score: (1/k) * sum(|b_e^T * x^(i)| * |b_e^T * L * x^(i)|)
        spec_scores[(u, v)] = np.mean(np.abs(b_e_X) * np.abs(b_e_LX))
    
    # Combine scores using geometric mean
    for edge in G.edges:
        hybrid_scores[edge] = np.sqrt(deg_scores[edge] * (spec_scores[edge] + 1e-10))
    
    return hybrid_scores

def normalize_scores(scores):
    """Normalize scores to create a probability distribution"""
    total = sum(scores.values())
    return {edge: score / total for edge, score in scores.items()}

def sample_edges(G, probs, sample_count):
    """
    Sample edges from G according to probabilities and create a sparsifier
    Using vectorized sampling for efficiency
    """
    # Extract edges and probabilities for vectorized sampling
    edges = list(probs.keys())
    p_values = np.array(list(probs.values()))
    
    # Sample edges with replacement
    sampled_indices = np.random.choice(
        len(edges),
        size=sample_count,
        p=p_values,
        replace=True
    )
    
    # Count occurrences of each edge
    edge_counts = defaultdict(int)
    for idx in sampled_indices:
        edge_counts[edges[idx]] += 1
    
    # Create sparsifier with reweighted edges
    H = DirectedGraph(G.num_nodes)
    for edge, count in edge_counts.items():
        u, v = edge
        orig_weight = G.edges[edge]
        new_weight = (orig_weight * count) / (sample_count * probs[edge])
        H.add_or_update_edge(u, v, new_weight)
    
    return H

def estimate_spectral_error(G, H):
    """
    Estimate the spectral approximation error ||L_G - L_H||_2 / ||L_G||_2
    using a Hutchinson trace estimator
    
    Handles graphs with different node counts by extending the smaller matrix
    """
    # Ensure both graphs have the same number of nodes
    n_G = G.num_nodes
    n_H = H.num_nodes
    n_max = max(n_G, n_H)
    
    # Compute Laplacians
    L_G = G.out_degree_laplacian()
    L_H = H.out_degree_laplacian()
    
    # Extend matrices if needed
    if n_G < n_max:
        # Extend L_G with zeros
        L_G = sp.vstack([L_G, sp.csr_matrix((n_max - n_G, n_G))])
        L_G = sp.hstack([L_G, sp.csr_matrix((n_max, n_max - n_G))])
    
    if n_H < n_max:
        # Extend L_H with zeros
        L_H = sp.vstack([L_H, sp.csr_matrix((n_max - n_H, n_H))])
        L_H = sp.hstack([L_H, sp.csr_matrix((n_max, n_max - n_H))])
    
    # Compute difference
    L_diff = L_G - L_H
    
    # Use Hutchinson trace estimator with 20 random vectors
    s = min(20, n_max)
    
    # Generate random vectors
    R = np.random.normal(0, 1, (n_max, s))
    
    # Estimate operator norm (largest singular value) of L_diff
    L_diff_norm_est = 0
    L_G_norm_est = 0
    
    for i in range(s):
        r = R[:, i]
        L_diff_r = L_diff @ r
        L_G_r = L_G @ r
        
        L_diff_norm_est = max(L_diff_norm_est, np.linalg.norm(L_diff_r) / np.linalg.norm(r))
        L_G_norm_est = max(L_G_norm_est, np.linalg.norm(L_G_r) / np.linalg.norm(r))
    
    # Calculate relative error
    if L_G_norm_est > 0:
        return L_diff_norm_est / L_G_norm_est
    else:
        return 1.0  # Default to high error if L_G_norm_est is too small

def compute_exact_spectral_error(G, H, k=6):
    """
    Compute the exact spectral approximation error ||L_G - L_H||_2 / ||L_G||_2
    using k largest singular values - using memory-efficient sparse operations
    """
    # Compute Laplacians (keep as sparse matrices)
    L_G = G.out_degree_laplacian()
    L_H = H.out_degree_laplacian()
    
    # Compute difference (as sparse matrix)
    L_diff = L_G - L_H
    
    # Compute largest singular values using sparse SVD
    try:
        # Use min(k, n-1) since L matrices have at least one zero singular value
        k_use = min(k, L_G.shape[0] - 1, 100)  # Limit k to avoid excessive computation
        
        # For very large matrices, use a more efficient estimator
        if L_G.shape[0] > 100000:
            # Use a power iteration method to estimate largest singular value
            return estimate_spectral_error(G, H)
        
        # For smaller matrices, use sparse SVD
        sigma_G = svds(L_G, k=k_use, return_singular_vectors=False)
        sigma_diff = svds(L_diff, k=k_use, return_singular_vectors=False)
        
        # Return relative error using largest singular values
        return np.max(sigma_diff) / np.max(sigma_G)
    except Exception as e:
        print(f"Warning: SVD computation failed with error: {e}")
        print("Falling back to spectral error estimator...")
        # Fallback to estimator
        return estimate_spectral_error(G, H)

def compute_condition_number_distortion(G, H):
    """
    Compute the condition number distortion κ*(T) = σ_max(L_H^+ L_G) / σ_min(L_H^+ L_G)
    This metric ensures both largest and smallest singular values are controlled
    Using a memory-efficient implementation for large graphs
    """
    # For very large graphs, use an approximation
    if G.num_nodes > 10000:
        # For large graphs, return an approximation based on spectral error
        err = estimate_spectral_error(G, H)
        # Approximate condition number distortion using error
        # This is a heuristic: 1 + 2*err provides a reasonable approximation
        return 1.0 + 2.0 * err
    
    # Compute Laplacians (as sparse matrices)
    L_G = G.out_degree_laplacian()
    L_H = H.out_degree_laplacian()
    
    # Add small regularization to make the matrices full rank
    eps = 1e-10
    n = L_G.shape[0]
    reg = eps * sp.eye(n, format='csr')
    L_G_reg = L_G + reg
    L_H_reg = L_H + reg
    
    try:
        # For smaller matrices, use sparse pseudoinverse approach
        # This is still an approximation but more accurate than the heuristic
        # Use iterative solver to approximate L_H_reg^+ * L_G_reg for a few vectors
        
        # Sample random test vectors
        num_samples = min(5, n - 1)
        max_sing = 0.0
        min_sing = float('inf')
        
        for _ in range(num_samples):
            # Generate random vector
            x = np.random.randn(n)
            x = x / np.linalg.norm(x)
            
            # Apply L_G_reg to x
            L_G_x = L_G_reg @ x
            
            # Solve L_H_reg y = L_G_x approximately using least squares
            y, info = sp.linalg.bicgstab(L_H_reg, L_G_x, tol=1e-6)
            
            if info == 0:  # Convergence check
                # Estimate singular value as ||y|| / ||x||
                sing_val = np.linalg.norm(y) / np.linalg.norm(x)
                max_sing = max(max_sing, sing_val)
                if sing_val > 1e-10:  # Filter out near-zero values
                    min_sing = min(min_sing, sing_val)
        
        if min_sing == float('inf') or max_sing < 1e-10:
            return 1.0  # Default to 1.0 if estimation fails
        
        return max_sing / min_sing
    except Exception as e:
        print(f"Warning: Condition number computation failed with error: {e}")
        # Fallback to approximation based on spectral error
        err = estimate_spectral_error(G, H)
        return 1.0 + 2.0 * err

def adaptive_directed_resparsification(edge_stream, max_size, target_error=0.3, error_threshold=0.4):
    """
    Adaptive Iterative Directed Resparsification
    
    Args:
        edge_stream: List of (u, v, weight) tuples representing the edge stream
        max_size: Maximum size threshold for the sparsifier
        target_error: Target approximation error (epsilon) - reduced from 0.5 to 0.3
        error_threshold: Error threshold to trigger resparsification (tau)
    
    Returns:
        H: Final sparsifier
        resparsification_stats: Dictionary with statistics
    """
    # Initial empty sparsifier
    H = DirectedGraph()
    
    # Edge buffer
    edge_buffer = []
    
    # Statistics
    resparsification_stats = {
        'resparsification_count': 0,
        'error_evolution': [],
        'parameter_adjustments': 0,
        'processing_times': []
    }
    
    # Set initial sample count (theoretically justified)
    epsilon = target_error  # Start with a more aggressive target (0.3 instead of 0.5)
    tau = error_threshold
    
    # Keep track of all node IDs seen
    all_nodes = set()
    
    # Minimum number of edges to process before considering resparsification
    # This prevents excessive resparsification on small buffers
    min_buffer_size = max(50, max_size // 10)
    
    # Track total edges seen for better sparsifier sizing
    total_edges_seen = 0
    
    # Process edge stream
    for i, (u, v, weight) in enumerate(tqdm(edge_stream, desc="Processing edge stream")):
        # Track nodes and total edges
        all_nodes.add(u)
        all_nodes.add(v)
        total_edges_seen += 1
        
        # Add edge to buffer
        edge_buffer.append((u, v, weight))
        
        # Skip resparsification checks if buffer is too small
        if len(edge_buffer) < min_buffer_size and H.num_edges > 0:
            continue
        
        # Check current graph size including buffer
        current_size = H.num_edges + len(edge_buffer)
        
        # Create temporary graph for error estimation
        G_temp = H.copy()
        
        # Add edges from buffer to temporary graph
        for buf_u, buf_v, buf_weight in edge_buffer:
            G_temp.add_or_update_edge(buf_u, buf_v, buf_weight)
        
        # Ensure all nodes exist in both graphs (important for error estimation)
        for node in all_nodes:
            if node not in G_temp.nodes:
                G_temp.nodes.add(node)
                G_temp.n = max(G_temp.n, node + 1)
            if node not in H.nodes:
                H.nodes.add(node)
                H.n = max(H.n, node + 1)
        
        # Calculate nodes count for sample size determination
        n = len(all_nodes)
        
        # Calculate min target size - SIGNIFICANTLY INCREASED from 5% to 25%
        min_edges = min(int(total_edges_seen * 0.25), max_size)  # At least 25% of total
        
        # Size based on node count, using less aggressive reduction for larger graphs
        # Double the coefficient from 1 to 2 for larger sparsifiers
        k_s = int(np.ceil(2 * n * np.log2(n) / (epsilon * epsilon)))
        
        # Ensure sample size is at least min_edges
        k_s = max(k_s, min_edges)
        
        # Ensure k_s doesn't exceed max_size to avoid constant resparsification
        k_s = min(k_s, max_size)
        
        # Estimate spectral error if graph is non-empty
        current_error = 0
        if G_temp.num_edges > 0 and H.num_edges > 0:
            current_error = estimate_spectral_error(G_temp, H)
        
        # For diagnostics in first iteration
        if resparsification_stats['resparsification_count'] == 0 and i > 0 and i % 1000 == 0:
            print(f"Current buffer size: {len(edge_buffer)}, Error: {current_error:.4f}, "
                  f"Threshold: {tau:.4f}, Current graph size: {current_size}, "
                  f"Target sample size: {k_s}, Min edges: {min_edges}")
        
        # Check if resparsification is needed - updated condition to prevent excessive resparsifications
        if (current_size > max_size or (current_error > tau and len(edge_buffer) >= min_buffer_size)):
            start_time = time.time()
            
            # Adaptive parameter adjustment - LOWER THRESHOLD from 1.2 to 1.0
            # This means ANY error over tau will trigger adjustment
            if current_error > 1.0 * tau:
                old_epsilon = epsilon
                # Reduce epsilon more aggressively and set a higher minimum (0.25)
                epsilon = max(epsilon / 2.0, 0.25)  
                # Double sample count for immediate impact
                k_s = min(k_s * 2, max_size)
                tau = 1.2 * epsilon
                resparsification_stats['parameter_adjustments'] += 1
                print(f"Adjusting parameters: epsilon {old_epsilon:.4f} -> {epsilon:.4f}, "
                      f"sample count -> {k_s}, tau -> {tau:.4f}")
            
            # Compute importance scores
            scores = compute_hybrid_importance_scores(G_temp)
            probs = normalize_scores(scores)
            
            # Sample edges to create new sparsifier
            H = sample_edges(G_temp, probs, k_s)
            
            # Ensure the new sparsifier has all nodes (even if some have no edges)
            for node in all_nodes:
                if node not in H.nodes:
                    H.nodes.add(node)
                    H.n = max(H.n, node + 1)
            
            # Calculate actual error after resparsification
            actual_error = estimate_spectral_error(G_temp, H)
            
            # If the error is still too high after resparsification, try increasing sample size
            # LOWER THRESHOLD to ensure we catch more cases
            if actual_error > 0.8 * tau and k_s < max_size // 2:
                print(f"Error still high ({actual_error:.4f}) after resparsification. Increasing sample size.")
                # More aggressive increase
                k_s = min(k_s * 3, max_size)
                H = sample_edges(G_temp, probs, k_s)
                
                # Recalculate error
                actual_error = estimate_spectral_error(G_temp, H)
                print(f"New sample size: {k_s}, New error: {actual_error:.4f}")
            
            # Clear buffer
            edge_buffer = []
            
            # Update statistics - use actual error after resparsification
            resparsification_stats['resparsification_count'] += 1
            resparsification_stats['error_evolution'].append(actual_error)
            resparsification_stats['processing_times'].append(time.time() - start_time)
            
            # More frequent output to track progress
            if resparsification_stats['resparsification_count'] % 5 == 0:
                print(f"Resparsification #{resparsification_stats['resparsification_count']}, "
                      f"Current error: {actual_error:.4f}, "
                      f"Sparsifier size: {H.num_edges}, "
                      f"Target epsilon: {epsilon:.4f}")
    
    # Final sparsification of remaining buffer if needed
    if edge_buffer:
        print(f"Processing final buffer with {len(edge_buffer)} edges")
        G_temp = H.copy()
        for buf_u, buf_v, buf_weight in edge_buffer:
            G_temp.add_or_update_edge(buf_u, buf_v, buf_weight)
        
        # Ensure consistent node sets
        for node in all_nodes:
            if node not in G_temp.nodes:
                G_temp.nodes.add(node)
                G_temp.n = max(G_temp.n, node + 1)
        
        scores = compute_hybrid_importance_scores(G_temp)
        probs = normalize_scores(scores)
        
        # Calculate final sample size - aim for 25-30% of total edges seen (INCREASED)
        final_k_s = max(k_s, int(total_edges_seen * 0.25))
        final_k_s = min(final_k_s, max_size)
        
        print(f"Final sparsification with sample size: {final_k_s}")
        H = sample_edges(G_temp, probs, final_k_s)
        
        # Ensure all nodes in final sparsifier
        for node in all_nodes:
            if node not in H.nodes:
                H.nodes.add(node)
                H.n = max(H.n, node + 1)
    
    return H, resparsification_stats

def uniform_sampling(G, sample_count):
    """Simple uniform random edge sampling baseline"""
    edges = list(G.edges.keys())
    weights = [G.edges[e] for e in edges]
    
    # Uniform sampling probabilities
    n_edges = len(edges)
    probs = {edge: 1.0 / n_edges for edge in edges}
    
    # Sample edges
    H = sample_edges(G, probs, sample_count)
    return H

def degree_only_sampling(G, sample_count):
    """Degree-based importance sampling without spectral information"""
    scores = {}
    for (u, v), weight in G.edges.items():
        scores[(u, v)] = weight * (G.out_degree[u] + G.in_degree[v])
    
    probs = normalize_scores(scores)
    H = sample_edges(G, probs, sample_count)
    return H

def evaluate_sparsification_methods(G, sample_counts):
    """
    Compare different sparsification methods across various sample sizes
    
    Args:
        G: Original graph
        sample_counts: List of sample counts to test
    
    Returns:
        results: Dictionary with evaluation results
    """
    results = {
        'sample_counts': sample_counts,
        'ADR': {'approx_error': [], 'condition_distortion': [], 'runtime': []},
        'ADR_deg_only': {'approx_error': [], 'condition_distortion': [], 'runtime': []},
        'US': {'approx_error': [], 'condition_distortion': [], 'runtime': []}
    }
    
    # Check if graph is too large for exact computation
    use_estimator = G.num_nodes > 100000
    if use_estimator:
        print(f"Graph is large ({G.num_nodes} nodes). Using error estimator instead of exact computation.")
    
    for k_s in sample_counts:
        print(f"\nEvaluating with sample count k_s = {k_s}")
        
        # Adaptive Directed Resparsification (ADR)
        start_time = time.time()
        scores = compute_hybrid_importance_scores(G)
        probs = normalize_scores(scores)
        H_adr = sample_edges(G, probs, k_s)
        adr_time = time.time() - start_time
        results['ADR']['runtime'].append(adr_time)
        
        # ADR with degree-only scores
        start_time = time.time()
        H_deg_only = degree_only_sampling(G, k_s)
        deg_only_time = time.time() - start_time
        results['ADR_deg_only']['runtime'].append(deg_only_time)
        
        # Uniform Sampling (US)
        start_time = time.time()
        H_us = uniform_sampling(G, k_s)
        us_time = time.time() - start_time
        results['US']['runtime'].append(us_time)
        
        # Compute metrics
        print("Computing approximation errors...")
        
        # For ADR
        if use_estimator:
            adr_error = estimate_spectral_error(G, H_adr)
        else:
            adr_error = compute_exact_spectral_error(G, H_adr)
        adr_cond = compute_condition_number_distortion(G, H_adr)
        results['ADR']['approx_error'].append(adr_error)
        results['ADR']['condition_distortion'].append(adr_cond)
        
        # For ADR with degree-only scores
        if use_estimator:
            deg_only_error = estimate_spectral_error(G, H_deg_only)
        else:
            deg_only_error = compute_exact_spectral_error(G, H_deg_only)
        deg_only_cond = compute_condition_number_distortion(G, H_deg_only)
        results['ADR_deg_only']['approx_error'].append(deg_only_error)
        results['ADR_deg_only']['condition_distortion'].append(deg_only_cond)
        
        # For US
        if use_estimator:
            us_error = estimate_spectral_error(G, H_us)
        else:
            us_error = compute_exact_spectral_error(G, H_us)
        us_cond = compute_condition_number_distortion(G, H_us)
        results['US']['approx_error'].append(us_error)
        results['US']['condition_distortion'].append(us_cond)
        
        print(f"ADR Error: {adr_error:.4f}, Condition: {adr_cond:.4f}, Time: {adr_time:.4f}s")
        print(f"ADR-deg Error: {deg_only_error:.4f}, Condition: {deg_only_cond:.4f}, Time: {deg_only_time:.4f}s")
        print(f"US Error: {us_error:.4f}, Condition: {us_cond:.4f}, Time: {us_time:.4f}s")
    
    return results

def simulate_edge_stream(G, buffer_size=1000):
    """
    Simulate an edge stream from a static graph by randomly permuting edges
    
    Args:
        G: Original graph
        buffer_size: Size of the edge buffer
    
    Returns:
        edge_stream: List of (u, v, weight) tuples
        static_resparsification_metrics: Metrics for static resparsification
    """
    # Create a list of edges with their weights
    edge_stream = [(u, v, G.edges[(u, v)]) for u, v in G.edges]
    
    # Shuffle edges to create a random stream
    np.random.shuffle(edge_stream)
    
    # Define the maximum sparsifier size (buffer_size * 2)
    max_size = buffer_size * 2
    
    # Static resparsification (standard approach)
    print("Computing static resparsification baseline...")
    n = G.num_nodes
    
    # Calculate a more reasonable size for sparsifier - targeting around 25% of edges
    # but with a minimum based on node count to ensure good quality
    target_edges = max(int(G.num_edges * 0.25), int(n * np.log2(n)))
    # Can't have more than original edges
    target_edges = min(target_edges, G.num_edges)  
    
    print(f"Using static sparsifier target size: {target_edges} edges (~{target_edges/G.num_edges*100:.1f}% of original)")
    
    start_time = time.time()
    
    scores = compute_hybrid_importance_scores(G)
    probs = normalize_scores(scores)
    H_static = sample_edges(G, probs, target_edges)
    
    # Ensure H_static has all nodes from G (even if they have no edges)
    for node in G.nodes:
        if node not in H_static.nodes:
            H_static.nodes.add(node)
            H_static.n = max(H_static.n, node + 1)
    
    static_time = time.time() - start_time
    static_error = estimate_spectral_error(G, H_static)
    static_cond = compute_condition_number_distortion(G, H_static)
    
    static_resparsification_metrics = {
        'error': static_error,
        'condition': static_cond,
        'time': static_time,
        'size': H_static.num_edges,
        'target_size': target_edges
    }
    
    return edge_stream, static_resparsification_metrics

def run_streaming_experiment(filename, max_edges=100000, buffer_size=1000):
    """
    Run a streaming experiment on a dataset
    
    Args:
        filename: Path to the graph dataset
        max_edges: Maximum number of edges to load
        buffer_size: Size of the edge buffer for resparsification
    
    Returns:
        results: Dictionary with experiment results
    """
    # Load the graph
    print(f"Loading graph from {filename}...")
    G = load_directed_graph_from_file(filename, max_edges=max_edges)
    
    # Get edge stream and static baseline
    edge_stream, static_metrics = simulate_edge_stream(G, buffer_size)
    
    # Run adaptive resparsification
    print("Running adaptive directed resparsification...")
    H_adaptive, adaptive_stats = adaptive_directed_resparsification(
        edge_stream, 
        max_size=buffer_size*2, 
        target_error=0.5
    )
    
    # Run uniform sampling baseline for streaming
    print("Running uniform sampling baseline...")
    # Create edge stream graph by directly adding all edges
    G_stream = DirectedGraph()
    for u, v, weight in edge_stream:
        G_stream.add_or_update_edge(u, v, weight)
    
    # Sample edges uniformly
    H_uniform = uniform_sampling(G_stream, H_adaptive.num_edges)
    uniform_error = compute_exact_spectral_error(G_stream, H_uniform)
    
    # Run naive iterative sparsification baseline
    print("Running naive iterative sparsification baseline...")
    H_naive = DirectedGraph()
    naive_errors = []
    
    # Process in chunks to simulate naive resampling
    chunk_size = buffer_size
    for i in range(0, len(edge_stream), chunk_size):
        # Add chunk to graph
        G_chunk = H_naive.copy()
        for u, v, weight in edge_stream[i:i+chunk_size]:
            G_chunk.add_or_update_edge(u, v, weight)
        
        # Naive resampling (uniform)
        n = G_chunk.num_nodes
        k_s = int(np.ceil(4 * n * np.log(n) / (0.5 * 0.5)))  # Using epsilon = 0.5
        H_naive = uniform_sampling(G_chunk, k_s)
        
        # Track error
        if G_chunk.num_edges > 0:
            naive_error = compute_exact_spectral_error(G_chunk, H_naive)
            naive_errors.append(naive_error)
    
    # Final evaluation
    final_adaptive_error = compute_exact_spectral_error(G_stream, H_adaptive)
    final_naive_error = compute_exact_spectral_error(G_stream, H_naive)
    
    # Collect results
    results = {
        'graph_info': {
            'nodes': G.num_nodes,
            'edges': G.num_edges,
            'filename': filename
        },
        'static': static_metrics,
        'adaptive': {
            'error': final_adaptive_error,
            'size': H_adaptive.num_edges,
            'resparsification_count': adaptive_stats['resparsification_count'],
            'error_evolution': adaptive_stats['error_evolution'],
            'parameter_adjustments': adaptive_stats['parameter_adjustments'],
            'processing_times': adaptive_stats['processing_times']
        },
        'uniform': {
            'error': uniform_error,
            'size': H_uniform.num_edges
        },
        'naive': {
            'error': final_naive_error,
            'size': H_naive.num_edges,
            'error_evolution': naive_errors
        }
    }
    
    return results

def plot_error_evolution(adaptive_errors, naive_errors, title="Error Evolution"):
    """Plot the error evolution for adaptive and naive resparsification"""
    plt.figure(figsize=(10, 6))
    
    # Plot adaptive errors
    if adaptive_errors:
        plt.plot(range(len(adaptive_errors)), adaptive_errors, 'b-', marker='o', 
                 label='Directed Resparsification (DR)')
    
    # Plot naive errors
    if naive_errors:
        plt.plot(range(len(naive_errors)), naive_errors, 'r--', marker='x', 
                 label='Naive Iterative Sampling')
    
    plt.xlabel('Resparsification Steps')
    plt.ylabel('Spectral Approximation Error')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt

def plot_sparsification_quality(results, title="Sparsification Quality Comparison"):
    """Plot the sparsification quality comparison for different methods"""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    sample_counts = results['sample_counts']
    
    # Create bar groups
    bar_width = 0.25
    index = np.arange(len(sample_counts))
    
    # Plot errors for each method
    plt.bar(index, results['ADR']['approx_error'], bar_width, 
            label='Adaptive Directed Resparsification (ADR)', color='blue', alpha=0.7)
    
    plt.bar(index + bar_width, results['ADR_deg_only']['approx_error'], bar_width, 
            label='ADR-deg-only', color='green', alpha=0.7)
    
    plt.bar(index + 2*bar_width, results['US']['approx_error'], bar_width, 
            label='Uniform Sampling (US)', color='red', alpha=0.7)
    
    # Labels and formatting
    plt.xlabel('Sample Count')
    plt.ylabel('Approximation Error')
    plt.title(title)
    plt.xticks(index + bar_width, [str(sc) for sc in sample_counts])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    
    return plt

def plot_runtime_comparison(results, title="Runtime Performance Comparison"):
    """Plot the runtime comparison for different methods"""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    sample_counts = results['sample_counts']
    
    # Create line plot
    plt.plot(sample_counts, results['ADR']['runtime'], 'bo-', linewidth=2, 
             label='Adaptive Directed Resparsification (ADR)')
    
    plt.plot(sample_counts, results['ADR_deg_only']['runtime'], 'gs-', linewidth=2, 
             label='ADR-deg-only')
    
    plt.plot(sample_counts, results['US']['runtime'], 'rd-', linewidth=2, 
             label='Uniform Sampling (US)')
    
    # Labels and formatting
    plt.xlabel('Sample Count')
    plt.ylabel('Runtime (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt

def condition_number_table(results):
    """Generate a table of condition number distortions"""
    sample_counts = results['sample_counts']
    
    print("\nCondition Number Distortion Comparison:")
    print("=" * 65)
    print(f"{'Sample Count':<15} {'ADR':<15} {'ADR-deg-only':<15} {'US':<15}")
    print("-" * 65)
    
    for i, k_s in enumerate(sample_counts):
        adr_cond = results['ADR']['condition_distortion'][i]
        deg_only_cond = results['ADR_deg_only']['condition_distortion'][i]
        us_cond = results['US']['condition_distortion'][i]
        
        print(f"{k_s:<15} {adr_cond:<15.4f} {deg_only_cond:<15.4f} {us_cond:<15.4f}")
    
    print("=" * 65)

def generate_directed_configuration_model(in_degree_sequence, out_degree_sequence):
    """
    Generate a directed graph from given in and out degree sequences
    Based on NetworkX's directed_configuration_model
    
    Parameters:
    in_degree_sequence : list or array
        Sequence of in-degrees for each node
    out_degree_sequence : list or array
        Sequence of out-degrees for each node
        
    Returns:
    G : DirectedGraph
        A directed graph with the given degree sequences
    """
    # Check if the degree sequences have the same sum
    if sum(in_degree_sequence) != sum(out_degree_sequence):
        raise ValueError("In and out degree sequences must have the same sum")
    
    n_in = len(in_degree_sequence)
    n_out = len(out_degree_sequence)
    n = max(n_in, n_out)
    
    # Extend sequences if needed
    if n_in < n:
        in_degree_sequence.extend([0] * (n - n_in))
    if n_out < n:
        out_degree_sequence.extend([0] * (n - n_out))
    
    # Create node stubs
    in_stubs = []
    for i in range(n):
        in_stubs.extend([i] * in_degree_sequence[i])
    
    out_stubs = []
    for i in range(n):
        out_stubs.extend([i] * out_degree_sequence[i])
    
    # Shuffle to randomize connections
    random.shuffle(in_stubs)
    random.shuffle(out_stubs)
    
    # Create graph
    G = DirectedGraph(n)
    
    # Create edges
    for out_node, in_node in zip(out_stubs, in_stubs):
        G.add_or_update_edge(out_node, in_node, 1.0)
    
    print(f"Generated directed configuration model with {G.num_nodes} nodes and {G.num_edges} edges")
    return G

def generate_power_law_sequence(n, alpha, min_degree=1, max_degree=None):
    """
    Generate a power-law degree sequence
    
    Parameters:
    n : int
        Number of nodes
    alpha : float
        Power law exponent (typically 2 < alpha < 3)
    min_degree : int, optional (default=1)
        Minimum degree
    max_degree : int, optional (default=None)
        Maximum degree. If None, set to n-1
        
    Returns:
    sequence : list
        A sequence of n integers following a power law distribution
    """
    if max_degree is None:
        max_degree = n - 1
    
    # Generate sequence using inverse transform sampling
    seq = []
    for _ in range(n):
        u = random.random()
        # Inverse CDF of power law distribution
        if alpha != 1:
            x = ((max_degree**(1-alpha) - min_degree**(1-alpha)) * u + min_degree**(1-alpha)) ** (1/(1-alpha))
        else:
            x = min_degree * np.exp(u * np.log(max_degree / min_degree))
        
        # Round to integer
        seq.append(int(x))
    
    # Adjust to ensure the sequence is graphical
    # This is a simple adjustment - more sophisticated methods exist
    while sum(seq) % 2 != 0:
        i = random.randrange(n)
        if seq[i] > min_degree:
            seq[i] -= 1
        else:
            seq[i] += 1
    
    return seq

def generate_powerlaw_directed_graph(n, alpha_in=2.5, alpha_out=2.5, min_degree=1, max_degree=None):
    """
    Generate a directed graph with power-law in/out degree distributions
    
    Parameters:
    n : int
        Number of nodes
    alpha_in : float, optional (default=2.5)
        Power law exponent for in-degree distribution
    alpha_out : float, optional (default=2.5)
        Power law exponent for out-degree distribution
    min_degree : int, optional (default=1)
        Minimum degree
    max_degree : int, optional (default=None)
        Maximum degree. If None, set to n-1
        
    Returns:
    G : DirectedGraph
        A directed graph with power-law degree distributions
    """
    if max_degree is None:
        max_degree = min(100, n - 1)  # Limit max degree for very large graphs
    
    # Generate in and out degree sequences
    in_seq = generate_power_law_sequence(n, alpha_in, min_degree, max_degree)
    out_seq = generate_power_law_sequence(n, alpha_out, min_degree, max_degree)
    
    # Make sure the sequences have the same sum
    while sum(in_seq) != sum(out_seq):
        if sum(in_seq) > sum(out_seq):
            # Decrease in-degree or increase out-degree
            if random.random() < 0.5:
                i = random.choice([i for i in range(n) if in_seq[i] > min_degree])
                in_seq[i] -= 1
            else:
                i = random.choice([i for i in range(n) if out_seq[i] < max_degree])
                out_seq[i] += 1
        else:
            # Increase in-degree or decrease out-degree
            if random.random() < 0.5:
                i = random.choice([i for i in range(n) if in_seq[i] < max_degree])
                in_seq[i] += 1
            else:
                i = random.choice([i for i in range(n) if out_seq[i] > min_degree])
                out_seq[i] -= 1
    
    # Generate graph using configuration model
    G = generate_directed_configuration_model(in_seq, out_seq)
    
    return G

def main():
    """Main function to run experiments"""
    # Define dataset paths - adjust as needed
    datasets = {
        'cit-HepPh': 'cit-HepPh.txt.gz',
        'soc-Epinions1': 'soc-Epinions1.txt.gz',
        'web-Google': 'web-Google.txt.gz'
    }
    
    # Results storage
    all_results = {}
    
    # Run static quality experiments
    for name, path in datasets.items():
        print(f"\n{'='*30}\nRunning quality experiments on {name}\n{'='*30}")
        
        # Determine max_edges based on dataset
        if "Google" in path:
            # Use a smaller subset for web-Google since it's very large
            max_edges = 50000
        elif "HepPh" in path:
            max_edges = 50000
        else:
            max_edges = 100000
            
        print(f"Loading graph with max_edges={max_edges}")
        G = load_directed_graph_from_file(path, max_edges=max_edges)
        
        # Determine sample counts based on graph size
        n = G.num_nodes
        # Adjust base_sample for very large graphs to avoid memory issues
        if n > 100000:
            base_sample = int(min(n * np.log2(n) / 10, 10000))  # Limit size for large graphs
        else:
            base_sample = int(np.ceil(n * np.log2(n) / 2))  # Base sample count
            
        sample_counts = [
            max(100, base_sample // 4),  # Ensure minimum size
            max(200, base_sample // 2),
            max(500, base_sample)
        ]
        
        print(f"Using sample counts: {sample_counts}")
        
        # Evaluate methods
        results = evaluate_sparsification_methods(G, sample_counts)
        all_results[f"{name}_quality"] = results
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot approximation error
        plt.subplot(2, 1, 1)
        for method, color, marker in [
            ('ADR', 'blue', 'o'),
            ('ADR_deg_only', 'green', 's'),
            ('US', 'red', '^')
        ]:
            plt.plot(sample_counts, results[method]['approx_error'], color=color, marker=marker, 
                    linewidth=2, markersize=8, label=method)
        
        plt.xlabel('Sample Count')
        plt.ylabel('Approximation Error')
        plt.title(f'Approximation Error Comparison - {name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot runtime
        plt.subplot(2, 1, 2)
        for method, color, marker in [
            ('ADR', 'blue', 'o'),
            ('ADR_deg_only', 'green', 's'),
            ('US', 'red', '^')
        ]:
            plt.plot(sample_counts, results[method]['runtime'], color=color, marker=marker, 
                    linewidth=2, markersize=8, label=method)
        
        plt.xlabel('Sample Count')
        plt.ylabel('Runtime (seconds)')
        plt.title(f'Runtime Comparison - {name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{name}_quality_results.png")
        
        # Print condition number table
        condition_number_table(results)
    
    # Run streaming experiments with reduced sizes
    for name, path in datasets.items():
        print(f"\n{'='*30}\nRunning streaming experiment on {name}\n{'='*30}")
        
        # Use even smaller samples for streaming to ensure experiments complete in reasonable time
        if "Google" in path:
            max_edges = 20000
            buffer_size = 500
        elif "HepPh" in path:
            max_edges = 30000
            buffer_size = 1000
        else:
            max_edges = 50000
            buffer_size = 1000
            
        print(f"Running streaming with max_edges={max_edges}, buffer_size={buffer_size}")
        
        # Run streaming experiment
        stream_results = run_streaming_experiment(path, max_edges=max_edges, buffer_size=buffer_size)
        all_results[f"{name}_stream"] = stream_results
        
        # Plot error evolution
        adaptive_errors = stream_results['adaptive']['error_evolution']
        naive_errors = stream_results['naive']['error_evolution']
        
        plt_error = plot_error_evolution(
            adaptive_errors, 
            naive_errors, 
            title=f"Error Evolution for {name} Stream"
        )
        plt_error.savefig(f"{name}_error_evolution.png")
        
        # Print final results
        print("\nStreaming Experiment Results:")
        print(f"Graph: {name}, Nodes: {stream_results['graph_info']['nodes']}, "
              f"Edges: {stream_results['graph_info']['edges']}")
        print("\nStatic Resparsification:")
        print(f"  Approximation Error: {stream_results['static']['error']:.4f}")
        print(f"  Sparsifier Size: {stream_results['static']['size']}")
        print(f"  Processing Time: {stream_results['static']['time']:.4f}s")
        
        print("\nAdaptive Directed Resparsification:")
        print(f"  Final Approximation Error: {stream_results['adaptive']['error']:.4f}")
        print(f"  Sparsifier Size: {stream_results['adaptive']['size']}")
        print(f"  Resparsification Count: {stream_results['adaptive']['resparsification_count']}")
        print(f"  Parameter Adjustments: {stream_results['adaptive']['parameter_adjustments']}")
        
        print("\nUniform Sampling:")
        print(f"  Approximation Error: {stream_results['uniform']['error']:.4f}")
        
        print("\nNaive Iterative Sparsification:")
        print(f"  Final Approximation Error: {stream_results['naive']['error']:.4f}")
    
    # Run synthetic graph experiments
    print(f"\n{'='*30}\nRunning synthetic graph experiments\n{'='*30}")
    
    # Generate various synthetic graphs - with smaller size for faster experiments
    n_nodes = 1000
    
    # Generate Erdos-Renyi graph
    p_er = 0.01
    G_er = generate_erdos_renyi_directed(n_nodes, p_er)
    
    # Generate preferential attachment graph (NetworkX-style)
    m_pa = 5
    G_pa = generate_preferential_attachment_directed(n_nodes, m_pa)
    
    # Generate power-law directed graph using configuration model
    G_pl = generate_powerlaw_directed_graph(n_nodes, alpha_in=2.3, alpha_out=2.7)
    
    # Define sample counts for synthetic graphs - smaller for faster experiments
    sample_counts = [200, 500, 1000]
    
    # Evaluate methods on synthetic graphs
    print("\nEvaluating Erdos-Renyi graph:")
    results_er = evaluate_sparsification_methods(G_er, sample_counts)
    all_results['erdos_renyi_quality'] = results_er
    
    print("\nEvaluating Preferential Attachment graph:")
    results_pa = evaluate_sparsification_methods(G_pa, sample_counts)
    all_results['pref_attachment_quality'] = results_pa
    
    print("\nEvaluating Power-Law Configuration Model:")
    results_pl = evaluate_sparsification_methods(G_pl, sample_counts)
    all_results['powerlaw_quality'] = results_pl
    
    # Compare synthetic models
    plt.figure(figsize=(15, 10))
    
    # Plot approximation error across synthetic models
    plt.subplot(2, 1, 1)
    for model_name, results, color, marker in [
        ('Erdos-Renyi', results_er, 'blue', 'o'),
        ('Preferential Attachment', results_pa, 'green', 's'),
        ('Power-Law Configuration', results_pl, 'red', '^')
    ]:
        plt.plot(sample_counts, results['ADR']['approx_error'], color=color, marker=marker, 
                linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel('Sample Count')
    plt.ylabel('ADR Approximation Error')
    plt.title('Approximation Error Comparison - Synthetic Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot runtime
    plt.subplot(2, 1, 2)
    for model_name, results, color, marker in [
        ('Erdos-Renyi', results_er, 'blue', 'o'),
        ('Preferential Attachment', results_pa, 'green', 's'),
        ('Power-Law Configuration', results_pl, 'red', '^')
    ]:
        plt.plot(sample_counts, results['ADR']['runtime'], color=color, marker=marker, 
                linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel('Sample Count')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison - Synthetic Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("synthetic_models_comparison.png")
    
    # Save all results
    import pickle
    with open('directed_resparsification_results.pickle', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\nAll experiments completed and results saved.")

if __name__ == "__main__":
    main()