# compaction/algorithms/cluster.py
"""
Cluster-based KV cache compaction algorithm.

Clusters keys into k groups and selects the latest key (highest index)
from each cluster as its representative. The clustering method is
pluggable via CLUSTERING_REGISTRY.
"""
import torch
from typing import Tuple, Dict, Callable
from .base import CompactionAlgorithm


# ---------------------------------------------------------------------------
# Clustering functions
# ---------------------------------------------------------------------------
# Each function has signature: (K: Tensor, k: int, **kwargs) -> Tensor
# Returns assignments of shape (T,) with values in [0, k).

def _kmeans_cluster(K: torch.Tensor, k: int, max_iters: int = 50) -> torch.Tensor:
    """
    K-means clustering in pure torch.

    Uses k-means++ initialization and Lloyd's algorithm.
    Handles empty clusters by re-seeding from the largest cluster.

    Parameters
    ----------
    K : Tensor, shape (T, d)
        Key vectors to cluster.
    k : int
        Number of clusters.
    max_iters : int
        Maximum iterations (default: 50).

    Returns
    -------
    assignments : Tensor, shape (T,), dtype long
        Cluster assignment for each key, values in [0, k).
    """
    T, d = K.shape
    device = K.device

    if k >= T:
        # More clusters than keys: each key is its own cluster
        return torch.arange(T, device=device, dtype=torch.long)

    K_fp32 = K.to(torch.float32)

    # --- k-means++ initialization ---
    centroids = torch.empty(k, d, device=device, dtype=torch.float32)
    # Pick first centroid uniformly at random
    idx = torch.randint(T, (1,), device=device).item()
    centroids[0] = K_fp32[idx]

    # Squared distances to nearest chosen centroid
    dists = torch.sum((K_fp32 - centroids[0].unsqueeze(0)) ** 2, dim=1)  # (T,)

    for i in range(1, k):
        # Sample proportional to squared distance
        probs = dists / (dists.sum() + 1e-12)
        idx = torch.multinomial(probs, 1).item()
        centroids[i] = K_fp32[idx]
        # Update distances: min of current and distance to new centroid
        new_dists = torch.sum((K_fp32 - centroids[i].unsqueeze(0)) ** 2, dim=1)
        dists = torch.minimum(dists, new_dists)

    # --- Lloyd's iterations ---
    for _ in range(max_iters):
        # Assign each key to nearest centroid
        # (T, k) pairwise squared distances via expansion
        dots = K_fp32 @ centroids.T  # (T, k)
        centroid_sq = (centroids ** 2).sum(dim=1)  # (k,)
        key_sq = (K_fp32 ** 2).sum(dim=1, keepdim=True)  # (T, 1)
        sq_dists = key_sq + centroid_sq.unsqueeze(0) - 2 * dots  # (T, k)
        assignments = sq_dists.argmin(dim=1)  # (T,)

        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand_as(K_fp32), K_fp32)
        counts.scatter_add_(0, assignments, torch.ones(T, device=device, dtype=torch.float32))

        # Handle empty clusters: re-seed from the largest cluster
        empty_mask = counts == 0
        if empty_mask.any():
            largest_cluster = counts.argmax().item()
            members = (assignments == largest_cluster).nonzero(as_tuple=True)[0]
            empty_indices = empty_mask.nonzero(as_tuple=True)[0]
            for j, empty_idx in enumerate(empty_indices):
                # Pick a random member from the largest cluster
                donor = members[torch.randint(len(members), (1,), device=device).item()]
                new_centroids[empty_idx] = K_fp32[donor]
                counts[empty_idx] = 1.0

        # Normalize non-empty centroids
        nonzero = counts > 0
        new_centroids[nonzero] /= counts[nonzero].unsqueeze(1)

        # Check convergence
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < 1e-6:
            break

    # Final assignment
    dots = K_fp32 @ centroids.T
    centroid_sq = (centroids ** 2).sum(dim=1)
    key_sq = (K_fp32 ** 2).sum(dim=1, keepdim=True)
    sq_dists = key_sq + centroid_sq.unsqueeze(0) - 2 * dots
    assignments = sq_dists.argmin(dim=1)

    return assignments


# Registry of clustering functions.
# To add a new method, define a function with signature
#   (K: Tensor, k: int, **kwargs) -> Tensor
# and add it here.
CLUSTERING_REGISTRY: Dict[str, Callable] = {
    'kmeans': _kmeans_cluster,
}


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class ClusterCompaction(CompactionAlgorithm):
    """Cluster keys and select the latest key from each cluster."""

    def __init__(
        self,
        clustering_method: str = 'kmeans',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
        kmeans_max_iters: int = 50,
    ):
        """
        Parameters
        ----------
        clustering_method : str
            Key in CLUSTERING_REGISTRY (default: 'kmeans').
        beta_method : str
            'nnls' or 'zero' (default: 'nnls').
        c2_method : str
            'lsq' or 'direct' (default: 'lsq').
        nnls_iters : int
            Projected-gradient iterations for NNLS (0 = clamped lstsq).
        nnls_lower_bound, nnls_upper_bound : float, optional
            Box constraints for NNLS.
        c2_ridge_lambda, c2_solver, c2_ridge_scale :
            Parameters for C2 ridge regression (see base class).
        kmeans_max_iters : int
            Iteration cap for k-means (default: 50).
        """
        if clustering_method not in CLUSTERING_REGISTRY:
            raise ValueError(
                f"Unknown clustering_method '{clustering_method}'. "
                f"Available: {list(CLUSTERING_REGISTRY.keys())}"
            )
        self.clustering_method = clustering_method
        if beta_method not in ('nnls', 'zero'):
            raise ValueError(f"beta_method must be 'nnls' or 'zero', got '{beta_method}'")
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.kmeans_max_iters = kmeans_max_iters

    def name(self) -> str:
        return "Cluster"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Cluster keys into t groups, pick the latest key per cluster.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        V : Tensor, shape (T, d)
            Original value matrix.
        queries : Tensor, shape (n, d)
            Query samples for beta / C2 computation.
        t : int
            Number of keys to keep (= number of clusters).
        attention_bias : Tensor, optional
            Additive attention bias (broadcastable to (n, T)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
        beta : Tensor, shape (t,)
        C2 : Tensor, shape (t, d)
        indices : list of int
        """
        T, d = K.shape
        device = K.device
        dtype_param = K.dtype

        # 1. Cluster keys
        cluster_fn = CLUSTERING_REGISTRY[self.clustering_method]
        kwargs = {}
        if self.clustering_method == 'kmeans':
            kwargs['max_iters'] = self.kmeans_max_iters
        assignments = cluster_fn(K, t, **kwargs)  # (T,) long, values in [0, t)

        # 2. From each cluster, pick the key with the highest index (latest)
        selected_indices = []
        for c in range(t):
            members = (assignments == c).nonzero(as_tuple=True)[0]
            if len(members) == 0:
                # Shouldn't happen with proper clustering, but fallback
                # to a random index not already selected
                remaining = set(range(T)) - set(selected_indices)
                selected_indices.append(min(remaining))
            else:
                selected_indices.append(members.max().item())

        selected_indices_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
        C1 = K[selected_indices_tensor]  # (t, d)

        # 3. Compute beta
        if self.beta_method == 'zero':
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        else:  # 'nnls'
            n = queries.shape[0]
            inv_sqrt_d = (1.0 / d) ** 0.5
            scores32 = (queries @ K.T).to(torch.float32) * inv_sqrt_d  # (n, T)
            if attention_bias is not None:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32), scores32.shape
                )
                scores32 = scores32 + bias32
            max_scores = scores32.max(dim=1, keepdim=True)[0]
            exp_scores = torch.exp(scores32 - max_scores)  # (n, T)

            target = exp_scores.sum(dim=1)  # (n,)
            M = exp_scores[:, selected_indices_tensor]  # (n, t)
            B = self._nnls_pg(
                M, target, self.nnls_iters,
                self.nnls_lower_bound, self.nnls_upper_bound
            )
            beta32 = torch.log(B)

        beta = beta32.to(dtype_param)

        # 4. Compute C2
        C2 = self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method=self.c2_method,
            indices=selected_indices,
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale,
        )

        return C1, beta, C2, selected_indices
