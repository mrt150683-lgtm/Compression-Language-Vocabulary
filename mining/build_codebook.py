"""
Build CLV codebook: Cluster phrase embeddings into discrete codes.

This module takes phrase embeddings and runs k-means (or FAISS k-means)
clustering to create a discrete codebook of K vectors.

Output: artifacts/clv_codebook.npy with shape (K, code_dim)
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not available, will use sklearn KMeans")

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    if not HAS_FAISS:
        raise ImportError("Neither faiss nor sklearn available. Install at least one.")


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Load phrase embeddings from disk.
    
    Args:
        embeddings_path: Path to phrase_embeddings.npy
    
    Returns:
        Embedding matrix (N x D)
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: shape {embeddings.shape}")
    
    # Ensure float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    return embeddings


def build_codebook_faiss(
    embeddings: np.ndarray,
    codebook_size: int,
    seed: int = 42
) -> np.ndarray:
    """
    Build codebook using FAISS k-means.
    
    Args:
        embeddings: Embedding matrix (N x D)
        codebook_size: Number of codes (K)
        seed: Random seed
    
    Returns:
        Codebook matrix (K x D)
    """
    print(f"Building codebook with FAISS k-means (K={codebook_size})...")
    
    n_samples, dim = embeddings.shape
    
    # FAISS k-means
    kmeans = faiss.Kmeans(
        dim,
        codebook_size,
        niter=100,
        verbose=True,
        seed=seed,
        gpu=False  # CPU-only for compatibility
    )
    
    # Train
    kmeans.train(embeddings)
    
    # Get centroids
    codebook = kmeans.centroids
    
    print(f"✓ Codebook built: shape {codebook.shape}")
    
    return codebook.astype(np.float32)


def build_codebook_sklearn(
    embeddings: np.ndarray,
    codebook_size: int,
    seed: int = 42
) -> np.ndarray:
    """
    Build codebook using sklearn KMeans.
    
    Args:
        embeddings: Embedding matrix (N x D)
        codebook_size: Number of codes (K)
        seed: Random seed
    
    Returns:
        Codebook matrix (K x D)
    """
    print(f"Building codebook with sklearn KMeans (K={codebook_size})...")
    
    n_samples, dim = embeddings.shape
    
    # sklearn KMeans
    kmeans = KMeans(
        n_clusters=codebook_size,
        random_state=seed,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    
    # Train
    kmeans.fit(embeddings)
    
    # Get centroids
    codebook = kmeans.cluster_centers_
    
    print(f"✓ Codebook built: shape {codebook.shape}")
    
    return codebook.astype(np.float32)


def build_codebook(
    embeddings: np.ndarray,
    codebook_size: int,
    code_dim: int,
    method: str = "auto",
    seed: int = 42
) -> np.ndarray:
    """
    Build codebook via clustering.
    
    Args:
        embeddings: Embedding matrix (N x D)
        codebook_size: Number of codes (K)
        code_dim: Dimension of code vectors (should match embedding dim)
        method: Clustering method ("auto", "kmeans", "faiss_kmeans")
        seed: Random seed for reproducibility
    
    Returns:
        Codebook matrix (K x code_dim)
    """
    n_samples, dim = embeddings.shape
    
    # Validate dimensions
    if dim != code_dim:
        print(f"Warning: Embedding dimension ({dim}) != code_dim ({code_dim})")
        print(f"  Using embedding dimension: {dim}")
        code_dim = dim
    
    if codebook_size > n_samples:
        raise ValueError(
            f"Codebook size ({codebook_size}) > number of samples ({n_samples})"
        )
    
    # Choose method
    if method == "auto":
        if HAS_FAISS:
            method = "faiss_kmeans"
        elif HAS_SKLEARN:
            method = "kmeans"
        else:
            raise RuntimeError("No clustering method available")
    
    # Build codebook
    if method == "faiss_kmeans":
        if not HAS_FAISS:
            raise ImportError("faiss not available, use method='kmeans'")
        codebook = build_codebook_faiss(embeddings, codebook_size, seed)
    elif method == "kmeans":
        if not HAS_SKLEARN:
            raise ImportError("sklearn not available, use method='faiss_kmeans'")
        codebook = build_codebook_sklearn(embeddings, codebook_size, seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return codebook


def save_codebook(codebook: np.ndarray, output_path: Path) -> None:
    """
    Save codebook to disk.
    
    Args:
        codebook: Codebook matrix (K x code_dim)
        output_path: Path to save .npy file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, codebook)
    print(f"✓ Codebook saved to {output_path}")


def main():
    """CLI entry point for codebook building."""
    parser = argparse.ArgumentParser(
        description="Build CLV codebook from phrase embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build codebook with default settings
  python mining/build_codebook.py --embeddings data/phrase_embeddings.npy --k 8000
  
  # Use FAISS (if available)
  python mining/build_codebook.py --embeddings data/phrase_embeddings.npy --k 8000 --method faiss_kmeans
  
  # Use sklearn
  python mining/build_codebook.py --embeddings data/phrase_embeddings.npy --k 8000 --method kmeans
        """
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Input path to phrase embeddings (.npy, default: data/phrase_embeddings.npy)"
    )
    parser.add_argument(
        "--input_embeddings",
        type=str,
        default=None,
        help="Input path to phrase embeddings (.npy, alias for --embeddings)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for codebook (.npy, default: artifacts/clv_codebook.npy)"
    )
    parser.add_argument(
        "--output_codebook",
        type=str,
        default=None,
        help="Output path for codebook (.npy, alias for --output)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8000,
        help="Number of codes K (default: 8000)"
    )
    parser.add_argument(
        "--code_dim",
        type=int,
        default=256,
        help="Dimension of code vectors (default: 256, auto-detected from embeddings)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "faiss_kmeans"],
        help="Clustering method (default: auto - uses faiss if available, else sklearn)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Handle argument aliases (prefer new names, fall back to old names, then defaults)
    embeddings_path_str = args.input_embeddings or args.embeddings or "data/phrase_embeddings.npy"
    output_path_str = args.output_codebook or args.output or "artifacts/clv_codebook.npy"
    
    # Validate inputs
    embeddings_path = Path(embeddings_path_str)
    if not embeddings_path.exists():
        parser.error(f"Embeddings file not found: {embeddings_path}")
    
    output_path = Path(output_path_str)
    
    if args.k < 1:
        parser.error("--k must be >= 1")
    
    if args.code_dim < 1:
        parser.error("--code_dim must be >= 1")
    
    # Set random seed
    np.random.seed(args.seed)
    # Note: FAISS doesn't have a direct seed_random function
    # The seed is passed to the Kmeans constructor
    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = load_embeddings(embeddings_path)
    print()
    
    # Build codebook
    codebook = build_codebook(
        embeddings=embeddings,
        codebook_size=args.k,
        code_dim=args.code_dim,
        method=args.method,
        seed=args.seed
    )
    
    print()
    
    # Save codebook
    save_codebook(codebook, output_path)
    
    print()
    print(f"✓ Codebook built: {codebook.shape} -> {output_path}")


if __name__ == "__main__":
    main()
