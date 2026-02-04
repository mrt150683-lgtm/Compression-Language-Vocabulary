"""
Build CLV mapping: Map phrases to codebook indices.

For each phrase, find the nearest codebook vector and create a mapping
from phrase string to code index.

Output: artifacts/clv_map.json
    {
        "i was thinking about": 123,
        "i want to talk about": 456,
        ...
    }
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not available, will use numpy for nearest neighbor search")


def load_codebook(codebook_path: Path) -> np.ndarray:
    """
    Load codebook from disk.
    
    Args:
        codebook_path: Path to clv_codebook.npy
    
    Returns:
        Codebook matrix (K x code_dim)
    """
    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook file not found: {codebook_path}")
    
    codebook = np.load(codebook_path)
    print(f"Loaded codebook: shape {codebook.shape}")
    
    # Ensure float32
    if codebook.dtype != np.float32:
        codebook = codebook.astype(np.float32)
    
    return codebook


def load_embeddings_and_index(
    embeddings_path: Path,
    index_path: Path
) -> tuple[np.ndarray, Dict[int, str]]:
    """
    Load embeddings and phrase index.
    
    Args:
        embeddings_path: Path to phrase_embeddings.npy
        index_path: Path to phrase_index.jsonl
    
    Returns:
        Tuple of (embeddings matrix, index mapping row -> phrase)
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    print(f"Loaded embeddings: shape {embeddings.shape}")
    
    # Load phrase index
    phrase_index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Support both "index" and "row" keys (for compatibility)
                idx = entry.get("index") or entry.get("row")
                phrase = entry.get("phrase")
                if idx is not None and phrase:
                    phrase_index[idx] = phrase
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(phrase_index)} phrase mappings")
    
    return embeddings, phrase_index


def find_nearest_codes_faiss(
    embeddings: np.ndarray,
    codebook: np.ndarray
) -> np.ndarray:
    """
    Find nearest codebook index for each embedding using FAISS.
    
    Args:
        embeddings: Embedding matrix (N x D)
        codebook: Codebook matrix (K x D)
    
    Returns:
        Array of code indices (N,)
    """
    print("Finding nearest codes using FAISS...")
    
    n_samples, dim = embeddings.shape
    k, code_dim = codebook.shape
    
    if dim != code_dim:
        raise ValueError(f"Dimension mismatch: embeddings {dim} != codebook {code_dim}")
    
    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(codebook)
    
    # Search
    distances, indices = index.search(embeddings, 1)
    
    # Return indices (squeeze to 1D)
    return indices.flatten().astype(np.int32)


def find_nearest_codes_numpy(
    embeddings: np.ndarray,
    codebook: np.ndarray
) -> np.ndarray:
    """
    Find nearest codebook index for each embedding using numpy.
    
    Args:
        embeddings: Embedding matrix (N x D)
        codebook: Codebook matrix (K x D)
    
    Returns:
        Array of code indices (N,)
    """
    print("Finding nearest codes using numpy (slower, but no faiss required)...")
    
    n_samples, dim = embeddings.shape
    k, code_dim = codebook.shape
    
    if dim != code_dim:
        raise ValueError(f"Dimension mismatch: embeddings {dim} != codebook {code_dim}")
    
    # Compute L2 distances
    # embeddings: (N, D), codebook: (K, D)
    # distances: (N, K)
    distances = np.sum((embeddings[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
    
    # Find nearest (argmin)
    indices = np.argmin(distances, axis=1)
    
    return indices.astype(np.int32)


def build_clv_map(
    embeddings: np.ndarray,
    phrase_index: Dict[int, str],
    codebook: np.ndarray,
    use_faiss: bool = True
) -> Dict[str, int]:
    """
    Build phrase -> code index mapping.
    
    Args:
        embeddings: Embedding matrix (N x D)
        phrase_index: Mapping from row index to phrase string
        codebook: Codebook matrix (K x D)
        use_faiss: Whether to use FAISS (faster) or numpy
    
    Returns:
        Dictionary mapping phrase -> code index
    """
    print(f"Building CLV map for {len(phrase_index)} phrases...")
    
    # Find nearest codes
    if use_faiss and HAS_FAISS:
        code_indices = find_nearest_codes_faiss(embeddings, codebook)
    else:
        code_indices = find_nearest_codes_numpy(embeddings, codebook)
    
    # Build mapping
    clv_map = {}
    for idx, phrase in phrase_index.items():
        if idx < len(code_indices):
            code_idx = int(code_indices[idx])
            clv_map[phrase] = code_idx
    
    # Handle duplicates: if multiple phrases map to same code, keep all
    # (This is expected behavior - multiple phrases can share a code)
    
    print(f"✓ Built mapping: {len(clv_map)} phrases -> codes")
    
    # Print some statistics
    code_usage = {}
    for phrase, code_idx in clv_map.items():
        code_usage[code_idx] = code_usage.get(code_idx, 0) + 1
    
    unique_codes = len(code_usage)
    print(f"  Unique codes used: {unique_codes}/{len(codebook)}")
    if code_usage:
        max_usage = max(code_usage.values())
        avg_usage = sum(code_usage.values()) / len(code_usage)
        print(f"  Max phrases per code: {max_usage}")
        print(f"  Avg phrases per code: {avg_usage:.2f}")
    
    return clv_map


def save_clv_map(clv_map: Dict[str, int], output_path: Path) -> None:
    """
    Save CLV mapping to disk.
    
    Args:
        clv_map: Dictionary mapping phrase -> code index
        output_path: Path to save .json file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clv_map, f, indent=2, ensure_ascii=False)
    print(f"✓ CLV map saved to {output_path}")


def main():
    """CLI entry point for CLV map building."""
    parser = argparse.ArgumentParser(
        description="Build CLV mapping from phrases to codebook indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build CLV map with default settings
  python mining/build_clv_map.py \
    --embeddings data/phrase_embeddings.npy \
    --index data/phrase_index.jsonl \
    --codebook artifacts/clv_codebook.npy
  
  # Force numpy (no faiss)
  python mining/build_clv_map.py \
    --embeddings data/phrase_embeddings.npy \
    --index data/phrase_index.jsonl \
    --codebook artifacts/clv_codebook.npy \
    --no_faiss
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
        "--index",
        type=str,
        default=None,
        help="Input path to phrase index (.jsonl, default: data/phrase_index.jsonl)"
    )
    parser.add_argument(
        "--phrase_index",
        type=str,
        default=None,
        help="Input path to phrase index (.jsonl, alias for --index)"
    )
    parser.add_argument(
        "--codebook",
        type=str,
        default="artifacts/clv_codebook.npy",
        help="Input path to codebook (.npy, default: artifacts/clv_codebook.npy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/clv_map.json",
        help="Output path for CLV mapping (.json, default: artifacts/clv_map.json)"
    )
    parser.add_argument(
        "--no_faiss",
        action="store_true",
        help="Force use of numpy instead of FAISS (slower but no dependencies)"
    )
    
    args = parser.parse_args()
    
    # Handle argument aliases (prefer new names, fall back to old names, then defaults)
    embeddings_path_str = args.input_embeddings or args.embeddings or "data/phrase_embeddings.npy"
    index_path_str = args.phrase_index or args.index or "data/phrase_index.jsonl"
    
    # Validate inputs
    embeddings_path = Path(embeddings_path_str)
    index_path = Path(index_path_str)
    codebook_path = Path(args.codebook)
    
    if not embeddings_path.exists():
        parser.error(f"Embeddings file not found: {embeddings_path}")
    if not index_path.exists():
        parser.error(f"Index file not found: {index_path}")
    if not codebook_path.exists():
        parser.error(f"Codebook file not found: {codebook_path}")
    
    output_path = Path(args.output)
    
    # Load inputs
    print(f"Loading codebook from {codebook_path}...")
    codebook = load_codebook(codebook_path)
    print()
    
    print(f"Loading embeddings and index...")
    embeddings, phrase_index = load_embeddings_and_index(embeddings_path, index_path)
    print()
    
    # Validate dimensions
    if embeddings.shape[0] != len(phrase_index):
        print(f"Warning: Embedding count ({embeddings.shape[0]}) != phrase index count ({len(phrase_index)})")
    
    # Build mapping
    use_faiss = not args.no_faiss and HAS_FAISS
    clv_map = build_clv_map(
        embeddings=embeddings,
        phrase_index=phrase_index,
        codebook=codebook,
        use_faiss=use_faiss
    )
    
    print()
    
    # Save mapping
    save_clv_map(clv_map, output_path)
    
    print()
    print(f"✓ CLV map built: {len(clv_map)} phrases -> {output_path}")


if __name__ == "__main__":
    main()
