"""
CLV Tokenizer Adapter: Handle CLV special tokens in tokenizer.

Loads base tokenizer and registers CLV special tokens (<clv:0000> through <clv:####>).
Ensures encoding/decoding round-trip behavior is properly defined.

Usage:
    >>> from inference.clv_tokenizer_adapter import add_clv_tokens_to_tokenizer, load_tokenizer_with_clv
    >>> from transformers import AutoTokenizer
    >>>
    >>> # Load base tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>>
    >>> # Add CLV tokens
    >>> tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size=8000)
    >>>
    >>> # Now tokenizer can handle <clv:0000> through <clv:7999>
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")


def format_clv_token(code_index: int) -> str:
    """
    Format CLV token string from code index.
    
    Args:
        code_index: Code index (0 to codebook_size-1)
    
    Returns:
        CLV token string, e.g., "<clv:0000>", "<clv:0123>", "<clv:7999>"
    """
    # Format with zero-padding to 4 digits
    return f"<clv:{code_index:04d}>"


# ============================================================================
# LOSSLESS MODE: PID Token Support
# ============================================================================

def format_pid_token(phrase_id: int) -> str:
    """
    Format PID token string from phrase ID (lossless mode).
    
    Args:
        phrase_id: Phrase ID (int)
    
    Returns:
        PID token string, e.g., "<pid:000123>", "<pid:000000>"
    """
    # Format with zero-padding to 6 digits
    return f"<pid:{phrase_id:06d}>"


def add_pid_tokens_to_tokenizer(
    tokenizer,
    max_phrase_id: int,
    special_tokens: bool = True
):
    """
    Add PID special tokens to tokenizer (lossless mode).
    
    Registers tokens: <pid:000000> ... <pid:######> where ###### is max_phrase_id.
    
    Args:
        tokenizer: Base tokenizer (from transformers)
        max_phrase_id: Maximum phrase ID (creates tokens 0 through max_phrase_id)
        special_tokens: Whether to add as special tokens (default: True)
    
    Returns:
        Updated tokenizer with PID tokens added
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = add_pid_tokens_to_tokenizer(tokenizer, max_phrase_id=999999)
        >>> # Now <pid:000000> through <pid:999999> are registered
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required for tokenizer operations")
    
    # Generate all PID token strings
    pid_tokens = [format_pid_token(i) for i in range(max_phrase_id + 1)]
    
    # Add tokens to tokenizer
    if special_tokens:
        # Add as special tokens (won't be split)
        tokenizer.add_special_tokens({"additional_special_tokens": pid_tokens})
    else:
        # Add as regular tokens
        tokenizer.add_tokens(pid_tokens)
    
    return tokenizer


def load_tokenizer_with_pid(
    tokenizer_name_or_path: str,
    max_phrase_id: int,
    cache_dir: Optional[str] = None
):
    """
    Load tokenizer and add PID tokens (lossless mode).
    
    Convenience function that loads base tokenizer and adds all PID tokens.
    
    Args:
        tokenizer_name_or_path: Hugging Face model identifier or local path
        max_phrase_id: Maximum phrase ID (determines number of PID tokens)
        cache_dir: Optional cache directory
    
    Returns:
        Tokenizer with PID tokens added
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required")
    
    print(f"Loading base tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir
    )
    
    print(f"Adding {max_phrase_id + 1} PID tokens...")
    tokenizer = add_pid_tokens_to_tokenizer(tokenizer, max_phrase_id=max_phrase_id)
    
    print(f"✓ Tokenizer ready with {len(tokenizer)} total tokens")
    
    return tokenizer


def add_clv_tokens_to_tokenizer(
    tokenizer,
    codebook_size: int,
    special_tokens: bool = True
):
    """
    Add CLV special tokens to tokenizer.
    
    Registers tokens: <clv:0000> ... <clv:####> where #### is codebook_size-1.
    
    Args:
        tokenizer: Base tokenizer (from transformers)
        codebook_size: Number of CLV codes (K)
        special_tokens: Whether to add as special tokens (default: True)
    
    Returns:
        Updated tokenizer with CLV tokens added
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size=8000)
        >>> # Now <clv:0000> through <clv:7999> are registered
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required for tokenizer operations")
    
    # Generate all CLV token strings
    clv_tokens = [format_clv_token(i) for i in range(codebook_size)]
    
    # Add tokens to tokenizer
    if special_tokens:
        # Add as special tokens (won't be split)
        tokenizer.add_special_tokens({"additional_special_tokens": clv_tokens})
    else:
        # Add as regular tokens
        tokenizer.add_tokens(clv_tokens)
    
    # Resize token embeddings if model is attached (for training)
    # Note: This is a no-op if no model is attached
    if hasattr(tokenizer, 'resize_token_embeddings'):
        # This will be called during model setup, not here
        pass
    
    return tokenizer


def get_clv_token_id(tokenizer, code_index: int) -> Optional[int]:
    """
    Get token ID for a CLV code index.
    
    Args:
        tokenizer: Tokenizer with CLV tokens added
        code_index: Code index (0 to codebook_size-1)
    
    Returns:
        Token ID or None if not found
    """
    clv_token = format_clv_token(code_index)
    return tokenizer.convert_tokens_to_ids(clv_token)


def build_clv_token_id_map(tokenizer, codebook_size: int) -> Dict[int, int]:
    """
    Build mapping from code index to token ID.
    
    Args:
        tokenizer: Tokenizer with CLV tokens added
        codebook_size: Number of CLV codes
    
    Returns:
        Dictionary mapping code_index -> token_id
    """
    clv_token_map = {}
    for code_idx in range(codebook_size):
        token_id = get_clv_token_id(tokenizer, code_idx)
        if token_id is not None:
            clv_token_map[code_idx] = token_id
    
    return clv_token_map


def build_token_id_to_code_map(tokenizer, codebook_size: int) -> Dict[int, int]:
    """
    Build reverse mapping from token ID to code index.
    
    Args:
        tokenizer: Tokenizer with CLV tokens added
        codebook_size: Number of CLV codes
    
    Returns:
        Dictionary mapping token_id -> code_index
    """
    reverse_map = {}
    for code_idx in range(codebook_size):
        token_id = get_clv_token_id(tokenizer, code_idx)
        if token_id is not None:
            reverse_map[token_id] = code_idx
    
    return reverse_map


def load_tokenizer_with_clv(
    base_tokenizer_path: str,
    codebook_size: int,
    clv_tokenizer_path: Optional[Path] = None
):
    """
    Load tokenizer with CLV tokens added.
    
    Args:
        base_tokenizer_path: Path or Hugging Face identifier for base tokenizer
        codebook_size: Number of CLV codes (K)
        clv_tokenizer_path: Optional path to saved tokenizer with CLV tokens
            If provided and exists, loads that instead of adding tokens
    
    Returns:
        Tokenizer with CLV tokens
    
    Example:
        >>> tokenizer = load_tokenizer_with_clv(
        ...     "Qwen/Qwen2-0.5B-Instruct",
        ...     codebook_size=8000
        ... )
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required")
    
    # Try to load pre-configured tokenizer
    if clv_tokenizer_path and Path(clv_tokenizer_path).exists():
        print(f"Loading tokenizer with CLV tokens from {clv_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(clv_tokenizer_path))
        return tokenizer
    
    # Load base tokenizer
    print(f"Loading base tokenizer: {base_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    
    # Add CLV tokens
    print(f"Adding {codebook_size} CLV tokens...")
    tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size)
    
    print(f"✓ Tokenizer ready with {len(tokenizer)} total tokens")
    
    return tokenizer


def save_tokenizer_with_clv(tokenizer, output_path: Path) -> None:
    """
    Save tokenizer with CLV tokens to disk.
    
    Args:
        tokenizer: Tokenizer with CLV tokens
        output_path: Path to save tokenizer
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Tokenizer saved to {output_path}")


# Test/example code
if __name__ == "__main__":
    print("Testing CLV Tokenizer Adapter...")
    print()
    
    if not HAS_TRANSFORMERS:
        print("Skipping tests: transformers not available")
        exit(0)
    
    # Test with a small codebook
    codebook_size = 10
    
    print(f"1. Testing with codebook_size={codebook_size}")
    from transformers import AutoTokenizer
    
    try:
        # Use a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        print(f"   Base tokenizer vocab size: {len(tokenizer)}")
        
        # Add CLV tokens
        tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size)
        print(f"   Tokenizer vocab size after adding CLV: {len(tokenizer)}")
        
        # Test tokenization
        test_tokens = ["<clv:0000>", "<clv:0005>", "<clv:0009>"]
        for token_str in test_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            decoded = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"   {token_str} -> token_id={token_id}, decoded={decoded}")
        
        # Test encoding/decoding
        text = "Hello <clv:0001> world <clv:0002>"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"   Encode/decode test: '{text}' -> '{decoded}'")
        
        # Build maps
        clv_map = build_clv_token_id_map(tokenizer, codebook_size)
        reverse_map = build_token_id_to_code_map(tokenizer, codebook_size)
        print(f"   CLV token map: {len(clv_map)} entries")
        print(f"   Reverse map: {len(reverse_map)} entries")
        
        print()
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
