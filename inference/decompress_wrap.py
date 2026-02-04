"""
Decompression wrapper: Map CLV tokens back to canonical phrases.

For each <clv:####> token, map back to canonical phrase string.
Used to:
- Humanize logs
- Sanity-check semantics
- Reconstruct original text (approximately)

Usage:
    >>> from inference.decompress_wrap import decompress_text_with_clv
    >>> from inference.clv_tokenizer_adapter import load_tokenizer_with_clv
    >>>
    >>> tokenizer = load_tokenizer_with_clv("Qwen/Qwen2-0.5B-Instruct", codebook_size=8000)
    >>> clv_map = load_clv_map("artifacts/clv_map.json")
    >>>
    >>> compressed_text = "Hello <clv:0001> world <clv:0002>"
    >>> decompressed = decompress_text_with_clv(compressed_text, tokenizer, clv_map)
    >>> print(decompressed)  # "Hello [original phrase] world [original phrase]"
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")


def load_clv_map(clv_map_path: Path) -> Dict[str, int]:
    """
    Load CLV mapping from disk.
    
    Args:
        clv_map_path: Path to clv_map.json
    
    Returns:
        Dictionary mapping phrase -> code index
    """
    if not clv_map_path.exists():
        raise FileNotFoundError(f"CLV map not found: {clv_map_path}")
    
    with open(clv_map_path, "r", encoding="utf-8") as f:
        clv_map = json.load(f)
    
    if not isinstance(clv_map, dict):
        raise ValueError(f"CLV map must be a dictionary, got {type(clv_map)}")
    
    return clv_map


# ============================================================================
# LOSSLESS MODE: Phrase Index Loading
# ============================================================================

def load_phrase_index(phrase_index_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load phrase index from JSONL file (for lossless mode).
    
    Args:
        phrase_index_path: Path to phrase_index.jsonl
    
    Returns:
        Tuple of:
        - phrase_to_id: Dictionary mapping phrase -> phrase ID (int)
        - id_to_phrase: Dictionary mapping phrase ID -> phrase (str)
    """
    if not phrase_index_path.exists():
        raise FileNotFoundError(f"Phrase index not found: {phrase_index_path}")
    
    phrase_to_id = {}
    id_to_phrase = {}
    
    with open(phrase_index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Support both "index" and "row" keys for compatibility
                phrase_id = entry.get("index") or entry.get("row")
                phrase = entry.get("phrase")
                
                if phrase_id is not None and phrase:
                    phrase_id = int(phrase_id)
                    phrase_to_id[phrase] = phrase_id
                    id_to_phrase[phrase_id] = phrase
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    
    return phrase_to_id, id_to_phrase


def build_reverse_map(clv_map: Dict[str, int]) -> Dict[int, str]:
    """
    Build reverse mapping from code index to phrase.
    
    If multiple phrases map to the same code, uses the first one encountered.
    This should be rare if codebook is well-trained.
    
    Args:
        clv_map: Phrase -> code index mapping
    
    Returns:
        Code index -> phrase mapping
    """
    reverse_map = {}
    
    for phrase, code_idx in clv_map.items():
        # If code already exists, keep the first phrase (or could use most frequent)
        if code_idx not in reverse_map:
            reverse_map[code_idx] = phrase
    
    return reverse_map


def find_clv_tokens_in_text(text: str) -> List[Tuple[int, int, int]]:
    """
    Find all CLV tokens in text.
    
    Args:
        text: Input text string
    
    Returns:
        List of (start_pos, end_pos, code_index) tuples
    """
    # Pattern: <clv:####> where #### is 0-9 digits
    pattern = r'<clv:(\d{4})>'
    
    matches = []
    for match in re.finditer(pattern, text):
        start = match.start()
        end = match.end()
        code_str = match.group(1)
        code_idx = int(code_str)
        matches.append((start, end, code_idx))
    
    return matches


# ============================================================================
# LOSSLESS MODE: PID Token Detection
# ============================================================================

def find_pid_tokens_in_text(text: str) -> List[Tuple[int, int, int]]:
    """
    Find all PID tokens in text (lossless mode).
    
    Args:
        text: Input text string
    
    Returns:
        List of (start_pos, end_pos, phrase_id) tuples
    """
    # Pattern: <pid:######> where ###### is 0-9 digits (6 digits)
    pattern = r'<pid:(\d{6})>'
    
    matches = []
    for match in re.finditer(pattern, text):
        start = match.start()
        end = match.end()
        pid_str = match.group(1)
        phrase_id = int(pid_str)
        matches.append((start, end, phrase_id))
    
    return matches


def decompress_text_with_clv(
    compressed_text: str,
    tokenizer,
    clv_map: Dict[str, int],
    reverse_map: Optional[Dict[int, str]] = None
) -> str:
    """
    Decompress text by replacing CLV tokens with phrases.
    
    Args:
        compressed_text: Text containing <clv:####> tokens
        tokenizer: Tokenizer with CLV tokens (for validation)
        clv_map: Phrase -> code index mapping
        reverse_map: Optional pre-built reverse map (code -> phrase)
            If None, builds it from clv_map
    
    Returns:
        Decompressed text with CLV tokens replaced by phrases
    """
    if reverse_map is None:
        reverse_map = build_reverse_map(clv_map)
    
    # Find all CLV tokens
    matches = find_clv_tokens_in_text(compressed_text)
    
    if not matches:
        return compressed_text
    
    # Replace from end to start to preserve positions
    decompressed_text = compressed_text
    replacements = sorted(matches, key=lambda x: x[0], reverse=True)
    
    stats = {
        "total_clv_tokens": len(matches),
        "successful_replacements": 0,
        "failed_replacements": 0
    }
    
    for start, end, code_idx in replacements:
        # Lookup phrase
        if code_idx in reverse_map:
            phrase = reverse_map[code_idx]
            decompressed_text = decompressed_text[:start] + phrase + decompressed_text[end:]
            stats["successful_replacements"] += 1
        else:
            # Code not found in map - keep CLV token or use placeholder
            stats["failed_replacements"] += 1
            # Option: could replace with placeholder like "[UNKNOWN_CODE:1234]"
            pass
    
    return decompressed_text


# ============================================================================
# LOSSLESS MODE: Decompression Function
# ============================================================================

def decompress_text_lossless(
    compressed_text: str,
    id_to_phrase: Dict[int, str]
) -> str:
    """
    Decompress text by replacing PID tokens with exact phrases (lossless mode).
    
    Args:
        compressed_text: Text containing <pid:######> tokens
        id_to_phrase: Phrase ID -> phrase mapping
    
    Returns:
        Decompressed text with PID tokens replaced by exact original phrases
    """
    # Find all PID tokens
    matches = find_pid_tokens_in_text(compressed_text)
    
    if not matches:
        return compressed_text
    
    # Replace from end to start to preserve positions
    decompressed_text = compressed_text
    replacements = sorted(matches, key=lambda x: x[0], reverse=True)
    
    for start, end, phrase_id in replacements:
        # Lookup phrase
        if phrase_id in id_to_phrase:
            phrase = id_to_phrase[phrase_id]
            decompressed_text = decompressed_text[:start] + phrase + decompressed_text[end:]
        else:
            # Keep token if phrase not found (should not happen in lossless mode)
            pass
    
    return decompressed_text


def decompress_token_ids_with_clv(
    token_ids: List[int],
    tokenizer,
    clv_map: Dict[str, int],
    reverse_map: Optional[Dict[int, str]] = None
) -> str:
    """
    Decompress token IDs by replacing CLV token IDs with phrases.
    
    Args:
        token_ids: List of token IDs (may include CLV token IDs)
        tokenizer: Tokenizer with CLV tokens
        clv_map: Phrase -> code index mapping
        reverse_map: Optional pre-built reverse map
    
    Returns:
        Decompressed text string
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required")
    
    if reverse_map is None:
        reverse_map = build_reverse_map(clv_map)
    
    # Build token_id -> code_index mapping
    # Add parent directory to path for imports
    import sys
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from inference.clv_tokenizer_adapter import build_token_id_to_code_map, format_clv_token
    
    codebook_size = max(clv_map.values()) + 1 if clv_map else 0
    token_to_code = build_token_id_to_code_map(tokenizer, codebook_size)
    
    # Decode token IDs to text
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Replace CLV tokens in decoded text
    decompressed = decompress_text_with_clv(decoded_text, tokenizer, clv_map, reverse_map)
    
    return decompressed


def main():
    """CLI entry point for decompression."""
    parser = argparse.ArgumentParser(
        description="Decompress CLV tokens back to phrases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decompress from JSON file with token IDs
  python inference/decompress_wrap.py --input compressed.json
  
  # Decompress text directly
  python inference/decompress_wrap.py --text "Hello <clv:0001> world" --clv-map artifacts/clv_map.json
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file with compressed token IDs (JSON) or text"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text with CLV tokens (alternative to --input)"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Input text with CLV tokens (alias for --text)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for decompressed text (optional, prints to stdout if not provided)"
    )
    parser.add_argument(
        "--clv-map",
        dest="clv_map_hyphen",
        type=str,
        default=None,
        help="Path to CLV mapping (default: artifacts/clv_map.json)"
    )
    parser.add_argument(
        "--clv_map",
        dest="clv_map_underscore",
        type=str,
        default=None,
        help="Path to CLV mapping (alias for --clv-map)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Tokenizer name or path (default: Qwen/Qwen2-0.5B-Instruct)"
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=8000,
        help="Codebook size K (default: 8000)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["codebook", "lossless"],
        default="codebook",
        help="Decompression mode: 'codebook' (lossy, uses codebook clusters) or 'lossless' (exact phrase IDs, default: codebook)"
    )
    parser.add_argument(
        "--phrase-index",
        type=str,
        default=None,
        help="Path to phrase_index.jsonl (required for lossless mode, default: data/phrase_index.jsonl)"
    )
    parser.add_argument(
        "--phrase_index",
        dest="phrase_index_underscore",
        type=str,
        default=None,
        help="Path to phrase_index.jsonl (alias for --phrase-index)"
    )
    
    args = parser.parse_args()
    
    # Handle argument aliases
    clv_map_path_str = args.clv_map_underscore or args.clv_map_hyphen or "artifacts/clv_map.json"
    phrase_index_path_str = args.phrase_index_underscore or args.phrase_index or "data/phrase_index.jsonl"
    text_input = args.input_text or args.text
    
    # Add parent directory to path for imports
    import sys
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # ============================================================================
    # MODE-SPECIFIC SETUP
    # ============================================================================
    
    if args.mode == "lossless":
        # LOSSLESS MODE: Load phrase index
        phrase_index_path = Path(phrase_index_path_str)
        if not phrase_index_path.exists():
            parser.error(f"Phrase index not found: {phrase_index_path} (required for lossless mode)")
        
        phrase_to_id, id_to_phrase = load_phrase_index(phrase_index_path)
        print(f"Loaded phrase index: {len(phrase_to_id)} phrases (max ID: {max(id_to_phrase.keys()) if id_to_phrase else 0})")
        
    else:
        # CODEBOOK MODE: Load CLV map
        clv_map_path = Path(clv_map_path_str)
        if not clv_map_path.exists():
            parser.error(f"CLV map not found: {clv_map_path} (required for codebook mode)")
        
        clv_map = load_clv_map(clv_map_path)
        reverse_map = build_reverse_map(clv_map)
        print(f"Loaded CLV map: {len(clv_map)} phrases")
    
    # Determine input
    input_text = None
    token_ids = None
    
    if text_input:
        input_text = text_input
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {input_path}")
        
        # Try to load as JSON first
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                if "compressed_output" in data:
                    # Could be text or token IDs
                    compressed_output = data["compressed_output"]
                    if isinstance(compressed_output, list):
                        token_ids = compressed_output
                    else:
                        input_text = compressed_output
                elif "compressed_token_ids" in data:
                    token_ids = data["compressed_token_ids"]
                else:
                    input_text = str(data)
            elif isinstance(data, list):
                token_ids = data
            else:
                input_text = str(data)
        except json.JSONDecodeError:
            # Not JSON, treat as plain text
            with open(input_path, "r", encoding="utf-8") as f:
                input_text = f.read()
    else:
        parser.error("Must provide --text or --input")
    
    # ============================================================================
    # DECOMPRESSION (mode-specific)
    # ============================================================================
    
    if args.mode == "lossless":
        # LOSSLESS MODE: Decompress using exact phrase IDs
        if token_ids is not None:
            if not HAS_TRANSFORMERS:
                parser.error("transformers required for token ID decompression")
            
            from inference.clv_tokenizer_adapter import load_tokenizer_with_pid
            
            # Get max phrase ID from id_to_phrase
            max_phrase_id = max(id_to_phrase.keys()) if id_to_phrase else 0
            tokenizer = load_tokenizer_with_pid(args.tokenizer, max_phrase_id=max_phrase_id)
            
            # Decode token IDs to text
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            # Replace PID tokens with phrases
            decompressed = decompress_text_lossless(decoded_text, id_to_phrase)
        else:
            # Decompress text directly
            decompressed = decompress_text_lossless(input_text, id_to_phrase)
    
    else:
        # CODEBOOK MODE: Decompress using codebook clusters
        if token_ids is not None:
            if not HAS_TRANSFORMERS:
                parser.error("transformers required for token ID decompression")
            
            from inference.clv_tokenizer_adapter import load_tokenizer_with_clv
            tokenizer = load_tokenizer_with_clv(args.tokenizer, codebook_size=args.codebook_size)
            
            decompressed = decompress_token_ids_with_clv(
                token_ids, tokenizer, clv_map, reverse_map
            )
        else:
            # Decompress text directly
            if HAS_TRANSFORMERS:
                from inference.clv_tokenizer_adapter import load_tokenizer_with_clv
                tokenizer = load_tokenizer_with_clv(args.tokenizer, codebook_size=args.codebook_size)
            else:
                tokenizer = None  # Not needed for text-only decompression
            
            decompressed = decompress_text_with_clv(
                input_text, tokenizer, clv_map, reverse_map
            )
    
    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(decompressed)
        print(f"✓ Decompressed text saved to {args.output}")
    else:
        print(decompressed)


if __name__ == "__main__":
    main()
