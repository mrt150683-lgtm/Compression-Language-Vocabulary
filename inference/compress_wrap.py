"""
Compression wrapper: Runtime preprocessor for CLV phrase replacement.

Loads clv_map.json and performs longest-match phrase replacement:
- Scans text for phrase matches using longest-match algorithm
- Replaces matched spans with corresponding <clv:####> token
- Avoids overlapping matches; prefers longer spans
- Outputs compressed text or token ids with compression stats

Usage:
    >>> from inference.compress_wrap import compress_text_with_clv
    >>> from inference.clv_tokenizer_adapter import load_tokenizer_with_clv
    >>>
    >>> tokenizer = load_tokenizer_with_clv("Qwen/Qwen2-0.5B-Instruct", codebook_size=8000)
    >>> clv_map = load_clv_map("artifacts/clv_map.json")
    >>>
    >>> text = "i was thinking about this problem"
    >>> compressed_ids, stats = compress_text_with_clv(text, tokenizer, clv_map)
    >>> print(f"Compression ratio: {stats['compression_ratio']:.2%}")
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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


# ============================================================================
# LOSSLESS MODE: Phrase Matching (using phrase IDs)
# ============================================================================

def find_longest_matches_lossless(
    text: str,
    phrase_to_id: Dict[str, int],
    case_sensitive: bool = False
) -> List[Tuple[int, int, int, str]]:
    """
    Find longest-matching phrases in text using greedy longest-match algorithm (lossless mode).
    
    Args:
        text: Input text string
        phrase_to_id: Phrase -> phrase ID mapping
        case_sensitive: Whether matching is case-sensitive (default: False)
    
    Returns:
        List of (start_pos, end_pos, phrase_id, phrase) tuples, non-overlapping
        Sorted by start position
    """
    if not case_sensitive:
        text_lower = text.lower()
        # Create case-insensitive mapping
        phrase_to_id_lower = {phrase.lower(): pid for phrase, pid in phrase_to_id.items()}
        phrase_map_to_use = phrase_to_id_lower
        text_to_search = text_lower
    else:
        phrase_map_to_use = phrase_to_id
        text_to_search = text
    
    # Sort phrases by length (longest first) for longest-match
    phrases_sorted = sorted(phrase_map_to_use.keys(), key=len, reverse=True)
    
    matches = []
    used_positions = set()  # Track used character positions
    
    # Greedy longest-match: try longest phrases first
    for phrase in phrases_sorted:
        phrase_id = phrase_map_to_use[phrase]
        
        # Find all occurrences of this phrase
        start = 0
        while True:
            pos = text_to_search.find(phrase, start)
            if pos == -1:
                break
            
            end = pos + len(phrase)
            
            # Check if this position range overlaps with any existing match
            overlaps = False
            for used_start, used_end in used_positions:
                if not (end <= used_start or pos >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                # Add match
                matches.append((pos, end, phrase_id, phrase))
                # Mark positions as used
                for i in range(pos, end):
                    used_positions.add((i, i + 1))
            
            start = pos + 1
    
    # Sort matches by start position
    matches.sort(key=lambda x: x[0])
    
    return matches


def replace_phrases_with_pid(
    text: str,
    matches: List[Tuple[int, int, int, str]]
) -> Tuple[str, Dict[str, Any]]:
    """
    Replace matched phrases with PID tokens (lossless mode).
    
    Args:
        text: Original text
        matches: List of (start, end, phrase_id, phrase) tuples
    
    Returns:
        Tuple of (compressed_text, stats_dict)
    """
    if not matches:
        return text, {
            "original_length": len(text),
            "compressed_length": len(text),
            "num_replacements": 0,
            "compression_ratio": 1.0
        }
    
    # Build replacement map (process from end to start to preserve indices)
    replacements = sorted(matches, key=lambda x: x[0], reverse=True)
    
    compressed_text = text
    original_chars_saved = 0
    
    for start, end, phrase_id, phrase in replacements:
        # Format PID token
        pid_token = format_pid_token(phrase_id)
        
        # Replace
        compressed_text = compressed_text[:start] + pid_token + compressed_text[end:]
        
        # Track savings
        original_chars_saved += len(phrase) - len(pid_token)
    
    # Compute stats
    original_length = len(text)
    compressed_length = len(compressed_text)
    compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
    
    stats = {
        "original_length": original_length,
        "compressed_length": compressed_length,
        "num_replacements": len(matches),
        "chars_saved": original_chars_saved,
        "compression_ratio": compression_ratio,
        "matches": [
            {
                "phrase": phrase,
                "phrase_id": phrase_id,
                "position": (start, end)
            }
            for start, end, phrase_id, phrase in sorted(matches, key=lambda x: x[0])
        ]
    }
    
    return compressed_text, stats


# ============================================================================
# CODEBOOK MODE: Phrase Matching (existing implementation)
# ============================================================================

def find_longest_matches(
    text: str,
    clv_map: Dict[str, int],
    case_sensitive: bool = False
) -> List[Tuple[int, int, int, str]]:
    """
    Find longest-matching phrases in text using greedy longest-match algorithm.
    
    Args:
        text: Input text string
        clv_map: Phrase -> code index mapping
        case_sensitive: Whether matching is case-sensitive (default: False)
    
    Returns:
        List of (start_pos, end_pos, code_index, phrase) tuples, non-overlapping
        Sorted by start position
    """
    if not case_sensitive:
        text_lower = text.lower()
        # Create case-insensitive mapping
        clv_map_lower = {phrase.lower(): code_idx for phrase, code_idx in clv_map.items()}
        clv_map_to_use = clv_map_lower
        text_to_search = text_lower
    else:
        clv_map_to_use = clv_map
        text_to_search = text
    
    # Sort phrases by length (longest first) for longest-match
    phrases_sorted = sorted(clv_map_to_use.keys(), key=len, reverse=True)
    
    matches = []
    used_positions = set()  # Track used character positions
    
    # Greedy longest-match: try longest phrases first
    for phrase in phrases_sorted:
        code_idx = clv_map_to_use[phrase]
        
        # Find all occurrences of this phrase
        start = 0
        while True:
            pos = text_to_search.find(phrase, start)
            if pos == -1:
                break
            
            end = pos + len(phrase)
            
            # Check if this position range overlaps with any existing match
            overlaps = False
            for used_start, used_end in used_positions:
                if not (end <= used_start or pos >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                # Add match
                matches.append((pos, end, code_idx, phrase))
                # Mark positions as used
                for i in range(pos, end):
                    used_positions.add((i, i + 1))
            
            start = pos + 1
    
    # Sort matches by start position
    matches.sort(key=lambda x: x[0])
    
    return matches


def replace_phrases_with_clv(
    text: str,
    matches: List[Tuple[int, int, int, str]],
    codebook_size: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Replace matched phrases with CLV tokens.
    
    Args:
        text: Original text
        matches: List of (start, end, code_idx, phrase) tuples
        codebook_size: Codebook size (for formatting)
    
    Returns:
        Tuple of (compressed_text, stats_dict)
    """
    if not matches:
        return text, {
            "original_length": len(text),
            "compressed_length": len(text),
            "num_replacements": 0,
            "compression_ratio": 1.0
        }
    
    # Build replacement map (process from end to start to preserve indices)
    replacements = sorted(matches, key=lambda x: x[0], reverse=True)
    
    compressed_text = text
    original_chars_saved = 0
    
    for start, end, code_idx, phrase in replacements:
        # Format CLV token
        clv_token = f"<clv:{code_idx:04d}>"
        
        # Replace
        compressed_text = compressed_text[:start] + clv_token + compressed_text[end:]
        
        # Track savings
        original_chars_saved += len(phrase) - len(clv_token)
    
    # Compute stats
    original_length = len(text)
    compressed_length = len(compressed_text)
    compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
    
    stats = {
        "original_length": original_length,
        "compressed_length": compressed_length,
        "num_replacements": len(matches),
        "chars_saved": original_chars_saved,
        "compression_ratio": compression_ratio,
        "matches": [
            {
                "phrase": phrase,
                "code_index": code_idx,
                "position": (start, end)
            }
            for start, end, code_idx, phrase in sorted(matches, key=lambda x: x[0])
        ]
    }
    
    return compressed_text, stats


def compress_text_with_clv(
    text: str,
    tokenizer,
    clv_map: Dict[str, int],
    codebook_size: int,
    return_text: bool = False,
    case_sensitive: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compress text using CLV phrase replacement (codebook mode).
    
    Args:
        text: Input text string
        tokenizer: Tokenizer with CLV tokens (from clv_tokenizer_adapter)
        clv_map: Phrase -> code index mapping
        codebook_size: Number of CLV codes
        return_text: If True, return compressed text; if False, return token IDs
        case_sensitive: Whether matching is case-sensitive
    
    Returns:
        Tuple of:
        - compressed_output: Compressed text (if return_text) or token IDs (list)
        - stats_dict: Compression statistics
    """
    # Find phrase matches
    matches = find_longest_matches(text, clv_map, case_sensitive=case_sensitive)
    
    # Replace with CLV tokens
    compressed_text, text_stats = replace_phrases_with_clv(text, matches, codebook_size)
    
    if return_text:
        return compressed_text, text_stats
    
    # Tokenize compressed text
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required for tokenization")
    
    token_ids = tokenizer.encode(compressed_text, add_special_tokens=False)
    
    # Compute token-level stats
    original_token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    token_stats = {
        "original_tokens": len(original_token_ids),
        "compressed_tokens": len(token_ids),
        "token_compression_ratio": len(token_ids) / len(original_token_ids) if original_token_ids else 1.0,
        "tokens_saved": len(original_token_ids) - len(token_ids)
    }
    
    # Combine stats
    stats = {**text_stats, **token_stats}
    
    return token_ids, stats


# ============================================================================
# LOSSLESS MODE: Compression Function
# ============================================================================

def compress_text_lossless(
    text: str,
    tokenizer,
    phrase_to_id: Dict[str, int],
    return_text: bool = False,
    case_sensitive: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compress text using exact phrase IDs (lossless mode).
    
    Args:
        text: Input text string
        tokenizer: Tokenizer with PID tokens (from clv_tokenizer_adapter)
        phrase_to_id: Phrase -> phrase ID mapping
        return_text: If True, return compressed text; if False, return token IDs
        case_sensitive: Whether matching is case-sensitive
    
    Returns:
        Tuple of:
        - compressed_output: Compressed text (if return_text) or token IDs (list)
        - stats_dict: Compression statistics
    """
    # Find phrase matches using phrase IDs
    matches = find_longest_matches_lossless(text, phrase_to_id, case_sensitive=case_sensitive)
    
    # Replace with PID tokens
    compressed_text, text_stats = replace_phrases_with_pid(text, matches)
    
    if return_text:
        return compressed_text, text_stats
    
    # Tokenize compressed text
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required for tokenization")
    
    token_ids = tokenizer.encode(compressed_text, add_special_tokens=False)
    
    # Compute token-level stats
    original_token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    token_stats = {
        "original_tokens": len(original_token_ids),
        "compressed_tokens": len(token_ids),
        "token_compression_ratio": len(token_ids) / len(original_token_ids) if original_token_ids else 1.0,
        "tokens_saved": len(original_token_ids) - len(token_ids)
    }
    
    # Combine stats
    stats = {**text_stats, **token_stats}
    
    return token_ids, stats


def main():
    """CLI entry point for compression."""
    parser = argparse.ArgumentParser(
        description="Compress text using CLV phrase replacement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress text string
  python inference/compress_wrap.py --text "i was thinking about this" --clv-map artifacts/clv_map.json
  
  # Compress from file
  python inference/compress_wrap.py --input-file data/sample.txt --output compressed.json
  
  # Get stats only
  python inference/compress_wrap.py --text "some text" --stats-only
        """
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text to compress"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Input text to compress (alias for --text)"
    )
    parser.add_argument(
        "--input-file",
        dest="input_file_hyphen",
        type=str,
        default=None,
        help="Input file path (alternative to --text)"
    )
    parser.add_argument(
        "--input_file",
        dest="input_file_underscore",
        type=str,
        default=None,
        help="Input file path (alias for --input-file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for compressed tokens (JSON)"
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
        "--stats-only",
        action="store_true",
        help="Only output compression statistics"
    )
    parser.add_argument(
        "--return-text",
        action="store_true",
        help="Return compressed text instead of token IDs"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive phrase matching"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["codebook", "lossless"],
        default="codebook",
        help="Compression mode: 'codebook' (lossy, uses codebook clusters) or 'lossless' (exact phrase IDs, default: codebook)"
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
    
    if not HAS_TRANSFORMERS:
        parser.error("transformers library required")
    
    # Handle argument aliases
    clv_map_path_str = args.clv_map_underscore or args.clv_map_hyphen or "artifacts/clv_map.json"
    phrase_index_path_str = args.phrase_index_underscore or args.phrase_index or "data/phrase_index.jsonl"
    text_input = args.input_text or args.text
    input_file_path = args.input_file_underscore or args.input_file_hyphen
    
    # Add parent directory to path for imports
    import sys
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from inference.clv_tokenizer_adapter import load_tokenizer_with_clv, load_tokenizer_with_pid
    
    # ============================================================================
    # MODE-SPECIFIC SETUP
    # ============================================================================
    
    if args.mode == "lossless":
        # LOSSLESS MODE: Load phrase index
        phrase_index_path = Path(phrase_index_path_str)
        if not phrase_index_path.exists():
            parser.error(f"Phrase index not found: {phrase_index_path} (required for lossless mode)")
        
        phrase_to_id, id_to_phrase = load_phrase_index(phrase_index_path)
        max_phrase_id = max(id_to_phrase.keys()) if id_to_phrase else 0
        print(f"Loaded phrase index: {len(phrase_to_id)} phrases (max ID: {max_phrase_id})")
        
        # Load tokenizer with PID tokens
        tokenizer = load_tokenizer_with_pid(
            args.tokenizer,
            max_phrase_id=max_phrase_id
        )
        print(f"✓ Tokenizer ready with PID tokens (0-{max_phrase_id})")
        
    else:
        # CODEBOOK MODE: Load CLV map
        clv_map_path = Path(clv_map_path_str)
        if not clv_map_path.exists():
            parser.error(f"CLV map not found: {clv_map_path} (required for codebook mode)")
        
        clv_map = load_clv_map(clv_map_path)
        print(f"Loaded CLV map: {len(clv_map)} phrases")
        
        # Load tokenizer with CLV tokens
        tokenizer = load_tokenizer_with_clv(
            args.tokenizer,
            codebook_size=args.codebook_size
        )
        print(f"✓ Tokenizer ready with CLV tokens (0-{args.codebook_size-1})")
    
    # Get input text
    if text_input:
        text = text_input
    elif input_file_path:
        input_path = Path(input_file_path)
        if not input_path.exists():
            parser.error(f"Input file not found: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        parser.error("Must provide --text or --input-file")
    
    # ============================================================================
    # COMPRESSION (mode-specific)
    # ============================================================================
    
    if args.mode == "lossless":
        # LOSSLESS MODE: Use exact phrase IDs
        compressed_output, stats = compress_text_lossless(
            text,
            tokenizer,
            phrase_to_id,
            return_text=args.return_text,
            case_sensitive=args.case_sensitive
        )
    else:
        # CODEBOOK MODE: Use codebook clusters
        compressed_output, stats = compress_text_with_clv(
            text,
            tokenizer,
            clv_map,
            codebook_size=args.codebook_size,
            return_text=args.return_text,
            case_sensitive=args.case_sensitive
        )
    
    # Output
    if args.stats_only:
        print(json.dumps(stats, indent=2))
    else:
        output_data = {
            "compressed_output": compressed_output,
            "stats": stats
        }
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        
        # Print summary
        print()
        print("Compression Summary:")
        print(f"  Original length: {stats['original_length']} chars")
        print(f"  Compressed length: {stats['compressed_length']} chars")
        if 'original_tokens' in stats:
            print(f"  Original tokens: {stats['original_tokens']}")
            print(f"  Compressed tokens: {stats['compressed_tokens']}")
        print(f"  Replacements: {stats['num_replacements']}")
        print(f"  Compression ratio: {stats['compression_ratio']:.2%}")
        if 'token_compression_ratio' in stats:
            print(f"  Token compression ratio: {stats['token_compression_ratio']:.2%}")


if __name__ == "__main__":
    main()
