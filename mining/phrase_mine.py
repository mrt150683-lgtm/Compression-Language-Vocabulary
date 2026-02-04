"""
Phrase mining pipeline: Extract n-grams and compute frequencies & PMI.

This module extracts candidate phrases (n-grams of length 2-6) from raw text,
computes their frequencies and Pointwise Mutual Information (PMI), and filters
them based on frequency thresholds and cohesion criteria.

Output: data/mined_phrases.jsonl with entries:
    {
        "phrase": "i was thinking about",
        "count": 8123,
        "pmi": 2.45
    }
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Set

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using whitespace tokenization")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not available, cannot load HF datasets")


# Common stopwords (English)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "her", "its",
    "our", "their", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "now"
}


def load_text_from_sources(input_sources: List[str]) -> List[str]:
    """
    Load text from files or Hugging Face datasets.

    Args:
        input_sources: List of paths (files or "hf:dataset_name:config:split")

    Returns:
        List of text strings
    """
    texts = []

    for source in input_sources:
        if source.startswith("hf:"):
            # Hugging Face dataset format: hf:dataset_name:config:split
            if not HAS_DATASETS:
                raise ValueError("datasets library required for HF dataset loading")

            parts = source[3:].split(":")
            if len(parts) == 3:
                dataset_name, config, split = parts
            elif len(parts) == 2:
                dataset_name, split = parts
                config = None
            else:
                raise ValueError(f"Invalid HF dataset format: {source}")

            print(f"Loading dataset {dataset_name} {config or ''} {split}...")
            if config:
                dataset = load_dataset(dataset_name, config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)

            # Extract text from the dataset
            if hasattr(dataset, 'column_names') and 'text' in dataset.column_names:
                texts.extend(dataset['text'])
            else:
                # Fallback: assume first column or concatenate all string columns
                for item in dataset:
                    if 'text' in item:
                        texts.append(item['text'])
                    else:
                        # Concatenate all string values
                        text_parts = [str(v) for v in item.values() if isinstance(v, str)]
                        texts.append(' '.join(text_parts))

        else:
            # Regular file path
            file_path = Path(source)
            texts.extend(load_text_from_file(file_path))

    return texts


def load_text_from_file(file_path: Path) -> List[str]:
    """
    Load text from a file (plain text or JSONL).

    Args:
        file_path: Path to input file

    Returns:
        List of text strings
    """
    texts = []

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Try to detect JSONL format (first line is JSON)
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{") and "text" in first_line:
            # JSONL format
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "text" in data:
                        texts.append(data["text"])
                except json.JSONDecodeError:
                    continue
        else:
            # Plain text format
            texts.append(f.read())

    return texts


def tokenize_text(text: str, tokenizer=None) -> List[str]:
    """
    Tokenize text using tokenizer or whitespace.
    
    Args:
        text: Input text
        tokenizer: Optional Hugging Face tokenizer
    
    Returns:
        List of tokens
    """
    if tokenizer is not None:
        # Use Hugging Face tokenizer
        tokens = tokenizer.tokenize(text)
        # Remove special tokens and subword prefixes
        tokens = [t.replace("##", "").replace("Ġ", "").strip() for t in tokens]
        tokens = [t for t in tokens if t]
    else:
        # Simple whitespace tokenization
        # Normalize whitespace and split
        text = re.sub(r'\s+', ' ', text.lower())
        tokens = text.split()
    
    return tokens


def extract_ngrams(tokens: List[str], min_n: int = 2, max_n: int = 6) -> List[str]:
    """
    Extract n-grams from tokenized text.
    
    Args:
        tokens: List of tokens
        min_n: Minimum n-gram length
        max_n: Maximum n-gram length
    
    Returns:
        List of n-gram strings (space-separated)
    """
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            ngrams.append(ngram)
    return ngrams


def is_stopword_only(phrase: str) -> bool:
    """
    Check if phrase contains only stopwords.
    
    Args:
        phrase: Phrase string
    
    Returns:
        True if phrase is only stopwords
    """
    words = phrase.lower().split()
    return all(word in STOPWORDS for word in words)


def compute_pmi(
    phrase: str,
    phrase_count: int,
    total_phrases: int,
    word_counts: Dict[str, int],
    total_words: int
) -> float:
    """
    Compute Pointwise Mutual Information for a phrase.
    
    PMI = log2(P(phrase) / (P(word1) * P(word2) * ...))
    Simplified: log2(count(phrase) * N / (count(word1) * count(word2) * ...))
    
    Args:
        phrase: The phrase to compute PMI for
        phrase_count: Frequency of the phrase
        total_phrases: Total number of phrases in corpus
        word_counts: Dictionary of word -> count
        total_words: Total number of words
    
    Returns:
        PMI score
    """
    import math
    
    if phrase_count == 0 or total_phrases == 0:
        return 0.0
    
    words = phrase.split()
    if len(words) < 2:
        return 0.0
    
    # Compute product of individual word probabilities
    word_prob_product = 1.0
    for word in words:
        word_count = word_counts.get(word.lower(), 1)
        word_prob = word_count / total_words
        word_prob_product *= word_prob
    
    if word_prob_product == 0:
        return 0.0
    
    # Phrase probability
    phrase_prob = phrase_count / total_phrases
    
    # PMI = log2(P(phrase) / product(P(words)))
    pmi = math.log2(phrase_prob / word_prob_product) if phrase_prob > 0 else 0.0
    
    return pmi


def mine_phrases(
    input_sources: List[str],
    output_path: Path,
    min_frequency: int = 50,
    min_n: int = 2,
    max_n: int = 6,
    tokenizer_name: str = None,
    max_phrases: int = None
) -> None:
    """
    Main phrase mining function.

    Args:
        input_sources: List of paths (files or "hf:dataset_name:config:split")
        output_path: Output path for mined_phrases.jsonl
        min_frequency: Minimum frequency threshold
        min_n: Minimum n-gram length
        max_n: Maximum n-gram length
        tokenizer_name: Optional Hugging Face tokenizer name
        max_phrases: Maximum number of phrases to output (None for all)
    """
    print(f"Mining phrases from {len(input_sources)} input source(s)...")

    # Load tokenizer if specified
    tokenizer = None
    if tokenizer_name and HAS_TRANSFORMERS:
        try:
            print(f"Loading tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
            print("Falling back to whitespace tokenization")

    # Load all texts from sources (files or HF datasets)
    print("Loading texts from sources...")
    all_texts = load_text_from_sources(input_sources)

    if not all_texts:
        raise ValueError("No texts loaded from sources")

    print(f"Processing {len(all_texts)} text sample(s)...")

    # Extract all n-grams
    phrase_counts = Counter()
    word_counts = Counter()
    total_phrases = 0
    total_words = 0

    for i, text in enumerate(all_texts):
        if i % 100 == 0:
            print(f"  Processing text {i+1}/{len(all_texts)}")
        texts = [text]  # Wrap in list for compatibility
        
        for text in texts:
            tokens = tokenize_text(text, tokenizer)
            if not tokens:
                continue
            
            # Count words
            for token in tokens:
                word_counts[token.lower()] += 1
                total_words += 1
            
            # Extract n-grams
            ngrams = extract_ngrams(tokens, min_n, max_n)
            for ngram in ngrams:
                phrase_counts[ngram.lower()] += 1
                total_phrases += 1
    
    print(f"Extracted {len(phrase_counts)} unique phrases")
    print(f"Total phrase occurrences: {total_phrases}")
    print(f"Total words: {total_words}")
    
    # Filter phrases
    print(f"Filtering phrases (min_frequency={min_frequency})...")
    filtered_phrases = []
    
    for phrase, count in phrase_counts.items():
        # Filter by frequency
        if count < min_frequency:
            continue
        
        # Filter stopword-only phrases
        if is_stopword_only(phrase):
            continue
        
        # Compute PMI
        pmi = compute_pmi(phrase, count, total_phrases, word_counts, total_words)
        
        filtered_phrases.append({
            "phrase": phrase,
            "count": count,
            "pmi": round(pmi, 4)
        })
    
    # Sort by count (descending)
    filtered_phrases.sort(key=lambda x: x["count"], reverse=True)

    # Apply max_phrases limit if specified
    if max_phrases is not None:
        filtered_phrases = filtered_phrases[:max_phrases]

    print(f"Filtered to {len(filtered_phrases)} phrases")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(filtered_phrases):
            # Add index field for phrase ID (required by load_phrase_index)
            entry_with_index = {"index": i, **entry}
            f.write(json.dumps(entry_with_index, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(filtered_phrases)} phrases to {output_path}")


def main():
    """CLI entry point for phrase mining."""
    parser = argparse.ArgumentParser(
        description="Mine phrases from raw text shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine from single file
  python mining/phrase_mine.py --input_files data/sample.txt --output data/mined_phrases.jsonl

  # Mine from directory
  python mining/phrase_mine.py --input_dir data/texts/ --min_freq 10

  # Mine from Hugging Face dataset
  python mining/phrase_mine.py --input_files "hf:wikitext:wikitext-103-raw-v1:test" --min_freq 5

  # Use Hugging Face tokenizer
  python mining/phrase_mine.py --input_files data/sample.txt --tokenizer Qwen/Qwen2-0.5B-Instruct
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Paths to input text files or HF datasets (plain text, JSONL with 'text' field, or 'hf:dataset_name:config:split')"
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing text files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/mined_phrases.jsonl",
        help="Output path for mined phrases (JSONL format, default: data/mined_phrases.jsonl)"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=50,
        help="Minimum frequency threshold (default: 50)"
    )
    parser.add_argument(
        "--min_ngram",
        type=int,
        default=2,
        help="Minimum n-gram length (default: 2)"
    )
    parser.add_argument(
        "--max_ngram",
        type=int,
        default=6,
        help="Maximum n-gram length (default: 6)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional Hugging Face tokenizer name (default: whitespace tokenization)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42, currently unused)"
    )
    parser.add_argument(
        "--max_phrases",
        type=int,
        default=None,
        help="Maximum number of phrases to output (default: None, output all)"
    )
    
    args = parser.parse_args()

    # Determine input sources (can be files or HF datasets)
    if args.input_files:
        input_sources = args.input_files  # Keep as strings
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            parser.error(f"Input directory not found: {input_dir}")
        # For directories, we need to expand to file paths
        input_sources = []
        for pattern in ["*.txt", "*.jsonl"]:
            input_sources.extend([str(p) for p in input_dir.glob(pattern)])

        if not input_sources:
            parser.error(f"No .txt or .jsonl files found in directory: {input_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate n-gram range
    if args.min_ngram < 1:
        parser.error("--min_ngram must be >= 1")
    if args.max_ngram < args.min_ngram:
        parser.error("--max_ngram must be >= --min_ngram")

    try:
        mine_phrases(
            input_sources=input_sources,
            output_path=output_path,
            min_frequency=args.min_freq,
            min_n=args.min_ngram,
            max_n=args.max_ngram,
            tokenizer_name=args.tokenizer,
            max_phrases=args.max_phrases
        )
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        exit(1)


if __name__ == "__main__":
    main()
