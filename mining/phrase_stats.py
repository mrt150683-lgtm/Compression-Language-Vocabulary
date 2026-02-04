"""
Phrase statistics: Summarize mined phrases for codebook size decisions.

This module analyzes mined phrases to produce:
- Top phrases by frequency/PMI
- Coverage statistics
- Histograms of phrase lengths and frequencies

Used to decide K (codebook size) and frequency thresholds.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, some statistics will be limited")


def load_phrases(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load mined phrases from JSONL file.
    
    Args:
        input_path: Path to mined_phrases.jsonl
    
    Returns:
        List of phrase dictionaries
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    phrases = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                phrases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    return phrases


def compute_top_phrases(
    phrases: List[Dict[str, Any]],
    top_k: int = 100,
    sort_by: str = "count"
) -> List[Dict[str, Any]]:
    """
    Get top K phrases by specified metric.
    
    Args:
        phrases: List of phrase dictionaries
        top_k: Number of top phrases to return
        sort_by: Metric to sort by ("count", "pmi")
    
    Returns:
        List of top K phrases
    """
    if sort_by == "count":
        sorted_phrases = sorted(phrases, key=lambda x: x.get("count", 0), reverse=True)
    elif sort_by == "pmi":
        sorted_phrases = sorted(phrases, key=lambda x: x.get("pmi", 0), reverse=True)
    else:
        raise ValueError(f"Invalid sort_by: {sort_by}")
    
    return sorted_phrases[:top_k]


def compute_coverage_stats(phrases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute coverage statistics.
    
    Args:
        phrases: List of phrase dictionaries
    
    Returns:
        Dictionary with coverage statistics
    """
    total_phrases = len(phrases)
    total_occurrences = sum(p.get("count", 0) for p in phrases)
    
    # Phrase length distribution
    length_counts = Counter()
    for phrase in phrases:
        phrase_text = phrase.get("phrase", "")
        length = len(phrase_text.split())
        length_counts[length] += 1
    
    # Count distribution
    counts = [p.get("count", 0) for p in phrases]
    if HAS_NUMPY and counts:
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        mean_count = float(np.mean(counts))
        median_count = float(np.median(counts))
    else:
        if counts:
            sorted_counts = sorted(counts)
            min_count = sorted_counts[0]
            max_count = sorted_counts[-1]
            mean_count = sum(counts) / len(counts)
            median_count = sorted_counts[len(sorted_counts) // 2]
        else:
            min_count = max_count = mean_count = median_count = 0
    
    # PMI distribution
    pmis = [p.get("pmi", 0) for p in phrases if "pmi" in p]
    if HAS_NUMPY and pmis:
        min_pmi = float(np.min(pmis))
        max_pmi = float(np.max(pmis))
        mean_pmi = float(np.mean(pmis))
        median_pmi = float(np.median(pmis))
    else:
        if pmis:
            sorted_pmis = sorted(pmis)
            min_pmi = sorted_pmis[0]
            max_pmi = sorted_pmis[-1]
            mean_pmi = sum(pmis) / len(pmis)
            median_pmi = sorted_pmis[len(sorted_pmis) // 2]
        else:
            min_pmi = max_pmi = mean_pmi = median_pmi = 0.0
    
    return {
        "total_unique_phrases": total_phrases,
        "total_occurrences": total_occurrences,
        "length_distribution": dict(length_counts),
        "count_stats": {
            "min": min_count,
            "max": max_count,
            "mean": round(mean_count, 2),
            "median": median_count
        },
        "pmi_stats": {
            "min": round(min_pmi, 4),
            "max": round(max_pmi, 4),
            "mean": round(mean_pmi, 4),
            "median": round(median_pmi, 4)
        }
    }


def print_summary(phrases: List[Dict[str, Any]], top_k: int = 20) -> None:
    """
    Print summary statistics to console.
    
    Args:
        phrases: List of phrase dictionaries
        top_k: Number of top phrases to display
    """
    print("=" * 60)
    print("Phrase Statistics Summary")
    print("=" * 60)
    print()
    
    # Basic stats
    total_phrases = len(phrases)
    total_occurrences = sum(p.get("count", 0) for p in phrases)
    
    print(f"Total unique phrases: {total_phrases:,}")
    print(f"Total phrase occurrences: {total_occurrences:,}")
    print()
    
    # Coverage stats
    coverage = compute_coverage_stats(phrases)
    
    print("Count Statistics:")
    count_stats = coverage["count_stats"]
    print(f"  Min: {count_stats['min']:,}")
    print(f"  Max: {count_stats['max']:,}")
    print(f"  Mean: {count_stats['mean']:.2f}")
    print(f"  Median: {count_stats['median']:,}")
    print()
    
    if coverage["pmi_stats"]["mean"] > 0:
        print("PMI Statistics:")
        pmi_stats = coverage["pmi_stats"]
        print(f"  Min: {pmi_stats['min']:.4f}")
        print(f"  Max: {pmi_stats['max']:.4f}")
        print(f"  Mean: {pmi_stats['mean']:.4f}")
        print(f"  Median: {pmi_stats['median']:.4f}")
        print()
    
    # Length distribution
    print("Phrase Length Distribution:")
    length_dist = coverage["length_distribution"]
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        percentage = (count / total_phrases * 100) if total_phrases > 0 else 0
        print(f"  {length}-grams: {count:,} ({percentage:.1f}%)")
    print()
    
    # Top phrases by count
    print(f"Top {top_k} Phrases by Count:")
    top_by_count = compute_top_phrases(phrases, top_k=top_k, sort_by="count")
    for i, phrase in enumerate(top_by_count, 1):
        phrase_text = phrase.get("phrase", "")
        count = phrase.get("count", 0)
        pmi = phrase.get("pmi", 0)
        print(f"  {i:3d}. {phrase_text[:50]:<50} count={count:6d}, PMI={pmi:6.2f}")
    print()
    
    # Top phrases by PMI
    if any("pmi" in p for p in phrases):
        print(f"Top {top_k} Phrases by PMI:")
        top_by_pmi = compute_top_phrases(phrases, top_k=top_k, sort_by="pmi")
        for i, phrase in enumerate(top_by_pmi, 1):
            phrase_text = phrase.get("phrase", "")
            count = phrase.get("count", 0)
            pmi = phrase.get("pmi", 0)
            print(f"  {i:3d}. {phrase_text[:50]:<50} PMI={pmi:6.2f}, count={count:6d}")
        print()
    
    # Recommendations
    print("Recommendations for Codebook Size (K):")
    if total_phrases < 1000:
        print(f"  Small dataset: Consider K={total_phrases // 2} to {total_phrases}")
    elif total_phrases < 10000:
        print(f"  Medium dataset: Consider K=2000 to 8000")
    else:
        print(f"  Large dataset: Consider K=8000 to 32000")
    print(f"  Current unique phrases: {total_phrases:,}")
    print()


def main():
    """CLI entry point for phrase statistics."""
    parser = argparse.ArgumentParser(
        description="Analyze mined phrases and generate statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic statistics
  python mining/phrase_stats.py --input data/mined_phrases.jsonl
  
  # Top 50 phrases
  python mining/phrase_stats.py --input data/mined_phrases.jsonl --top-k 50
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/mined_phrases.jsonl",
        help="Input path to mined phrases (JSONL format, default: data/mined_phrases.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for statistics and plots (default: reports)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top phrases to display (default: 20)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load phrases
    print(f"Loading phrases from {input_path}...")
    phrases = load_phrases(input_path)
    print(f"Loaded {len(phrases)} phrases")
    print()
    
    # Generate statistics
    coverage_stats = compute_coverage_stats(phrases)
    top_phrases = compute_top_phrases(phrases, top_k=args.top_k)
    
    # Print summary
    print_summary(phrases, top_k=args.top_k)
    
    # Save statistics to JSON
    stats_output = output_dir / "phrase_stats.json"
    stats_data = {
        "total_phrases": len(phrases),
        "top_phrases": top_phrases,
        "coverage": coverage_stats
    }
    
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Statistics saved to {stats_output}")


if __name__ == "__main__":
    main()
