"""
Evaluate latency and cost: Measure inference performance.

This is a minimal stub implementation for local testing.
For production evaluation on A100, extend with real model inference and timing.

Measure:
- tokens/sec
- ms/token
- peak VRAM

Compare:
- Baseline text
- CLV-compressed text

Target: ≥20% reduction in ms/token or $/effective-token
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_model_and_tokenizer(
    model_path: str = None,
    base_model_name: str = None,
    use_clv: bool = False,
    clv_adapter_path: str = None,
    clv_map_path: str = None,
    phrase_index_path: str = None,
    use_lossless: bool = False,
    device: str = "cpu"
) -> tuple:
    """Load model and tokenizer with optional CLV adapter."""
    if not HAS_TRANSFORMERS:
        print("⚠ transformers not available")
        return None, None
    
    # Determine model name
    model_name = model_path or base_model_name
    if not model_name:
        if device == "cpu":
            model_name = "Qwen/Qwen2-0.5B-Instruct"
        else:
            model_name = "Qwen/Qwen2-7B-Instruct"
    
    print(f"Loading base model: {model_name}")
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Always add CLV tokens when use_clv=True, regardless of adapter presence
    if use_clv:
        from pathlib import Path

        # Check if CLV tokenizer already exists (preferred)
        tokenizer_path = None
        if clv_adapter_path:
            tokenizer_path = Path(clv_adapter_path).parent / "tokenizer"

        if tokenizer_path and tokenizer_path.exists():
            print(f"Loading CLV tokenizer from: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            print("✓ CLV tokenizer loaded")
        else:
            # Add CLV tokens manually BEFORE loading model
            print("Adding CLV tokens to tokenizer...")
            codebook_size = 8192
            if use_lossless and phrase_index_path:
                # For lossless mode, size based on phrase index
                try:
                    from inference.compress_wrap import load_phrase_index
                    _, id_to_phrase = load_phrase_index(phrase_index_path)
                    codebook_size = max(id_to_phrase.keys()) + 1 if id_to_phrase else 8192
                except:
                    codebook_size = 8192
            elif clv_map_path and Path(clv_map_path).exists():
                # For codebook mode, size based on CLV map
                import json
                with open(clv_map_path) as f:
                    clv_map = json.load(f)
                codebook_size = max(clv_map.values()) + 1 if clv_map else 8192

            import sys
            script_dir = Path(__file__).parent
            parent_dir = script_dir.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from inference.clv_tokenizer_adapter import add_clv_tokens_to_tokenizer
            tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size=codebook_size)
            print(f"✓ Added {codebook_size} CLV tokens to tokenizer")
    
    # Load base model
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Resize model embeddings if tokenizer was expanded
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"Resizing model embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        print("✓ Model embeddings resized")
    
    if device == "cpu":
        model = model.to(device)
    
    # Load CLV adapter if requested
    if use_clv and clv_adapter_path:
        try:
            from peft import PeftModel
            print(f"Loading CLV LoRA adapter from: {clv_adapter_path}")
            model = PeftModel.from_pretrained(model, clv_adapter_path)
            print("✓ CLV adapter loaded")
        except Exception as e:
            print(f"⚠ Failed to load CLV adapter: {e}")
            import traceback
            traceback.print_exc()
    
    model.eval()
    return model, tokenizer


def create_dummy_texts(num_texts: int = 5) -> List[str]:
    """Create dummy input texts."""
    base_texts = [
        "This is a sample text for latency measurement. " * 10,
        "Another example text with different content. " * 10,
        "Machine learning models require efficient inference. " * 10,
    ]
    return (base_texts * (num_texts // 3 + 1))[:num_texts]


def measure_inference_speed_mock(
    texts: List[str],
    device: str = "cpu"
) -> Dict[str, float]:
    """Mock speed measurement."""
    # Simulate timing
    time.sleep(0.1)  # Simulate processing
    
    total_tokens = sum(len(text.split()) for text in texts)
    elapsed = 0.1 * len(texts)
    
    return {
        "total_tokens": total_tokens,
        "total_time_sec": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "ms_per_token": (elapsed / total_tokens * 1000) if total_tokens > 0 else 0.0
    }


def measure_inference_speed(
    model,
    tokenizer,
    input_texts: List[str],
    device: str = "cuda",
    num_runs: int = 10
) -> Dict[str, float]:
    """Measure inference speed."""
    if model is None or tokenizer is None:
        return measure_inference_speed_mock(input_texts, device)
    
    # Real measurement: time tokenization + forward pass
    import torch
    
    total_tokens = 0
    total_time = 0.0
    
    # Warmup
    if len(input_texts) > 0:
        warmup_text = input_texts[0]
        inputs = tokenizer(warmup_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Actual measurement
    for text in input_texts:
        # Time tokenization
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        num_tokens = inputs["input_ids"].shape[1]
        
        # Time forward pass
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_inf = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        total_time += elapsed
        total_tokens += num_tokens
    
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    ms_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0.0
    
    return {
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token
    }


def measure_memory_usage(
    model,
    tokenizer,
    input_texts: List[str],
    device: str = "cuda"
) -> Dict[str, float]:
    """Measure peak memory usage."""
    if device == "cpu" or model is None:
        return {
            "peak_memory_mb": 0.0,
            "allocated_memory_mb": 0.0
        }
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Run inference
        for text in input_texts[:1]:  # Just one for stub
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
        
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {
            "peak_memory_mb": peak_mb,
            "allocated_memory_mb": allocated_mb
        }
    
    return {"peak_memory_mb": 0.0, "allocated_memory_mb": 0.0}


def compute_compression_ratio(
    baseline_texts: List[str],
    clv_texts: List[str],
    tokenizer
) -> float:
    """Compute compression ratio."""
    if tokenizer is None:
        # Simple word count
        baseline_tokens = sum(len(t.split()) for t in baseline_texts)
        clv_tokens = sum(len(t.split()) for t in clv_texts)
    else:
        baseline_tokens = sum(len(tokenizer.encode(t)) for t in baseline_texts)
        clv_tokens = sum(len(tokenizer.encode(t)) for t in clv_texts)
    
    return clv_tokens / baseline_tokens if baseline_tokens > 0 else 1.0


def evaluate(
    model_path: str = None,
    base_model_name: str = None,
    use_clv: bool = False,
    clv_adapter_path: str = None,
    clv_map_path: str = None,
    phrase_index_path: str = None,
    use_lossless: bool = False,
    num_texts: int = 5,
    num_runs: int = 3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluate latency and cost."""
    print(f"Evaluating {'CLV' if use_clv else 'baseline'} model latency...")
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        base_model_name=base_model_name,
        use_clv=use_clv,
        clv_adapter_path=clv_adapter_path,
        clv_map_path=clv_map_path,
        phrase_index_path=phrase_index_path,
        use_lossless=use_lossless,
        device=device
    )
    
    texts = create_dummy_texts(num_texts)
    
    # Measure speed
    speed_metrics = measure_inference_speed(model, tokenizer, texts, device, num_runs)
    
    # Measure memory
    memory_metrics = measure_memory_usage(model, tokenizer, texts, device)
    
    results = {
        **speed_metrics,
        **memory_metrics,
        "mode": "clv" if use_clv else "baseline",
        "num_texts": len(texts),
        "num_runs": num_runs
    }
    
    print(f"✓ Tokens/sec: {speed_metrics['tokens_per_sec']:.2f}")
    print(f"✓ ms/token: {speed_metrics['ms_per_token']:.4f}")
    
    return results


def generate_report(
    baseline_metrics: Dict[str, Any],
    clv_metrics: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate latency/cost report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_tps = baseline_metrics["tokens_per_sec"]
    clv_tps = clv_metrics["tokens_per_sec"]
    improvement_pct = ((clv_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0.0
    
    report = {
        "baseline": baseline_metrics,
        "clv": clv_metrics,
        "improvement": {
            "tokens_per_sec_delta": clv_tps - baseline_tps,
            "tokens_per_sec_improvement_pct": improvement_pct,
            "ms_per_token_reduction_pct": -improvement_pct,  # Inverse
            "meets_target": improvement_pct >= 20.0
        }
    }
    
    json_path = output_dir / "latency_metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    print("Latency Evaluation Results")
    print("=" * 60)
    print(f"Baseline: {baseline_tps:.2f} tokens/sec, {baseline_metrics['ms_per_token']:.4f} ms/token")
    print(f"CLV:      {clv_tps:.2f} tokens/sec, {clv_metrics['ms_per_token']:.4f} ms/token")
    print(f"Improvement: {improvement_pct:+.2f}%")
    print(f"Meets target (≥20%): {'✓' if report['improvement']['meets_target'] else '✗'}")
    print(f"\nReport saved to: {json_path}")


def main():
    """CLI entry point for latency/cost evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate inference latency and cost")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--base-model-name", type=str, help="Base model name")
    parser.add_argument("--use-clv", action="store_true", help="Use CLV model")
    parser.add_argument("--clv-adapter", type=str, help="CLV adapter path")
    parser.add_argument("--clv-map", type=str, default="artifacts/clv_map.json", help="CLV map path (for codebook mode)")
    parser.add_argument("--phrase-index", type=str, default="data/phrase_index.jsonl", help="Path to phrase index (for lossless mode)")
    parser.add_argument("--use-lossless", action="store_true", help="Use lossless (PID) mode instead of codebook (CLV) mode")
    parser.add_argument("--num-texts", type=int, default=5, help="Number of texts (stub)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--clv-only", action="store_true")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    baseline_metrics = None
    clv_metrics = None
    
    if not args.clv_only:
        baseline_metrics = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=False,
            num_texts=args.num_texts,
            num_runs=args.num_runs,
            device=args.device
        )
    
    if not args.baseline_only:
        clv_metrics = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=True,
            clv_adapter_path=args.clv_adapter,
            clv_map_path=args.clv_map,
            phrase_index_path=args.phrase_index,
            use_lossless=args.use_lossless,
            num_texts=args.num_texts,
            num_runs=args.num_runs,
            device=args.device
        )
    
    if baseline_metrics and clv_metrics:
        generate_report(baseline_metrics, clv_metrics, output_dir)


if __name__ == "__main__":
    main()
