"""
Evaluate perplexity: Compute PPL for baseline vs CLV-augmented model.

This is a minimal stub implementation for local testing.
For production evaluation on A100, extend with real dataset loading and model inference.

Datasets:
- Wikitext-103 test
- Small curated corpus of long-form text

Compute PPL for:
- Baseline model
- CLV-augmented model (with compression)

Report:
- ΔPPL %
- Confidence it's <3%
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using mock model")


class MockModel(nn.Module):
    """Mock model for local testing."""
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lm_head = nn.Linear(128, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        logits = self.lm_head(x.mean(dim=1))
        return type('Output', (), {'logits': logits.unsqueeze(1).expand(-1, input_ids.size(1), -1)})()


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    use_clv: bool = False,
    clv_adapter_path: Optional[str] = None,
    clv_map_path: Optional[str] = None,
    phrase_index_path: Optional[str] = None,
    use_lossless: bool = False,
    device: str = "cpu"
) -> tuple:
    """Load model and tokenizer with optional CLV adapter."""
    if not HAS_TRANSFORMERS:
        print("⚠ transformers not available, using mock model")
        model = MockModel()
        return model, None
    
    # Determine model name
    model_name = model_path or base_model_name
    if not model_name:
        if device == "cpu":
            model_name = "Qwen/Qwen2-0.5B-Instruct"  # Small model for CPU
        else:
            model_name = "Qwen/Qwen2-7B-Instruct"  # Full model for GPU
    
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
            print("  Continuing with base model only")
    
    model.eval()
    
    return model, tokenizer


def load_wikitext_dataset(num_samples: Optional[int] = None, split: str = "test") -> list:
    """Load Wikitext-103 dataset."""
    try:
        from datasets import load_dataset
        print(f"Loading Wikitext-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        # Filter out empty texts and get text field
        texts = [item["text"] for item in dataset if len(item.get("text", "").strip()) > 0]
        
        if num_samples:
            texts = texts[:num_samples]
        
        print(f"✓ Loaded {len(texts)} samples from Wikitext-103")
        return texts
    except Exception as e:
        print(f"⚠ Failed to load Wikitext-103: {e}")
        print("  Falling back to dummy dataset")
        return create_dummy_dataset(num_samples or 10)


def create_dummy_dataset(num_samples: int = 10) -> list:
    """Create dummy dataset for local testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Transformers have revolutionized the field of NLP.",
    ] * (num_samples // 5 + 1)
    return texts[:num_samples]


def compute_perplexity_mock(
    model,
    tokenizer,
    texts: list,
    device: str = "cpu"
) -> float:
    """Compute mock perplexity (placeholder for real implementation)."""
    # Mock computation
    import random
    base_ppl = 15.0 + random.uniform(-2.0, 2.0)
    return base_ppl


def get_clv_pid_token_ids(tokenizer):
    """Get set of CLV and PID token IDs using added vocab (fast and reliable)."""
    ids = set()

    # Use added vocab keys (fast, doesn't scan full vocab)
    added_vocab = getattr(tokenizer, 'get_added_vocab', lambda: {})()
    for tok, tid in added_vocab.items():
        if tok.startswith("<clv:") or tok.startswith("<pid:"):
            ids.add(tid)

    return ids


def compute_perplexity(
    model,
    tokenizer,
    texts: list,
    device: str = "cpu",
    max_length: int = 512,
    batch_size: int = 4,
    mask_clv_tokens: bool = True,
    decompressor = None
) -> float:
    """Compute perplexity on dataset, optionally masking CLV/PID tokens."""
    if not HAS_TRANSFORMERS or isinstance(model, MockModel):
        return compute_perplexity_mock(model, tokenizer, texts, device)
    
    # Get CLV/PID token IDs to mask
    clv_pid_ids = get_clv_pid_token_ids(tokenizer) if mask_clv_tokens else set()
    ignore_index = -100  # Standard ignore index for loss
    
    if clv_pid_ids:
        print(f"Masking {len(clv_pid_ids)} CLV/PID tokens from perplexity computation")
    
    # Real perplexity computation
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    print(f"Computing perplexity on {len(texts)} samples...")
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            # Create labels: decompress if decompressor provided, otherwise mask CLV/PID
            if decompressor:
                # Decompress the input text to get raw target
                # This implements "input-only CLV" - model sees compressed input but predicts raw tokens
                batch_raw_texts = []
                for text in batch_texts:
                    try:
                        raw_text = decompressor(text)
                        batch_raw_texts.append(raw_text)
                    except Exception as e:
                        # Fallback to original if decompression fails
                        batch_raw_texts.append(text)
                
                # Tokenize decompressed targets
                target_inputs = tokenizer(
                    batch_raw_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                labels = target_inputs["input_ids"].to(device)
                
                # Align labels with input_ids length (truncate if needed)
                if labels.shape[1] < input_ids.shape[1]:
                    # Pad labels
                    padding = torch.full((labels.shape[0], input_ids.shape[1] - labels.shape[1]), 
                                        ignore_index, device=device, dtype=labels.dtype)
                    labels = torch.cat([labels, padding], dim=1)
                elif labels.shape[1] > input_ids.shape[1]:
                    # Truncate labels
                    labels = labels[:, :input_ids.shape[1]]
            else:
                # Standard: use input_ids as labels, mask CLV/PID tokens
                labels = input_ids.clone()
                if clv_pid_ids:
                    # Set CLV/PID tokens to ignore_index
                    for token_id in clv_pid_ids:
                        labels[labels == token_id] = ignore_index
            
            # Forward pass with labels
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Count non-padding, non-CLV/PID tokens
            valid_mask = (labels != ignore_index) & (attention_mask == 1)
            num_tokens = valid_mask.sum().item()
            
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            
            num_batches += 1
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)} samples...")
    
    if total_tokens == 0:
        print("⚠ No valid tokens processed, returning default perplexity")
        return 100.0
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"✓ Computed perplexity: {ppl:.4f} (avg_loss: {avg_loss:.4f}, valid_tokens: {total_tokens:,})")
    
    return ppl


def evaluate(
    model_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    use_clv: bool = False,
    clv_adapter_path: Optional[str] = None,
    clv_map_path: Optional[str] = None,
    phrase_index_path: Optional[str] = None,
    use_lossless: bool = False,
    dataset_name: str = "wikitext",
    device: str = "cpu",
    num_samples: Optional[int] = None,
    batch_size: int = 4,
    max_length: int = 512
) -> Dict[str, Any]:
    """Evaluate model perplexity."""
    print(f"Evaluating {'CLV' if use_clv else 'baseline'} model...")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_name}" + (f" (samples: {num_samples})" if num_samples else " (all available)"))
    
    # Load model
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
    
    # Load dataset
    if dataset_name == "wikitext":
        texts = load_wikitext_dataset(num_samples=num_samples)
        if len(texts) == 0:
            print("⚠ No texts loaded, using dummy dataset")
            texts = create_dummy_dataset(num_samples or 10)
    else:
        texts = create_dummy_dataset(num_samples or 10)
    
    # CRITICAL: For CLV evaluation, we compress INPUT but decompress TARGET
    # The model should predict raw tokens, not CLV/PID tokens
    # This matches "input-only CLV" - compression for transport, not generation
    
    compressed_inputs = None
    decompressor = None
    
    if use_clv:
        print(f"Setting up {'lossless (PID)' if use_lossless else 'codebook (CLV)'} compression for INPUT only...")
        print("  (Targets will be decompressed before perplexity computation)")
        print(f"  use_lossless: {use_lossless}")
        print(f"  phrase_index_path: {phrase_index_path}")

        # Check if phrase index exists
        from pathlib import Path
        phrase_index_exists = phrase_index_path and Path(phrase_index_path).exists()
        print(f"  phrase_index_exists: {phrase_index_exists}")

        try:
            import sys
            script_dir = Path(__file__).parent
            parent_dir = script_dir.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            if use_lossless and phrase_index_exists:
                print("✓ Using LOSSLESS mode")
                # Lossless mode: use PID tokens
                from inference.compress_wrap import compress_text_lossless, load_phrase_index
                from inference.decompress_wrap import decompress_text_lossless

                print(f"Loading phrase index from: {phrase_index_path}")
                phrase_index_path_obj = Path(phrase_index_path)
                phrase_to_id, id_to_phrase = load_phrase_index(phrase_index_path_obj)
                print(f"Loaded {len(phrase_to_id)} phrases, {len(id_to_phrase)} IDs")
                
                # Compress inputs
                compressed_inputs = []
                for i, text in enumerate(texts):
                    compressed_text, stats = compress_text_lossless(
                        text, tokenizer, phrase_to_id, return_text=True
                    )
                    compressed_inputs.append(compressed_text)
                    if i == 0:  # Debug first sample
                        print(f"Sample compression: '{text[:50]}...' → '{compressed_text[:50]}...'")
                
                # Store decompressor for targets
                decompressor = lambda text: decompress_text_lossless(text, id_to_phrase)
                
                print(f"✓ Compressed {len(compressed_inputs)} inputs with lossless (PID) mode")
            else:
                print("⚠ Using CODEBOOK mode (fallback)")
                # Codebook mode: use CLV tokens
                from inference.compress_wrap import compress_text_with_clv
                from inference.decompress_wrap import decompress_text_with_clv

                if not clv_map_path:
                    raise ValueError("clv_map_path required for codebook mode")
                
                # Load CLV map for compression
                import json
                with open(clv_map_path) as f:
                    clv_map = json.load(f)
                
                # Build reverse map for decompression
                reverse_map = {v: k for k, v in clv_map.items()}
                
                # Determine codebook size from CLV map
                codebook_size = max(clv_map.values()) + 1 if clv_map else 8192
                
                # Compress inputs
                compressed_inputs = []
                for text in texts:
                    compressed_text, stats = compress_text_with_clv(
                        text, tokenizer, clv_map, codebook_size, return_text=True
                    )
                    compressed_inputs.append(compressed_text)
                
                # Store decompressor for targets
                decompressor = lambda text: decompress_text_with_clv(text, tokenizer, clv_map, reverse_map)
                
                print(f"✓ Compressed {len(compressed_inputs)} inputs with codebook (CLV) mode")
        except Exception as e:
            print(f"⚠ Failed to setup compression: {e}")
            import traceback
            traceback.print_exc()
            print("  Evaluating on raw text (may give poor results)")
            compressed_inputs = None
    
    # Use compressed inputs if available, otherwise raw
    input_texts = compressed_inputs if compressed_inputs else texts
    
    # Compute perplexity with decompression of targets
    ppl = compute_perplexity(
        model, tokenizer, input_texts, device, 
        max_length=max_length, batch_size=batch_size,
        decompressor=decompressor
    )
    
    # Determine actual mode used
    actual_mode = "baseline"
    if use_clv:
        if use_lossless and phrase_index_path and Path(phrase_index_path).exists():
            actual_mode = "lossless"
        else:
            actual_mode = "codebook"

    results = {
        "perplexity": ppl,
        "num_samples": len(texts),
        "dataset": dataset_name,
        "mode": actual_mode
    }
    
    print(f"✓ Perplexity: {ppl:.4f}")
    
    return results


def generate_report(
    baseline_results: Dict[str, Any],
    clv_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_ppl = baseline_results["perplexity"]
    clv_ppl = clv_results["perplexity"]
    delta_ppl = clv_ppl - baseline_ppl
    delta_ppl_pct = (delta_ppl / baseline_ppl) * 100 if baseline_ppl > 0 else 0.0
    
    report = {
        "baseline": baseline_results,
        "clv": clv_results,
        "delta_ppl": delta_ppl,
        "delta_ppl_percent": delta_ppl_pct,
        "meets_target": abs(delta_ppl_pct) < 3.0
    }
    
    # Save JSON
    json_path = output_dir / "perplexity_metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print()
    print("=" * 60)
    print("Perplexity Evaluation Results")
    print("=" * 60)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"CLV PPL:      {clv_ppl:.4f}")
    print(f"ΔPPL:         {delta_ppl:+.4f} ({delta_ppl_pct:+.2f}%)")
    print(f"Meets target (<3%): {'✓' if report['meets_target'] else '✗'}")
    print()
    print(f"Report saved to: {json_path}")


def main():
    """CLI entry point for perplexity evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity for baseline vs CLV model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model (local)"
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        help="Base model name (Hugging Face identifier)"
    )
    parser.add_argument(
        "--use-clv",
        action="store_true",
        help="Use CLV-augmented model"
    )
    parser.add_argument(
        "--clv-adapter",
        type=str,
        help="Path to CLV LoRA adapter"
    )
    parser.add_argument(
        "--clv-map",
        type=str,
        default="artifacts/clv_map.json",
        help="Path to CLV mapping (for codebook mode)"
    )
    parser.add_argument(
        "--phrase-index",
        type=str,
        default="data/phrase_index.jsonl",
        help="Path to phrase index (for lossless mode)"
    )
    parser.add_argument(
        "--use-lossless",
        action="store_true",
        help="Use lossless (PID) mode instead of codebook (CLV) mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name (default: wikitext)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu, use cuda for GPU)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: None = all available)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate baseline"
    )
    parser.add_argument(
        "--clv-only",
        action="store_true",
        help="Only evaluate CLV"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results = None
    clv_results = None
    
    # Evaluate baseline
    if not args.clv_only:
        baseline_results = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=False,
            dataset_name=args.dataset,
            device=args.device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    
    # Evaluate CLV
    if not args.baseline_only:
        clv_results = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=True,
            clv_adapter_path=args.clv_adapter,
            clv_map_path=args.clv_map,
            phrase_index_path=args.phrase_index,
            use_lossless=args.use_lossless,
            dataset_name=args.dataset,
            device=args.device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    
    # Generate report
    if baseline_results and clv_results:
        generate_report(baseline_results, clv_results, output_dir)
    elif baseline_results:
        json_path = output_dir / "perplexity_baseline.json"
        with open(json_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Baseline results saved to: {json_path}")
    elif clv_results:
        json_path = output_dir / "perplexity_clv.json"
        with open(json_path, "w") as f:
            json.dump(clv_results, f, indent=2)
        print(f"CLV results saved to: {json_path}")


if __name__ == "__main__":
    main()
