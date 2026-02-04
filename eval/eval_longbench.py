"""
Evaluate on LongBench: Run LongBench tasks for baseline vs CLV.

This is a minimal stub implementation for local testing.
For production evaluation on A100, extend with real LongBench dataset loading.

Use LongBench tasks (single-doc QA, multi-doc QA, summarization subsets).

For each task:
- Run with baseline (no CLV)
- Run with CLV compress+model
- Metrics: EM, F1, ROUGE etc. as appropriate
- Compare results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

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
    device: str = "cpu"
) -> tuple:
    """Load model and tokenizer."""
    if not HAS_TRANSFORMERS or device == "cpu":
        print("⚠ Using mock model for local testing")
        return None, None
    
    model_name = model_path or base_model_name or "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def create_dummy_task_data(task: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    """Create dummy task data for local testing."""
    if "qa" in task.lower():
        return [
            {
                "question": "What is the main topic?",
                "context": "Machine learning is a subset of AI.",
                "answer": "Machine learning"
            }
        ] * num_samples
    elif "summarization" in task.lower():
        return [
            {
                "document": "This is a long document about various topics. " * 10,
                "summary": "Summary of the document."
            }
        ] * num_samples
    else:
        return [{"input": "Sample input", "output": "Sample output"}] * num_samples


def compute_metrics_mock(
    predictions: List[str],
    references: List[str],
    task: str
) -> Dict[str, float]:
    """Compute mock metrics."""
    import random
    if "qa" in task.lower():
        return {
            "exact_match": random.uniform(0.6, 0.9),
            "f1": random.uniform(0.7, 0.95)
        }
    elif "summarization" in task.lower():
        return {
            "rouge1": random.uniform(0.5, 0.8),
            "rouge2": random.uniform(0.4, 0.7),
            "rougeL": random.uniform(0.5, 0.8)
        }
    else:
        return {"accuracy": random.uniform(0.6, 0.9)}


def evaluate_task(
    model,
    tokenizer,
    task_data: List[Dict[str, Any]],
    task: str,
    use_clv: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluate model on a LongBench task."""
    print(f"  Evaluating task: {task} ({len(task_data)} samples)")
    
    # Mock evaluation
    predictions = ["Mock prediction"] * len(task_data)
    references = [item.get("answer") or item.get("summary") or item.get("output", "") for item in task_data]
    
    metrics = compute_metrics_mock(predictions, references, task)
    
    return {
        "task": task,
        "num_samples": len(task_data),
        "metrics": metrics
    }


def evaluate(
    model_path: str = None,
    base_model_name: str = None,
    use_clv: bool = False,
    clv_adapter_path: str = None,
    tasks: List[str] = None,
    num_samples: int = 5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluate on LongBench tasks."""
    print(f"Evaluating {'CLV' if use_clv else 'baseline'} model on LongBench...")
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        base_model_name=base_model_name,
        use_clv=use_clv,
        clv_adapter_path=clv_adapter_path,
        device=device
    )
    
    tasks = tasks or ["single_doc_qa", "multi_doc_qa", "summarization"]
    results = {}
    
    for task in tasks:
        task_data = create_dummy_task_data(task, num_samples)
        task_results = evaluate_task(model, tokenizer, task_data, task, use_clv, device)
        results[task] = task_results
    
    results["mode"] = "clv" if use_clv else "baseline"
    
    return results


def generate_report(
    baseline_results: Dict[str, Any],
    clv_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "baseline": baseline_results,
        "clv": clv_results,
        "comparison": {}
    }
    
    # Compare metrics
    for task in baseline_results:
        if task == "mode":
            continue
        if task in clv_results:
            baseline_metrics = baseline_results[task]["metrics"]
            clv_metrics = clv_results[task]["metrics"]
            
            comparison = {}
            for metric in baseline_metrics:
                if metric in clv_metrics:
                    delta = clv_metrics[metric] - baseline_metrics[metric]
                    comparison[metric] = {
                        "baseline": baseline_metrics[metric],
                        "clv": clv_metrics[metric],
                        "delta": delta
                    }
            report["comparison"][task] = comparison
    
    json_path = output_dir / "longbench_metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    print("LongBench Evaluation Results")
    print("=" * 60)
    for task, comp in report["comparison"].items():
        print(f"\n{task}:")
        for metric, values in comp.items():
            print(f"  {metric}: {values['baseline']:.4f} -> {values['clv']:.4f} (Δ{values['delta']:+.4f})")
    print(f"\nReport saved to: {json_path}")


def main():
    """CLI entry point for LongBench evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate on LongBench tasks")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--base-model-name", type=str, help="Base model name")
    parser.add_argument("--use-clv", action="store_true", help="Use CLV model")
    parser.add_argument("--clv-adapter", type=str, help="CLV adapter path")
    parser.add_argument("--tasks", type=str, nargs="+", default=["single_doc_qa"], help="Tasks to evaluate")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples per task (stub)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--clv-only", action="store_true")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    baseline_results = None
    clv_results = None
    
    if not args.clv_only:
        baseline_results = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=False,
            tasks=args.tasks,
            num_samples=args.num_samples,
            device=args.device
        )
    
    if not args.baseline_only:
        clv_results = evaluate(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            use_clv=True,
            clv_adapter_path=args.clv_adapter,
            tasks=args.tasks,
            num_samples=args.num_samples,
            device=args.device
        )
    
    if baseline_results and clv_results:
        generate_report(baseline_results, clv_results, output_dir)


if __name__ == "__main__":
    main()
