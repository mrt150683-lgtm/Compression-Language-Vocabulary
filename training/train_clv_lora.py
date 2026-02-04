"""
Train CLV LoRA: Fine-tune base LLM with CLV tokens and VQ module.

⚠️ IMPORTANT: This script is designed for remote GPU execution (A100 80GB recommended).
Local CPU runs are only for smoke tests with tiny models and toy datasets.

Training Flow:
1. Load base model and tokenizer
2. Add CLV special tokens (<clv:0000> ... <clv:####>)
3. Load codebook + CLV map
4. Attach LoRA adapters (preserves base weights)
5. Attach VQ module at specified layer
6. Wrap dataset with CLV phrase replacement
7. Train with accelerate (mixed precision, gradient accumulation)

Training Objective:
- Standard LM loss on output tokens
- VQ loss on positions participating in CLV spans
- Combined: L_total = w_lm * L_lm + w_vq * L_vq

Key Constraints:
- No full fine-tune; adapters only (LoRA)
- Mixed precision (fp16/bf16) for efficiency
- Gradient accumulation for effective large batch size
- Config via configs/train_clv_lora_vq.yaml

Outputs:
- artifacts/clv_lora_adapter.pt (LoRA weights)
- artifacts/clv_tokenizer_added.json (tokenizer with CLV tokens)
- artifacts/clv_vq_state.pt (VQ module state)

Usage:
    # Remote GPU (production):
    accelerate launch training/train_clv_lora.py --config configs/train_clv_lora_vq.yaml
    
    # Local CPU (smoke test only):
    python training/train_clv_lora.py --config configs/train_clv_lora_vq.yaml --device cpu --smoke-test
"""

import argparse
import json
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from accelerate import Accelerator
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    print(f"Warning: Missing dependencies: {e}")
    print("Install: transformers, peft, accelerate")

from training.vq_module import VectorQuantizer


# ============================================================================
# Helper Functions for Robust Model Layer Detection
# ============================================================================

def _resolve_attr(root, dotted):
    """Resolve a dotted attribute path (e.g., 'base_model.model.layers')."""
    cur = root
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _get_transformer_layers_any(model):
    """
    Return an ordered list/ModuleList of transformer blocks for a wide set of HF models,
    handling PEFT wrappers (PeftModel*) and raw bases.
    """
    candidates = [
        # Common PEFT / Qwen2 / LLaMA / Mistral
        "base_model.model.model.layers",
        "base_model.model.layers",
        "model.model.layers",
        "model.layers",
        # GPT-NeoX family
        "gpt_neox.layers",
        # OPT
        "model.decoder.layers",
        # Falcon
        "transformer.h",
        # Last-ditch
        "layers",
    ]
    
    for path in candidates:
        layers = _resolve_attr(model, path)
        if layers is not None and hasattr(layers, "__len__"):
            return layers
    
    return None


class SimpleCLVDataset(Dataset):
    """
    Simple CLV dataset wrapper for training.
    
    Applies CLV phrase replacement to base dataset sequences.
    For smoke testing, can generate synthetic data.
    """
    
    def __init__(
        self,
        texts: list,
        clv_map: Dict[str, int],
        tokenizer,
        max_length: int = 512,
        replacement_prob: float = 0.4,
        seed: int = 42
    ):
        self.texts = texts
        self.clv_map = clv_map
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.replacement_prob = replacement_prob
        self.rng = random.Random(seed)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Simple phrase replacement (for training, use compress_wrap logic)
        # For now, just tokenize - full replacement can be added later
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_base_model(config: Dict[str, Any], device: str = "cpu"):
    """Load base model and tokenizer."""
    if not HAS_DEPENDENCIES:
        raise ImportError("Required dependencies not available")
    
    # Get model name from config (tolerant of multiple key names)
    base_model_name = (
        config.get("base_model_name")
        or config.get("base_model")
        or config.get("model_name_or_path")
    )
    
    if not base_model_name:
        # Try loading from base_model_config
        base_config_path = config.get("base_model_config")
        if base_config_path:
            with open(base_config_path, "r") as f:
                base_config = yaml.safe_load(f)
            base_model_name = (
                base_config.get("model_name")
                or base_config.get("base_model_name")
                or base_config.get("base_model")
                or base_config.get("model_name_or_path")
            )
    
    if not base_model_name:
        raise ValueError("base_model_name not found in config (tried: base_model_name, base_model, model_name_or_path)")
    
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✓ Model loaded: {model.config.model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def add_clv_tokens(tokenizer, codebook_size: int):
    """Add CLV special tokens to tokenizer."""
    from inference.clv_tokenizer_adapter import add_clv_tokens_to_tokenizer

    print(f"Adding {codebook_size} CLV tokens...")
    tokenizer = add_clv_tokens_to_tokenizer(tokenizer, codebook_size)
    print(f"✓ Tokenizer vocab size: {len(tokenizer)}")

    return tokenizer


def add_pid_tokens(tokenizer, phrase_index_path: Path):
    """Add PID special tokens to tokenizer for lossless compression."""
    from inference.clv_tokenizer_adapter import add_pid_tokens_to_tokenizer

    print(f"Loading phrase index from {phrase_index_path}...")
    from inference.compress_wrap import load_phrase_index
    phrase_to_id, _ = load_phrase_index(phrase_index_path)
    max_pid = max(phrase_to_id.values()) if phrase_to_id else 0

    print(f"Adding PID tokens for {max_pid + 1} phrases...")
    tokenizer = add_pid_tokens_to_tokenizer(tokenizer, max_pid + 1)
    print(f"✓ Tokenizer vocab size: {len(tokenizer)}")

    return tokenizer


def setup_lora(model, config: Dict[str, Any]):
    """Setup LoRA adapters on model."""
    if not HAS_DEPENDENCIES:
        raise ImportError("peft required for LoRA")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none"
    )
    
    print("Setting up LoRA adapters...")
    print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA setup complete")
    print(f"  Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    
    return model


def setup_vq_module(model, config: Dict[str, Any], tokenizer) -> Tuple[nn.Module, VectorQuantizer]:
    """
    Setup VQ module at specified layer.
    
    Returns:
        Tuple of (modified model, vq_module)
    """
    # Get layer index
    vq_layer = config.get("vq_layer", -2)
    
    # Find transformer layers (robust across wrappers)
    layers = _get_transformer_layers_any(model)
    if layers is None:
        raise ValueError("Could not find transformer layers in model (after PEFT).")
    
    num_layers = len(layers)
    
    # Resolve layer index
    if vq_layer < 0:
        layer_idx = num_layers + vq_layer
    else:
        layer_idx = vq_layer
    
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer index: {vq_layer} (model has {num_layers} layers)")
    
    print(f"Setting up VQ module at layer {layer_idx} (index {vq_layer})")
    
    # Hidden size: prefer config; fall back to proj dims if needed
    hidden_dim = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_dim is None:
        try:
            if hasattr(layers[0], "self_attn") and hasattr(layers[0].self_attn, "q_proj"):
                hidden_dim = layers[0].self_attn.q_proj.in_features
        except Exception:
            pass
    if hidden_dim is None:
        hidden_dim = int(config.get("vq_code_dim", 256))  # conservative fallback
    
    # Create VQ module
    vq_module = VectorQuantizer(
        codebook_size=config.get("vq_codebook_size", 8000),
        code_dim=config.get("vq_code_dim", 256),
        input_dim=hidden_dim,
        commitment_cost=config.get("vq_commitment_cost", 0.25),
        ema_decay=config.get("vq_ema_decay", 0.99)
    )
    
    # Store VQ module reference (will be used in forward hook)
    model.vq_module = vq_module
    model.vq_layer_idx = layer_idx
    
    print(f"✓ VQ module setup:")
    print(f"  Codebook size: {vq_module.codebook_size}")
    print(f"  Code dim: {vq_module.code_dim}")
    print(f"  Input dim: {hidden_dim}")
    
    return model, vq_module


def load_codebook_and_map(config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """Load codebook and CLV map."""
    codebook_path = Path(config.get("codebook_path", "artifacts/clv_codebook.npy"))
    clv_map_path = Path(config.get("clv_map_path", "artifacts/clv_map.json"))
    
    codebook = None
    if codebook_path.exists():
        codebook = np.load(codebook_path)
        print(f"✓ Loaded codebook: {codebook.shape}")
    else:
        print(f"⚠ Codebook not found: {codebook_path} (will use random initialization)")
    
    clv_map = {}
    if clv_map_path.exists():
        with open(clv_map_path, "r") as f:
            clv_map = json.load(f)
        print(f"✓ Loaded CLV map: {len(clv_map)} phrases")
    else:
        print(f"⚠ CLV map not found: {clv_map_path}")
    
    return codebook, clv_map


def create_dataset(config: Dict[str, Any], tokenizer, clv_map: Dict[str, int], smoke_test: bool = False) -> Dataset:
    """Create training dataset."""
    dataset_path = config.get("dataset_path", "")
    
    if smoke_test or not dataset_path:
        # Generate synthetic data for smoke testing
        print("⚠ Using synthetic dataset for smoke testing")
        texts = [
            "This is a sample text for training.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ] * 10  # Repeat for small dataset
    else:
        # TODO: Load real dataset
        # For now, use synthetic
        print(f"⚠ Dataset loading from {dataset_path} not yet implemented, using synthetic")
        texts = ["Sample text"] * 100
    
    dataset = SimpleCLVDataset(
        texts=texts,
        clv_map=clv_map,
        tokenizer=tokenizer,
        max_length=config.get("max_seq_length", 512),
        replacement_prob=config.get("clv_replacement_prob", 0.4),
        seed=config.get("seed", 42)
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    return dataset


def compute_loss(
    model,
    batch: Dict[str, torch.Tensor],
    vq_module: VectorQuantizer,
    config: Dict[str, Any],
    device: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss.
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Language modeling loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    lm_loss_fct = nn.CrossEntropyLoss()
    lm_loss = lm_loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # VQ loss (simplified - in full implementation, extract hidden states at VQ layer)
    # For now, use a placeholder
    vq_loss = torch.tensor(0.0, device=device)
    
    # TODO: Extract hidden states at vq_layer_idx and compute VQ loss
    # This requires forward hook or model modification
    
    # Combine losses
    lm_weight = config.get("lm_loss_weight", 1.0)
    vq_weight = config.get("vq_loss_weight", 0.1)
    
    total_loss = lm_weight * lm_loss + vq_weight * vq_loss
    
    loss_dict = {
        "lm_loss": lm_loss.item(),
        "vq_loss": vq_loss.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, loss_dict


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    vq_module: VectorQuantizer,
    config: Dict[str, Any],
    device: str,
    accelerator: Optional[Any] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0
    
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    for batch_idx, batch in enumerate(dataloader):
        # Compute loss
        loss, loss_dict = compute_loss(model, batch, vq_module, config, device)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if accelerator:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        total_loss += loss_dict["total_loss"] * gradient_accumulation_steps
        total_lm_loss += loss_dict["lm_loss"]
        total_vq_loss += loss_dict["vq_loss"]
        num_batches += 1
        
        # Update weights (every gradient_accumulation_steps)
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if accelerator:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss_dict['total_loss']:.4f}")
    
    return {
        "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "avg_lm_loss": total_lm_loss / num_batches if num_batches > 0 else 0.0,
        "avg_vq_loss": total_vq_loss / num_batches if num_batches > 0 else 0.0
    }


def save_checkpoint(
    model,
    tokenizer,
    vq_module: VectorQuantizer,
    output_dir: Path,
    config: Dict[str, Any]
) -> None:
    """Save training artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapter
    adapter_path = output_dir / config.get("adapter_output", "clv_lora_adapter.pt")
    if hasattr(model, "save_pretrained"):
        # PEFT model
        model.save_pretrained(str(output_dir / "lora_adapter"))
        print(f"✓ LoRA adapter saved to {output_dir / 'lora_adapter'}")
    else:
        # Manual save
        adapter_state = {k: v for k, v in model.named_parameters() if "lora" in k.lower()}
        torch.save(adapter_state, adapter_path)
        print(f"✓ LoRA adapter saved to {adapter_path}")
    
    # Save tokenizer
    tokenizer_path = output_dir / config.get("tokenizer_output", "clv_tokenizer_added.json")
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    print(f"✓ Tokenizer saved to {output_dir / 'tokenizer'}")
    
    # Save VQ module
    vq_path = output_dir / config.get("vq_state_output", "clv_vq_state.pt")
    # Save entire VQ module state (includes codebook parameter and buffers)
    vq_state = {
        "codebook": vq_module.codebook.data.clone(),  # Save codebook parameter data
        "config": {
            "codebook_size": vq_module.codebook_size,
            "code_dim": vq_module.code_dim,
            "commitment_cost": vq_module.commitment_cost,
            "ema_decay": vq_module.ema_decay
        }
    }
    # Also save EMA buffers if they exist
    if hasattr(vq_module, "_ema_cluster_size"):
        vq_state["ema_cluster_size"] = vq_module._ema_cluster_size.data.clone()
    if hasattr(vq_module, "_ema_w"):
        vq_state["ema_w"] = vq_module._ema_w.data.clone()
    if hasattr(vq_module, "_n_updates"):
        vq_state["n_updates"] = vq_module._n_updates.data.clone()
    
    torch.save(vq_state, vq_path)
    print(f"✓ VQ module saved to {vq_path}")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train CLV LoRA model with VQ module",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_clv_lora_vq.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (overrides config, 'cpu' or 'cuda')"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with tiny model and synthetic data (for local testing)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLV-Lang Training Script")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: This script is designed for remote GPU execution.")
    print("   Local CPU runs are only for smoke tests.")
    print()
    
    if args.smoke_test:
        print("🧪 Running in SMOKE TEST mode (tiny model, synthetic data)")
        print()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"✓ Loaded config from {config_path}")
    
    # Override output dir if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    output_dir = Path(config.get("output_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = args.device or config.get("device", "cuda")
    if device == "cpu" and not args.smoke_test:
        print("⚠️  WARNING: Training on CPU is extremely slow.")
        print("   Use --smoke-test for local testing, or run on remote GPU.")
    
    # Set random seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"✓ Random seed: {seed}")
    print()
    
    # ========================================================================
    # STEP 1: Load base model and tokenizer
    # ========================================================================
    print("Step 1: Loading base model and tokenizer...")
    model, tokenizer = load_base_model(config, device=device)
    print()
    
    # ========================================================================
    # STEP 2: Add CLV/PID tokens
    # ========================================================================
    print("Step 2: Adding CLV/PID tokens...")
    codebook_size = config.get("vq", {}).get("codebook_size", 8000)

    # Always add CLV tokens (for codebook mode compatibility)
    tokenizer = add_clv_tokens(tokenizer, codebook_size)

    # Add PID tokens if lossless mode is enabled
    if config.get("lossless", False):
        phrase_index_path = Path(config.get("phrase_index", "data/phrase_index.jsonl"))
        if phrase_index_path.exists():
            tokenizer = add_pid_tokens(tokenizer, phrase_index_path)
        else:
            print(f"⚠ Lossless mode enabled but phrase_index not found: {phrase_index_path}")

    # Resize model embeddings if needed
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    print()
    
    # ========================================================================
    # STEP 3: Load codebook and CLV map
    # ========================================================================
    print("Step 3: Loading codebook and CLV map...")
    codebook, clv_map = load_codebook_and_map(config)
    print()
    
    # ========================================================================
    # STEP 4: Setup LoRA adapters
    # ========================================================================
    print("Step 4: Setting up LoRA adapters...")
    model = setup_lora(model, config)
    print()
    
    # ========================================================================
    # STEP 5: Setup VQ module
    # ========================================================================
    print("Step 5: Setting up VQ module...")
    if args.smoke_test and config.get("skip_vq_on_smoke", True):
        print("⚠ Skipping VQ module in smoke-test (set skip_vq_on_smoke: false to enable).")
        vq_module = VectorQuantizer(
            codebook_size=config.get("vq_codebook_size", 8000),
            code_dim=config.get("vq_code_dim", 256),
            input_dim=config.get("vq_code_dim", 256),
            commitment_cost=config.get("vq_commitment_cost", 0.25),
            ema_decay=config.get("vq_ema_decay", 0.99)
        )
        model.vq_module = vq_module
        model.vq_layer_idx = -1
    else:
        model, vq_module = setup_vq_module(model, config, tokenizer)
    print()
    
    # ========================================================================
    # STEP 6: Create dataset
    # ========================================================================
    print("Step 6: Creating dataset...")
    dataset = create_dataset(config, tokenizer, clv_map, smoke_test=args.smoke_test)
    
    # Create dataloader
    batch_size = config.get("batch_size", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    print(f"✓ DataLoader created: batch_size={batch_size}, batches={len(dataloader)}")
    print()
    
    # ========================================================================
    # STEP 7: Setup optimizer
    # ========================================================================
    print("Step 7: Setting up optimizer...")
    learning_rate = config.get("learning_rate", 2e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.get("weight_decay", 0.01)
    )
    print(f"✓ Optimizer: AdamW, lr={learning_rate}")
    print()
    
    # ========================================================================
    # STEP 8: Training loop
    # ========================================================================
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    print("⚠️  NOTE: Full training loop requires remote GPU (A100 80GB).")
    print("   Current run is for smoke testing only.")
    print()
    
    num_epochs = config.get("num_epochs", 3) if not args.smoke_test else 1
    
    # Setup accelerator (if available and on GPU)
    accelerator = None
    if device == "cuda" and HAS_DEPENDENCIES:
        try:
            accelerator = Accelerator()
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
            print("✓ Accelerate setup complete")
        except Exception as e:
            print(f"⚠ Accelerate setup failed: {e}, continuing without it")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    if args.smoke_test:
        # For smoke testing, just run a few batches:
        print("🧪 Running smoke test (few batches only)...")
        model.train()
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Just 2 batches for smoke test
                break
            loss, loss_dict = compute_loss(model, batch, vq_module, config, device)
            print(f"  Batch {i}: {loss_dict}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("✓ Smoke test complete")
    else:
        # Full training loop for production
        print("🚀 Starting full training loop...")
        model.train()
        
        # Get training config
        max_steps = config.get("max_steps", None)
        save_steps = config.get("save_steps", 500)
        logging_steps = config.get("logging_steps", 10)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Forward pass
                loss, loss_dict = compute_loss(model, batch, vq_module, config, device)
                
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Update weights (after accumulation)
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    global_step += 1
                    total_loss += loss_dict["total_loss"]
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / logging_steps
                        print(f"  Step {global_step}: loss={avg_loss:.4f}, {loss_dict}")
                        total_loss = 0.0
                    
                    # Save checkpoint periodically (based on steps)
                    if save_steps > 0 and global_step % save_steps == 0:
                        print(f"\n💾 Saving checkpoint at step {global_step}...")
                        save_checkpoint(model, tokenizer, vq_module, output_dir, config)
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Check max_steps
                if max_steps is not None and global_step >= max_steps:
                    print(f"\n✓ Reached max_steps ({max_steps}), stopping training.")
                    break
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"\nEpoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")
            
            # Check max_steps
            if max_steps is not None and global_step >= max_steps:
                break
        
        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"{'='*70}")
    
    print()
    
    # ========================================================================
    # STEP 9: Save final checkpoint
    # ========================================================================
    print("Step 9: Saving checkpoint...")
    save_checkpoint(model, tokenizer, vq_module, output_dir, config)
    print()
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print()
    print("Checkpoints saved to:", output_dir)
    print("  - LoRA adapter: lora_adapter/")
    print("  - Tokenizer: tokenizer/")
    print("  - VQ module: clv_vq_state.pt")
    print()
    print("Next steps:")
    print("  1. Run evaluations: bash scripts/run_all_evals.sh")
    print("  2. Test inference with trained model")
    print("  3. For distributed training, configure accelerate")
    print()


if __name__ == "__main__":
    main()
