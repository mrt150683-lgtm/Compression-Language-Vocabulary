"""
Embed phrase spans: Extract embeddings for mined phrases using base model.

This module loads the base model and extracts hidden state representations
for each phrase occurrence. For each phrase:
- Encode the sequence containing the phrase
- Extract hidden states for that span (from layer L-2)
- Aggregate via mean-pooling

CPU-friendly mode:
- Uses small model (configurable via --model_name)
- For POC, allows random vectors if no model available

Output:
- data/phrase_embeddings.npy: Embedding matrix
- data/phrase_index.jsonl: Maps embeddings row → phrase string
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, will use random vectors")


def load_phrases(phrases_path: Path) -> List[Dict[str, Any]]:
    """
    Load mined phrases from JSONL.
    
    Args:
        phrases_path: Path to mined_phrases.jsonl
    
    Returns:
        List of phrase dictionaries
    """
    if not phrases_path.exists():
        raise FileNotFoundError(f"Phrases file not found: {phrases_path}")
    
    phrases = []
    with open(phrases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                phrases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return phrases


def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    """
    Load model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer) or (None, None) if failed
    """
    if not HAS_TRANSFORMERS:
        return None, None
    
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        print(f"✓ Model loaded on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Failed to load model {model_name}: {e}")
        return None, None


def extract_phrase_embedding(
    model,
    tokenizer,
    phrase: str,
    layer_idx: int = -2,
    device: str = "cpu"
) -> np.ndarray:
    """
    Extract embedding for a phrase using model.
    
    Args:
        model: Model (or None for random)
        tokenizer: Tokenizer (or None)
        phrase: Phrase string
        layer_idx: Layer to extract from (default: -2)
        device: Device to run on
    
    Returns:
        Embedding vector
    """
    if model is None or tokenizer is None:
        # Fallback: random vector (for POC/testing)
        # Use deterministic seed based on phrase
        random.seed(hash(phrase) % (2**32))
        embedding_dim = 256  # Default embedding dimension
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    # Tokenize phrase
    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    # Extract from specified layer
    if layer_idx < 0:
        layer_idx = len(hidden_states) + layer_idx
    
    # Get hidden states for the phrase tokens
    # Shape: (batch_size, seq_len, hidden_dim)
    layer_hidden = hidden_states[layer_idx]
    
    # Mean pool over sequence length (excluding padding)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        # Mask out padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = layer_hidden * mask
        embedding = masked_hidden.sum(dim=1) / mask.sum(dim=1)
    else:
        embedding = layer_hidden.mean(dim=1)
    
    # Convert to numpy
    embedding = embedding.cpu().numpy().squeeze()
    
    return embedding.astype(np.float32)


def embed_all_phrases(
    phrases: List[Dict[str, Any]],
    model,
    tokenizer,
    layer_idx: int = -2,
    device: str = "cpu",
    batch_size: int = 8
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Embed all phrases from mined phrases.
    
    Args:
        phrases: List of phrase dictionaries
        model: Model (or None for random)
        tokenizer: Tokenizer (or None)
        layer_idx: Layer to extract from
        device: Device to run on
        batch_size: Batch size (for model inference, not used for random)
    
    Returns:
        Tuple of (embedding matrix, phrase index mapping)
    """
    print(f"Embedding {len(phrases)} phrases...")
    
    embeddings = []
    phrase_index = []
    
    for idx, phrase_dict in enumerate(phrases):
        phrase = phrase_dict.get("phrase", "")
        if not phrase:
            continue
        
        # Extract embedding
        embedding = extract_phrase_embedding(
            model, tokenizer, phrase, layer_idx, device
        )
        
        embeddings.append(embedding)
        phrase_index.append({
            "index": idx,
            "phrase": phrase
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(phrases)} phrases...")
    
    # Stack into matrix
    embedding_matrix = np.stack(embeddings, axis=0)
    
    print(f"✓ Embedded {len(embeddings)} phrases")
    print(f"  Embedding shape: {embedding_matrix.shape}")
    
    return embedding_matrix, phrase_index


def save_embeddings(
    embeddings: np.ndarray,
    phrase_index: List[Dict[str, Any]],
    embeddings_path: Path,
    index_path: Path
) -> None:
    """
    Save embeddings and index to disk.
    
    Args:
        embeddings: Embedding matrix (N x D)
        phrase_index: List of index mappings
        embeddings_path: Path to save .npy file
        index_path: Path to save .jsonl file
    """
    # Save embeddings
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings to {embeddings_path}")
    
    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        for entry in phrase_index:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Saved phrase index to {index_path}")


def main():
    """CLI entry point for embedding extraction."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases_json", required=True)
    ap.add_argument("--output_embeddings", required=True)
    ap.add_argument("--output_index", required=True)
    ap.add_argument("--mode", choices=["hf", "dummy"], default="hf")
    ap.add_argument("--model_name", default="hf-internal-testing/tiny-random-gpt2")
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--max_phrases", type=int, default=10000)
    
    args = ap.parse_args()
    
    # Load phrases
    phrases = []
    with open(args.phrases_json, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            phrases.append(obj["phrase"])
            if len(phrases) >= args.max_phrases:
                break
    
    if args.mode == "dummy":
        rng = np.random.default_rng(42)
        embs = rng.normal(size=(len(phrases), args.embedding_dim)).astype(np.float32)
    else:
        # very small HF model path to keep CPU-friendly
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tok = AutoTokenizer.from_pretrained(args.model_name)
        mdl = AutoModelForCausalLM.from_pretrained(args.model_name)
        mdl.eval()
        
        # simple mean-pooled last hidden state per phrase
        embs_list = []
        for ph in phrases:
            with torch.no_grad():
                ids = tok(ph, return_tensors="pt")
                out = mdl(**ids, output_hidden_states=True)
                # last hidden states: [1, seq, hidden]
                hs = out.hidden_states[-1][0].mean(dim=0).cpu().numpy()
                embs_list.append(hs)
        
        # pad/trim to a fixed dim if needed
        emb_dim = len(embs_list[0])
        embs = np.stack(embs_list).astype(np.float32)
    
    # Save
    output_emb_dir = os.path.dirname(args.output_embeddings)
    if output_emb_dir:
        os.makedirs(output_emb_dir, exist_ok=True)
    np.save(args.output_embeddings, embs)
    
    output_idx_dir = os.path.dirname(args.output_index)
    if output_idx_dir:
        os.makedirs(output_idx_dir, exist_ok=True)
    with open(args.output_index, "w", encoding="utf-8") as w:
        for i, ph in enumerate(phrases):
            w.write(json.dumps({"row": i, "phrase": ph}) + "\n")


if __name__ == "__main__":
    main()
