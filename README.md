# CLV-Lang

**Compression Language Vocabulary:** Discrete latent phrase codes for cheaper, longer context without trashing quality.

## What This Is

A reproducible demonstration of a discrete phrase-code layer (CLV) for LLMs. This project trains an augmenting layer + wrappers so an existing open LLM can operate over compressed semantic phrase codes (`<clv:####>`) that:

- Reduce token count for real-world inputs (~25–30% compression target)
- Preserve task performance (negligible loss in perplexity / downstream quality)
- Improve effective long-context capacity at fixed window size
- Provide runtime & KV cache savings

## What We Show

- **Compression:** ~25–30% token compression on real inputs
- **Quality:** Negligible loss in perplexity / downstream quality
- **Long Context:** Improved long-context effectiveness under fixed window
- **Efficiency:** Runtime & KV cache savings

## What This Is NOT

- Not changing transformer math
- Not RAG
- Not hand-wavy "AI magic"

## Compression Modes

CLV-Lang supports two compression modes:

### Codebook Mode (Default, Lossy)

The default mode uses a learned codebook to map phrases to cluster codes. This provides:
- **Compression:** ~25–30% token reduction
- **Efficiency:** Fixed vocabulary size (typically 8K codes)
- **Trade-off:** Slight semantic loss due to clustering (multiple phrases may share a code)

**Usage:**
```bash
# Compression
python inference/compress_wrap.py \
  --mode codebook \
  --clv_map artifacts/clv_map.json \
  --input_text "your text here"

# Decompression
python inference/decompress_wrap.py \
  --mode codebook \
  --clv_map artifacts/clv_map.json \
  --input_text "<clv:0001> your text"
```

### Lossless Mode (Exact Phrase IDs)

A new mode that uses exact phrase IDs for perfect round-trip decompression. This provides:
- **Perfect Reconstruction:** Exact phrase recovery (no semantic loss)
- **Larger Vocabulary:** One token per unique phrase (vocabulary size = number of phrases)
- **Use Cases:** When exact text reconstruction is required, debugging, or when phrase count is manageable

**Usage:**
```bash
# Compression
python inference/compress_wrap.py \
  --mode lossless \
  --phrase_index data/phrase_index.jsonl \
  --input_text "your text here"

# Decompression
python inference/decompress_wrap.py \
  --mode lossless \
  --phrase_index data/phrase_index.jsonl \
  --input_text "<pid:000123> your text"
```

**Key Differences:**
- **Codebook mode:** Uses `<clv:####>` tokens (4 digits, 0-7999 typically)
- **Lossless mode:** Uses `<pid:######>` tokens (6 digits, one per phrase)
- **Lossless mode** requires `data/phrase_index.jsonl` instead of `artifacts/clv_map.json`
- **Lossless mode** automatically adds all PID tokens to the tokenizer based on max phrase ID

## Quick Start

### Requirements

- **Hardware:** 1× A100 80GB or equivalent (for training)
- **Software:** Python 3.10+, CUDA 12.x compatible environment

### One-Liner Reproduction

```bash
./scripts/bootstrap.sh && ./scripts/train_poc.sh && ./scripts/run_all_evals.sh
```

**Note:** GPU-specific steps (training, heavy inference) must be run on remote GPU. Local CPU-only setup is for development/testing only.

### Manual Steps

1. **Bootstrap environment:**
   ```bash
   ./scripts/bootstrap.sh
   ```

2. **Train POC model:**
   ```bash
   ./scripts/train_poc.sh
   ```
   ⚠️ **Requires GPU:** This step must be run on remote A100.

3. **Run all evaluations:**
   ```bash
   ./scripts/run_all_evals.sh
   ```

## Repository Structure

```
clv-lang/
├── docker/              # Docker environment setup
├── configs/             # YAML configuration files
├── data/                # Data storage and documentation
├── mining/              # Phrase mining pipeline
├── training/            # VQ module and LoRA training
├── inference/           # Compression/decompression wrappers
├── eval/                # Evaluation suite
├── ablations/           # Ablation study scripts
├── scripts/             # High-level orchestration scripts
├── mcp/                 # MCP server for distributed processing
├── artifacts/           # Model artifacts (codebook, adapters, etc.)
└── reports/             # Evaluation reports and plots
```

## Success Criteria

- **Compression ratio:** ≥ 0.25 (25%+ reduction) on eval corpora
- **Perplexity:** ΔPPL ≤ +3% on Wikitext-103 and held-out sets
- **LongBench/SCROLLS:** Degradation ≤ 2% absolute
- **Needle-in-a-Haystack:** CLV@16k ≥ Baseline@16k, approaches Baseline@32k
- **Latency:** ≥ 20% reduction in ms/token or $/effective-token

## License

# TODO: Add license information

## Citation

```bibtex
# TODO: Add citation once published
```

