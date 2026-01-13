# LLM Personalization Project - Progress Tracker

Based on: "Efficient Personalization of LLMs via Compact Context Representations"

## Goal
Implement and compare methods for injecting user context into LLMs efficiently.

---

## Implemented

### 1. Text Baseline
- [x] `TextBaselineLLM` - Injects user context as raw text prefix
- [x] Configurable templates for context formatting
- [x] Context truncation support
- [x] Demo script: `scripts/run_baseline.py`

**Result:** Works but uses 40-100+ tokens for user context.

### 2. E2P (Embedding-to-Prefix) - 1 Token
- [x] `SentenceTransformerEncoder` - Encodes user context to 384-dim embedding
- [x] `PrefixProjector` - MLP that maps embedding to soft tokens
- [x] `E2PLLM` - Full model with soft prefix injection
- [x] Training script: `scripts/train_e2p.py`
- [x] Trained on GPU (100 epochs)

**Result:**
- Compression: **40-92x** (user context → 1 soft token)
- Loss: 4.08 → **0.67** after 100 epochs
- **Exact match** on 4/12 training samples (greetings, style adaptation)
- Trainable params: **462K**

### 3. E2P (Embedding-to-Prefix) - 5 Tokens
- [x] Multi-token variant for richer representation
- [x] Trained on GPU (100 epochs)

**Result:**
- Compression: **8-18x**
- Loss: → **0.001** (overfits on small dataset)
- Trainable params: **5.3M**
- More capacity but needs more data to generalize

### 4. PERSOMA (Two-Stage Encoder)
- [x] `HistoryEncoder` - Encodes each history item separately
- [x] `PerceiverResampler` - Cross-attention compression to N tokens
- [x] `MLPAdapter` - Simple mean-pool + MLP alternative
- [x] `PERSOMALLM` - Full model integration
- [x] Training script: `scripts/train_persoma.py`
- [x] Trained Perceiver variant on GPU

**Result:**
- Handles variable-length histories naturally
- Loss: 3.79 → **2.56** (needs more data)
- Trainable params: **8.2M** (Perceiver)
- Best suited for long/diverse user histories

### 5. Comparison Framework
- [x] `scripts/compare_methods.py` - Side-by-side evaluation
- [x] All methods compared on same test cases

---

## Key Results

| Method              | Soft Tokens | Params | Compression | Final Loss | Quality |
|---------------------|-------------|--------|-------------|------------|---------|
| Text Baseline       | 40-100      | 0      | 1x          | N/A        | Good    |
| E2P (1 token)       | 1           | 462K   | 40-92x      | 0.67       | **Best**|
| E2P (5 tokens)      | 5           | 5.3M   | 8-18x       | 0.001      | Overfit |
| PERSOMA (Perceiver) | 4           | 8.2M   | 10-25x      | 2.56       | Needs data |

**Winner for small data:** E2P (1 token) - best compression, learns exact patterns

---

## Trained Models

```
e2p_projector.pt      # E2P 1-token, 100 epochs, loss 0.67
e2p_5tok.pt           # E2P 5-token, 100 epochs, loss 0.001
persoma_perceiver.pt  # PERSOMA Perceiver, 100 epochs, loss 2.56
```

---

### 6. LaMP Benchmark Training
- [x] LaMP dataset loader with auto-download
- [x] E2P training on LaMP-2 (Movie Tagging)
- [x] Evaluation script with accuracy metrics

**Training Results (LaMP-2, 500 samples, 5 epochs):**
- Train loss: 1.28 → **0.065**
- Dev loss: **0.186**
- Model learned to output tag format (vs narratives)

**Evaluation Results (100 dev samples):**
- Exact Match Accuracy: **2%**
- Tag Extraction Accuracy: **6%**
- Model heavily biased toward romance/fantasy tags

**Analysis:** E2P learns the output format but struggles with 15-way classification on limited data. The user context encoding is working, but needs more training data to discriminate between tag categories.

---

## TODO

### Cross-Attention Memory (Flamingo-style)
- [ ] Add gated cross-attention layers to LLM
- [ ] Implement memory bank for user context
- [ ] Compare with prefix-based approaches

### Retrieval-Augmented Personalization
- [ ] Add FAISS vector index for user history
- [ ] Implement retrieval + projection pipeline
- [ ] Test with large user histories

### Evaluation
- [ ] Implement BLEU/ROUGE metrics
- [ ] User study / preference ranking
- [ ] Latency benchmarking

### Scale Up
- [ ] Larger training dataset (1000+ samples)
- [ ] Larger LLM (1B+ params)
- [ ] Real user data (LaMP benchmark)

---

## Files Structure

```
personalize/
├── src/personalize/
│   ├── datasets/          # User context & datasets
│   ├── encoders/          # User encoder, prefix projector, PERSOMA
│   ├── models/            # TextBaseline, E2P, PERSOMA LLM
│   └── training/          # Training utilities
├── scripts/
│   ├── run_baseline.py    # Test text baseline
│   ├── run_e2p.py         # Test E2P (--projector-path)
│   ├── train_e2p.py       # Train E2P projector
│   ├── train_persoma.py   # Train PERSOMA adapter
│   └── compare_methods.py # Compare all methods
├── configs/
│   └── baseline.json
├── e2p_projector.pt       # Trained E2P 1-token
├── e2p_5tok.pt            # Trained E2P 5-token
└── persoma_perceiver.pt   # Trained PERSOMA
```

---

## Notes

- Using SmolLM2-135M-Instruct for fast iteration
- Remote GPU: ani@10.0.0.41 (RTX A2000 8GB)
- 12 training samples in demo dataset
- E2P (1 token) is the sweet spot for small data scenarios
