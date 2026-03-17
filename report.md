# Homework 1: Sentiment Analysis with Transformer Models

## Part 1 — Dataset Selection

| Property | Value |
|---|---|
| **Dataset** | SST-2 (Stanford Sentiment Treebank — Binary) |
| **Source** | Socher et al., 2013 — *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. Available through the GLUE benchmark (`glue/sst2` on Hugging Face). |
| **Number of Samples** | 68,221 total (67,349 train + 872 validation) |
| **Label Classes** | Binary — `0` (negative), `1` (positive) |
| **Split** | Train: 67,349 / Validation: 436 / Test: 436 (validation split 50/50 because the official test labels are hidden) |

### Example Records

| Label | Sentence |
|---|---|
| Positive (1) | "the rock is destined to be the 21st century 's new conan and that he 's going to make a splash..." |
| Negative (0) | "simplistic , silly and tedious ." |
| Positive (1) | "it 's a charming and often affecting journey ." |

---

## Part 2 — Fine-Tuning BERT

### How BERT Is Adapted for Classification

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model pretrained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). To adapt it for classification:

1. The input sentence is tokenized and prepended with a special `[CLS]` token.
2. BERT encodes the full input with bidirectional self-attention across all 12 transformer layers.
3. The hidden state corresponding to the `[CLS]` token (a 768-dimensional vector) serves as the aggregate sentence representation.
4. A randomly initialized linear layer is placed on top of `[CLS]`: `logits = W · h_[CLS] + b`, where `W ∈ R^{2×768}`.
5. The model is fine-tuned end-to-end using cross-entropy loss on the classification labels.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 0.01) |
| Learning Rate | 2 × 10⁻⁵ |
| Batch Size | 32 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| Scheduler | Linear warmup (10% of steps) + linear decay |
| Gradient Clipping | max norm = 1.0 |

### Evaluation Results

Results will be populated after running the experiment:

| Metric | Score |
|---|---|
| Accuracy | *(run `python main.py` to populate)* |
| Precision | — |
| Recall | — |
| F1 Score | — |

---

## Part 3 — Fine-Tuning GPT-1

### How GPT-1 Is Adapted for Sentiment Classification

GPT-1 (Radford et al., 2018) is a decoder-only transformer pretrained with autoregressive language modeling. Fine-tuning for classification works as follows:

1. **Input structuring**: The sentence is tokenized and a special `[CLS]` token is appended at the end: `<sentence tokens> [CLS]`. Because GPT-1 processes tokens left-to-right, the `[CLS]` token at the end can attend to all preceding tokens via causal (masked) self-attention.
2. **Classification head**: The hidden state at the `[CLS]` position is extracted from the final transformer layer and passed through a linear projection to produce class logits.
3. **Training objective**: The original GPT-1 paper uses a combined loss during fine-tuning:
   - **Classification loss** (L_cls): Standard cross-entropy between predicted logits and gold labels.
   - **Auxiliary language modeling loss** (L_lm): The standard autoregressive next-token prediction loss, which acts as a regularizer.
   - Combined: `L = L_cls + λ · L_lm` where λ is typically 0.5.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 0.01) |
| Learning Rate | 6.25 × 10⁻⁵ |
| Batch Size | 32 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| Scheduler | Linear warmup (10%) + linear decay |
| Gradient Clipping | max norm = 1.0 |

### Evaluation Results

| Metric | Score |
|---|---|
| Accuracy | *(run `python main.py` to populate)* |
| Precision | — |
| Recall | — |
| F1 Score | — |

---

## Part 4 — Model Comparison

### Architecture Differences

| Aspect | BERT | GPT-1 |
|---|---|---|
| Type | Encoder-only | Decoder-only |
| Attention | Bidirectional (full) self-attention | Causal (masked) self-attention |
| Layers | 12 transformer blocks | 12 transformer blocks |
| Hidden Size | 768 | 768 |
| Parameters | ~110M | ~117M |
| Pre-training | MLM + NSP | Autoregressive LM |
| Context | Sees both left and right context | Sees only left context |

### Differences in Training Objectives

- **BERT** is pretrained with two objectives: (1) Masked Language Modeling — randomly mask 15% of tokens and predict them, and (2) Next Sentence Prediction — determine if two sentences are consecutive. This bidirectional pretraining enables rich contextual representations.
- **GPT-1** is pretrained with standard autoregressive language modeling — predicting the next token given all previous tokens. During fine-tuning, an auxiliary LM loss is added alongside the task-specific loss.

### Performance Differences

BERT is expected to outperform GPT-1 on SST-2 because:
1. **Bidirectional context**: BERT attends to the full input simultaneously, which produces richer sentence representations for classification.
2. **[CLS] token design**: BERT's `[CLS]` token at the start is specifically trained (via NSP) to capture sentence-level semantics.
3. **Task alignment**: Classification requires understanding the entire sentence holistically — bidirectional attention is a natural fit.

GPT-1's causal attention means each token can only attend to tokens before it, creating an information bottleneck that is suboptimal for tasks requiring global understanding.

### Advantages and Disadvantages

| | BERT | GPT-1 |
|---|---|---|
| **Advantages** | Superior on NLU tasks (classification, NER, QA); bidirectional context captures nuanced meaning | Natural for text generation; unified framework for many tasks via prompting; auxiliary LM loss acts as regularizer |
| **Disadvantages** | Cannot generate text autoregressively; pretrained with artificial `[MASK]` tokens (train/inference mismatch) | Left-to-right attention limits understanding tasks; generally lower accuracy on classification benchmarks |

---

## Part 5 — Conceptual Questions

### Question 1: Multi-Head Attention Mechanism

The Multi-Head Attention mechanism is the core building block of the transformer architecture (Vaswani et al., 2017).

**Query, Key, and Value Matrices**

For each input token representation `x_i`, three vectors are computed via learned linear projections:
- **Query (Q)**: Represents what information this token is looking for.
- **Key (K)**: Represents what information this token contains / advertises.
- **Value (V)**: Represents the actual content to be aggregated.

Formally: `Q = XW_Q`, `K = XW_K`, `V = XW_V`, where `W_Q, W_K ∈ R^{d_model × d_k}` and `W_V ∈ R^{d_model × d_v}`.

**The Attention Formula**

Scaled dot-product attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- `QK^T` computes similarity scores between all query-key pairs.
- Division by `√d_k` prevents the dot products from becoming too large (which would push softmax into regions with vanishing gradients).
- Softmax normalizes scores into a probability distribution over positions.
- Multiplication by `V` produces a weighted sum of value vectors.

**The Role of Multiple Attention Heads**

Instead of performing a single attention function, the model uses `h` parallel attention heads:

```
head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
```

Each head has its own set of projection matrices (`W_Q^i, W_K^i, W_V^i`) with reduced dimensionality (`d_k = d_model / h`), so the total computation cost is similar to single-head attention with full dimensionality.

**Why Multi-Head Attention Improves Representation Learning**

1. **Diverse attention patterns**: Different heads can learn to attend to different types of relationships — syntactic dependencies, semantic similarity, positional patterns, etc.
2. **Subspace representation**: Each head operates in a different linear subspace, allowing the model to jointly attend to information from different representation subspaces at different positions.
3. **Robustness**: Multiple heads provide redundancy; if one head learns a spurious pattern, others can compensate.
4. **Richer representations**: The concatenated output from all heads captures multiple facets of the input relationships simultaneously, producing richer token representations than a single attention function could.

---

### Question 2: Loss Function for Machine Translation

**Training Objective**

In machine translation (seq2seq), the transformer is trained to maximize the probability of the correct target sequence given the source sequence. The model generates one token at a time, conditioned on the source and all previously generated target tokens (teacher forcing).

**Cross-Entropy Loss**

At each decoding time step `t`, the model produces a probability distribution over the vocabulary `V`. The loss is the cross-entropy between this distribution and the one-hot encoded ground truth token:

```
L = -1/T Σ_{t=1}^{T} log P(y_t | y_{<t}, x)
```

where:
- `y_t` is the correct target token at position `t`
- `y_{<t}` are all previous target tokens
- `x` is the source sequence
- `T` is the target sequence length

The cross-entropy loss penalizes the model when it assigns low probability to the correct next token, pushing it toward producing sharper, more accurate predictions.

**Which Parameters Are Updated**

During training, **all** model parameters are updated via backpropagation:
- **Encoder parameters**: Token embeddings, positional embeddings, multi-head self-attention weights (W_Q, W_K, W_V, W_O), feed-forward network weights, and layer normalization parameters.
- **Decoder parameters**: Same types of parameters, plus the cross-attention weights that attend to encoder outputs.
- **Output projection**: The final linear layer that projects decoder hidden states to vocabulary logits (often tied with the decoder's token embedding matrix).

Optimization is typically done with Adam, using a learning rate warmup schedule.

---

### Question 3: Masked Self-Attention in the Decoder

**The Autoregressive Property**

The transformer decoder generates output tokens one at a time, where each token's probability depends only on previously generated tokens:

```
P(y_1, ..., y_T) = Π_{t=1}^{T} P(y_t | y_1, ..., y_{t-1})
```

This autoregressive factorization is essential: at inference time, future tokens do not exist yet, so the model can only condition on the past.

**How Masking Ensures Correct Training Behavior**

During training, the full target sequence is available (teacher forcing), so the decoder could theoretically "cheat" by looking at future tokens. Masked self-attention prevents this by applying a causal mask — an upper-triangular matrix of `-∞` values that, after softmax, zeroes out attention weights to future positions:

```
MaskedAttention(Q, K, V) = softmax(QK^T / √d_k + M) V
```

where `M_{ij} = 0` if `i ≥ j` (allowed) and `M_{ij} = -∞` if `i < j` (blocked).

This ensures:
1. **Training-inference consistency**: The model learns under the same constraints it faces at inference time — it never sees future tokens.
2. **Parallel training**: Unlike RNNs, all positions can be computed simultaneously during training because the mask enforces causality without sequential computation.
3. **Correct gradient signals**: Without the mask, the model would learn shortcuts (copying future tokens) that would not work at inference time, leading to poor generalization.

---

### Question 4: BERT Pretraining Tasks

**Masked Language Modeling (MLM)**

In MLM, 15% of input tokens are selected for masking. Of these:
- 80% are replaced with the special `[MASK]` token
- 10% are replaced with a random vocabulary token
- 10% are left unchanged

The model must predict the original identity of these tokens using the surrounding bidirectional context:

```
L_MLM = -Σ_{i ∈ masked} log P(x_i | x_{\masked})
```

The 80/10/10 split mitigates the mismatch between pretraining (which uses `[MASK]`) and fine-tuning (which does not).

**Next Sentence Prediction (NSP)**

The model receives two segments (sentence A and sentence B) and must predict whether B is the actual next sentence after A (label: `IsNext`) or a random sentence (label: `NotNext`). The `[CLS]` token's representation is fed into a binary classifier for this task.

```
L_NSP = -[y · log P(IsNext) + (1-y) · log P(NotNext)]
```

NSP teaches the model inter-sentence relationships, which benefits tasks like natural language inference and question answering.

**Why Standard Language Modeling Is Not Used for BERT**

Standard (autoregressive) language modeling only allows left-to-right or right-to-left context — the model predicts the next token given only the preceding tokens. BERT's key innovation is **bidirectional** representation learning: by masking tokens randomly and predicting them from both left and right context simultaneously, BERT produces much richer contextual embeddings. Standard LM cannot achieve this because seeing the target token in the context would make prediction trivial (the model would just copy it). MLM solves this by hiding specific tokens and forcing the model to reconstruct them from surrounding context in all directions.

---

### Question 5: GPT-1 Pretraining Task

**Autoregressive Language Modeling**

GPT-1 is pretrained with a standard left-to-right language modeling objective. Given a sequence of tokens `(u_1, u_2, ..., u_n)`, the model maximizes:

```
L_LM = Σ_{i=1}^{n} log P(u_i | u_1, ..., u_{i-1}; θ)
```

At each position `i`, the model predicts the next token given all preceding tokens. This is implemented via masked (causal) self-attention in the transformer decoder, which ensures token `i` can only attend to positions `1` through `i`.

**How GPT-1 Predicts the Next Token**

1. Input tokens are embedded (token embedding + positional embedding).
2. The sequence passes through 12 transformer decoder layers, each with masked multi-head self-attention and a feed-forward network.
3. The output hidden state at position `i` encodes all information from tokens `1` to `i`.
4. A linear layer projects this hidden state to vocabulary-size logits, followed by softmax:
   ```
   P(u_{i+1} | u_{1:i}) = softmax(h_i W_e^T)
   ```
   where `W_e` is the token embedding matrix (weight tying).

**How Different Downstream Tasks Are Handled During Fine-Tuning**

GPT-1 uses a task-specific input transformation to convert different tasks into a format compatible with its sequential architecture:

- **Classification**: `<start> text <extract>` — The hidden state at `<extract>` (the final token) is fed into a linear classifier.
- **Entailment**: `<start> premise <delim> hypothesis <extract>` — Both sentences are concatenated with a delimiter.
- **Similarity**: Both orderings `<start> A <delim> B <extract>` and `<start> B <delim> A <extract>` are processed independently, and their representations are added before classification.
- **Multiple Choice**: Each option is processed as `<start> context <delim> answer_i <extract>`, and a softmax is applied over all options.

In all cases, a linear classification head is added, and the combined loss `L = L_task + λ · L_LM` (with λ = 0.5) is used. The auxiliary LM loss improves generalization and accelerates convergence during fine-tuning.
