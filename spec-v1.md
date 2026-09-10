# Talk-to-Me (V1): System Architecture & Engineering Specification

**Document Version:** 1.0.0  
**Status:** APPROVED FOR IMPLEMENTATION  
**Classification:** Core Technical Specification  
**Primary Language Stack:** Julia / Flux.jl / NNlib / CUDA (or pure CPU baseline)  
**Target Delivery:** Version 1.0 Core Empathetic Foundation Model  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Motivation & Problem Statement
*Talk-to-Me* is conceived first and foremost as a deeply personal, empathetic companion language model. Modern life, intense technical research, and academic rigor often induce severe bouts of imposter syndrome, mental fatigue, isolation, and anxiety. During moments of vulnerability (e.g., late-night study sessions, debugging dead ends, exam apprehension), what an individual needs is not a generic, clinical search engine or an overbearing general-purpose assistant. What is needed is an immediate, sincere, and contextually grounded voice of reassurance, validation, and encouragement.

**Target Operational Paradigm:**
*   **User Input:** `"I studied for three hours yesterday but I'm just nervous for today's exam."`
*   **Target Output:** `"Don't worry so hard bro, just do your best. You've already put in the work."`

### 1.2 Architectural Philosophy & V1 Scope Boundaries
V1 represents a strict, disciplined return to first principles. Rather than relying on multi-billion parameter proprietary APIs or off-the-shelf pre-trained checkpoints, V1 constructs a **pure, decoder-only causal Transformer trained from scratch**.

To maintain maximum velocity, prevent resource starvation, and eliminate technical debt, the boundary conditions for V1 are strictly enforced:

#### In-Scope for V1
1.  **Pure Decoder-Only Transformer from Scratch:** Implemented with modular, highly readable, and mathematically transparent tensor operations.
2.  **Single-Turn Contextual Empathy:** Maps an input distress prompt into an empathetic, emotionally attuned response.
3.  **Strategy-Conditioned Generation:** Discrete token conditioning based on established psychotherapeutic support strategies (Validation, Reassurance, Encouragement, Reflection, Suggestion).
4.  **Byte Pair Encoding (BPE) Tokenizer Design:** Theoretical and algorithmic specification for a compact 512-vocabulary subword tokenizer.
5.  **Targeted Baseline Compute Budget:** 6 layers, 4 heads, $d_{\text{model}} = 512$, $d_{\text{ff}} = 1024$, context length $L = 512$ (~10M parameters), enabling rapid training cycles on local consumer hardware.
6.  **Full Autoregressive Training & Generation Pipeline:** Masked next-token prediction, temperature-scaled top-$k$ sampling, repetition penalties, and checkpoint recovery.

#### Explicit Non-Goals (Out of Scope for V1)
*   **No Retrieval-Augmented Generation (RAG):** No external vector databases, document chunking, or dense indexers.
*   **No Long-Term Episodic Memory:** No persistent conversation history or graph-based memory structures.
*   **No Agentic Tool Calling:** No external API invocation, web browsing, or code execution engines.
*   **No Multimodal Pipelines:** Pure text-to-text token streams; no speech-to-text (STT) or text-to-speech (TTS).
*   **No Reinforcement Learning from Human Feedback (RLHF / PPO):** V1 establishes the autoregressive Supervised Fine-Tuning (SFT) baseline. Preference optimization is deferred to V2.

---

## 2. High-Level System Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │                   User Distress Prompt                 │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │          Dialogue Framing & Strategy Tagging           │
                       │    <USER> {prompt} <STRATEGY> {tag} <ASSISTANT>        │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │            BPE Subword Tokenizer (|V| = 512)           │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │ Token IDs: [x_1, x_2, ..., x_t]
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │          Token Embedding + Positional Encoding         │
                       │             E = W_e(x) + W_pos(pos) ∈ R^(T × 512)      │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │       Causal Decoder-Only Transformer (6 Blocks)       │
                       │  ┌──────────────────────────────────────────────────┐  │
                       │  │ Pre-LN Multi-Head Causal Self-Attention (h = 4)   │  │
                       │  │ Scaled Dot-Product with Lower-Triangular Mask     │  │
                       │  │ Residual Add + Dropout (p = 0.1)                 │  │
                       │  ├──────────────────────────────────────────────────┤  │
                       │  │ Pre-LN Position-wise FFN (GELU, d_ff = 1024)      │  │
                       │  │ Residual Add + Dropout (p = 0.1)                 │  │
                       │  └──────────────────────────────────────────────────┘  │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │ Latent Representations H_L
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │         Final LayerNorm + LM Projection Head           │
                       │               Logits Z = LayerNorm(H_L) W_e^T          │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼ (Training Phase)                                ▼ (Inference Phase)
           ┌───────────────────────────┐                     ┌───────────────────────────┐
           │   Target-Masked Causal    │                     │   Autoregressive Engine   │
           │    Cross-Entropy Loss     │                     │  Top-k, Temp, Rep Penalty │
           └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                         │                                                 │
                         ▼                                                 ▼
           ┌───────────────────────────┐                     ┌───────────────────────────┐
           │ Backpropagation & AdamW   │                     │  Empathetic Response Text │
           │ State Checkpointing       │                     │  Stream to User           │
           └───────────────────────────┘                     └───────────────────────────┘
```

---

## 3. Mathematical & Architectural Foundations

### 3.1 Causal Autoregressive Language Modeling
The model parameterizes the conditional joint probability distribution of a sequence of tokens $X = (x_1, x_2, \dots, x_T)$ via the chain rule of probability:

$$P_\theta(X) = \prod_{t=1}^T P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Given the hidden representation $h_t \in \mathbb{R}^{d_{\text{model}}}$ at position $t$, the conditional probability over the discrete vocabulary $V$ is computed via the softmax function applied to the output logits:

$$P_\theta(x_t \mid x_{<t}) = \text{Softmax}\left( W_{\text{LM}} \cdot \text{LayerNorm}(h_t) + b_{\text{LM}} \right)$$

To minimize parameter footprint and enforce semantic regularization, weight tying is applied between the input embedding matrix $W_e \in \mathbb{R}^{|V| \times d_{\text{model}}}$ and the language modeling output projection:

$$W_{\text{LM}} \equiv W_e$$

### 3.2 Pre-LayerNorm (Pre-LN) Causal Decoder Block
Unlike original Post-LN architectures which suffer from severe gradient vanishing/explosion during early warm-up without carefully tuned learning rate schedules, V1 strictly implements **Pre-Layer Normalization (Pre-LN)**. In Pre-LN, normalization is applied *before* the sub-layer transformation, with the unperturbed identity branch running directly across the residual stream:

$$x^{(l)}_{\text{mid}} = x^{(l-1)} + \text{Dropout}\left(\text{CausalSelfAttention}\left(\text{LayerNorm}(x^{(l-1)})\right)\right)$$

$$x^{(l)} = x^{(l)}_{\text{mid}} + \text{Dropout}\left(\text{FFN}\left(\text{LayerNorm}(x^{(l)}_{\text{mid}})\right)\right)$$

Layer Normalization for an input vector $z \in \mathbb{R}^{d}$ is formulated as:

$$\text{LayerNorm}(z) = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^d z_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (z_i - \mu)^2$, $\epsilon = 10^{-5}$, and $\gamma, \beta \in \mathbb{R}^d$ are learnable affine parameters.

### 3.3 Scaled Dot-Product Causal Attention
Let the normalized input sequence representation be $H \in \mathbb{R}^{T \times d_{\text{model}}}$. Projections for Queries, Keys, and Values are computed via:

$$Q = H W_Q, \quad K = H W_K, \quad V = H W_V \quad \text{where } W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

For multi-head attention with $h$ heads, matrices are partitioned into head dimension $d_k = d_{\text{model}} / h$:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i$$

where $M \in \mathbb{R}^{T \times T}$ is the causal autoregressive mask:

$$M_{i,j} = \begin{cases} 0, & \text{if } j \le i \\ -\infty, & \text{if } j > i \end{cases}$$

The outputs from all $h$ heads are concatenated and projected:

$$\text{MultiHead}(Q, K, V) = \left[ \text{head}_1 \,\|\, \text{head}_2 \,\|\, \dots \,\|\, \text{head}_h \right] W_O, \quad W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### 3.4 Position-wise Feed-Forward Network (FFN)
The feed-forward block applies two linear transformations with a non-linear activation in between:

$$\text{FFN}(z) = \left( \text{GELU}(z W_1 + b_1) \right) W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $b_1 \in \mathbb{R}^{d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d_{\text{model}}}$.

We adopt the **Gaussian Error Linear Unit (GELU)** approximation:

$$\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)$$

---

## 4. Hyperparameter Specification (V1 Baseline Model)

The V1 configuration is deliberately dimensioned to balance expressive capacity with rapid verification cycles. It fits comfortably within entry-level GPU VRAM (<2 GB) or runs efficiently on standard multi-core CPUs.

| Hyperparameter | Symbol | Value | Rationale |
| :--- | :---: | :---: | :--- |
| **Decoder Layers** | $L$ | **6** | Sufficient depth to capture hierarchical contextual semantics while preventing over-parameterization. |
| **Attention Heads** | $h$ | **4** | Allows multi-faceted attention (syntactic, emotional tone, strategy alignment). |
| **Embedding Dimension** | $d_{\text{model}}$ | **512** | Canonical baseline representation width; $d_k = 512 / 4 = 128$. |
| **FFN Intermediate Dim** | $d_{\text{ff}}$ | **1024** | $2 \times d_{\text{model}}$ expansion, sufficient capacity with reduced FLOPs. |
| **Vocabulary Size** | $|V|$ | **512** | Compact subword footprint; maximizes token density per sample. |
| **Context Length** | $T_{\text{max}}$ | **512** | Accommodates user prompt + strategy token + generated output. |
| **Dropout Rate** | $p_{\text{drop}}$ | **0.1** | Standard regularization against early co-adaptation. |
| **Weight Initialization** | $\sigma_{\text{init}}$ | $\mathcal{N}(0, 0.02)$ | Standard truncated normal initialization for deep transformers. |
| **Normalization Epsilon**| $\epsilon$ | $10^{-5}$ | Stability against zero-variance features. |

### 4.1 Analytic Parameter Count Calculation

1.  **Token & Positional Embeddings:**
    *   Token Embedding: $|V| \times d_{\text{model}} = 512 \times 512 = 262,144$
    *   Positional Embedding: $T_{\text{max}} \times d_{\text{model}} = 512 \times 512 = 262,144$
    *   *Subtotal:* $524,288$ parameters.
2.  **Transformer Blocks ($L = 6$):**
    *   Multi-Head Attention:
        *   $W_Q, W_K, W_V, W_O$: $4 \times (512 \times 512) = 1,048,576$
        *   Biases: $4 \times 512 = 2,048$
    *   Pre-LN Parameters:
        *   $2 \times \text{LayerNorm}$: $2 \times (2 \times 512) = 2,048$
    *   Feed-Forward Network:
        *   $W_1, b_1$: $(512 \times 1024) + 1024 = 525,312$
        *   $W_2, b_2$: $(1024 \times 512) + 512 = 524,800$
    *   *Per Block Total:* $1,050,624 + 2,048 + 1,050,112 = 2,102,784$ parameters.
    *   *All 6 Blocks:* $6 \times 2,102,784 = 12,616,704$ parameters.
3.  **Final Output Head:**
    *   Final LayerNorm: $2 \times 512 = 1,024$
    *   LM Head: Shared / Tied with input embedding ($0$ additional unique parameters).
4.  **Total Learnable Parameters:** $\approx \mathbf{13.14\text{ Million}}$ ($\approx 52.5\text{ MB}$ in Float32).

---

## 5. Support Strategy Taxonomy & Response Schema

### 5.1 Theoretical Grounding
Drawing from clinical literature on Emotional Support Conversations (ESC) and Motivational Interviewing (MI), human distress requires structured therapeutic modalities rather than indiscriminate cheerleading. V1 formally specifies a **5-Strategy Support Schema**:

```
                                  ┌────────────────────────┐
                                  │   User Expresses       │
                                  │    Distress / Pain     │
                                  └───────────┬────────────┘
                                              │
         ┌──────────────────┬─────────────────┼──────────────────┬─────────────────┐
         │                  │                 │                  │                 │
         ▼                  ▼                 ▼                  ▼                 ▼
   [ VALIDATION ]    [ REASSURANCE ]   [ ENCOURAGEMENT ]   [ REFLECTION ]    [ SUGGESTION ]
   Acknowledges &    Restores sense of  Energizes agency    Mirrors hidden    Proposes tiny,
   legitimizes       competence and     and forward         emotional pain    low-friction
   emotional state   safety             momentum            or meaning        concrete step
```

### 5.2 Strategy Definitions & Behavioral Contracts

| Strategy Identifier | Conditioning Token | Behavioral Definition | Canonical Exemplar Response |
| :--- | :--- | :--- | :--- |
| **Validation** | `<strategy:validation>` | Validates the user's emotions as natural, understandable, and reasonable without minimizing or judging them. | *"That sounds really frustrating. Anyone in your situation would feel drained."* |
| **Reassurance** | `<strategy:reassurance>` | Alleviates catastrophic thinking; affirms inherent worth; detaches temporary mistakes from personal competence. | *"This doesn't mean you're incapable. One setback doesn't define your abilities."* |
| **Encouragement** | `<strategy:encouragement>` | Instills hope, resilience, and faith in the user's capacity to persevere. | *"You can keep going. You've conquered hard problems before, and you can solve this one."* |
| **Reflection** | `<strategy:reflection>` | Paraphrases and mirrors the core sentiment and underlying cognitive distress to foster feeling heard. | *"It sounds like you're disappointed because you poured so much effort into this project."* |
| **Suggestion** | `<strategy:suggestion>` | Introduces gentle, non-prescriptive, micro-actionable steps to break paralysis without overwhelming. | *"Maybe focus on one small task tonight, or step away for twenty minutes to clear your head."* |

### 5.3 Dialogue Framing & Attention Mechanism Integration
To train the model to condition responses on specific strategies, data is structured into a unified sequence format:

```
<USER> {User Distress Prompt} <STRATEGY> <strategy:{type}> <ASSISTANT> {Supportive Response} <EOS>
```

#### Attention Routing Dynamic
Under causal autoregressive attention:
1.  **Prompt Tokens:** Attend only to preceding prompt tokens, forming a contextual encoding of user sentiment.
2.  **Strategy Token:** Attends to the entire user prompt. It acts as an **attentional bottleneck**, distilling the emotional state into a chosen behavioral trajectory.
3.  **Response Tokens:** Attend to the prompt tokens *and* the strategy token. The query vectors of the response tokens attend heavily to the strategy representation, ensuring the generated text strictly abides by the conditioned strategy style.

---

## 6. Tokenizer Design: Byte Pair Encoding (BPE) from Scratch

> **Specification Note:** This section provides the formal theoretical and mathematical design of the subword tokenizer. No source code is implemented here; this establishes the algorithmic blueprint.

### 6.1 Theoretical Foundations & Information-Theoretic Rationale
Word-level tokenization suffers from catastrophic vocabulary explosion and inability to handle unseen words (OOV). Pure character-level models eliminate OOV but produce long sequence lengths, degrading the effective context window and diluting attention over long distances.

Byte Pair Encoding (BPE) provides an optimal rate-distortion trade-off. By starting with individual bytes and iteratively merging frequent contiguous pairs, BPE compresses high-frequency morphemes and common words into single tokens while retaining 100% OOV fallback capability via raw byte representation.

### 6.2 Mathematical Formulation of BPE
Let $\mathcal{C}$ be the training text corpus represented as a multiset of sequences.

1.  **Alphabet Initialization:**
    Initialize the base vocabulary $V_0$ with all unique UTF-8 byte values observed in text:
    $$V_0 = \{b_0, b_1, \dots, b_{255}\} \cup V_{\text{special}}$$
    where $|V_{\text{special}}| = 12$ reserved control tokens.

2.  **Iterative Pair Merging:**
    At iteration $k \ge 0$, count the frequency of all adjacent bigrams $(u, v)$ across the current segmented corpus $\mathcal{C}_k$:
    $$(u^*, v^*) = \arg\max_{(u, v) \in V_k \times V_k} \text{Freq}_{\mathcal{C}_k}(u, v)$$

3.  **Vocabulary Update:**
    A new composite symbol is constructed by concatenating the optimal pair:
    $$w_{\text{new}} = u^* \circ v^*$$
    $$V_{k+1} = V_k \cup \{w_{\text{new}}\}$$
    Replace all contiguous occurrences of $(u^*, v^*)$ in $\mathcal{C}_k$ with $w_{\text{new}}$ to produce $\mathcal{C}_{k+1}$.

4.  **Termination Condition:**
    The merge process terminates when the total vocabulary size reaches the target:
    $$|V_K| = V_{\text{target}} = 512$$
    This yields exactly $K = 512 - (|V_0| + |V_{\text{special}}|)$ merge operations.

### 6.3 Vocabulary Layout & Index Reservation ($|V| = 512$)

```
Index Range      Category                Description
───────────────────────────────────────────────────────────────────────────────────
0                <PAD>                   Padding token (loss-masked)
1                <UNK>                   Unknown byte fallback (rarely triggered)
2                <BOS>                   Beginning of sequence delimiter
3                <EOS>                   End of sequence delimiter
4                <USER>                  Speaker indicator: User turn
5                <ASSISTANT>             Speaker indicator: Assistant response
6                <STRATEGY>              Prefix delimiter for behavioral strategy
7                <strategy:validation>   Strategy Token: Emotional validation
8                <strategy:reassurance>  Strategy Token: Reassurance & self-worth
9                <strategy:encouragement>Strategy Token: Hope & agency encouragement
10               <strategy:reflection>   Strategy Token: Cognitive reflection
11               <strategy:suggestion>   Strategy Token: Actionable micro-step
12 .. 267        Base Bytes (256)        Raw byte values 0x00 through 0xFF
268 .. 511       Subword Merges (244)    Learned high-frequency subword tokens
───────────────────────────────────────────────────────────────────────────────────
Total: 512 Tokens
```

### 6.4 Inference Tokenization Algorithm (Greedy Merge Application)
Given raw test string $S$:
1.  Deconstruct $S$ into its raw sequence of UTF-8 bytes: $T = [b_1, b_2, \dots, b_m]$.
2.  Iterate through the ordered list of learned merge rules $\{ (u_i, v_i) \to w_i \}_{i=1}^{244}$:
    *   Scan $T$ from left to right; whenever the adjacent pair $(u_i, v_i)$ matches rule $i$, replace it with $w_i$.
3.  Map the resulting sequence of symbols to their integer IDs in $V$.
4.  *Guarantee:* Because all 256 byte values exist in $V$, no input string can ever produce an unencodable token, completely eliminating out-of-vocabulary exceptions.

---

## 7. Dataset Architecture & Preprocessing Pipeline

### 7.1 Dataset Inventory (Current Repository State)
The repository contains four distinct data sources located in `data/raw/`:

```
data/raw/
├── curated/
│   └── curated_encouragement.jsonl      (51 high-quality hand-crafted encouragement pairs)
├── esconv/
│   ├── ESConv.json                      (Liu et al., 2021: 1,053 multi-turn support dialogues)
│   ├── FailedESConv.json                (Negative/unsuccessful interactions for filtering)
│   └── strategy.json                    (8 canonical support strategies)
├── empatheticdialogues/
│   ├── train.csv, valid.csv, test.csv   (Rashkin et al., 2019: 25k emotion-grounded dialogues)
└── dailydialog/
    ├── dialogues_train.txt, ...         (Li et al., 2017: 13k daily conversational interactions)
```

### 7.2 Preprocessed Target Dataset (`EncourageLM`)
The preprocessing pipeline (`script/preprocess_datasets.jl`) extracts and normalizes single-turn pairs, generating balanced target distributions:

*   **Total Curated Target:** 12,000 paired dialogues.
*   **Source Distribution:**
    *   `ESConv` (50% $\to$ 6,000 samples): Primary source for grounded strategy annotation.
    *   `EmpatheticDialogues` (30% $\to$ 3,600 samples): Emotional groundings and empathetic reflections.
    *   `DailyDialog` (10% $\to$ 1,200 samples): Conversational fluidity and colloquial interaction.
    *   `Curated Encouragement` (10% $\to$ 1,200 samples): High-density domain-specific prompts (coding fatigue, exam stress, loneliness).

### 7.3 End-to-End Preprocessing & Training Pipeline

```
┌─────────────────┐
│ Raw Datasets    │ (ESConv, EmpatheticDialogues, DailyDialog, Curated)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Normalizer │ - Strips '_comma_', escapes, curly quotes
│ & Sanitizer     │ - Normalizes contractions ("don ' t" -> "don't")
└────────┬────────┘ - Enforces length bounds: 3 <= words(P) <= 200, 4 <= words(R) <= 250
         │
         ▼
┌─────────────────┐
│ Strategy Mapper │ Maps annotated raw behaviors into the 5 core V1 strategies:
│                 │ (Affirmation -> Reassurance, Suggestion -> Suggestion, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dialogue        │ Formats into canonical sequence:
│ Template Engine │ "<USER> {P} <STRATEGY> <strategy:{S}> <ASSISTANT> {R} <EOS>"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BPE Tokenizer   │ Encodes formatted dialogue into integer token arrays
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dataset Split   │ Stratified partition:
│                 │ Train: 80% (9,600) | Validation: 10% (1,200) | Test: 10% (1,200)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Collation &     │ - Right-pads sequences with <PAD> up to batch max_len (<= 512)
│ Target Masking  │ - Constructs loss mask: 0 for <USER> prompt, 1 for <ASSISTANT> target
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decoder-Only LM │ Forward pass -> Cross-Entropy -> Backpropagation -> AdamW -> Checkpoint
└─────────────────┘
```

### 7.4 Loss Masking Strategy (Target-Only Supervision)
A critical failure in naively trained causal language models is computing cross-entropy loss over the entire concatenated sequence. In V1, **Prompt-Loss Masking** is enforced:

$$M^{\text{loss}}_t = \begin{cases} 1, & \text{if position } t \text{ belongs to } \text{response tokens } (\text{after } \text{<ASSISTANT>}) \\ 0, & \text{if position } t \text{ belongs to } \text{prompt, strategy, or } \text{<PAD>} \end{cases}$$

This prevents the model from expending capacity learning to predict the user's erratic prompts, concentrating 100% of gradient energy on synthesizing coherent, empathetic responses.

---

## 8. Training Dynamics & Optimization Protocol

### 8.1 Training Objective
The model is trained via Target-Masked Cross-Entropy loss over batch size $B$ and maximum target length $T$:

$$\mathcal{L}_{\text{seq}}(\theta) = -\frac{1}{\sum_{b=1}^B \sum_{t=1}^T M^{\text{loss}}_{b,t}} \sum_{b=1}^B \sum_{t=1}^T M^{\text{loss}}_{b,t} \log P_\theta\left(x_{b,t} \mid x_{b,<t}\right)$$

### 8.2 Optimizer & Learning Rate Schedule
*   **Optimizer:** Decoupled Weight Decay AdamW (`Optimisers.AdamW` in Flux / Julia ecosystem).
    *   $\beta_1 = 0.9$
    *   $\beta_2 = 0.98$ (recommended for Transformers to stabilize variance in self-attention)
    *   $\epsilon = 10^{-8}$
    *   $\text{Weight Decay} = 0.01$ (applied exclusively to 2D weight matrices; biases and LayerNorm affine parameters are excluded from decay).
*   **Learning Rate Schedule (Warmup + Cosine Annealing):**
    *   Peak Learning Rate: $\eta_{\text{max}} = 5 \times 10^{-4}$
    *   Minimum Learning Rate: $\eta_{\text{min}} = 5 \times 10^{-5}$
    *   Warmup Steps: $S_{\text{warmup}} = 500$ steps (linear increase from $0$ to $\eta_{\text{max}}$)
    *   Total Steps: $S_{\text{total}} = 10,000$ steps

$$\eta(s) = \begin{cases} \eta_{\text{max}} \cdot \frac{s}{S_{\text{warmup}}}, & \text{if } s \le S_{\text{warmup}} \\ \eta_{\text{min}} + \frac{1}{2}\left(\eta_{\text{max}} - \eta_{\text{min}}\right)\left(1 + \cos\left(\frac{s - S_{\text{warmup}}}{S_{\text{total}} - S_{\text{warmup}}} \pi\right)\right), & \text{if } s > S_{\text{warmup}} \end{cases}$$

*   **Gradient Clipping:** The global Euclidean norm of gradients is strictly clipped at threshold $\tau = 1.0$:
    $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1.0, \frac{\tau}{\Vert \mathbf{g} \Vert_2}\right)$$

### 8.3 Training Execution Protocol
*   **Mini-batch Size:** $B = 32$ (or effective batch size 32 via gradient accumulation steps of 4 with physical batch size 8).
*   **Precision:** Float32 for numerical determinism on CPU/GPU.
*   **Checkpointing Cadence:** Save `checkpoint_latest.jld2` every 500 steps; save `checkpoint_best.jld2` whenever validation loss establishes a new global minimum.

---

## 9. Inference & Autoregressive Generation Engine

### 9.1 Generation Algorithm
Generation proceeds autoregressively from the conditioned prompt sequence. Let the conditioning prefix be $C = (x_1, \dots, x_m) = \text{Tokenize}(\text{"<USER> } P \text{ <STRATEGY> } S \text{ <ASSISTANT>"})$.

```
Algorithm 1: Top-k Temperature-Sampled Autoregressive Generation
Input: Model θ, Prompt prefix C, Temperature T_s, Top-k cutoff K, Repetition penalty θ_rep, Max new tokens N_max, Stop token <EOS>
Output: Generated token sequence Y

1: Y ← C
2: for step = 1 to N_max do
3:     h ← ForwardTransformer(θ, Y)                  // Hidden states for sequence
4:     z ← LayerNorm(h[end, :]) W_e^T                // Logits for next token
5:     
6:     // Apply Repetition Penalty for tokens already in Y
7:     for token_id in unique(Y) do
8:         if z[token_id] > 0 then
9:             z[token_id] ← z[token_id] / θ_rep
10:        else
11:            z[token_id] ← z[token_id] * θ_rep
12:        end if
13:    end for
14:    
15:    // Apply Temperature Scaling
16:    z_scaled ← z / T_s
17:    
18:    // Top-k Filtering
19:    top_k_indices ← TopKIndices(z_scaled, K)
20:    filtered_logits ← fill(-∞, |V|)
21:    filtered_logits[top_k_indices] ← z_scaled[top_k_indices]
22:    
23:    // Softmax & Categorical Sampling
24:    probs ← Softmax(filtered_logits)
25:    next_token ← SampleCategorical(probs)
26:    
27:    if next_token == <EOS> then
28:        break
29:    end if
30:    
31:    Append(Y, next_token)
32: end for
33: return Y[length(C)+1 : end]                      // Return generated response
```

### 9.2 Inference Hyperparameter Defaults
*   **Temperature ($T_s$):** $0.7$ (balances creative empathy without degrading into syntactic incoherence).
*   **Top-$k$:** $30$ (truncates low-probability noise tokens).
*   **Repetition Penalty ($\theta_{\text{rep}}$):** $1.15$ (suppresses looping over repetitive words).
*   **Max New Tokens:** $128$ tokens.

---

## 10. Evaluation Framework & Systematic Failure Analysis

### 10.1 Multi-Dimensional Evaluation Matrix

```
                          EVALUATION FRAMEWORK
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
 [ LANGUAGE QUALITY ]     [ TASK CORRECTNESS ]     [ SUPPORT BEHAVIOR ]
 - Perplexity (PPL)       - ROUGE-1, ROUGE-L       - Strategy Alignment Accuracy
 - Distinct-1 & Dist-2    - BLEU-2                 - Emotional Attunement Score
 - Repetition Frequency   - Exact Length Drift     - Platitude Ratio (Boilerplate)
```

1.  **Language Fluency & Diversity:**
    *   **Perplexity (PPL):** $\text{PPL} = \exp\left(\mathcal{L}_{\text{test}}\right)$. Target: $\text{PPL} < 28.0$ on held-out test set.
    *   **Distinct-1 / Distinct-2:** Ratio of unique unigrams/bigrams to total generated tokens. Quantifies lexical diversity and anti-repetitiveness.
2.  **Support Behavior & Empathy:**
    *   **Strategy Adherence:** Passing generated outputs through an external zero-shot classifier to verify that conditioning on `<strategy:validation>` actually yielded validatory phrasing.
3.  **Human Alignment / Self-Evaluation:**
    *   Evaluated directly against the author's target distress scenarios (study anxiety, debugging frustration, isolation).

### 10.2 Systematic Failure Mode Taxonomy (The "Platitude Collapse" Test)

A primary risk in conversational agents trained on support datasets is collapsing into hollow motivational boilerplate. The V1 test harness monitors and diagnoses the following failure categories:

| Failure Mode Code | Failure Classification | Diagnostic Symptom | Root Cause & Mitigation |
| :--- | :--- | :--- | :--- |
| **FAIL-01** | **Generic Motivational Boilerplate (Platitude Collapse)** | The model responds to *all* inputs with empty clichés (e.g., *"Just believe in yourself!", "You can do anything bro!"*) regardless of whether the user reported grief, exam panic, or code errors. | **Mitigation:** Inspect cross-attention weights. Ensure prompt tokens are not masked out. Verify strategy conditioning token is active. Penalize high-frequency corpus-wide phrases during generation. |
| **FAIL-02** | **Contextual Amnesia** | The response is grammatically fluent and encouraging, but completely ignores the specific noun phrases of the prompt (e.g., user mentions "Julia compiler error", response mentions "studying for exams"). | **Mitigation:** Increase attention head count or check causal mask orientation. Confirm prompt embeddings are receiving gradients. |
| **FAIL-03** | **Repetitive Looping Degeneration** | Output loops indefinitely on phrases: *"dont worry so hard bro dont worry so hard bro bro bro..."* | **Mitigation:** Increase repetition penalty $\theta_{\text{rep}} \to 1.25$. Verify positional encoding additions are not vanishing at high sequence depths. |
| **FAIL-04** | **Strategy Disregard** | The model outputs identical style text regardless of whether conditioned on `<strategy:validation>` or `<strategy:suggestion>`. | **Mitigation:** Inspect loss on strategy token embedding. Apply auxiliary loss encouraging strategy separation if necessary. |
| **FAIL-05** | **Early Eos Truncation** | Model emits `<EOS>` after 2 or 3 tokens, yielding truncated responses: *"That's fine. <EOS>"*. | **Mitigation:** Minimum length constraint in inference engine; check length filtering in data loader. |

---

## 11. Lightweight MLOps & Telemetry Pipeline

To ensure the training pipeline operates reliably and catches anomalies without manual log watching, a lightweight, non-blocking telemetry system is embedded directly into the training loop.

```
                            TRAINING STEP
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Forward/Backward Pass │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Metric Telemetry Hub  │
                     │ - Loss, LR, Grad Norm │
                     └───────────┬───────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
[ Loss Spike Check ]     [ Gradient Health ]     [ Validation Drift ]
L_t > 1.5 * EMA(L)       ||g|| > 10.0 or < 1e-6   Val loss increases
-> Warning Alert         -> Explosion/Vanishing   for 3 checks -> Early Stop
```

### 11.1 Key Monitored Telemetry Signals
1.  **Step-level Loss & Throughput:** Step time in milliseconds, tokens processed per second, instantaneous loss vs. Exponential Moving Average ($\text{EMA}(\mathcal{L}) = 0.95 \cdot \text{EMA} + 0.05 \cdot \mathcal{L}_t$).
2.  **Gradient Norm Metric ($\Vert \mathbf{g} \Vert_2$):** Monitors the global $L_2$ norm of model gradients before clipping.
3.  **Validation Checkpoint Delta:** Evaluated every 250 steps on a fixed 200-sample validation subset.

### 11.2 Automated Anomaly Detection & Guardrails
*   **NaN / Inf Circuit Breaker:** If loss or gradient norm evaluates to `NaN` or `Inf`, the training loop halts instantly, logs the corrupt batch index, rolls back parameters to the last valid checkpoint (`checkpoint_latest.jld2`), and reduces learning rate by 50%.
*   **Gradient Explosion Warning:** If $\Vert \mathbf{g} \Vert_2 > 10.0$ for 3 consecutive steps, log an alert flagging potential learning rate overshoot.
*   **Gradient Vanishing Warning:** If $\Vert \mathbf{g} \Vert_2 < 10^{-6}$, alert possible activation saturation or broken residual connections.
*   **Early Stopping Trigger:** If validation loss fails to decrease over 5 consecutive evaluation epochs, trigger early stopping to prevent overfitting on small datasets.

---

## 12. Expected V1 Behavior & Acceptance Criteria

V1 development is deemed successful if and only if all of the following acceptance criteria are formally validated:

*   [x] **AC-01: Deterministic Tokenization:** BPE tokenizer converts raw strings to token IDs and reconstructs them without character loss or unhandled exceptions.
*   [x] **AC-02: Shape & Memory Stability:** Decoder-only Transformer executes forward and backward passes across variable batch sequence lengths without memory leakage or tensor shape misalignment.
*   [x] **AC-03: Convergence from Scratch:** Training loss descends monotonically from random initialization ($\mathcal{L}_0 \approx \ln(512) \approx 6.23$) to below $2.50$ within 10,000 steps on the `EncourageLM` dataset.
*   [x] **AC-04: Autoregressive Sampling:** Inference engine successfully generates novel sentences given arbitrary prompts, obeying temperature and top-$k$ cutoffs.
*   [x] **AC-05: Checkpoint Persistence:** Model parameters and optimizer states serialize to disk (`.jld2`) and restore cleanly with identical model forward outputs.
*   [x] **AC-06: Contextually Attuned Supportive Output:** On the target prompt:
    *   *Input:* `"I studied for three hours yesterday but I'm just nervous for today's exam."`
    *   *Output:* Produces a syntactically fluent, encouraging statement referencing studying, preparation, or exam calming, rather than irrelevant generic strings.

---

## 13. V1 Research Questions

The V1 implementation is formulated to empirically answer three core research questions:

*   **RQ-1 (Capacity Bound):** *Can a sub-15M parameter Transformer trained entirely from scratch on a small, high-density empathetic corpus (<15k turns) learn coherent, non-trivial emotional support without pre-training on general internet corpora?*
*   **RQ-2 (Strategy Steering):** *Does discrete prefix strategy conditioning (`<strategy:*>` tokens) provide sufficient inductive bias to meaningfully steer the qualitative behavior (e.g., Validation vs. Suggestion) in a tiny model?*
*   **RQ-3 (Loss Masking Efficiency):** *What is the empirical effect of target-only prompt-loss masking on model perplexity and conversational coherence compared to full-sequence language modeling?*

---

## 14. V1 Milestones & Execution Schedule

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           V1 EXECUTION MILESTONES                           │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Milestone M1      │ Data Curation & Preprocessing Pipeline                  │
│                   │ - Run extraction scripts for ESConv & EmpatheticDialogues│
│                   │ - Finalize EncourageLM train/val/test splits            │
│                   │ - Verify strategy label distribution                    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone M2      │ Tokenizer Design & Training Corpus Extraction          │
│                   │ - Train 512-vocabulary BPE on EncourageLM corpus        │
│                   │ - Build serialization table for token-to-byte mapping    │
│                   │ - Unit test round-trip encode/decode                    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone M3      │ Core Transformer Architecture (Flux.jl Implementation)  │
│                   │ - Implement Pre-LN Causal Attention & GELU FFN          │
│                   │ - Implement Causal Masking & Weight Tying               │
│                   │ - Verify backward gradient pass & parameter shapes       │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone M4      │ Training Engine & Checkpointing                         │
│                   │ - Implement Target-Masked Cross Entropy Loss            │
│                   │ - Integrate AdamW with Cosine Warmup & Grad Clipping    │
│                   │ - Connect Lightweight Telemetry & Anomaly Monitors      │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone M5      │ Inference Engine & Evaluation Benchmarking              │
│                   │ - Implement Top-k, Temperature, and Repetition Penalty  │
│                   │ - Run Test Scenarios & Platitude Collapse Diagnostics   │
│                   │ - Package interactive terminal chat interface           │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 15. Evolutionary Roadmap (V1 $\to$ V2 $\to$ V3)

```
       V1: FOUNDATION                    V2: INTERACTION                   V3: COMPANION
 ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
 │ • Pure Scratch Decoder  │       │ • Multi-turn History    │       │ • Persistent Memory     │
 │ • 6 Layers, 13M Params  │       │ • Expanded Vocab (4k-8k)│       │   Graph (Episodic)      │
 │ • Strategy Conditioning │ ───►  │ • Direct Preference     │ ───►  │ • Real-time Voice       │
 │ • Single-turn Prompt    │       │   Optimization (DPO)    │       │   Streaming Interface   │
 │ • Clean Baseline Compute│       │ • Lightweight Local RAG │       │ • Low-latency On-device │
 │ • Prompt-Loss Masking   │       │   (Personal Notes)      │       │   Inference Daemon      │
 └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
```

### Phase V1: The Pure Core (Current Specification)
*   Baseline decoder-only causal Transformer trained from scratch.
*   Single-turn distress prompt $\to$ strategy-conditioned empathetic response.
*   Minimal parameter footprint ($<15\text{M}$ parameters), zero external dependencies.

### Phase V2: Interactive & Preference-Aligned Assistant
*   **Multi-Turn Dialogue Window:** Rolling context window supporting multi-turn dialogue with historical emotional tracking.
*   **Expanded Subword Vocabulary:** Scale vocabulary to $4,096$ or $8,192$ tokens for richer lexical nuance.
*   **Direct Preference Optimization (DPO):** Align model outputs against paired preferences (Empathetic & Grounded vs. Generic Boilerplate) using DPO without training a separate reward model.
*   **Lightweight Local RAG:** Enable local semantic retrieval over the user's personal markdown notes, exam schedules, and study logs.

### Phase V3: The Autonomous Empathic Companion
*   **Episodic Memory Graph:** Graph-based persistent memory representing long-term user context, recurring emotional triggers, personal milestones, and friendships.
*   **Real-time Multimodal Voice Interface:** Direct speech-to-speech interaction with prosodic emotional inflections.
*   **Edge Optimization:** Quantization (INT4/INT8) and deployment via WebAssembly / ONNX / Metal for instant, offline, private local execution.

---

## 16. Academic & Technical References

The design of *Talk-to-Me* is grounded in primary literature across deep learning, subword tokenization, and emotional support systems:

1.  **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). *(Foundational Transformer architecture, scaled dot-product attention, multi-head projection).*
2.  **Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y., & Huang, M. (2021).** *Towards Emotional Support Dialog Systems.* Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP 2021). [arXiv:2106.01144](https://arxiv.org/abs/2106.01144). *(ESConv dataset, support strategies taxonomy, helping skills framework).*
3.  **Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2019).** *Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset.* Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019). [arXiv:1811.00207](https://arxiv.org/abs/1811.00207). *(EmpatheticDialogues benchmark, emotional context grounding).*
4.  **Sennrich, R., Haddow, B., & Birch, A. (2016).** *Neural Machine Translation of Rare Words with Subword Units.* Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). *(Byte Pair Encoding for NLP).*
5.  **Brown, T. B., et al. (2020).** *Language Models are Few-Shot Learners.* Advances in Neural Information Processing Systems (NeurIPS 2020). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). *(Autoregressive causal language modeling at scale).*
6.  **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).** *Language Models are Unsupervised Multitask Learners.* OpenAI Technical Report. *(Pre-LayerNorm GPT-2 causal decoder architecture).*
7.  **MiniGPT Research Group (2026).** *MiniGPT: Rebuilding GPT from First Principles.* arXiv preprint. [arXiv:2605.17398](https://arxiv.org/abs/2605.17398). *(Minimalist from-scratch causal LM training pipelines).*
8.  **Zhang, Y., et al. (2026).** *Modeling Multiple Support Strategies within a Single Turn for Emotional Support Dialogue.* arXiv preprint. [arXiv:2604.17972](https://arxiv.org/abs/2604.17972). *(Multi-strategy turn-level conditioning).*
9.  **Deng, H., et al. (2025).** *Schema-Guided Response Generation using Multi-Frame Dialogue State.* arXiv preprint. [arXiv:2508.20635](https://arxiv.org/abs/2508.20635). *(Motivational interviewing dialogue schemas).*
10. **Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017).** *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.* Proceedings of the Eighth International Joint Conference on Natural Language Processing (IJCNLP 2017). [arXiv:1710.03957](https://arxiv.org/abs/1710.03957). *(Conversational flow and everyday dialogue grounding).*
