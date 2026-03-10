# Tiny LLM in Ruby

This is a minimal LLM implemented in pure Ruby as a single file.  
The goal is not performance. The goal is to make the full flow easy to follow in code:
`Tokenizer -> Embedding -> Self-Attention -> FFN -> Training -> Generation`.

The entire implementation lives in [tiny_llm.rb](./tiny_llm.rb). No external gems are required.

## Requirements

- Ruby 3.0 or later
- No external gems

Example:

```bash
ruby -v
```

## Quick Start

By default, the script runs with a lightweight `demo` profile.

```bash
ruby tiny_llm.rb
```

The script runs these steps:

1. Basic `Tensor` and `Tokenizer` sanity checks
2. Shape checks for each layer
3. Training
4. Text generation

Example output:

```text
=== Runtime Profile ===
profile: demo
embed_dim=8, context_len=8, hidden_dim=16
epochs=180, batch_size=8, lr=0.01
optimizer=adam
generation=top_k, temperature=0.55, top_k=4, no_repeat_ngram=5
seed=3

=== Training ===
epoch   1/180  loss=...
...
Final loss: ...
Best loss:  ...

=== Generation ===
To be ...
```

## `demo` vs `full`

The file keeps the specification constants at the top, but pure Ruby scalar autograd is slow.  
Because of that, the default run uses a smaller `demo` configuration.

- `demo`: default, intended for normal local runs
- `full`: runs with the specification-sized constants

Run the full profile:

```bash
env TINY_LLM_PROFILE=full ruby tiny_llm.rb
```

Notes:

- `full` is much slower
- On some machines, training can take a long time

## Runtime Environment Variables

You can override the runtime settings with environment variables.

| Environment Variable | Default | Description |
|---|---:|---|
| `TINY_LLM_PROFILE` | `demo` | Set to `full` to use the specification constants |
| `TINY_LLM_RUNTIME_EMBED_DIM` | `8` | Embedding dimension for demo runs |
| `TINY_LLM_RUNTIME_CONTEXT_LEN` | `8` | Context length for demo runs |
| `TINY_LLM_RUNTIME_HIDDEN_DIM` | `16` | FFN hidden dimension for demo runs |
| `TINY_LLM_RUNTIME_LR` | `0.01` | Learning rate |
| `TINY_LLM_RUNTIME_EPOCHS` | `180` | Number of epochs |
| `TINY_LLM_RUNTIME_BATCH_SIZE` | `8` | Batch size |
| `TINY_LLM_OPTIMIZER` | `adam` | `adam` or `sgd` |
| `TINY_LLM_SEED` | `3` | Random seed |
| `TINY_LLM_GENERATION_TEMPERATURE` | `0.55` | Sampling temperature during generation |
| `TINY_LLM_GENERATION_STRATEGY` | `top_k` | `greedy`, `sample`, or `top_k` |
| `TINY_LLM_GENERATION_TOP_K` | `4` | Candidate count used by `top_k` generation |
| `TINY_LLM_NO_REPEAT_NGRAM` | `5` | Blocks exact local n-gram loops during generation; `0` disables it |

Examples:

```bash
env TINY_LLM_RUNTIME_EPOCHS=300 TINY_LLM_RUNTIME_LR=0.02 ruby tiny_llm.rb
```

```bash
env TINY_LLM_SEED=11 TINY_LLM_RUNTIME_BATCH_SIZE=8 ruby tiny_llm.rb
```

```bash
env TINY_LLM_OPTIMIZER=sgd TINY_LLM_GENERATION_STRATEGY=greedy ruby tiny_llm.rb
```

```bash
env TINY_LLM_NO_REPEAT_NGRAM=0 TINY_LLM_RUNTIME_BATCH_SIZE=4 ruby tiny_llm.rb
```

## What Is Implemented

- `Tensor`: scalar-based automatic differentiation
- `Tokenizer`: character-level tokenizer
- `Embedding` / `PositionEmbedding`
- `LayerNorm`
- causal-masked `SelfAttention`
- `FeedForward`
- `TransformerBlock`
- `TinyLLM`
- cross-entropy loss
- Adam / SGD training
- greedy / sampling / top-k generation

## Changing the Training Data

The training data is the `TRAIN_DATA` constant inside `tiny_llm.rb`.  
Replace that string to experiment with different inputs.

Good candidates:

- short English text
- text with repeated patterns
- small samples of a few hundred characters

## Recommended Reading Order

If you want to understand the implementation, this order is the easiest:

1. `Tensor`
2. `Tokenizer`
3. `Embedding` / `PositionEmbedding`
4. `SelfAttention`
5. `TransformerBlock`
6. `TinyLLM`
7. `train`
8. `generate`

## Limitations

- This is not a production model
- The tokenizer is character-level
- Attention is single-head
- The default optimizer is Adam, but the model is still tiny
- The demo profile favors cleaner output over raw speed
- Autograd is scalar-based, not tensor-based
- Training is slow because of that

## Common Commands

Normal run:

```bash
ruby tiny_llm.rb
```

Full run:

```bash
env TINY_LLM_PROFILE=full ruby tiny_llm.rb
```

Run longer training:

```bash
env TINY_LLM_RUNTIME_EPOCHS=300 ruby tiny_llm.rb
```

Compare different random seeds:

```bash
env TINY_LLM_SEED=21 ruby tiny_llm.rb
```
