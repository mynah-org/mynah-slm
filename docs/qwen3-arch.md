# Qwen3 — verified architecture

The reference for `src/arch_qwen3.c`. Every value here was read off a real
checkpoint (`mynah-slm inspect --meta --tensors`) or off the upstream
`config.json` committed in `reference/qwen3-0.6b/`, not from a model card.

Verified 2026-08-06 against `Qwen3-0.6B-Q4_K_M.gguf` (unsloth) and the
safetensors header of `Qwen/Qwen3-0.6B`.

---

## Shape of the model

| | Qwen3-0.6B | source |
|---|---|---|
| layers | 28 | `qwen3.block_count` |
| hidden_size | 1024 | `qwen3.embedding_length` |
| intermediate | 3072 | `qwen3.feed_forward_length` |
| heads / kv_heads | 16 / 8 | `qwen3.attention.head_count{,_kv}` |
| head_dim | 128 | `qwen3.attention.key_length` = `value_length` |
| vocab | 151936 | tokenizer array length |
| context | 40960 | `qwen3.context_length` |
| rope_theta | 1e6 | `qwen3.rope.freq_base` |
| rms_norm_eps | 1e-6 | `qwen3.attention.layer_norm_rms_epsilon` |
| tie_word_embeddings | true | `config.json`; and no `output.weight` in the GGUF |

**`head_dim × heads ≠ hidden_size`** — 16 × 128 = 2048, hidden is 1024. The Q
projection is `[2048 x 1024]` and the output projection `[1024 x 2048]`. Never
derive `head_dim` from `hidden / heads`; read `attention.key_length`.

## Forward pass

```
x = embed(token)                                    # no scaling (unlike Gemma)
for layer in 0..27:
    h = RMSNorm(x, attn_norm, eps)
    q, k, v = Wq·h, Wk·h, Wv·h                      # 2048, 1024, 1024
    q = RMSNorm_per_head(q, q_norm, eps)            # QK-norm, weight is [128]
    k = RMSNorm_per_head(k, k_norm, eps)
    q, k = RoPE_neox(q, pos), RoPE_neox(k, pos)     # split-half, theta 1e6
    a = GQA_causal_attention(q, k, v)               # 16 q-heads over 8 kv-heads
    x = x + Wo·a
    h = RMSNorm(x, ffn_norm, eps)
    x = x + Wdown·(SiLU(Wgate·h) ⊙ Wup·h)           # SwiGLU
x = RMSNorm(x, output_norm, eps)
logits = embed_matrix · x                           # tied: same tensor as embed
```

**RoPE is NeoX split-half, not interleaved**: `x[i]` rotates against
`x[i + head_dim/2]`. `qwen-tts` ships *both* variants
(`apply_rope_neox_inplace` in `qwen_tts_talker.c` vs
`qwen_apply_rope_interleaved` in `qwen_tts_kernels.h`); taking the wrong one
produces a model that runs and talks nonsense.

---

## Weight name mapping — GGUF ⇄ safetensors

One table, two containers, one in-memory model. `N` is the layer index.

| GGUF | safetensors | shape (row-major) |
|---|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` | `[151936, 1024]` |
| `output_norm.weight` | `model.norm.weight` | `[1024]` |
| `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` | `[1024]` |
| `blk.N.attn_q.weight` | `model.layers.N.self_attn.q_proj.weight` | `[2048, 1024]` |
| `blk.N.attn_k.weight` | `model.layers.N.self_attn.k_proj.weight` | `[1024, 1024]` |
| `blk.N.attn_v.weight` | `model.layers.N.self_attn.v_proj.weight` | `[1024, 1024]` |
| `blk.N.attn_output.weight` | `model.layers.N.self_attn.o_proj.weight` | `[1024, 2048]` |
| `blk.N.attn_q_norm.weight` | `model.layers.N.self_attn.q_norm.weight` | `[128]` |
| `blk.N.attn_k_norm.weight` | `model.layers.N.self_attn.k_norm.weight` | `[128]` |
| `blk.N.ffn_norm.weight` | `model.layers.N.post_attention_layernorm.weight` | `[1024]` |
| `blk.N.ffn_gate.weight` | `model.layers.N.mlp.gate_proj.weight` | `[3072, 1024]` |
| `blk.N.ffn_up.weight` | `model.layers.N.mlp.up_proj.weight` | `[3072, 1024]` |
| `blk.N.ffn_down.weight` | `model.layers.N.mlp.down_proj.weight` | `[1024, 3072]` |
| *(absent)* | `lm_head.weight` | `[151936, 1024]` — see below |

Count: 28 × 11 + 2 = **310 in the GGUF**, **311 in the safetensors**.

The naming is mechanical apart from three traps:

- `input_layernorm` → `attn_norm` and `post_attention_layernorm` → `ffn_norm`.
  The HF names describe *position*, the GGUF names describe *what they
  normalize for*. They do not sort into the same order, so a mapping built by
  zipping two sorted lists is wrong.
- `o_proj` → `attn_output`, not `attn_o`.
- Shapes agree **exactly** once GGUF `ne` is reversed into row-major. That
  reversal is not cosmetic: it is what makes this table checkable, and
  `tests/test_inspect.c` pins it.

### The `lm_head.weight` trap

`config.json` says `tie_word_embeddings: true`, and the GGUF has no
`output.weight`. But the safetensors file **does** contain `lm_head.weight`,
at its own non-overlapping offsets:

```
lm_head.weight            data_offsets [0, 311164928]
model.embed_tokens.weight data_offsets [311164928, 622329856]
```

Two separate 297 MiB regions. Verified byte-identical by range-reading 96 bytes
from each at three positions (start, middle, end) — a ~200 byte download rather
than 1.2 GB.

So the checkpoint physically stores the tied matrix twice. The loader must:

1. prefer `model.embed_tokens.weight`;
2. ignore `lm_head.weight` when `tie_word_embeddings` is true — mapping it to a
   second tensor wastes 297 MiB of address space for a duplicate;
3. **not** assume it is absent — a checkpoint with untied embeddings would have
   a genuinely different `lm_head`, and the config flag is what decides, not the
   presence of the tensor.

---

## Metadata keys we depend on

From `inspect --meta`. GGUF names them `<arch>.*`, so the prefix follows
`general.architecture` and must not be hardcoded to `qwen3.`.

```
general.architecture                     qwen3
qwen3.block_count                        28
qwen3.context_length                     40960
qwen3.embedding_length                   1024
qwen3.feed_forward_length                3072
qwen3.attention.head_count               16
qwen3.attention.head_count_kv            8
qwen3.attention.key_length               128
qwen3.attention.value_length             128
qwen3.attention.layer_norm_rms_epsilon   1e-06
qwen3.rope.freq_base                     1e+06
tokenizer.ggml.model                     gpt2
tokenizer.ggml.pre                       qwen2
tokenizer.ggml.tokens                    [str x 151936]
tokenizer.ggml.token_type                [i32 x 151936]
tokenizer.ggml.merges                    [str x 151387]
tokenizer.ggml.eos_token_id              151645
tokenizer.ggml.padding_token_id          151654
tokenizer.ggml.add_bos_token             false
tokenizer.chat_template                  (the full Jinja template, tools included)
```

Three things worth flagging:

- **`rms_norm_eps` and `rope.freq_base` are FLOAT32 KVs.** ingot's integer
  accessors convert them without complaining, so a reader that tries `_u64`
  before `_f64` gets `rms_norm_eps = 0` — a plausible-looking value, not an
  error. Switch on the declared KV type. `tests/test_inspect.c` has the
  regression.
- **The GGUF carries one EOS, the HF `generation_config.json` carries two**
  (`151645` and `151643`). Generation must stop on either; take the set from
  the config, not from `tokenizer.ggml.eos_token_id` alone.
- **The chat template ships inside the GGUF.** We do not need to bundle one for
  this family — but the unsloth build has 32 KV keys and the official Qwen
  build has 28, so never assume a fixed key set.

Recommended sampling, from `generation_config.json`: `temperature 0.6`,
`top_k 20`, `top_p 0.95`, `do_sample true`.

---

## Reference artifacts

Committed under `reference/qwen3-0.6b/`, all obtained without downloading the
1.2 GB checkpoint (safetensors puts a JSON header at byte 8, so a range request
reads the tensor table for 35 KB):

- `config.json`, `generation_config.json`
- `safetensors_header.json` — 311 tensors with dtypes, shapes and offsets
