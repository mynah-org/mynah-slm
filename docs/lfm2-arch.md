# LFM2 — verified architecture

Everything here was read out of the actual checkpoint on 2026-08-08, never off
the model card. Two independent sources agree on every line: the GGUF metadata
and tensor census of `LFM2.5-2.6B-Q4_K_M.gguf` (`mynah-slm inspect`), and
`reference/lfm2.5-2.6b/config.json` + `model.safetensors.index.json` from
`LiquidAI/LFM2.5-2.6B`. Where they disagree, the disagreement is written down
rather than resolved silently.

`src/arch_lfm2.c` must match this document stage by stage.

> **Licence, before anything else.** LFM Open License v1.0 (`general.license =
> other`, `general.license.name = lfm1.0` in the GGUF itself). Commercial use
> only under $10M annual revenue. This is not MIT/Apache: LFM2 is an **opt-in**
> target, never a bundled download and never the shipped default. Say so
> wherever its numbers appear.

---

## Shape of the model

| | LFM2.5-2.6B | source |
|---|---|---|
| layers | 30 — **22 conv + 8 attention** | `lfm2.block_count`, `layer_types` |
| hidden_size | 2048 | `lfm2.embedding_length`, `block_dim` |
| intermediate | 10752 | `lfm2.feed_forward_length` |
| heads / kv_heads | 32 / 8 (GQA) | `lfm2.attention.head_count`, `num_key_value_heads` |
| head_dim | **64** | **derived** — see the trap below |
| vocab | 128000 | `lfm2.vocab_size` |
| context | 128000 **or** 131072 | see the trap below |
| rope_theta | 1e7 | `lfm2.rope.freq_base` |
| rms_norm_eps | 1e-5 | `lfm2.attention.layer_norm_rms_epsilon` |
| conv FIR length | **3** | `lfm2.shortconv.l_cache`, `conv_L_cache` |
| conv_dim | 2048 | `conv_dim` (= hidden_size) |
| conv bias | none | `conv_bias: false` |
| MLP | SwiGLU | `block_use_swiglu: true` |
| tie_word_embeddings | true | `config.json`; and no `output.weight` in the GGUF |

Overall 4.94 bits/weight at Q4_K_M — 1.6 GiB of a 2.69B-parameter model.

## The layer map is irregular — read the array

```
layer  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
       c  c  A  c  c  A  c  c  c  A  c  c  c  A  c  c  c  A  c  c  c  A  c  c  A  c  c  A  c  c

attention at 2, 5, 9, 13, 17, 21, 24, 27   →   gaps of 3, 4, 4, 4, 4, 3, 3
```

There is **no ratio to hardcode.** "1 in 4" is right for the middle of the
model and wrong at both ends; it would misplace six of the eight attention
layers. Gemma's 5:1 and 4:1 patterns were regular and this one is not, which is
exactly why rule 1 exists.

**The split is in the GGUF, so no `#define` is needed.** It is encoded as a
per-layer array:

```
lfm2.attention.head_count_kv    [i32 x 30]    0 on conv layers, 8 on attention layers
```

A conv layer declares zero KV heads. That single array is the layer map, and it
agrees element-for-element with `layer_types` in `config.json` and with which
tensors each block actually carries.

---

## Forward pass

```
x = embed(token)                          # NO sqrt(d) scaling — this is not Gemma

for each layer L in 0..29:
    h = rmsnorm(x, operator_norm[L])      # GGUF calls this attn_norm on BOTH kinds
    if layer_types[L] == conv:
        h = short_conv(h, L)
    else:
        h = attention(h, L)
    x = x + h                             # one residual, no post-norm

    h = rmsnorm(x, ffn_norm[L])
    x = x + swiglu(h, L)                  # w2(silu(w1(h)) * w3(h))

x = rmsnorm(x, embedding_norm)            # the FINAL norm — see the naming trap
logits = x @ embed.T                      # tied, no separate lm_head
```

Two norms per layer, not four. No logit softcapping, no embedding scale, no
residual scale — none of the Gemma machinery applies here.

### The short conv block — 22 of the 30 layers

```
BCx   = in_proj(h)                    # [2048] -> [6144], split into three [2048]
B, C, x = BCx[0:2048], BCx[2048:4096], BCx[4096:6144]

Bx    = B * x                         # elementwise gate BEFORE the filter
conv  = depthwise_causal_fir(Bx, W)   # W is [2048, 3], per channel, length 3
y     = C * conv                      # elementwise gate AFTER the filter
out   = out_proj(y)                   # [2048] -> [2048]
```

The filter is **depthwise**: channel `c` of the output depends only on channel
`c` of the input, over the last 3 positions. There is no mixing across
channels, no bias, no activation inside the block, and no data-dependent state
transition. That is what makes it cheap and what makes it *not* Mamba — there
is no selective scan to write.

**One code path for prefill and decode (rule 2).** The same filter is a full
causal convolution over the prompt and a 3-tap sliding window at decode; write
it as "for each position, dot the last 3 inputs with W" and both fall out of
the same loop. The state to carry between decode steps is the last 2 input
values per channel — `conv_L_cache: 3` counts the taps, so 3 slots hold the
window including the current position.

`blk.N.shortconv.conv.weight` is **F32 in the Q4_K_M file**, 24 KiB a layer.
It is never quantized and should not be: 2048×3 floats is 0.03% of the model.

### Attention — 8 layers only

Standard GQA with per-head QK-RMSNorm, which is the same shape Qwen3 already
needs, so `src/arch_qwen3.c` is the thing to port from rather than write anew:

- 32 query heads, 8 KV heads, head_dim 64
- **per-head RMSNorm on Q and K** before RoPE (`q_layernorm`, `k_layernorm`,
  both `[64]`)
- RoPE theta **1e7**, `rope_type: default` — a single base, no scaling, no
  partial rotary, unlike Gemma 4
- full causal attention, no sliding window anywhere in this model
- `use_pos_enc: true`; the conv layers carry no positional encoding at all —
  position is implicit in the causal filter

**Only 8 layers hold a KV cache.** Allocate 8, not 30. Together with the
constant conv state this is the reason the model is interesting on a CPU: at
long context the memory that grows per token comes from 8 layers instead of 30,
and the other 22 carry a fixed 2048×3 f32 state (24 KiB each, 528 KiB total,
independent of context length).

---

## Weight name mapping — GGUF ⇄ safetensors

266 tensors in both, and they correspond one to one.

| GGUF | safetensors | shape (row-major) |
|---|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` | `[128000, 2048]` |
| `token_embd_norm.weight` | `model.embedding_norm.weight` | `[2048]` |
| `blk.N.attn_norm.weight` | `model.layers.N.operator_norm.weight` | `[2048]` |
| `blk.N.ffn_norm.weight` | `model.layers.N.ffn_norm.weight` | `[2048]` |
| `blk.N.ffn_gate.weight` | `model.layers.N.feed_forward.w1.weight` | `[10752, 2048]` |
| `blk.N.ffn_down.weight` | `model.layers.N.feed_forward.w2.weight` | `[2048, 10752]` |
| `blk.N.ffn_up.weight` | `model.layers.N.feed_forward.w3.weight` | `[10752, 2048]` |
| **conv layers only** | | |
| `blk.N.shortconv.in_proj.weight` | `model.layers.N.conv.in_proj.weight` | `[6144, 2048]` |
| `blk.N.shortconv.conv.weight` | `model.layers.N.conv.conv.weight` | `[2048, 3]` |
| `blk.N.shortconv.out_proj.weight` | `model.layers.N.conv.out_proj.weight` | `[2048, 2048]` |
| **attention layers only** | | |
| `blk.N.attn_q.weight` | `model.layers.N.self_attn.q_proj.weight` | `[2048, 2048]` |
| `blk.N.attn_k.weight` | `model.layers.N.self_attn.k_proj.weight` | `[512, 2048]` |
| `blk.N.attn_v.weight` | `model.layers.N.self_attn.v_proj.weight` | `[512, 2048]` |
| `blk.N.attn_output.weight` | `model.layers.N.self_attn.out_proj.weight` | `[2048, 2048]` |
| `blk.N.attn_q_norm.weight` | `model.layers.N.self_attn.q_layernorm.weight` | `[64]` |
| `blk.N.attn_k_norm.weight` | `model.layers.N.self_attn.k_layernorm.weight` | `[64]` |
| *(absent)* | *(absent)* | no `lm_head` on either side — tied |

---

## Traps

### `attn_norm` on a conv layer is not an attention norm

GGUF names the pre-operator norm `blk.N.attn_norm.weight` on **all 30 layers**,
including the 22 that have no attention. Upstream it is honestly called
`operator_norm`. Loading code that keys off the name will happily find
`attn_norm` on layer 0 and conclude layer 0 has attention; it does not. **The
layer map comes from `head_count_kv`, never from which tensors are present** —
though the two do agree, and checking that they agree is a cheap load-time
assertion worth having.

### `token_embd_norm` is the FINAL norm, not an embedding norm

The name reads like a norm applied to the embedding on the way in. It is not:
upstream it is `model.embedding_norm` and it is applied **once at the end**, to
the last hidden state before the tied LM head — the slot every other
architecture in this engine calls `output_norm`. Apply it after the embedding
lookup and the model produces fluent-looking garbage, which is the worst kind
of wrong.

### head_dim is in neither file

Qwen3 publishes `attention.key_length`. LFM2 publishes **nothing**:

```
lfm2.attention.key_length        absent
lfm2.attention.value_length      absent
lfm2.rope.dimension_count        absent
```

`src/model.c` requires one of `key_length` or `rope.dimension_count` and will
refuse the file until LFM2 is allowed to fall back to
`embedding_length / head_count` = 2048 / 32 = **64**. That fallback is only
safe because LFM2's projections are square (`attn_q` is `[2048, 2048]`) —
exactly the assumption Qwen3 breaks, where 16 heads × 128 ≠ 1024. Two
independent confirmations that 64 is right: `attn_q_norm.weight` is `[64]`
(per-head norm, so its length *is* head_dim), and `attn_k` is `[512, 2048]`
with 8 KV heads → 512/8 = 64.

Derive it per architecture, do not make the fallback global.

### The two context lengths disagree

`lfm2.context_length` says **128000**, `max_position_embeddings` in
`config.json` says **131072**. They are not the same number and neither is a
typo of the other; the GGUF value is the conservative one and is what the
engine should use, because it is what the file it is reading declares. Note the
discrepancy rather than picking the bigger one.

### `head_count_kv` is an array and stops the loader today

Measured, on this exact file:

```
$ mynah-slm run -m models/LFM2.5-2.6B-Q4_K_M.gguf -p "..." -n 5
mynah-slm: lfm2.attention.head_count_kv varies per layer (0 at 0, 8 at 2) - unsupported
```

`kv_u32_uniform()` in `src/model.c` is doing its job — it refuses a per-layer
array rather than silently taking element 0 (which would be **zero KV heads**,
i.e. a model with no attention at all). The fix is a per-layer array in the
config, not a relaxation of the check.

### Thinking is on by default, from inside the template

`add_generation_prompt` emits `<|im_start|>assistant\n<think>` — the opening
`<think>` tag is **part of the prompt**, not something the model chooses to
generate. There is no `enable_thinking` flag in this template. `--think off`
therefore cannot be implemented the Qwen3 way (append `/no_think`); it has to
mean "do not emit the opening tag", which changes the prompt bytes rather than
the instructions.

---

## Chat template

ChatML with a leading BOS. `reference/lfm2.5-2.6b/chat_template.jinja` is the
committed copy; the GGUF carries a byte-identical one in
`tokenizer.chat_template`.

```
<BOS><|im_start|>system
{system}<|im_end|>
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
<think>
```

| | |
|---|---|
| bos | 124894 |
| eos | 124900 (`<|im_end|>`) |
| pad | 124893 |
| tokenizer | `gpt2` BPE, `tokenizer.ggml.pre = lfm2`, 128000 tokens, 293320 merges |

### Tool calls are Pythonic, and the schemas ride in the system prompt

**Schemas in** — appended to the system message, not a block of their own:

```
List of tools: [{"type":"function","function":{...}}, {...}]
```

**Calls out** — a Python call *expression list*, not JSON:

```
<|tool_call_start|>[get_weather(city='Verona'), set_timer(minutes=5)]<|tool_call_end|>
```

Kwargs, not a JSON object. Strings are single-quoted with `\\`, `\'`, `\n`,
`\r` escaped; mappings and lists are emitted as JSON *inside* the expression;
numbers and booleans go through Python's `str()`, so it is `True`, not `true`.
Several calls are comma-separated inside one pair of brackets.

`src/tools.c` parses none of this today. Per PLAN.md the trial must score
**both** this native format and the JSON-via-system-prompt override, because
the native one is what the model was trained on and the detour may cost exactly
the accuracy the model is being shopped for.

---

## Reference artifacts

Committed under `reference/lfm2.5-2.6b/`, fetched 2026-08-08 from
`LiquidAI/LFM2.5-2.6B`:

- `config.json` — the architecture, including the full `layer_types` array
- `chat_template.jinja` — 125 lines, the tool-call macros included
- `tokenizer_config.json`, `generation_config.json`

The weights themselves live on the NAS and are an **opt-in** download:
`scripts/download_model.sh --model lfm2.5-2.6b-q4`.
