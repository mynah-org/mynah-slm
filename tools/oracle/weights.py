"""Read a Qwen3 GGUF into f32 numpy arrays.

Deliberately the SAME file the C engine reads. If the oracle loaded bf16
safetensors while the engine loaded a Q4_K GGUF, every comparison would mix our
bugs with quantization error and no tolerance could be tight enough to be
useful. Consuming identical weights means a mismatch is ours.

Names and shapes follow docs/qwen3-arch.md.
"""

from __future__ import annotations

import os

import numpy as np
from gguf import GGUFReader, dequantize


class Qwen3Config:
    """Everything the forward pass needs, read from GGUF metadata.

    No value is hardcoded: the key prefix comes from general.architecture, so
    this class does not know the string "qwen3" either.
    """

    def __init__(self, reader: GGUFReader):
        def kv(key, default=None):
            field = reader.fields.get(key)
            if field is None:
                if default is None:
                    raise KeyError(f"missing GGUF metadata key: {key}")
                return default
            return field.contents()

        self.arch = kv("general.architecture")
        p = f"{self.arch}."

        self.n_layers = int(kv(p + "block_count"))
        self.d_model = int(kv(p + "embedding_length"))
        self.d_ff = int(kv(p + "feed_forward_length"))
        self.n_heads = int(kv(p + "attention.head_count"))
        self.n_kv_heads = int(kv(p + "attention.head_count_kv"))
        # head_dim is its own key on purpose: for Qwen3-0.6B it is 128 while
        # d_model/n_heads would give 64. Deriving it is the classic bug.
        self.head_dim = int(kv(p + "attention.key_length"))
        self.n_ctx = int(kv(p + "context_length"))
        # These two are FLOAT32 KVs. Reading them through an integer accessor
        # yields 0 without an error — the same trap that hit inspect --meta.
        self.rms_eps = float(kv(p + "attention.layer_norm_rms_epsilon"))
        self.rope_theta = float(kv(p + "rope.freq_base"))

        self.eos_ids = {int(kv("tokenizer.ggml.eos_token_id"))}
        # generation_config.json carries a second terminator that the GGUF does
        # not; the caller adds it. Kept as a set for exactly that reason.

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def __repr__(self) -> str:
        return (
            f"Qwen3Config(arch={self.arch}, layers={self.n_layers}, d_model={self.d_model}, "
            f"heads={self.n_heads}/{self.n_kv_heads}, head_dim={self.head_dim}, "
            f"d_ff={self.d_ff}, eps={self.rms_eps:g}, theta={self.rope_theta:g})"
        )


class Qwen3Weights:
    def __init__(self, path: str):
        if not os.path.exists(path):
            # models/ is a symlink to the NAS, which drops its SMB session more
            # often than is comfortable. Say which of the two it is instead of
            # letting np.memmap raise a bare FileNotFoundError.
            hint = ""
            root = os.path.dirname(os.path.abspath(path))
            if os.path.islink(root) or "models" in path:
                hint = ("\nmodels/ is a symlink to /Volumes/shared (the NAS). "
                        "If it is not mounted, remount it — see CLAUDE.md.")
            raise SystemExit(f"oracle: checkpoint not found: {path}{hint}")

        reader = GGUFReader(path)
        self.config = Qwen3Config(reader)

        self._t = {}
        for t in reader.tensors:
            self._t[t.name] = t

        self.embed = self.get("token_embd.weight")
        self.out_norm = self.get("output_norm.weight")
        self.layers = [self._layer(i) for i in range(self.config.n_layers)]

    def get(self, name: str) -> np.ndarray:
        """Dequantize to f32. GGUF stores ne fastest-first; dequantize() already
        hands back the row-major shape, which is what docs/qwen3-arch.md lists."""
        t = self._t.get(name)
        if t is None:
            raise KeyError(f"tensor not found: {name}")
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    def _layer(self, i: int) -> dict:
        p = f"blk.{i}."
        return {
            "attn_norm": self.get(p + "attn_norm.weight"),
            "wq": self.get(p + "attn_q.weight"),
            "wk": self.get(p + "attn_k.weight"),
            "wv": self.get(p + "attn_v.weight"),
            "wo": self.get(p + "attn_output.weight"),
            "q_norm": self.get(p + "attn_q_norm.weight"),
            "k_norm": self.get(p + "attn_k_norm.weight"),
            "ffn_norm": self.get(p + "ffn_norm.weight"),
            "gate": self.get(p + "ffn_gate.weight"),
            "up": self.get(p + "ffn_up.weight"),
            "down": self.get(p + "ffn_down.weight"),
        }
