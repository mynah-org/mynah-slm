#!/usr/bin/env python3
"""R1 — post-training ternarization feasibility for Qwen3-0.6B.

This is a research tool, not part of the runtime. It answers, in order:

  census    Phase 0 — tensor/module census straight off the safetensors header.
            No torch, no weights read: the header carries every shape and dtype.
  budget    Phase 0 — storage cost of the checkpoint under every representation
            we might ship, computed from the census.
  calib     Phase 1 — build and cache the calibration corpus.
  sensitivity  Phase 3 — per-tensor reconstruction and layer-output error.
  coverage  Phase 4 — degradation curves vs fraction of linears quantized.
  mixed     Phase 5 — mixed-precision search.
  shapes    Phase 7 — matrix-shape census and weight traffic per token.

Phases that are not written yet exit 2 and say so. They do not print a
placeholder number. See .work/engineering-method.md, "every tool declares a
refusal".

Every subcommand writes machine-readable JSON (and CSV where a table is the
natural shape) under --output-dir, so a later phase never has to re-derive an
earlier one's numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Representations we compare against. Bits per weight are the *packed block
# rate* of the format, i.e. what the bytes on disk actually cost, including the
# format's own scales. Sources:
#   - ggml block geometry: third_party/ingot/docs/QUANTS.md
#   - measured whole-file rates for this checkpoint: docs/models.md
# A scheme's `bpw` covers the quantized linears only; what happens to the
# embedding is a separate axis, because for Qwen3 it dominates.
# ---------------------------------------------------------------------------

FORMATS = {
    # name:        (bits per weight, note)
    "bf16": (16.0, "the checkpoint as published"),
    "q8_0": (8.5, "ggml Q8_0, 34 B / 32 weights"),
    "q6_k": (6.5625, "ggml Q6_K, 210 B / 256 weights"),
    "q5_k": (5.5, "ggml Q5_K, 176 B / 256 weights"),
    "q4_k": (4.5, "ggml Q4_K, 144 B / 256 weights"),
    "q4_0": (4.5, "ggml Q4_0, 18 B / 32 weights"),
    "q3_k": (3.4375, "ggml Q3_K, 110 B / 256 weights"),
    "iq3_xxs": (3.0625, "ggml IQ3_XXS codebook"),
    "q2_k": (2.625, "ggml Q2_K, 84 B / 256 weights"),
    "tq2_0": (2.0625, "ggml TQ2_0 ternary, 66 B / 256 weights — ingot decodes it"),
    "iq2_xxs": (2.0625, "ggml IQ2_XXS codebook"),
    "tq1_0": (1.6875, "ggml TQ1_0 ternary, 54 B / 256 weights — ingot decodes it"),
    "iq1_s": (1.5625, "ggml IQ1_S codebook"),
}

# Ternary schemes under study. `planes` is how many {-1,0,+1} matrices are
# stored; `scale_bits_per_weight` is the scale/metadata overhead ON TOP of the
# packed trits, computed later from the real group geometry when the scheme is
# grouped. `container` says whether an existing ggml type can hold it.
TERNARY_SCHEMES = {
    "w158_rowwise": dict(
        planes=1,
        trit_pack="tq1_0",
        group=None,
        scales_per_group=1,
        label="single-plane W1.58, one fp16 scale per output row",
        container="TQ1_0 (exact fit)",
    ),
    "w158_g128": dict(
        planes=1,
        trit_pack="tq1_0",
        group=128,
        scales_per_group=2,  # (alpha, mu) as in PT2-LLM
        label="single-plane W1.58, (alpha, mu) fp16 per group of 128",
        container="TQ1_0 + a side scale tensor",
    ),
    "ptqtp_rowwise": dict(
        planes=2,
        trit_pack="tq1_0",
        group=None,
        scales_per_group=1,
        label="PTQTP dual trit-plane, one fp16 alpha per row per plane",
        container="two TQ1_0 tensors — no single ggml type holds this",
    ),
    "ptqtp_g128": dict(
        planes=2,
        trit_pack="tq1_0",
        group=128,
        scales_per_group=1,
        label="PTQTP dual trit-plane, grouped alpha (G=128), the paper's variant",
        container="two TQ1_0 tensors + side scales — no single ggml type",
    ),
}

FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Loading the census without loading the model
# ---------------------------------------------------------------------------


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str
    params: int
    block: int | None  # transformer block index, None for the rest
    family: str  # q_proj / ... / embedding / lm_head / norm / other
    rows: int  # output channels, for scale accounting
    cols: int  # input channels


class Refusal(SystemExit):
    """Exit 2 with a message, never a placeholder number."""

    def __init__(self, msg: str):
        print(f"REFUSED: {msg}", file=sys.stderr)
        super().__init__(2)


def classify(name: str) -> tuple[int | None, str]:
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    block = int(m.group(1)) if m else None
    tail = m.group(2) if m else name
    for fam in FAMILIES:
        if tail.endswith(f"{fam}.weight"):
            return block, fam
    if name == "model.embed_tokens.weight":
        return None, "embedding"
    if name == "lm_head.weight":
        return None, "lm_head"
    if "norm" in tail:
        return block, "norm"
    return block, "other"


def read_header(model_dir: Path) -> dict:
    st = model_dir / "model.safetensors"
    if not st.is_file():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            hint = ""
            if str(model_dir).startswith("models/") or "/Volumes/shared" in str(model_dir):
                hint = " — is the NAS mounted? `ls /Volumes/shared` should not be empty."
            raise Refusal(f"no safetensors under {model_dir}{hint}")
        header = {}
        for shard in shards:
            header.update(read_one_header(shard))
        return header
    return read_one_header(st)


def read_one_header(path: Path) -> dict:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    for v in header.values():
        v["_file"] = path.name
    return header


def census(model_dir: Path) -> tuple[list[Tensor], dict]:
    header = read_header(model_dir)
    tensors: list[Tensor] = []
    for name, meta in sorted(header.items()):
        shape = tuple(meta["shape"])
        params = 1
        for d in shape:
            params *= d
        block, family = classify(name)
        rows = shape[0] if shape else 1
        cols = shape[1] if len(shape) > 1 else 1
        tensors.append(Tensor(name, shape, meta["dtype"], params, block, family, rows, cols))

    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    return tensors, cfg


def tie_check(model_dir: Path, tensors: list[Tensor], cfg: dict) -> dict:
    """Are lm_head and embed_tokens the same bytes? Never assume the config."""
    names = {t.name for t in tensors}
    declared = bool(cfg.get("tie_word_embeddings"))
    if "lm_head.weight" not in names:
        return dict(declared=declared, both_stored=False, byte_identical=None,
                    note="no lm_head tensor: the checkpoint stores the tied weight once")
    header = read_header(model_dir)
    a, b = header["lm_head.weight"], header["model.embed_tokens.weight"]
    if a["_file"] != b["_file"]:
        return dict(declared=declared, both_stored=True, byte_identical=None,
                    note="the two tensors live in different shards; not compared")
    path = model_dir / a["_file"]
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        base = 8 + n
        blobs = []
        for t in (a, b):
            s, e = t["data_offsets"]
            f.seek(base + s)
            blobs.append(f.read(e - s))
    same = blobs[0] == blobs[1]
    return dict(
        declared=declared,
        both_stored=True,
        byte_identical=same,
        note=(
            "the checkpoint stores the tied weight TWICE; the duplicate is "
            f"{len(blobs[0]) / 2**20:.1f} MiB of the file and is dropped by any "
            "converter that honours tie_word_embeddings"
            if same
            else "lm_head and embed_tokens DIFFER despite the config — do not tie them"
        ),
    )


# ---------------------------------------------------------------------------
# Phase 0 — census
# ---------------------------------------------------------------------------


def cmd_census(args) -> int:
    model_dir = Path(args.model)
    tensors, cfg = census(model_dir)
    tie = tie_check(model_dir, tensors, cfg)

    dedup = tie.get("byte_identical") is True
    kept = [t for t in tensors if not (dedup and t.name == "lm_head.weight")]
    total = sum(t.params for t in kept)
    stored = sum(t.params for t in tensors)

    by_family: dict[str, dict] = {}
    for t in kept:
        e = by_family.setdefault(t.family, dict(params=0, count=0, shapes=set()))
        e["params"] += t.params
        e["count"] += 1
        e["shapes"].add(t.shape)

    rows = []
    for fam, e in sorted(by_family.items(), key=lambda kv: -kv[1]["params"]):
        rows.append(
            dict(
                family=fam,
                tensors=e["count"],
                shape=" / ".join(str(list(s)) for s in sorted(e["shapes"])),
                params=e["params"],
                pct_of_model=100 * e["params"] / total,
            )
        )

    linears = sum(by_family.get(f, {}).get("params", 0) for f in FAMILIES)
    attn = sum(by_family.get(f, {}).get("params", 0) for f in FAMILIES[:4])
    mlp = sum(by_family.get(f, {}).get("params", 0) for f in FAMILIES[4:])
    emb = by_family.get("embedding", {}).get("params", 0)
    norms = by_family.get("norm", {}).get("params", 0)

    out = dict(
        model=str(model_dir),
        revision=args.revision,
        config={k: cfg.get(k) for k in
                ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
                 "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                 "tie_word_embeddings", "torch_dtype")},
        tie_check=tie,
        params_distinct=total,
        params_stored=stored,
        families=rows,
        groups=dict(
            transformer_linears=dict(params=linears, pct=100 * linears / total),
            attention_linears=dict(params=attn, pct=100 * attn / total,
                                   pct_of_linears=100 * attn / linears if linears else 0),
            mlp_linears=dict(params=mlp, pct=100 * mlp / total,
                             pct_of_linears=100 * mlp / linears if linears else 0),
            embedding=dict(params=emb, pct=100 * emb / total),
            norms=dict(params=norms, pct=100 * norms / total),
        ),
    )

    write_json(args, "census.json", out)
    write_csv(args, "census_families.csv", rows)
    write_csv(
        args,
        "census_tensors.csv",
        [dict(name=t.name, block=t.block, family=t.family, shape=str(list(t.shape)),
              dtype=t.dtype, params=t.params, rows=t.rows, cols=t.cols) for t in kept],
    )

    print(f"# Phase 0 census — {model_dir}")
    if args.revision:
        print(f"# revision {args.revision}")
    print()
    print(f"distinct parameters : {total:,}")
    print(f"stored parameters   : {stored:,}"
          + ("  (the tied weight is written twice)" if stored != total else ""))
    print(f"tie check           : {tie['note']}")
    print()
    print(f"{'family':<12}{'tensors':>8}{'params':>15}{'% model':>9}  shape")
    for r in rows:
        print(f"{r['family']:<12}{r['tensors']:>8}{r['params']:>15,}"
              f"{r['pct_of_model']:>8.2f}%  {r['shape']}")
    print()
    for k, v in out["groups"].items():
        extra = f"  ({v['pct_of_linears']:.1f}% of linears)" if "pct_of_linears" in v else ""
        print(f"{k:<22}{v['params']:>15,}{v['pct']:>8.2f}%{extra}")
    return 0


# ---------------------------------------------------------------------------
# Phase 0 — storage budget
# ---------------------------------------------------------------------------


def group_scale_bits(tensors: list[Tensor], group: int | None, scales_per_group: int,
                     scale_bits: int = 16) -> tuple[int, int]:
    """Scale bits and weight count over the quantizable linears."""
    bits = 0
    weights = 0
    for t in tensors:
        if t.family not in FAMILIES:
            continue
        weights += t.params
        groups_per_row = 1 if group is None else max(1, t.cols // group)
        bits += t.rows * groups_per_row * scales_per_group * scale_bits
    return bits, weights


def cmd_budget(args) -> int:
    model_dir = Path(args.model)
    tensors, cfg = census(model_dir)
    tie = tie_check(model_dir, tensors, cfg)
    dedup = tie.get("byte_identical") is True
    kept = [t for t in tensors if not (dedup and t.name == "lm_head.weight")]

    total = sum(t.params for t in kept)
    lin = sum(t.params for t in kept if t.family in FAMILIES)
    emb = sum(t.params for t in kept if t.family in ("embedding", "lm_head"))
    rest = total - lin - emb  # norms and anything else: always f32/bf16

    rest_bytes = rest * 4  # norms stay f32, per the repo's quantization policy

    def mib(b: float) -> float:
        return b / 2**20

    rows = []

    def add(label: str, lin_bits: float, emb_fmt: str, note: str):
        emb_bits = FORMATS[emb_fmt][0]
        b = lin * lin_bits / 8 + emb * emb_bits / 8 + rest_bytes
        rows.append(
            dict(
                scheme=label,
                linear_bpw=round(lin_bits, 4),
                embedding=emb_fmt,
                total_MiB=round(mib(b), 1),
                whole_model_bpw=round(b * 8 / total, 3),
                note=note,
            )
        )

    # Conventional baselines, the honest controls.
    add("BF16 (as published)", 16.0, "bf16", "1 tensor written twice on disk; deduped here")
    add("Q8_0 everywhere", 8.5, "q8_0", "docs/models.md measures 8.50 bpw whole-file")

    # The real Q4_K_M recipe, so the model can be checked against a file we have
    # actually measured. llama.cpp bumps attn_v and ffn_down to Q6_K on 14 of the
    # 28 layers (docs/models.md lists the indices). If this row does not land on
    # the measured 372.7 MiB / 5.24 bpw, the whole table below is wrong.
    bumped = sum(t.params for t in kept
                 if t.family in ("v_proj", "down_proj")
                 and t.block in (0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27))
    b = ((lin - bumped) * 4.5 + bumped * 6.5625) / 8 + emb * 6.5625 / 8 + rest_bytes
    rows.append(dict(scheme="Q4_K_M, exact recipe (what we ship)", linear_bpw=4.5,
                     embedding="q6_k", total_MiB=round(mib(b), 1),
                     whole_model_bpw=round(b * 8 / total, 3),
                     note="MEASURED on the real file: 372.7 MiB, 5.24 bpw"))
    modelled = mib(b)

    add("Q4_K flat (no Q6_K bumps)", 4.5, "q6_k",
        "the same recipe without llama.cpp's per-layer bumps")
    add("Q3_K_M", 3.4375, "q6_k", "granite-350m says this survives but is a poor trade")
    add("IQ2_XXS linears", 2.0625, "q6_k", "ingot decodes it today; generic kernel")
    add("IQ1_S linears", 1.5625, "q6_k", "the conventional floor; quality unmeasured here")

    # Ternary schemes, with their real scale overhead.
    for key, s in TERNARY_SCHEMES.items():
        trit_bits = FORMATS[s["trit_pack"]][0] * s["planes"]
        scale_bits, weights = group_scale_bits(kept, s["group"], s["scales_per_group"])
        # TQ1_0 already carries one f16 per 256-weight block. A row-wise scheme
        # needs nothing more; a grouped scheme pays for its own side scales.
        extra = 0.0 if s["group"] is None else scale_bits / weights
        lin_bits = trit_bits + extra * s["planes"]
        for emb_fmt in ("bf16", "q6_k"):
            add(f"{key} + {emb_fmt} embedding", lin_bits, emb_fmt, s["container"])

    write_json(args, "budget.json",
               dict(model=str(model_dir), revision=args.revision,
                    params_distinct=total, linears=lin, embedding=emb, other=rest,
                    rows=rows))
    write_csv(args, "budget.csv", rows)

    print(f"# Phase 0 storage budget — {model_dir}")
    print(f"# {total:,} distinct parameters: {lin:,} linear ({100*lin/total:.1f}%), "
          f"{emb:,} embedding ({100*emb/total:.1f}%), {rest:,} other")
    print()
    print(f"{'scheme':<44}{'lin bpw':>9}{'embed':>8}{'MiB':>9}{'model bpw':>11}")
    for r in rows:
        print(f"{r['scheme']:<44}{r['linear_bpw']:>9.3f}{r['embedding']:>8}"
              f"{r['total_MiB']:>9.1f}{r['whole_model_bpw']:>11.3f}")
    print()
    # Validate the model against the one file we have actually weighed.
    MEASURED_Q4KM_MIB = 372.7
    err = abs(modelled - MEASURED_Q4KM_MIB) / MEASURED_Q4KM_MIB
    verdict = "OK" if err < 0.01 else "MODEL DISAGREES WITH THE MEASURED FILE"
    print(f"check: Q4_K_M exact recipe modelled {modelled:.1f} MiB vs "
          f"{MEASURED_Q4KM_MIB} MiB measured (docs/models.md) — "
          f"{err*100:.2f}% [{verdict}]")
    if err >= 0.01:
        print("REFUSED: the budget model does not reproduce a file we have weighed; "
              "every other row is unreliable.", file=sys.stderr)
        return 2
    print()
    print("The last column is the number that matters. A scheme whose linears are")
    print("1.7 bits and whose whole-model rate is above 5.24 has not beaten the")
    print("Q4_K_M file we already ship.")
    return 0


# ---------------------------------------------------------------------------
# Phases not written yet — they refuse rather than print a placeholder.
# ---------------------------------------------------------------------------


def not_yet(phase: str, what: str):
    def run(args) -> int:
        raise Refusal(
            f"{phase} is not implemented. {what} "
            "This tool does not emit a placeholder number for a phase that has "
            "not run; see .work/ternary-feasibility.md for the gate that unlocks it."
        )

    return run


# ---------------------------------------------------------------------------


def write_json(args, name: str, obj) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_csv(args, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen_ternary_feasibility.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="models-local/qwen3-0.6b-bf16",
                   help="directory holding config.json + model.safetensors")
    p.add_argument("--revision", default=None,
                   help="HF commit sha of --model, recorded in every output file")
    p.add_argument("--calibration-dataset", default="wikitext-2-raw-v1")
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--layer", type=int, default=None, help="restrict to one block")
    p.add_argument("--family", choices=FAMILIES, default=None)
    p.add_argument("--coverage", type=float, default=None,
                   help="fraction of eligible linear weights to quantize, 0..1")
    p.add_argument("--method", choices=["naive", "pt2", "twla", "ptqtp"], default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="reports/ternary")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census", help="Phase 0 — tensor/module census").set_defaults(fn=cmd_census)
    sub.add_parser("budget", help="Phase 0 — storage under every representation").set_defaults(fn=cmd_budget)
    sub.add_parser("calib", help="Phase 1 — calibration corpus").set_defaults(
        fn=not_yet("Phase 1 (calib)", "Nothing has been tokenized or cached."))
    sub.add_parser("sensitivity", help="Phase 3 — per-tensor sensitivity").set_defaults(
        fn=not_yet("Phase 3 (sensitivity)", "No weights have been ternarized."))
    sub.add_parser("coverage", help="Phase 4 — degradation curves").set_defaults(
        fn=not_yet("Phase 4 (coverage)", "Phase 3 must rank the tensors first."))
    sub.add_parser("mixed", help="Phase 5 — mixed-precision search").set_defaults(
        fn=not_yet("Phase 5 (mixed)", "Phase 4 must produce the curves first."))
    sub.add_parser("shapes", help="Phase 7 — matrix shapes and traffic/token").set_defaults(
        fn=not_yet("Phase 7 (shapes)", "Gated on a GO verdict from Phase 5/6."))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
