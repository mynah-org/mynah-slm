/* test_synth.c — the forward pass, on a checkpoint written by the test itself.
 *
 * test_batch holds batched prefill to the one-token path on a real checkpoint,
 * and SKIPS wherever there is none — which is CI, and this cloud branch. This
 * runs the same gate on tests/fixture_model.c's synthetic Qwen3: noise for
 * weights, real architecture, real tensor names, real tokenizer machinery.
 * What it can prove is that the engine agrees with itself; it cannot say
 * anything about text quality.
 *
 * Two fixtures: every matrix F32 (the batched product is then a plain sgemm
 * against the matvec), and the Q4_K_M type mix (Q4_K / Q8_0 / Q6_K), which is
 * the path a real checkpoint takes — our own Q4_K matvec on one side, the
 * dequantized strip + sgemm on the other.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "fixture_model.h"
#include "generate.h"
#include "sampler.h"
#include "model.h"
#include "mynah_slm.h"
#include "threads.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s\n     <- %s\n", what, detail ? detail : ""); failures++; }
}

static double rel_diff(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > num) num = d;
        const double m = fabs((double)a[i]);
        if (m > den) den = m;
    }
    return den > 0.0 ? num / den : num;
}

/* Reset AND poison the cache. state_reset only rewinds n_past: the bytes stay,
 * and the previous pass wrote the very same K/V at the very same positions, so
 * a batched path that skipped or misplaced a position would read back correct
 * stale values and pass. Found by mutation (K written one slot late): it
 * survived until this existed. 0xFF bytes are NaN in f32 and in bf16. */
static void reset_poisoned(mynah_slm_state *st) {
    mynah_slm_state_reset(st);
    const mynah_slm_kv *kv = &st->own.kv;
    const size_t positions = (size_t)kv->n_layers * kv->n_ctx;
    memset(kv->k, 0xff, positions * kv->pos_bytes_k);
    memset(kv->v, 0xff, positions * kv->pos_bytes_v);
}

static size_t argmax(const float *v, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* The same text test_batch uses: mixed scripts, so nothing depends on
 * one-byte tokens, and long enough to cross several batch boundaries. */
static const char *TEXT =
    "Il carbone che alimentava le fornaci arrivava dal nord. Die Fabrik lief "
    "Tag und Nacht. The ledger for 1887 lists twelve names. \xe6\xb8\xaf\xe3\x81\xab"
    "\xe3\x81\xaf\xe8\x88\xb9\xe3\x81\x8c\xe4\xb8\xa6\xe3\x82\x93\xe3\x81\xa7.";

/* ── generate() against a hand-written greedy loop ───────────────────────── */

typedef struct { uint32_t ids[64]; size_t n; char text[1024]; size_t len; } collect;

static int collect_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    collect *c = ctx;
    if (len == 0 && id == 0) return 0;               /* the end-of-stream flush */
    if (c->n < 64) c->ids[c->n++] = id;
    if (c->len + len < sizeof c->text) { memcpy(c->text + c->len, text, len); c->len += len; }
    c->text[c->len] = '\0';
    return 0;
}

#define GEN_MAX 24

/* The definition generate() has to agree with: forward one token at a time,
 * argmax, stop on EOS. No sampler, no channels, no batching. */
static size_t greedy_ref(mynah_slm_state *st, const uint32_t *prompt, size_t n_prompt,
                         uint32_t eos, uint32_t vocab, uint32_t *out) {
    mynah_slm_state_reset(st);
    for (size_t i = 0; i + 1 < n_prompt; i++) mynah_slm_forward(st, prompt[i], NULL);
    uint32_t next = prompt[n_prompt - 1];
    size_t n = 0;
    for (int step = 0; step < GEN_MAX; step++) {
        if (mynah_slm_forward(st, next, st->logits) != 0) break;
        const uint32_t id = (uint32_t)argmax(st->logits, vocab);
        if (id == eos) break;
        out[n++] = id;
        next = id;
    }
    return n;
}

static void check_generate(mynah_slm_model_t *m, mynah_slm_tokenizer *tok) {
    const char *prompt_text = "<|im_start|>user\nCiao! Come stai?<|im_end|>\n<|im_start|>assistant\n";
    uint32_t prompt[128];
    const long n_prompt = mynah_slm_tokenize(tok, prompt_text, 1, prompt, 128);
    const uint32_t eos = mynah_slm_tokenizer_eos(tok);
    const uint32_t vocab = mynah_slm_vocab_size(m);
    char err[256];

    mynah_slm_state st;
    if (mynah_slm_state_init_kv(&st, m, (uint32_t)n_prompt + GEN_MAX + 8, MYNAH_SLM_KV_F32,
                                MYNAH_SLM_KV_F32, err, sizeof err) != 0) {
        check("generate state", 0, err);
        return;
    }
    uint32_t ref[GEN_MAX];
    const size_t n_ref = greedy_ref(&st, prompt, (size_t)n_prompt, eos, vocab, ref);

    mynah_slm_sampler_params sp;
    mynah_slm_sampler_defaults(&sp);
    sp.temp = 0.0f;
    mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, vocab);

    collect got;
    memset(&got, 0, sizeof got);
    mynah_slm_gen_params gp;
    mynah_slm_gen_params_init(&gp);
    gp.prompt = prompt; gp.n_prompt = (size_t)n_prompt;
    gp.max_new = GEN_MAX;
    gp.eos = &eos; gp.n_eos = 1;
    gp.cb = collect_cb; gp.cb_ctx = &got;
    mynah_slm_timing tm;
    mynah_slm_timing_reset(&tm);
    mynah_slm_timing_start(&tm);
    mynah_slm_state_reset(&st);
    const long produced = mynah_slm_generate(&st, tok, sam, &gp, &tm);

    char detail[160];
    snprintf(detail, sizeof detail, "%zu ids vs %zu from the reference loop, produced %ld",
             got.n, n_ref, produced);
    check("generate() == a hand-written greedy loop, id for id",
          got.n == n_ref && memcmp(got.ids, ref, n_ref * sizeof *ref) == 0, detail);
    printf("     %s\n", detail);
    check("generate() reports its decode timing", tm.n_gen == (uint32_t)produced &&
          tm.n_prompt == (uint32_t)n_prompt, "timing counters disagree");

    /* The same generation driven by hand the way a scheduler drives it: a
     * workspace with no sequence of its own, a separate sequence, the prompt
     * in slices of 5, then one accept_logits per forward. */
    {
        mynah_slm_state ws;
        mynah_slm_seq q;
        memset(&q, 0, sizeof q);
        int ok = mynah_slm_state_init_workspace(&ws, m, 256, err, sizeof err) == 0 &&
                 mynah_slm_seq_reserve(&q, m, (uint32_t)n_prompt + GEN_MAX + 8,
                                       MYNAH_SLM_KV_F32, MYNAH_SLM_KV_F32, err, sizeof err) == 0;
        check("a workspace and a separate sequence", ok, err);
        if (ok) {
            mynah_slm_sampler *s2 = mynah_slm_sampler_new(&sp, vocab);
            collect hand;
            memset(&hand, 0, sizeof hand);
            gp.cb_ctx = &hand;
            mynah_slm_gen g;
            mynah_slm_gen_start(&g, tok, s2, &gp, NULL);
            int slices = 0, rc;
            while ((rc = mynah_slm_gen_prefill(&g, &ws, &q, 5)) == 0) slices++;
            while (rc == 1 && mynah_slm_gen_wants_step(&g)) {
                if (mynah_slm_seq_forward(&ws, &q, mynah_slm_gen_next_token(&g), ws.logits) != 0) {
                    mynah_slm_gen_fail(&g);
                    break;
                }
                mynah_slm_gen_accept_logits(&g, ws.logits);
            }
            mynah_slm_gen_finish(&g);
            snprintf(detail, sizeof detail, "%d+1 prefill slices, %zu ids vs %zu, stop %d",
                     slices, hand.n, got.n, (int)g.stop);
            check("a sliced, hand-driven gen == generate(), id for id",
                  rc == 1 && hand.n == got.n &&
                  memcmp(hand.ids, got.ids, got.n * sizeof *got.ids) == 0 &&
                  strcmp(hand.text, got.text) == 0, detail);
            printf("     %s\n", detail);
            mynah_slm_sampler_free(s2);
        }
        mynah_slm_seq_free(&q);
        mynah_slm_state_free(&ws);
    }

    mynah_slm_sampler_free(sam);
    mynah_slm_state_free(&st);
}

/* ── per-request state: what a sequence owns, and only that ─────────────── */

/* Two sequences advanced in lockstep on ONE workspace must each get, bit for
 * bit, the logits it gets alone on a state of its own: the workspace holds
 * nothing a sequence can see. memcmp, not a tolerance — the arithmetic is the
 * same calls in the same order, so anything else is shared state leaking. */
static void check_two_sequences(mynah_slm_model_t *m, const uint32_t *ids, long n_tok) {
    char err[256] = "";
    const uint32_t vocab = mynah_slm_vocab_size(m);
    const long half = n_tok / 2;
    const uint32_t *pa = ids, *pb = ids + half;           /* two different prompts */
    const long na = half, nb = n_tok - half;

    float *alone_a = malloc(vocab * sizeof(float)), *alone_b = malloc(vocab * sizeof(float));
    float *mix_a = malloc(vocab * sizeof(float)), *mix_b = malloc(vocab * sizeof(float));
    mynah_slm_state sa, sb, ws;
    mynah_slm_seq qa, qb;
    memset(&sa, 0, sizeof sa); memset(&sb, 0, sizeof sb); memset(&ws, 0, sizeof ws);
    memset(&qa, 0, sizeof qa); memset(&qb, 0, sizeof qb);
    int ok = mynah_slm_state_init_kv(&sa, m, 256, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err) == 0 &&
             mynah_slm_state_init_kv(&sb, m, 256, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err) == 0 &&
             mynah_slm_state_init_workspace(&ws, m, 256, err, sizeof err) == 0 &&
             mynah_slm_seq_reserve(&qa, m, 128, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err) == 0 &&
             mynah_slm_seq_reserve(&qb, m, 200, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err) == 0;
    check("two sequences and a workspace", ok, err);
    if (!ok) goto out;

    /* Alone: a prefill in batches of 9, then 6 greedy steps. */
    uint32_t gen_a[6], gen_b[6];
    for (int which = 0; which < 2; which++) {
        mynah_slm_state *st = which ? &sb : &sa;
        const uint32_t *p = which ? pb : pa;
        const long n = which ? nb : na;
        float *lg = which ? alone_b : alone_a;
        for (long i = 0; i + 1 < n; i += 9) {
            long take = (n - 1) - i;
            if (take > 9) take = 9;
            mynah_slm_forward_batch(st, p + i, (uint32_t)take, NULL);
        }
        uint32_t next = p[n - 1];
        for (int g = 0; g < 6; g++) {
            mynah_slm_forward(st, next, lg);
            next = (uint32_t)argmax(lg, vocab);
            (which ? gen_b : gen_a)[g] = next;
        }
    }

    /* Interleaved: a's slice, b's slice, a's step, b's step... */
    long ia = 0, ib = 0;
    while (ia + 1 < na || ib + 1 < nb) {
        if (ia + 1 < na) {
            long t = (na - 1) - ia;
            if (t > 9) t = 9;
            mynah_slm_seq_forward_batch(&ws, &qa, pa + ia, (uint32_t)t, NULL);
            ia += t;
        }
        if (ib + 1 < nb) {
            long t = (nb - 1) - ib;
            if (t > 9) t = 9;
            mynah_slm_seq_forward_batch(&ws, &qb, pb + ib, (uint32_t)t, NULL);
            ib += t;
        }
    }
    uint32_t a_next = pa[na - 1], b_next = pb[nb - 1];
    int same_ids = 1;
    for (int g = 0; g < 6; g++) {
        mynah_slm_seq_forward(&ws, &qa, a_next, mix_a);
        mynah_slm_seq_forward(&ws, &qb, b_next, mix_b);
        a_next = (uint32_t)argmax(mix_a, vocab);
        b_next = (uint32_t)argmax(mix_b, vocab);
        if (a_next != gen_a[g] || b_next != gen_b[g]) same_ids = 0;
    }
    check("two sequences interleaved on one workspace == each alone (memcmp)",
          same_ids && memcmp(mix_a, alone_a, vocab * sizeof(float)) == 0 &&
          memcmp(mix_b, alone_b, vocab * sizeof(float)) == 0, "logits or ids differ");

    /* A sequence bigger than the workspace's capacity is refused, not run
     * past the end of the scratch and the RoPE table. */
    {
        mynah_slm_seq big;
        memset(&big, 0, sizeof big);
        if (mynah_slm_seq_reserve(&big, m, 400, MYNAH_SLM_KV_F32, MYNAH_SLM_KV_F32,
                                  err, sizeof err) == 0)
            check("a sequence wider than the workspace is refused",
                  mynah_slm_seq_forward(&ws, &big, pa[0], NULL) == -1, "it ran");
        mynah_slm_seq_free(&big);
    }

    /* Pooled reuse: a big-enough slot is reset in place, a small one grows. */
    {
        mynah_slm_seq r;
        memset(&r, 0, sizeof r);
        mynah_slm_seq_reserve(&r, m, 100, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err);
        const void *k0 = r.kv.k;
        r.n_past = 42;
        mynah_slm_seq_reserve(&r, m, 60, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err);
        const int reused = r.kv.k == k0 && r.n_past == 0 && r.n_ctx == 100;
        mynah_slm_seq_reserve(&r, m, 300, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, err, sizeof err);
        const int grown = r.n_ctx == 300 && r.n_past == 0;
        mynah_slm_seq_reserve(&r, m, 50, MYNAH_SLM_KV_F32, MYNAH_SLM_KV_F32, err, sizeof err);
        const int retyped = r.kv.type_k == MYNAH_SLM_KV_F32 && r.n_ctx == 50;
        check("seq_reserve reuses a big-enough cache, regrows a small one, retypes",
              reused && grown && retyped, "reserve did not reuse/grow/retype as specified");
        mynah_slm_seq_free(&r);
    }

out:
    mynah_slm_seq_free(&qa);
    mynah_slm_seq_free(&qb);
    mynah_slm_state_free(&sa);
    mynah_slm_state_free(&sb);
    mynah_slm_state_free(&ws);
    free(alone_a); free(alone_b); free(mix_a); free(mix_b);
}

/* ── one decode step for several sequences ───────────────────────────────── */

#define MULTI_MAX   8
#define MULTI_STEPS 6

/* Prefill sequence b with its own prompt: a different slice of the text and a
 * different length, so the B sequences sit at B different positions. */
static int prefill_seq(mynah_slm_state *ws, mynah_slm_seq *q, const uint32_t *ids,
                       long n_tok, uint32_t b, uint32_t *last) {
    const long off = (long)(b * 3) % (n_tok / 2);
    long len = 4 + 13 * (long)b;
    if (off + len > n_tok) len = n_tok - off;
    /* Reset AND poison: the previous round wrote the same values at the same
     * positions, which would mask a misplaced write (see reset_poisoned). */
    mynah_slm_seq_reset(q);
    const size_t positions = (size_t)q->kv.n_layers * q->kv.n_ctx;
    memset(q->kv.k, 0xff, positions * q->kv.pos_bytes_k);
    memset(q->kv.v, 0xff, positions * q->kv.pos_bytes_v);
    for (long i = 0; i + 1 < len; i += 11) {
        long take = (len - 1) - i;
        if (take > 11) take = 11;
        if (mynah_slm_seq_forward_batch(ws, q, ids + off + i, (uint32_t)take, NULL) != 0)
            return -1;
    }
    *last = ids[off + len - 1];
    return 0;
}

static void check_multi(mynah_slm_model_t *m, const uint32_t *ids, long n_tok, int matmat) {
    char err[256] = "", what[128], detail[256];
    const uint32_t vocab = mynah_slm_vocab_size(m);
    mynah_slm_state ws;
    mynah_slm_seq solo[MULTI_MAX], batch[MULTI_MAX];
    memset(solo, 0, sizeof solo);
    memset(batch, 0, sizeof batch);
    float *ref = malloc(vocab * sizeof *ref);
    float *gotbuf = malloc((size_t)MULTI_MAX * vocab * sizeof *gotbuf);
    mynah_slm_decode_product_set(matmat);

    int ok = mynah_slm_state_init_workspace(&ws, m, 256, err, sizeof err) == 0 &&
             mynah_slm_state_init_decode(&ws, MULTI_MAX, err, sizeof err) == 0;
    /* Mixed precisions in one batch: even sequences keep an f32 cache, odd
     * ones bf16, so each (sequence, head) task must find its own format. */
    for (uint32_t b = 0; ok && b < MULTI_MAX; b++) {
        const mynah_slm_kv_type t = (b % 2) ? MYNAH_SLM_KV_BF16 : MYNAH_SLM_KV_F32;
        ok = mynah_slm_seq_reserve(&solo[b], m, 160, t, t, err, sizeof err) == 0 &&
             mynah_slm_seq_reserve(&batch[b], m, 160, t, t, err, sizeof err) == 0;
    }
    snprintf(what, sizeof what, "[%s] a decode workspace for %d sequences",
             mynah_slm_decode_product_name(), MULTI_MAX);
    check(what, ok, err);
    if (!ok) goto out;

    double worst_all = 0.0;
    for (uint32_t B = 1; B <= MULTI_MAX; B++) {
        uint32_t next[MULTI_MAX];
        mynah_slm_seq *bp[MULTI_MAX];
        for (uint32_t b = 0; b < B; b++) {
            uint32_t last_s = 0, last_b = 0;
            prefill_seq(&ws, &solo[b], ids, n_tok, b, &last_s);
            prefill_seq(&ws, &batch[b], ids, n_tok, b, &last_b);
            next[b] = last_s;
            bp[b] = &batch[b];
        }
        double worst = 0.0;
        int argmax_ok = 1, bit_ok = 1, past_ok = 1, rc_ok = 1;
        for (int step = 0; step < MULTI_STEPS; step++) {
            float *lg[MULTI_MAX];
            if (mynah_slm_forward_multi(&ws, bp, next, B, lg) != 0) { rc_ok = 0; break; }
            /* Copy the batch rows out: the solo calls below reuse the
             * workspace, and B == 1 hands back ws.logits itself. */
            float *got[MULTI_MAX];
            for (uint32_t b = 0; b < B; b++) {
                got[b] = gotbuf + (size_t)b * vocab;
                memcpy(got[b], lg[b], vocab * sizeof(float));
            }
            for (uint32_t b = 0; b < B; b++) {
                if (mynah_slm_seq_forward(&ws, &solo[b], next[b], ref) != 0) { rc_ok = 0; break; }
                const double rel = rel_diff(ref, got[b], vocab);
                if (rel > worst) worst = rel;
                if (argmax(ref, vocab) != argmax(got[b], vocab)) argmax_ok = 0;
                if (memcmp(ref, got[b], vocab * sizeof(float)) != 0) bit_ok = 0;
                if (solo[b].n_past != batch[b].n_past) past_ok = 0;
                /* Both copies are fed the SOLO choice, so the histories stay
                 * the same tokens and only the arithmetic is compared. */
                next[b] = (uint32_t)argmax(ref, vocab);
            }
        }
        if (worst > worst_all) worst_all = worst;
        snprintf(what, sizeof what, "[%s] B=%u: each sequence in the batch == its solo decode",
                 mynah_slm_decode_product_name(), B);
        snprintf(detail, sizeof detail, "%d steps, worst rel %.2e, argmax %s, bit-identical %s",
                 MULTI_STEPS, worst, argmax_ok ? "same" : "DIFFERS", bit_ok ? "yes" : "no");
        /* B == 1 is the single-token path verbatim, and the matvec product is
         * the same kernels in the same order as the solo path: both must be
         * memcmp-identical. matmat is a reorder: the 1e-4 gate test_batch
         * holds a real checkpoint to, and the same argmax. */
        const int exact_required = (B == 1 || !matmat);
        check(what, rc_ok && past_ok && argmax_ok && worst < 1e-4 &&
              (!exact_required || bit_ok), detail);
        printf("     %s\n", detail);
    }
    printf("     worst rel over B=1..%d: %.2e\n", MULTI_MAX, worst_all);

    /* Isolation: a batch with one sequence that cannot step (its cache is
     * full) fails as a whole and moves NOBODY. */
    {
        uint32_t next[3];
        mynah_slm_seq *bp[3] = { &batch[0], &batch[1], &batch[2] };
        for (uint32_t b = 0; b < 3; b++) prefill_seq(&ws, &batch[b], ids, n_tok, b, &next[b]);
        const uint32_t p0 = batch[0].n_past, p2 = batch[2].n_past;
        const uint32_t saved = batch[1].n_past;
        batch[1].n_past = batch[1].n_ctx;
        float *lg[3];
        const int rc = mynah_slm_forward_multi(&ws, bp, next, 3, lg);
        snprintf(what, sizeof what, "[%s] a batch with a full sequence fails and moves nobody",
                 mynah_slm_decode_product_name());
        check(what, rc == -1 && batch[0].n_past == p0 && batch[2].n_past == p2, "it ran or moved");
        batch[1].n_past = saved;
        bp[1] = &batch[0];                       /* the same sequence twice */
        check("     ... and so does a batch naming one sequence twice",
              mynah_slm_forward_multi(&ws, bp, next, 3, lg) == -1 && batch[0].n_past == p0,
              "it ran");
    }

out:
    for (uint32_t b = 0; b < MULTI_MAX; b++) {
        mynah_slm_seq_free(&solo[b]);
        mynah_slm_seq_free(&batch[b]);
    }
    mynah_slm_state_free(&ws);
    free(ref);
    free(gotbuf);
    mynah_slm_decode_product_set(-1);
}

static void run(const char *label, int quant) {
    printf("\n-- %s fixture --\n", label);
    fixture_spec spec;
    fixture_spec_tiny(&spec, quant);

    char path[256], err[256];
    if (fixture_write_temp(path, sizeof path, &spec, err, sizeof err) != 0) {
        check("fixture written", 0, err);
        return;
    }
    mynah_slm_model_t *m = mynah_slm_load(path, err, sizeof err);
    check("the loader accepts the fixture", m != NULL, err);
    if (!m) { unlink(path); return; }
    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    check("the tokenizer loads from it", tok != NULL, err);
    if (!tok) { mynah_slm_free(m); unlink(path); return; }

    const long n_tok = mynah_slm_tokenize(tok, TEXT, 0, NULL, 0);
    uint32_t *ids = malloc((size_t)(n_tok > 0 ? n_tok : 1) * sizeof *ids);
    mynah_slm_tokenize(tok, TEXT, 0, ids, (size_t)n_tok);

    /* The vocabulary round-trips: decode every id back and compare bytes. */
    {
        mynah_slm_detok d;
        mynah_slm_detok_init(&d, tok);
        char out[4096], piece[64];
        size_t o = 0;
        for (long i = 0; i < n_tok; i++) {
            const long w = mynah_slm_detok_feed(&d, ids[i], piece, sizeof piece);
            if (w > 0 && o + (size_t)w < sizeof out) { memcpy(out + o, piece, (size_t)w); o += (size_t)w; }
        }
        out[o] = '\0';
        char detail[128];
        snprintf(detail, sizeof detail, "%ld ids, %zu bytes back of %zu", n_tok, o, strlen(TEXT));
        check("tokenize then detokenize is the identity", strcmp(out, TEXT) == 0, detail);
        printf("     %s\n", detail);
    }

    const uint32_t vocab = mynah_slm_vocab_size(m);
    float *ref = malloc(vocab * sizeof *ref);
    float *got = malloc(vocab * sizeof *got);

    mynah_slm_state st;
    if (mynah_slm_state_init_kv(&st, m, (uint32_t)n_tok + 64, MYNAH_SLM_KV_F32,
                                MYNAH_SLM_KV_F32, err, sizeof err) != 0) {
        check("state", 0, err);
        goto out;
    }

    for (long i = 0; i + 1 < n_tok; i++) mynah_slm_forward(&st, ids[i], NULL);
    mynah_slm_forward(&st, ids[n_tok - 1], ref);
    const uint32_t past_ref = st.own.n_past;
    const size_t pick_ref = argmax(ref, vocab);

    const uint32_t widths[] = { (uint32_t)n_tok, 7, 13 };
    double worst = 0.0;
    for (size_t k = 0; k < sizeof widths / sizeof *widths; k++) {
        uint32_t width = widths[k];
        if (width > mynah_slm_batch_max(&st)) width = mynah_slm_batch_max(&st);
        reset_poisoned(&st);
        long i = 0;
        int bad = 0;
        while (i + 1 < n_tok) {
            long take = (n_tok - 1) - i;
            if (take > (long)width) take = width;
            if (mynah_slm_forward_batch(&st, ids + i, (uint32_t)take, NULL) != 0) { bad = 1; break; }
            i += take;
        }
        if (bad || mynah_slm_forward(&st, ids[n_tok - 1], got) != 0) {
            check("forward_batch runs", 0, "failed");
            continue;
        }
        const double rel = rel_diff(ref, got, vocab);
        if (rel > worst) worst = rel;
        char what[96], detail[160];
        snprintf(what, sizeof what, "batched prefill at width %u == one token at a time", width);
        snprintf(detail, sizeof detail, "rel=%.2e, argmax %zu vs %zu, n_past %u vs %u",
                 rel, argmax(got, vocab), pick_ref, st.own.n_past, past_ref);
        /* The same 1e-4 test_batch holds a real checkpoint to; a reorder lands
         * orders of magnitude under it. */
        check(what, rel < 1e-4 && argmax(got, vocab) == pick_ref && st.own.n_past == past_ref, detail);
        printf("     %s\n", detail);
    }

    /* Greedy continuation after the prompt: wrong K/V positions would still
     * score the prompt and then read a history that was never there. */
    uint32_t gen[2][12];
    for (int pass = 0; pass < 2; pass++) {
        reset_poisoned(&st);
        long i = 0;
        while (i + 1 < n_tok) {
            long take = (n_tok - 1) - i;
            const uint32_t w = pass == 0 ? 1u : 16u;
            if (take > (long)w) take = w;
            mynah_slm_forward_batch(&st, ids + i, (uint32_t)take, NULL);
            i += take;
        }
        uint32_t next = ids[n_tok - 1];
        for (int g = 0; g < 12; g++) {
            if (mynah_slm_forward(&st, next, got) != 0) { gen[pass][g] = (uint32_t)-1; continue; }
            next = (uint32_t)argmax(got, vocab);
            gen[pass][g] = next;
        }
    }
    check("greedy continuation is identical after a batched prefill",
          memcmp(gen[0], gen[1], sizeof gen[0]) == 0, "token ids differ");
    printf("     worst rel over the widths: %.2e\n", worst);

    mynah_slm_state_free(&st);
    check_generate(m, tok);
    check_two_sequences(m, ids, n_tok);
    check_multi(m, ids, n_tok, 1);
    check_multi(m, ids, n_tok, 0);
out:
    free(ref); free(got); free(ids);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);
    unlink(path);
}

int main(void) {
    mynah_slm_threads_init(0);
    run("F32", 0);
    run("Q4_K_M-mix", 1);
    mynah_slm_threads_shutdown();
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
