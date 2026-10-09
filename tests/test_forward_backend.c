/* test_forward_backend.c — the backend-driven forward == arch_qwen3.c, bit for bit.
 *
 * src/forward_backend.c issues arch_qwen3.c's op sequence through the backend
 * vtable. On the CPU backend every op is the kernel arch_qwen3.c calls, with
 * the same arguments in the same order, so the gate is memcmp — not a
 * tolerance. Anything but identical bits is a wiring bug: a wrong layer
 * offset, a K/V written at the wrong position, a norm on the wrong buffer, a
 * RoPE at the wrong position.
 *
 * Runs on tests/fixture_model.c's synthetic Qwen3 (noise weights, the real
 * tensor names and GQA/QK-norm/NeoX/tied-head structure), both fixtures:
 * every matrix F32, and the Q4_K_M type mix (Q4_K / Q8_0 / Q6_K). Model-free,
 * part of `make test`.
 *
 * Gates, per fixture and per KV precision (f32, bf16):
 *   - one token at a time over the whole prompt: logits at EVERY position
 *     memcmp-identical to mynah_slm_forward;
 *   - batched prefill at three slice widths, then 16 greedy tokens: the
 *     prompt's last logits and every decode step's logits memcmp-identical
 *     to mynah_slm_forward_batch + mynah_slm_forward, ids identical;
 *   - the greedy argmax step (4 bytes back) picks the same ids;
 *   - refusals touch nothing: a full sequence, a token past the vocabulary,
 *     a prefill wider than the forward was sized for.
 *
 * And the slot-scheduler step (mynah_slm_bfwd_multi), per fixture, KV and
 * product (n matvecs / one matmat), over sequences held in a
 * mynah_slm_bslots pool:
 *   - every row of every step memcmp-identical to mynah_slm_forward_multi on
 *     the reference with the same live set (same product selected there);
 *   - with the matvec product every row also memcmp-identical to that
 *     sequence stepped ALONE through the backend (solo == batched); with
 *     matmat, within 1e-4 relative and the same argmax (a sgemm reorder);
 *   - a row cancelled mid-generation (slot released, row left out of the
 *     next step) changes nobody else's bits, and the slot it frees serves a
 *     NEW request whose rows again match the reference;
 *   - a step with a full sequence, or one sequence named twice, fails and
 *     moves nobody.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "backend.h"
#include "fixture_model.h"
#include "forward_backend.h"
#include "generate.h"
#include "model.h"
#include "mynah_slm.h"
#include "sampler.h"
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

static uint32_t argmax(const float *v, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return (uint32_t)best;
}

static const char *TEXT =
    "Il carbone che alimentava le fornaci arrivava dal nord. Die Fabrik lief "
    "Tag und Nacht. The ledger for 1887 lists twelve names.";

#define GEN 16

typedef struct {
    mynah_slm_model_t *m;
    const uint32_t *ids;
    long n_tok;
    uint32_t vocab;
} fx;

/* Reference: arch_qwen3.c on its own state. Prefill in slices of `width`
 * (1 = one token at a time, logits taken at every position into `per_pos`),
 * then GEN greedy steps. */
static int run_ref(const fx *x, mynah_slm_kv_type kt, uint32_t width, float *per_pos,
                   float *last, float *steps, uint32_t *ids_out) {
    char err[256];
    mynah_slm_state st;
    if (mynah_slm_state_init_kv(&st, x->m, (uint32_t)x->n_tok + GEN + 8, kt, kt,
                                err, sizeof err) != 0) return -1;
    long i = 0;
    while (i + 1 < x->n_tok) {
        long take = (x->n_tok - 1) - i;
        if (take > (long)width) take = width;
        if (width == 1) {
            if (mynah_slm_forward(&st, x->ids[i], per_pos + (size_t)i * x->vocab) != 0) goto bad;
        } else if (mynah_slm_forward_batch(&st, x->ids + i, (uint32_t)take, NULL) != 0) goto bad;
        i += take;
    }
    if (mynah_slm_forward(&st, x->ids[x->n_tok - 1], last) != 0) goto bad;
    uint32_t next = argmax(last, x->vocab);
    for (int g = 0; g < GEN; g++) {
        ids_out[g] = next;
        float *lg = steps + (size_t)g * x->vocab;
        if (mynah_slm_forward(&st, next, lg) != 0) goto bad;
        next = argmax(lg, x->vocab);
    }
    mynah_slm_state_free(&st);
    return 0;
bad:
    mynah_slm_state_free(&st);
    return -1;
}

/* The same through the backend forward. */
static int run_bfwd(const fx *x, mynah_slm_bfwd *f, uint32_t width, float *per_pos,
                    float *last, float *steps, uint32_t *ids_out, char *err, size_t errsz) {
    mynah_slm_bseq q;
    if (mynah_slm_bseq_init(f, &q, (uint32_t)x->n_tok + GEN + 8, err, errsz) != 0) return -1;
    long i = 0;
    int rc = -1;
    while (i + 1 < x->n_tok) {
        long take = (x->n_tok - 1) - i;
        if (take > (long)width) take = width;
        float *lg = width == 1 ? per_pos + (size_t)i * x->vocab : NULL;
        if (mynah_slm_bfwd_prefill(f, &q, x->ids + i, (uint32_t)take, lg, err, errsz) != 0)
            goto out;
        i += take;
    }
    if (mynah_slm_bfwd_step(f, &q, x->ids[x->n_tok - 1], last, err, errsz) != 0) goto out;
    uint32_t next = argmax(last, x->vocab);
    for (int g = 0; g < GEN; g++) {
        ids_out[g] = next;
        float *lg = steps + (size_t)g * x->vocab;
        if (mynah_slm_bfwd_step(f, &q, next, lg, err, errsz) != 0) goto out;
        next = argmax(lg, x->vocab);
    }
    rc = 0;
out:
    mynah_slm_bseq_free(f, &q);
    return rc;
}

static void check_parity(const fx *x, mynah_slm_backend *b, mynah_slm_kv_type kt,
                         const char *label) {
    char err[256] = "", what[160], detail[200];
    const size_t V = x->vocab;
    mynah_slm_bfwd_desc d = { .ctx_cap = 256, .batch_max = 64, .dec_max = 1,
                              .kv_k = kt, .kv_v = kt };
    mynah_slm_bfwd *f = NULL;
    const int crc = mynah_slm_bfwd_create(b, x->m, &d, &f, err, sizeof err);
    snprintf(what, sizeof what, "[%s] backend forward created (KV %s)", label,
             mynah_slm_kv_type_name(kt));
    check(what, crc == 0, err);
    if (crc != 0) return;

    float *ref_pos = malloc((size_t)x->n_tok * V * sizeof(float));
    float *got_pos = malloc((size_t)x->n_tok * V * sizeof(float));
    float *ref_last = malloc(V * sizeof(float)), *got_last = malloc(V * sizeof(float));
    float *ref_steps = malloc((size_t)GEN * V * sizeof(float));
    float *got_steps = malloc((size_t)GEN * V * sizeof(float));
    uint32_t ref_ids[GEN], got_ids[GEN];

    const uint32_t widths[] = { 1, (uint32_t)x->n_tok - 1, 7, 13 };
    for (size_t w = 0; w < sizeof widths / sizeof *widths; w++) {
        uint32_t width = widths[w];
        if (width > 64) width = 64;
        memset(ref_pos, 0, (size_t)x->n_tok * V * sizeof(float));
        memset(got_pos, 0, (size_t)x->n_tok * V * sizeof(float));
        const int r1 = run_ref(x, kt, width, ref_pos, ref_last, ref_steps, ref_ids);
        const int r2 = run_bfwd(x, f, width, got_pos, got_last, got_steps, got_ids,
                                err, sizeof err);
        if (r1 != 0 || r2 != 0) {
            snprintf(what, sizeof what, "[%s/%s] width %u runs", label,
                     mynah_slm_kv_type_name(kt), width);
            check(what, 0, r2 != 0 ? err : "reference failed");
            continue;
        }
        int pos_same = 1;
        if (width == 1)
            pos_same = memcmp(ref_pos, got_pos, (size_t)(x->n_tok - 1) * V * sizeof(float)) == 0;
        const int last_same = memcmp(ref_last, got_last, V * sizeof(float)) == 0;
        int steps_same = 1, first_bad = -1;
        for (int g = 0; g < GEN; g++)
            if (memcmp(ref_steps + (size_t)g * V, got_steps + (size_t)g * V, V * sizeof(float))) {
                steps_same = 0;
                if (first_bad < 0) first_bad = g;
            }
        const int ids_same = memcmp(ref_ids, got_ids, sizeof ref_ids) == 0;
        snprintf(what, sizeof what,
                 "[%s/%s] prefill width %u + %d greedy: logits memcmp-identical", label,
                 mynah_slm_kv_type_name(kt), width, GEN);
        snprintf(detail, sizeof detail, "%s%s%s, first differing step %d, ids %s",
                 width == 1 ? (pos_same ? "every prompt position same, " : "a PROMPT POSITION differs, ") : "",
                 last_same ? "last prompt logits same" : "LAST PROMPT LOGITS DIFFER",
                 steps_same ? ", every step same" : "", first_bad,
                 ids_same ? "same" : "DIFFER");
        check(what, pos_same && last_same && steps_same && ids_same, detail);
        printf("     %s\n", detail);
    }

    /* Greedy through the 4-byte argmax: the same ids as host argmax. */
    {
        mynah_slm_bseq q;
        int ok = mynah_slm_bseq_init(f, &q, (uint32_t)x->n_tok + GEN + 8, err, sizeof err) == 0;
        for (long i = 0; ok && i + 1 < x->n_tok; i += 64) {
            long take = (x->n_tok - 1) - i;
            if (take > 64) take = 64;
            ok = mynah_slm_bfwd_prefill(f, &q, x->ids + i, (uint32_t)take, NULL,
                                        err, sizeof err) == 0;
        }
        uint32_t next = x->ids[x->n_tok - 1], ids[GEN + 1];
        for (int g = 0; ok && g <= GEN; g++) {
            ok = mynah_slm_bfwd_step_argmax(f, &q, next, &next, err, sizeof err) == 0;
            ids[g] = next;
        }
        /* ref_ids came from the last width's run: the same prompt and
         * history, so ids[0..GEN-1] must be ref_ids[0..] shifted by the
         * first pick (ref_ids[0] is argmax of the last prompt logits). */
        snprintf(what, sizeof what, "[%s/%s] argmax steps pick the same %d ids", label,
                 mynah_slm_kv_type_name(kt), GEN);
        check(what, ok && memcmp(ids, ref_ids, sizeof ref_ids) == 0 && q.n_past ==
              (uint32_t)x->n_tok + GEN, ok ? "ids differ" : err);
        mynah_slm_bseq_free(f, &q);
    }

    /* Refusals touch nothing. */
    {
        mynah_slm_bseq q;
        mynah_slm_bseq_init(f, &q, 8, err, sizeof err);
        const uint32_t ids8[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const int filled = mynah_slm_bfwd_prefill(f, &q, ids8, 8, NULL, err, sizeof err) == 0;
        const int full = mynah_slm_bfwd_step(f, &q, 1, NULL, err, sizeof err) == -1;
        mynah_slm_bseq_reset(&q);
        const int oov = mynah_slm_bfwd_step(f, &q, x->vocab, NULL, err, sizeof err) == -1;
        mynah_slm_bseq_free(f, &q);
        mynah_slm_bseq_init(f, &q, 200, err, sizeof err);
        uint32_t wide[65];
        for (int i = 0; i < 65; i++) wide[i] = (uint32_t)i;
        const int toowide = mynah_slm_bfwd_prefill(f, &q, wide, 65, NULL, err, sizeof err) == -1;
        snprintf(what, sizeof what, "[%s/%s] full / out-of-vocab / too-wide calls refused, "
                 "n_past unmoved", label, mynah_slm_kv_type_name(kt));
        check(what, filled && full && oov && toowide && q.n_past == 0, err);
        mynah_slm_bseq_free(f, &q);
    }

    mynah_slm_bfwd_free(f);
    free(ref_pos); free(got_pos); free(ref_last); free(got_last);
    free(ref_steps); free(got_steps);
}

/* ── generate.c's token loop on the backend forward ───────────────────── */

typedef struct { uint32_t ids[64]; size_t n; } collect;

static int collect_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)text;
    collect *c = ctx;
    if (len == 0 && id == 0) return 0;
    if (c->n < 64) c->ids[c->n++] = id;
    return 0;
}

/* generate() on the reference state and generate_driver() on the backend
 * forward, same params and sampler seed: the same ids, the same counts.
 * Sampled, not greedy, so the whole logits row matters, not just its max. */
static void check_driver(const fx *x, mynah_slm_backend *b, const mynah_slm_tokenizer *tok,
                         const char *label) {
    char err[256] = "", what[128], detail[400];
    mynah_slm_sampler_params sp;
    mynah_slm_sampler_defaults(&sp);
    sp.temp = 0.8f;
    sp.seed = 11;
    mynah_slm_gen_params gp;
    mynah_slm_gen_params_init(&gp);
    gp.prompt = x->ids;
    gp.n_prompt = (size_t)x->n_tok;
    gp.max_new = 24;

    collect a, c;
    memset(&a, 0, sizeof a);
    memset(&c, 0, sizeof c);
    long na = -1, nc = -1;
    {
        mynah_slm_state st;
        if (mynah_slm_state_init_kv(&st, x->m, (uint32_t)x->n_tok + 32, MYNAH_SLM_KV_BF16,
                                    MYNAH_SLM_KV_BF16, err, sizeof err) == 0) {
            mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, x->vocab);
            gp.cb = collect_cb; gp.cb_ctx = &a;
            na = mynah_slm_generate(&st, tok, sam, &gp, NULL);
            mynah_slm_sampler_free(sam);
            mynah_slm_state_free(&st);
        }
    }
    {
        /* batch 16 < the prompt: the driver's slicing is exercised too */
        mynah_slm_bfwd_desc d = { .ctx_cap = 256, .batch_max = 16, .dec_max = 1,
                                  .kv_k = MYNAH_SLM_KV_BF16, .kv_v = MYNAH_SLM_KV_BF16 };
        mynah_slm_bfwd *f = NULL;
        mynah_slm_bseq q;
        memset(&q, 0, sizeof q);
        if (mynah_slm_bfwd_create(b, x->m, &d, &f, err, sizeof err) == 0 &&
            mynah_slm_bseq_init(f, &q, (uint32_t)x->n_tok + 32, err, sizeof err) == 0) {
            mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, x->vocab);
            mynah_slm_bfwd_run r;
            memset(&r, 0, sizeof r);
            r.f = f;
            r.q = &q;
            mynah_slm_gen_driver drv;
            mynah_slm_bfwd_driver(&r, &drv);
            gp.cb = collect_cb; gp.cb_ctx = &c;
            nc = mynah_slm_generate_driver(&drv, tok, sam, &gp, NULL);
            if (r.err[0]) snprintf(err, sizeof err, "%s", r.err);
            mynah_slm_sampler_free(sam);
        }
        mynah_slm_bseq_free(f, &q);
        mynah_slm_bfwd_free(f);
    }
    snprintf(what, sizeof what, "[%s] generate_driver(backend forward) == generate(), sampled",
             label);
    snprintf(detail, sizeof detail, "%ld vs %ld tokens, %zu vs %zu ids %s", na, nc, a.n, c.n,
             err);
    check(what, na > 0 && na == nc && a.n == c.n && memcmp(a.ids, c.ids, a.n * sizeof *a.ids) == 0,
          detail);
    printf("     %s\n", detail);
}

/* ── several sequences per step, over a slot pool ─────────────────────── */

#define MROWS  4
#define MSTEPS 8

static double rel_diff(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > num) num = d;
        if (fabs((double)a[i]) > den) den = fabs((double)a[i]);
    }
    return den > 0.0 ? num / den : num;
}

/* Request r's prompt: a different slice and length per request, so rows sit
 * at different positions. */
static void prompt_of(const fx *x, uint32_t r, const uint32_t **p, uint32_t *len) {
    const long off = (long)(r * 5) % (x->n_tok / 2);
    long l = 4 + 11 * (long)r;
    if (off + l > x->n_tok) l = x->n_tok - off;
    *p = x->ids + off;
    *len = (uint32_t)l;
}

static int ref_prefill(mynah_slm_state *ws, mynah_slm_seq *q, const uint32_t *p, uint32_t len) {
    mynah_slm_seq_reset(q);
    for (uint32_t i = 0; i + 1 < len; i += 11) {
        uint32_t t = (len - 1) - i;
        if (t > 11) t = 11;
        if (mynah_slm_seq_forward_batch(ws, q, p + i, t, NULL) != 0) return -1;
    }
    return 0;
}

static int b_prefill(mynah_slm_bfwd *f, mynah_slm_bseq *q, const uint32_t *p, uint32_t len,
                     char *err, size_t errsz) {
    mynah_slm_bseq_reset(q);
    for (uint32_t i = 0; i + 1 < len; i += 11) {
        uint32_t t = (len - 1) - i;
        if (t > 11) t = 11;
        if (mynah_slm_bfwd_prefill(f, q, p + i, t, NULL, err, errsz) != 0) return -1;
    }
    return 0;
}

typedef struct {
    int      live;
    uint32_t slot, req;
    uint64_t gen;
    uint32_t next;
    mynah_slm_seq  ref;     /* the reference sequence */
    mynah_slm_bseq bs;      /* borrowed: the slot's KV */
    mynah_slm_bseq solo;    /* owned: the same request stepped alone */
} mrow;

static void check_multi(const fx *x, mynah_slm_backend *b, mynah_slm_kv_type kt,
                        int matmat, const char *label) {
    char err[256] = "", what[192], detail[256];
    const size_t V = x->vocab;
    mynah_slm_state ws;
    mynah_slm_bfwd *f = NULL;
    mynah_slm_bslots *pool = NULL;
    mrow R[MROWS];
    memset(R, 0, sizeof R);
    memset(&ws, 0, sizeof ws);
    float *glog = malloc((size_t)MROWS * V * sizeof(float));
    float *solo = malloc(V * sizeof(float));
    mynah_slm_decode_product_set(matmat);

    mynah_slm_bfwd_desc d = { .ctx_cap = 256, .batch_max = 16, .dec_max = MROWS,
                              .kv_k = kt, .kv_v = kt };
    int ok = mynah_slm_state_init_workspace(&ws, x->m, 256, err, sizeof err) == 0 &&
             mynah_slm_state_init_decode(&ws, MROWS, err, sizeof err) == 0 &&
             mynah_slm_bfwd_create(b, x->m, &d, &f, err, sizeof err) == 0;
    if (ok) {
        mynah_slm_bfwd_set_multi_product(f, matmat);
        mynah_slm_bslots_desc pd;
        memset(&pd, 0, sizeof pd);
        mynah_slm_bfwd_kv_desc(f, 160, &pd.kv);
        pd.n_slots = MROWS;
        ok = mynah_slm_backend_slots_create(b, &pd, &pool, err, sizeof err) == 0;
    }
    snprintf(what, sizeof what, "[%s/%s/%s] reference workspace, backend forward, %d-slot pool",
             label, mynah_slm_kv_type_name(kt), matmat ? "matmat" : "matvec", MROWS);
    if (!ok) { check(what, 0, err); goto out; }

    /* Admit: acquire a slot, bind the sequence to its KV, prefill. */
    uint32_t next_req = 0;
#define ADMIT(i) do {                                                                   \
        mrow *a = &R[i];                                                                \
        const uint32_t *p; uint32_t len;                                                \
        a->req = next_req++;                                                            \
        prompt_of(x, a->req, &p, &len);                                                 \
        ok = ok && mynah_slm_backend_slot_acquire(b, pool, &a->slot, &a->gen, err,      \
                                                  sizeof err) == 0 &&                   \
             mynah_slm_bseq_bind(f, &a->bs, mynah_slm_backend_slot_kv(pool, a->slot),   \
                                 err, sizeof err) == 0 &&                               \
             mynah_slm_seq_reserve(&a->ref, x->m, 160, kt, kt, err, sizeof err) == 0 && \
             (a->solo.kv || mynah_slm_bseq_init(f, &a->solo, 160, err, sizeof err) == 0) && \
             ref_prefill(&ws, &a->ref, p, len) == 0 &&                                  \
             b_prefill(f, &a->bs, p, len, err, sizeof err) == 0 &&                      \
             b_prefill(f, &a->solo, p, len, err, sizeof err) == 0;                      \
        a->next = p[len - 1];                                                           \
        a->live = 1;                                                                    \
    } while (0)
    for (uint32_t i = 0; i < MROWS; i++) ADMIT(i);
    check(what, ok, err);
    if (!ok) goto out;

    int ref_same = 1, solo_same = 1, argmax_same = 1, rc_ok = 1, reused_ok = 1;
    double worst_solo = 0.0;
    uint32_t cancelled_slot = 0;
    uint64_t cancelled_gen = 0;
    for (int step = 0; step < MSTEPS && rc_ok; step++) {
        /* Step 3: the client of row 1 leaves. Its slot is released (never
         * waits) and the row is simply not in the next step's table. */
        if (step == 3) {
            cancelled_slot = R[1].slot;
            cancelled_gen = R[1].gen;
            if (mynah_slm_backend_slot_release(b, pool, R[1].slot, err, sizeof err) != 0) {
                rc_ok = 0;
                break;
            }
            R[1].live = 0;
        }
        /* Step 5: a new request takes the freed slot — the lowest free
         * index, with a newer generation — and the KV it inherits is NOT
         * cleared: attention must read only what the new request wrote. */
        if (step == 5) {
            ADMIT(1);
            reused_ok = ok && R[1].slot == cancelled_slot && R[1].gen > cancelled_gen;
            if (!ok) { rc_ok = 0; break; }
        }

        mynah_slm_seq *rs[MROWS];
        mynah_slm_bseq *bs[MROWS];
        uint32_t tok[MROWS], idx[MROWS], n = 0;
        for (uint32_t i = 0; i < MROWS; i++) {
            if (!R[i].live) continue;
            rs[n] = &R[i].ref;
            bs[n] = &R[i].bs;
            tok[n] = R[i].next;
            idx[n] = i;
            n++;
        }
        float *rl[MROWS];
        if (mynah_slm_forward_multi(&ws, rs, tok, n, rl) != 0 ||
            mynah_slm_bfwd_multi(f, bs, tok, n, glog, err, sizeof err) != 0) {
            rc_ok = 0;
            break;
        }
        for (uint32_t r = 0; r < n; r++) {
            mrow *a = &R[idx[r]];
            const float *g = glog + (size_t)r * V;
            if (memcmp(rl[r], g, V * sizeof(float)) != 0) ref_same = 0;
            if (mynah_slm_bfwd_step(f, &a->solo, tok[r], solo, err, sizeof err) != 0) {
                rc_ok = 0;
                break;
            }
            if (memcmp(solo, g, V * sizeof(float)) != 0) solo_same = 0;
            const double rel = rel_diff(solo, g, V);
            if (rel > worst_solo) worst_solo = rel;
            if (argmax(solo, V) != argmax(g, V)) argmax_same = 0;
            if (a->bs.n_past != a->ref.n_past || a->bs.n_past != a->solo.n_past) rc_ok = 0;
            a->next = argmax(rl[r], V);
        }
    }

    snprintf(what, sizeof what,
             "[%s/%s/%s] multi step == forward_multi, every row, %d steps (memcmp)", label,
             mynah_slm_kv_type_name(kt), matmat ? "matmat" : "matvec", MSTEPS);
    check(what, rc_ok && ref_same, rc_ok ? "a row's logits differ" : err);
    snprintf(what, sizeof what, "[%s/%s/%s] multi step == each sequence alone %s", label,
             mynah_slm_kv_type_name(kt), matmat ? "matmat" : "matvec",
             matmat ? "(1e-4, same argmax)" : "(memcmp)");
    snprintf(detail, sizeof detail, "worst rel %.2e, bit-identical %s, argmax %s", worst_solo,
             solo_same ? "yes" : "no", argmax_same ? "same" : "DIFFERS");
    check(what, rc_ok && argmax_same && worst_solo < 1e-4 && (matmat || solo_same), detail);
    printf("     %s\n", detail);
    snprintf(what, sizeof what, "[%s/%s/%s] a cancelled row's slot is released, re-acquired "
             "with a newer generation, and serves a new request", label,
             mynah_slm_kv_type_name(kt), matmat ? "matmat" : "matvec");
    check(what, rc_ok && reused_ok, "slot or generation not as expected");

    /* Isolation: a full row, or a sequence twice, fails the step whole. */
    {
        mynah_slm_bseq *bs[3] = { &R[0].bs, &R[1].bs, &R[2].bs };
        const uint32_t tok[3] = { 1, 2, 3 };
        const uint32_t p0 = R[0].bs.n_past, p2 = R[2].bs.n_past, saved = R[1].bs.n_past;
        R[1].bs.n_past = R[1].bs.n_ctx;
        const int full = mynah_slm_bfwd_multi(f, bs, tok, 3, glog, err, sizeof err) == -1;
        R[1].bs.n_past = saved;
        bs[1] = &R[0].bs;
        const int twice = mynah_slm_bfwd_multi(f, bs, tok, 3, glog, err, sizeof err) == -1;
        snprintf(what, sizeof what, "[%s/%s/%s] a step with a full row or a row twice fails "
                 "and moves nobody", label, mynah_slm_kv_type_name(kt), matmat ? "matmat" : "matvec");
        check(what, full && twice && R[0].bs.n_past == p0 && R[2].bs.n_past == p2 &&
              R[1].bs.n_past == saved, "it ran or moved");
    }
#undef ADMIT

out:
    for (uint32_t i = 0; i < MROWS; i++) {
        mynah_slm_seq_free(&R[i].ref);
        mynah_slm_bseq_free(f, &R[i].solo);
        mynah_slm_bseq_free(f, &R[i].bs);
    }
    mynah_slm_backend_slots_destroy(b, pool);
    mynah_slm_bfwd_free(f);
    mynah_slm_state_free(&ws);
    mynah_slm_decode_product_set(-1);
    free(glog);
    free(solo);
}

static void run(const char *label, int quant) {
    printf("\n-- %s fixture --\n", label);
    fixture_spec spec;
    fixture_spec_tiny(&spec, quant);
    /* Untied too: the separate head is a second Q6_K upload, and the tied
     * fixture alone could not tell head from embedding. */
    for (int untied = 0; untied < 2; untied++) {
        spec.untied = untied;
        char path[256], err[256], lab[64];
        snprintf(lab, sizeof lab, "%s%s", label, untied ? ",untied" : "");
        if (fixture_write_temp(path, sizeof path, &spec, err, sizeof err) != 0) {
            check("fixture written", 0, err);
            return;
        }
        mynah_slm_model_t *m = mynah_slm_load(path, err, sizeof err);
        mynah_slm_tokenizer *tok = m ? mynah_slm_tokenizer_load(m->gguf, err, sizeof err) : NULL;
        check("fixture loads", m && tok, err);
        if (!m || !tok) { mynah_slm_free(m); unlink(path); return; }

        const long n_tok = mynah_slm_tokenize(tok, TEXT, 0, NULL, 0);
        uint32_t *ids = malloc((size_t)n_tok * sizeof *ids);
        mynah_slm_tokenize(tok, TEXT, 0, ids, (size_t)n_tok);
        fx x = { m, ids, n_tok, mynah_slm_vocab_size(m) };
        printf("     %ld prompt tokens, vocab %u, %u layers, %u/%u heads of %u\n", n_tok,
               x.vocab, m->cfg.n_layers, m->cfg.n_heads, m->cfg.n_kv_heads, m->cfg.head_dim);

        mynah_slm_backend *b = NULL;
        if (mynah_slm_backend_open(MYNAH_SLM_DEVICE_CPU, &b, err, sizeof err) != 0) {
            check("cpu backend opens", 0, err);
        } else {
            check_parity(&x, b, MYNAH_SLM_KV_F32, lab);
            check_parity(&x, b, MYNAH_SLM_KV_BF16, lab);
            check_driver(&x, b, tok, lab);
            for (int mm = 0; mm < 2; mm++) {
                check_multi(&x, b, MYNAH_SLM_KV_F32, mm, lab);
                check_multi(&x, b, MYNAH_SLM_KV_BF16, mm, lab);
            }
        }
        mynah_slm_backend_close(b);
        free(ids);
        mynah_slm_tokenizer_free(tok);
        mynah_slm_free(m);
        unlink(path);
    }
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
