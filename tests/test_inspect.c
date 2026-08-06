/* test_inspect.c — end-to-end check of the checkpoint census, with no model.
 *
 * ingot's writers exist partly for this: "a reader whose tests can only consume
 * files somebody else wrote is a reader with a blind spot". So we synthesize a
 * GGUF shaped like a real Q4_K_M recipe — Q4_K for most weights, Q6_K for
 * ffn_down, F32 for the norms — and assert that inspect reports the mix rather
 * than the filename.
 *
 * SPDX-License-Identifier: MIT */
#include "mynah_slm.h"

#include "ingot/dtype.h"
#include "ingot/write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void fail(const char *what, const char *detail) {
    printf("FAIL %s: %s\n", what, detail);
    failures++;
}

/* A deterministic ramp: values don't matter, byte accounting does. */
static float *ramp(size_t n) {
    float *v = malloc(n * sizeof *v);
    if (!v) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < n; i++) v[i] = (float)((i % 251) - 125) * 0.01f;
    return v;
}

/* Writes a miniature two-layer model. Returns 0 on success. */
static int write_fixture(const char *path, char *err, size_t errsz) {
    enum { DIM = 256, FF = 512 };

    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (!w) { snprintf(err, errsz, "writer_new failed"); return -1; }

    ingot_gguf_kv_string(w, "general.architecture", "qwen3");
    ingot_gguf_kv_u32(w, "qwen3.block_count", 2);
    ingot_gguf_kv_u32(w, "qwen3.embedding_length", DIM);

    float *attn = ramp(DIM * DIM);
    float *down = ramp(FF * DIM);
    float *norm = ramp(DIM);
    int rc = 0;

    for (int layer = 0; layer < 2 && rc == 0; layer++) {
        char name[64];
        const uint64_t ne_attn[2] = { DIM, DIM };
        const uint64_t ne_down[2] = { FF, DIM };
        const uint64_t ne_norm[1] = { DIM };

        snprintf(name, sizeof name, "blk.%d.attn_q.weight", layer);
        rc |= ingot_gguf_add_f32(w, name, INGOT_TYPE_Q4_K, 2, ne_attn, attn);

        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", layer);
        rc |= ingot_gguf_add_f32(w, name, INGOT_TYPE_Q6_K, 2, ne_down, down);

        /* Norms stay f32 — that is the policy, and it is why the overall
         * bits/weight of a "4-bit" checkpoint is never 4.00. */
        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", layer);
        rc |= ingot_gguf_add_f32(w, name, INGOT_TYPE_F32, 1, ne_norm, norm);
    }

    if (rc == 0) rc = ingot_gguf_writer_save(w, path, err, errsz);
    else snprintf(err, errsz, "add_f32 failed");

    ingot_gguf_writer_free(w);
    free(attn); free(down); free(norm);
    return rc;
}

/* Look up one row of the census. NULL when the type is absent. */
static const mynah_slm_type_use *find_type(const mynah_slm_report *r, int type) {
    for (size_t i = 0; i < r->n_types; i++)
        if (r->types[i].type == type) return &r->types[i];
    return NULL;
}

int main(void) {
    char path[] = "/tmp/mynah_slm_fixture_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 1; }
    close(fd);

    char err[256] = {0};
    if (write_fixture(path, err, sizeof err) != 0) {
        printf("FAIL could not build the fixture: %s\n", err);
        unlink(path);
        return 1;
    }

    mynah_slm_report r;
    if (mynah_slm_inspect_gguf(&r, path, err, sizeof err) != 0) {
        printf("FAIL inspect: %s\n", err);
        unlink(path);
        return 1;
    }

    printf("-- census of a synthetic Q4_K_M-shaped checkpoint --\n");
    for (size_t i = 0; i < r.n_types; i++)
        printf("ok   %-6s %2zu tensors, %llu bytes\n", r.types[i].type_name,
               r.types[i].tensors, (unsigned long long)r.types[i].bytes);

    if (strcmp(r.arch, "qwen3") != 0) fail("arch", r.arch);
    if (r.n_tensors != 6) fail("tensor count", "expected 6");
    if (r.n_types != 3)   fail("type count", "expected 3 distinct types");
    if (!r.runnable)      fail("runnable", "every type here decodes");

    const mynah_slm_type_use *q4k = find_type(&r, INGOT_TYPE_Q4_K);
    const mynah_slm_type_use *q6k = find_type(&r, INGOT_TYPE_Q6_K);
    const mynah_slm_type_use *f32 = find_type(&r, INGOT_TYPE_F32);
    if (!q4k || q4k->tensors != 2) fail("Q4_K", "expected 2 tensors");
    if (!q6k || q6k->tensors != 2) fail("Q6_K", "expected 2 tensors");
    if (!f32 || f32->tensors != 2) fail("F32",  "expected 2 tensors");

    /* The census must be sorted heaviest-first: ffn_down at Q6_K outweighs the
     * attention matrices here, so Q6_K has to lead. */
    if (r.n_types >= 2 && r.types[0].bytes < r.types[1].bytes)
        fail("ordering", "census is not sorted by size");

    /* The headline number: more than 4 bits/weight even though the biggest
     * tensors are 4-bit, because the norms are f32. */
    printf("ok   overall %.2f bits/weight\n", r.bits_per_weight);
    if (r.bits_per_weight <= 4.5 || r.bits_per_weight >= 8.0)
        fail("bits_per_weight", "outside the plausible range for this mix");

    mynah_slm_report_free(&r);
    unlink(path);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
