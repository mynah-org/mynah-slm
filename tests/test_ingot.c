/* test_ingot.c — pin the container-layer contract we depend on.
 *
 * Needs no model. It exists so that a bad `make update-ingot` fails HERE, with
 * a name attached, instead of three modules later as wrong numbers.
 *
 * The block geometries below are transcribed from ingot's docs/QUANTS.md, which
 * in turn is generated against ggml itself. If one of these ever fails, the
 * subtree changed something load-bearing.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/dtype.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void check_geometry(int type, const char *name,
                           uint64_t want_elems, uint64_t want_bytes) {
    uint64_t elems = 0, bytes = 0;
    if (ingot_type_geometry(type, &elems, &bytes) != 0) {
        printf("FAIL %-8s ingot does not know this type at all\n", name);
        failures++;
        return;
    }
    if (elems != want_elems || bytes != want_bytes) {
        printf("FAIL %-8s geometry %llu/%llu, expected %llu/%llu (elems/bytes)\n",
               name, (unsigned long long)elems, (unsigned long long)bytes,
               (unsigned long long)want_elems, (unsigned long long)want_bytes);
        failures++;
        return;
    }
    /* The name matters as much as the geometry: our error messages quote it,
     * and docs/QUANTS.md promises "IQ4_XS" rather than "unknown type 23". */
    const char *got = ingot_type_name(type);
    if (!got || strcmp(got, name) != 0) {
        printf("FAIL %-8s ingot_type_name says '%s'\n", name, got ? got : "(null)");
        failures++;
        return;
    }
    printf("ok   %-8s %llu weights in %llu bytes (%.2f bits/w)\n", name,
           (unsigned long long)elems, (unsigned long long)bytes,
           (double)bytes * 8.0 / (double)elems);
}

static void check_dequant(int type, const char *name, int want) {
    int got = ingot_type_can_dequant(type) ? 1 : 0;
    if (got != want) {
        printf("FAIL %-8s can_dequant=%d, expected %d\n", name, got, want);
        failures++;
        return;
    }
    printf("ok   %-8s %s\n", name, want ? "decodes" : "does NOT decode (known gap)");
}

int main(void) {
    printf("-- block geometry (docs/QUANTS.md) --\n");
    /* The default distribution formats. Q4_0 is what Google's Gemma 4 QAT
     * checkpoint ships, so its 18-byte block is load-bearing for us. */
    check_geometry(INGOT_TYPE_Q4_0, "Q4_0",  32,  18);
    check_geometry(INGOT_TYPE_Q8_0, "Q8_0",  32,  34);
    /* A "Q4_K_M" file is these three, not a type called Q4_K_M. */
    check_geometry(INGOT_TYPE_Q4_K, "Q4_K", 256, 144);
    check_geometry(INGOT_TYPE_Q6_K, "Q6_K", 256, 210);
    /* The sub-4-bit candidates from the quantization policy. */
    check_geometry(INGOT_TYPE_Q3_K, "Q3_K", 256, 110);
    check_geometry(INGOT_TYPE_Q2_K, "Q2_K", 256,  84);

    printf("\n-- dequant coverage --\n");
    check_dequant(INGOT_TYPE_Q4_0,   "Q4_0",   1);
    check_dequant(INGOT_TYPE_Q8_0,   "Q8_0",   1);
    check_dequant(INGOT_TYPE_Q4_K,   "Q4_K",   1);
    check_dequant(INGOT_TYPE_Q6_K,   "Q6_K",   1);
    check_dequant(INGOT_TYPE_Q3_K,   "Q3_K",   1);
    check_dequant(INGOT_TYPE_Q2_K,   "Q2_K",   1);
    check_dequant(INGOT_TYPE_IQ2_XXS,"IQ2_XXS",1);
    check_dequant(INGOT_TYPE_IQ1_S,  "IQ1_S",  1);
    check_dequant(INGOT_TYPE_TQ2_0,  "TQ2_0",  1);
    check_dequant(INGOT_TYPE_BF16,   "BF16",   1);
    check_dequant(INGOT_TYPE_F16,    "F16",    1);
    /* The one documented gap: llama.cpp's own reference package has no decoder
     * for Q1_0 either, so nothing could validate one. Asserted so that the day
     * it changes, we find out from a test rather than from a rumour. */
    check_dequant(INGOT_TYPE_Q1_0,   "Q1_0",   0);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
