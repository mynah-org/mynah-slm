/* mynah-slm — CLI.
 *
 * Only `inspect` exists so far. Subcommands are added when the engine behind
 * them works; a `chat` that prints "not implemented" would be worse than no
 * `chat` at all.
 *
 * SPDX-License-Identifier: MIT */
#include "mynah_slm.h"

#include <stdio.h>
#include <string.h>

static void usage(FILE *f) {
    fprintf(f,
        "mynah-slm %s — CPU-first SLM inference engine\n"
        "\n"
        "usage:\n"
        "  mynah-slm inspect <model.gguf>   what is inside a checkpoint\n"
        "  mynah-slm --version\n"
        "\n"
        "Weights live on the NAS (models/ is a symlink); see CLAUDE.md.\n",
        mynah_slm_version());
}

static void human(uint64_t bytes, char *buf, size_t n) {
    const char *unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = (double)bytes;
    size_t u = 0;
    while (v >= 1024.0 && u + 1 < sizeof unit / sizeof *unit) { v /= 1024.0; u++; }
    snprintf(buf, n, "%.1f %s", v, unit[u]);
}

static int cmd_inspect(const char *path) {
    mynah_slm_report r;
    char err[256];

    if (mynah_slm_inspect_gguf(&r, path, err, sizeof err) != 0) {
        fprintf(stderr, "mynah-slm: %s\n", err);
        return 1;
    }

    char total[32];
    human(r.total_bytes, total, sizeof total);

    printf("%s\n", r.path);
    printf("  arch          %s\n", r.arch[0] ? r.arch : "(unset)");
    printf("  gguf          v%u", r.gguf_version);
    if (r.shards > 1) printf(", %u shards", r.shards);
    printf("\n");
    printf("  tensors       %zu   metadata keys %zu\n", r.n_tensors, r.n_kv);
    printf("  weights       %s   (%.2f bits/weight overall)\n",
           total, r.bits_per_weight);
    printf("\n");

    printf("  %-10s %8s %10s  %s\n", "type", "tensors", "size", "");
    for (size_t i = 0; i < r.n_types; i++) {
        const mynah_slm_type_use *t = &r.types[i];
        char sz[32];
        human(t->bytes, sz, sizeof sz);
        printf("  %-10s %8zu %10s  %s\n", t->type_name, t->tensors, sz,
               t->can_dequant ? "" : "<- ingot cannot decode this");
    }
    printf("\n");

    if (!r.runnable)
        printf("  NOT RUNNABLE: at least one type above has no decoder.\n");

    mynah_slm_report_free(&r);
    return r.runnable ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(stderr); return 2; }

    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-v")) {
        printf("%s\n", mynah_slm_version());
        return 0;
    }
    if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage(stdout);
        return 0;
    }
    if (!strcmp(argv[1], "inspect")) {
        if (argc < 3) { fprintf(stderr, "mynah-slm: inspect needs a path\n"); return 2; }
        return cmd_inspect(argv[2]);
    }

    fprintf(stderr, "mynah-slm: unknown command '%s'\n", argv[1]);
    usage(stderr);
    return 2;
}
