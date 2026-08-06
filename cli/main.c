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
        "  mynah-slm inspect <model.gguf>            per-type census\n"
        "  mynah-slm inspect <model.gguf> --tensors  ... plus every tensor\n"
        "  mynah-slm inspect <model.gguf> --tensors Q6_K   ... only that type\n"
        "  mynah-slm inspect <model.gguf> --meta     ... plus the metadata KV\n"
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

/* "[1024 x 151936]" — row-major, the way a model card writes it. */
static void shape_str(const mynah_slm_tensor_info *t, char *buf, size_t n) {
    size_t off = 0;
    off += (size_t)snprintf(buf + off, n - off, "[");
    for (uint32_t d = 0; d < t->rank && off < n; d++)
        off += (size_t)snprintf(buf + off, n - off, "%s%llu",
                                d ? " x " : "", (unsigned long long)t->shape[d]);
    if (off < n) snprintf(buf + off, n - off, "]");
}

static int cmd_inspect(const char *path, int list_tensors, const char *only_type,
                       int list_meta) {
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

    if (list_meta) {
        printf("  %-40s %s\n", "metadata key", "value");
        for (size_t i = 0; i < r.n_meta; i++)
            printf("  %-40s %s\n", r.meta[i].name, r.meta[i].value);
        printf("\n");
    }

    if (list_tensors) {
        uint64_t shown_bytes = 0;
        size_t   shown = 0;
        printf("  %-34s %-6s %-22s %10s\n", "tensor", "type", "shape", "size");
        for (size_t i = 0; i < r.n_tensor_info; i++) {
            const mynah_slm_tensor_info *t = &r.tensors[i];
            if (only_type && strcmp(t->type_name, only_type) != 0) continue;
            char shape[64], sz[32];
            shape_str(t, shape, sizeof shape);
            human(t->nbytes, sz, sizeof sz);
            printf("  %-34s %-6s %-22s %10s\n", t->name, t->type_name, shape, sz);
            shown_bytes += t->nbytes;
            shown++;
        }
        char sub[32];
        human(shown_bytes, sub, sizeof sub);
        printf("\n  %zu tensors listed, %s (%.1f%% of the file)\n\n", shown, sub,
               r.total_bytes ? 100.0 * (double)shown_bytes / (double)r.total_bytes : 0.0);
    }

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
        int list = 0, meta = 0;
        const char *only = NULL;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "--tensors")) {
                list = 1;
                /* an optional bare type name may follow */
                if (i + 1 < argc && argv[i + 1][0] != '-') only = argv[++i];
            } else if (!strcmp(argv[i], "--meta")) {
                meta = 1;
            } else {
                fprintf(stderr, "mynah-slm: unknown option '%s'\n", argv[i]);
                return 2;
            }
        }
        return cmd_inspect(argv[2], list, only, meta);
    }

    fprintf(stderr, "mynah-slm: unknown command '%s'\n", argv[1]);
    usage(stderr);
    return 2;
}
