/* mynah-slm — CLI.
 *
 * Only `inspect` exists so far. Subcommands are added when the engine behind
 * them works; a `chat` that prints "not implemented" would be worse than no
 * `chat` at all.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "generate.h"
#include "model.h"
#include "mynah_slm.h"
#include "sampler.h"
#include "template.h"
#include "threads.h"
#include "timing.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
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
        "  mynah-slm tokenize -m <model.gguf> [--special] < text\n"
        "\n"
        "  mynah-slm run -m <model.gguf> -p \"prompt\" [-n 128] [--think off|low|on]\n"
        "                [--temp T] [--top-k K] [--top-p P] [--min-p M] [--seed S]\n"
        "                [--raw] [--no-stream] [--quiet] [--ctx N] [--show-think]\n"
        "                [-t N | --threads N]   (default: performance cores)\n"
        "\n"
        "  Reasoning NEVER reaches stdout: it is discarded, or written to stderr\n"
        "  with --show-think. stdout is the answer, so `| mynah-tts` is safe.\n"
        "\n"
        "  Timings go to stderr, so stdout stays pipeable into mynah-tts.\n"
        "  mynah-slm --version\n"
        "\n"
        "No checkpoint yet? scripts/download_model.sh --list\n",
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

/* Read all of stdin. Returns NULL on failure; caller frees. */
static char *slurp(void) {
    size_t cap = 4096, n = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (n + 1 >= cap) {
            cap *= 2;
            char *grown = realloc(buf, cap);
            if (!grown) { free(buf); return NULL; }
            buf = grown;
        }
        const size_t got = fread(buf + n, 1, cap - n - 1, stdin);
        n += got;
        if (got == 0) break;
    }
    buf[n] = '\0';
    return buf;
}

/* Token ids for the text on stdin, one per line. Deliberately machine-readable:
 * tools/eval/fuzz_tokenizer.py drives it against HF as an oracle. */
static int cmd_tokenize(const char *model_path, int parse_special) {
    char err[256];
    mynah_slm_model_t *m = mynah_slm_load(model_path, err, sizeof err);
    if (!m) { fprintf(stderr, "mynah-slm: %s\n", err); return 1; }

    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    if (!tok) { fprintf(stderr, "mynah-slm: %s\n", err); mynah_slm_free(m); return 1; }

    char *text = slurp();
    if (!text) { fprintf(stderr, "mynah-slm: cannot read stdin\n"); return 1; }

    const long n = mynah_slm_tokenize(tok, text, parse_special, NULL, 0);
    int rc = 0;
    if (n < 0) { fprintf(stderr, "mynah-slm: tokenize failed\n"); rc = 1; }
    else {
        uint32_t *ids = malloc((size_t)(n ? n : 1) * sizeof *ids);
        if (!ids) rc = 1;
        else {
            mynah_slm_tokenize(tok, text, parse_special, ids, (size_t)n);
            for (long i = 0; i < n; i++) printf("%u\n", ids[i]);
            free(ids);
        }
    }
    free(text);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);
    return rc;
}

/* Streaming: write each piece as it arrives and flush, or the pipeline
 * downstream waits for a 4 KB buffer instead of a sentence. */
static int stream_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)ctx; (void)id;
    if (len) { fwrite(text, 1, len, stdout); fflush(stdout); }
    return 0;
}

/* Thinking goes to STDERR, never stdout. stdout is the answer channel and the
 * thing that gets piped into mynah-tts; if reasoning could reach it, one
 * forgotten flag would have the speech stage read the model's monologue out
 * loud. Structural, not a filter. */
static int think_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)ctx; (void)id;
    if (len) { fwrite(text, 1, len, stderr); fflush(stderr); }
    return 0;
}

typedef struct { char *buf; size_t used, cap; } collect;

static int collect_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    collect *c = ctx;
    if (c->used + len + 1 > c->cap) {
        size_t cap = (c->used + len + 1) * 2;
        char *g = realloc(c->buf, cap);
        if (!g) return 1;
        c->buf = g; c->cap = cap;
    }
    memcpy(c->buf + c->used, text, len);
    c->used += len;
    c->buf[c->used] = '\0';
    return 0;
}

typedef struct {
    const char *model, *prompt, *system;
    int   max_new, raw, stream, quiet, think, ctx, show_think, threads;
    mynah_slm_sampler_params sp;
} run_opts;

static int cmd_run(run_opts *o) {
    char err[256];
    mynah_slm_timing tm;
    mynah_slm_timing_reset(&tm);
    tm.n_threads = mynah_slm_threads_init(o->threads);

    mynah_slm_timing_start(&tm);
    mynah_slm_model_t *m = mynah_slm_load(o->model, err, sizeof err);
    if (!m) { fprintf(stderr, "mynah-slm: %s\n", err); return 1; }

    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    if (!tok) { fprintf(stderr, "mynah-slm: %s\n", err); mynah_slm_free(m); return 1; }

    /* Render the chat turn unless --raw. The template inserts control tokens,
     * so THIS text is tokenized with parse_special on; user content reaching it
     * has already been escaped by being placed inside a turn. */
    char *text = NULL;
    if (o->raw) {
        text = strdup(o->prompt);
    } else {
        mynah_slm_message msgs[2];
        size_t n = 0;
        if (o->system) { msgs[n].role = MYNAH_SLM_ROLE_SYSTEM; msgs[n].content = o->system; n++; }
        msgs[n].role = MYNAH_SLM_ROLE_USER; msgs[n].content = o->prompt; n++;

        const long need = mynah_slm_render_chat(msgs, n, o->think, NULL, 0);
        if (need < 0) { fprintf(stderr, "mynah-slm: cannot render the prompt\n"); return 1; }
        text = malloc((size_t)need + 1);
        if (text) mynah_slm_render_chat(msgs, n, o->think, text, (size_t)need + 1);
    }
    if (!text) { fprintf(stderr, "mynah-slm: out of memory\n"); return 1; }

    const long n_prompt = mynah_slm_tokenize(tok, text, 1, NULL, 0);
    if (n_prompt <= 0) { fprintf(stderr, "mynah-slm: empty prompt\n"); return 1; }
    uint32_t *ids = malloc((size_t)n_prompt * sizeof *ids);
    if (!ids) return 1;
    mynah_slm_tokenize(tok, text, 1, ids, (size_t)n_prompt);

    mynah_slm_state st;
    const uint32_t want_ctx = o->ctx > 0 ? (uint32_t)o->ctx
                                         : (uint32_t)(n_prompt + o->max_new + 8);
    if (mynah_slm_state_init(&st, m, want_ctx, err, sizeof err) != 0) {
        fprintf(stderr, "mynah-slm: %s\n", err);
        return 1;
    }
    mynah_slm_timing_end_load(&tm);

    mynah_slm_sampler *sam = mynah_slm_sampler_new(&o->sp, mynah_slm_vocab_size(m));
    if (!sam) { fprintf(stderr, "mynah-slm: out of memory\n"); return 1; }

    /* Qwen3 has TWO terminators; the GGUF only names one. Stopping on a subset
     * runs past the end of the turn. */
    const uint32_t eos[] = { mynah_slm_tokenizer_eos(tok), 151643u };

    collect col = {0};
    mynah_slm_gen_params gp = {
        .prompt = ids, .n_prompt = (size_t)n_prompt,
        .max_new = (uint32_t)o->max_new,
        .eos = eos, .n_eos = sizeof eos / sizeof *eos,
        .cb = o->stream ? stream_cb : collect_cb,
        .cb_ctx = o->stream ? NULL : (void *)&col,
        /* Resolved by name: an id hardcoded here would be right for exactly
         * one vocabulary. -1 when the model has no think markers at all, which
         * simply disables the split. */
        .think_open  = mynah_slm_token_find(tok, "<think>"),
        .think_close = mynah_slm_token_find(tok, "</think>"),
        /* NULL discards the channel — the right default for a speech pipeline. */
        .cb_think     = o->show_think ? think_cb : NULL,
        .cb_think_ctx = NULL,
    };

    const long n = mynah_slm_generate(&st, tok, sam, &gp, &tm);
    if (!o->stream && col.buf) fputs(col.buf, stdout);
    fputc('\n', stdout);

    if (!o->quiet) {
        char line[256];
        mynah_slm_timing_format(&tm, line, sizeof line);
        fprintf(stderr, "%s\n", line);
    }

    free(col.buf);
    free(ids);
    free(text);
    mynah_slm_sampler_free(sam);
    mynah_slm_state_free(&st);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);
    mynah_slm_threads_shutdown();
    return n < 0 ? 1 : 0;
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
    if (!strcmp(argv[1], "run")) {
        run_opts o = { .max_new = 128, .stream = 1, .think = MYNAH_SLM_THINK_OFF };
        mynah_slm_sampler_defaults(&o.sp);
        for (int i = 2; i < argc; i++) {
            const char *a = argv[i];
            const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDV() do { if (!v) { fprintf(stderr, "mynah-slm: %s needs a value\n", a); return 2; } i++; } while (0)
            if      (!strcmp(a, "-m"))          { NEEDV(); o.model = v; }
            else if (!strcmp(a, "-p"))          { NEEDV(); o.prompt = v; }
            else if (!strcmp(a, "--system"))    { NEEDV(); o.system = v; }
            else if (!strcmp(a, "-n"))          { NEEDV(); o.max_new = atoi(v); }
            else if (!strcmp(a, "--ctx"))       { NEEDV(); o.ctx = atoi(v); }
            else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { NEEDV(); o.threads = atoi(v); }
            else if (!strcmp(a, "--temp"))      { NEEDV(); o.sp.temp = (float)atof(v); }
            else if (!strcmp(a, "--top-k"))     { NEEDV(); o.sp.top_k = (uint32_t)atoi(v); }
            else if (!strcmp(a, "--top-p"))     { NEEDV(); o.sp.top_p = (float)atof(v); }
            else if (!strcmp(a, "--min-p"))     { NEEDV(); o.sp.min_p = (float)atof(v); }
            else if (!strcmp(a, "--seed"))      { NEEDV(); o.sp.seed = strtoull(v, NULL, 10); }
            else if (!strcmp(a, "--repeat-penalty")) { NEEDV(); o.sp.repeat_penalty = (float)atof(v); }
            else if (!strcmp(a, "--think")) {
                NEEDV();
                o.think = !strcmp(v, "on")  ? MYNAH_SLM_THINK_ON
                        : !strcmp(v, "low") ? MYNAH_SLM_THINK_LOW
                                            : MYNAH_SLM_THINK_OFF;
            }
            else if (!strcmp(a, "--show-think")) o.show_think = 1;
            else if (!strcmp(a, "--raw"))       o.raw = 1;
            else if (!strcmp(a, "--no-stream")) o.stream = 0;
            else if (!strcmp(a, "--quiet"))     o.quiet = 1;
            else { fprintf(stderr, "mynah-slm: unknown option '%s'\n", a); return 2; }
#undef NEEDV
        }
        if (!o.model || !o.prompt) {
            fprintf(stderr, "mynah-slm: run needs -m <model.gguf> and -p <prompt>\n");
            return 2;
        }
        return cmd_run(&o);
    }
    if (!strcmp(argv[1], "tokenize")) {
        const char *model = NULL;
        int special = 0;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
            else if (!strcmp(argv[i], "--special")) special = 1;
            else { fprintf(stderr, "mynah-slm: unknown option '%s'\n", argv[i]); return 2; }
        }
        if (!model) { fprintf(stderr, "mynah-slm: tokenize needs -m <model.gguf>\n"); return 2; }
        return cmd_tokenize(model, special);
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
