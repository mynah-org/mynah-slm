/* mynah-slm-server — OpenAI-compatible HTTP, no framework.
 *
 * Concurrency model, stated because it is a decision and not an oversight:
 * INFERENCE IS SERIALIZED. One request runs at a time, using every thread.
 *
 * The alternative — N concurrent inferences with 1/N threads each — is worse
 * for a single CPU-bound model. The work is memory-bandwidth bound, so
 * splitting cores across requests does not increase aggregate throughput; it
 * just makes every request slower and multiplies the KV cache footprint. And
 * mynah-asr already paid for the general lesson: several inferences fighting
 * over one BLAS pool collapsed aggregate throughput from four concurrent
 * requests up.
 *
 * So connections are accepted concurrently, parsed concurrently, and queued at
 * the model. A caller sees the queue as latency, never as an error.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "generate.h"
#include "http.h"
#include "json.h"
#include "model.h"
#include "mynah_slm.h"
#include "sampler.h"
#include "template.h"
#include "threads.h"
#include "timing.h"
#include "tokenizer.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mynah_slm_model_t   *model;
    mynah_slm_tokenizer *tok;
    const char          *model_name;
    uint32_t             n_ctx;

    pthread_mutex_t infer_mu;    /* the serialization point */

    /* Rolling decode t/s, so /health shows a regression in production rather
     * than only in a benchmark. */
    pthread_mutex_t stat_mu;
    double   recent_tok_s[16];
    unsigned n_stat, stat_head;

    /* Monotonic request counter. The obvious id — the connection pointer — is
     * a stack address that repeats across threads, so two different requests
     * would share an id and a client correlating logs would merge them. */
    unsigned long next_id;
} server_ctx;

static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

static void record_tok_s(server_ctx *c, double v) {
    pthread_mutex_lock(&c->stat_mu);
    c->recent_tok_s[c->stat_head] = v;
    c->stat_head = (c->stat_head + 1) % 16;
    if (c->n_stat < 16) c->n_stat++;
    pthread_mutex_unlock(&c->stat_mu);
}

/* ── streaming context ─────────────────────────────────────────────────────
 * The answer channel writes SSE frames as tokens arrive. The thinking channel
 * is DISCARDED unless the caller opted in — same rule as the CLI, for the same
 * reason: a client piping this into TTS must not receive reasoning. */

typedef struct {
    http_conn *conn;
    const char *id;
    const char *model_name;
    int   stream;
    char *buf;            /* non-streaming accumulation */
    size_t used, cap;
    int   failed;
} emit_ctx;

static int emit_append(emit_ctx *e, const char *text, size_t len) {
    if (e->used + len + 1 > e->cap) {
        size_t cap = (e->used + len + 1) * 2;
        char *g = realloc(e->buf, cap);
        if (!g) { e->failed = 1; return 1; }
        e->buf = g;
        e->cap = cap;
    }
    memcpy(e->buf + e->used, text, len);
    e->used += len;
    e->buf[e->used] = '\0';
    return 0;
}

static int answer_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    emit_ctx *e = ctx;
    if (len == 0) return 0;

    if (!e->stream) return emit_append(e, text, len);

    char esc[2048], frame[3072];
    json_escape(text, len, esc, sizeof esc);
    const int n = snprintf(frame, sizeof frame,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"model\":%s,\"choices\":[{\"index\":0,\"delta\":{\"content\":%s},"
        "\"finish_reason\":null}]}\n\n",
        e->id, e->model_name, esc);
    if (http_write(e->conn, frame, (size_t)n) != 0) { e->failed = 1; return 1; }
    return 0;
}

/* ── handlers ─────────────────────────────────────────────────────────────── */

static void handle_health(server_ctx *c, http_conn *conn) {
    double avg = 0.0;
    pthread_mutex_lock(&c->stat_mu);
    for (unsigned i = 0; i < c->n_stat; i++) avg += c->recent_tok_s[i];
    if (c->n_stat) avg /= c->n_stat;
    const unsigned n = c->n_stat;
    pthread_mutex_unlock(&c->stat_mu);

    char body[512];
    const int len = snprintf(body, sizeof body,
        "{\"status\":\"ok\",\"model\":\"%s\",\"n_ctx\":%u,\"threads\":%d,"
        "\"recent_requests\":%u,\"recent_decode_tok_s\":%.2f}\n",
        c->model_name, c->n_ctx, mynah_slm_threads_count(), n, avg);
    http_respond(conn, 200, "application/json", body, (size_t)len);
}

static void handle_models(server_ctx *c, http_conn *conn) {
    char body[512];
    const int len = snprintf(body, sizeof body,
        "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
        "\"owned_by\":\"mynah\"}]}\n", c->model_name);
    http_respond(conn, 200, "application/json", body, (size_t)len);
}

static void handle_tokenize(server_ctx *c, http_conn *conn,
                            const char *body, size_t blen) {
    json_val root, v;
    if (json_parse(body, body + blen, &root) != 0 || root.kind != JSON_OBJECT) {
        http_error(conn, 400, "malformed JSON body");
        return;
    }
    if (json_object_get(&root, "content", &v) != 0 || v.kind != JSON_STRING) {
        http_error(conn, 400, "expected a string field \"content\"");
        return;
    }
    char *text = json_string_dup(&v);
    if (!text) { http_error(conn, 500, "out of memory"); return; }

    /* parse_special OFF: this is caller-supplied content, and text that merely
     * looks like a control token must not become one. Same boundary the CLI
     * and the chat handler enforce. */
    const long n = mynah_slm_tokenize(c->tok, text, 0, NULL, 0);
    if (n < 0) { free(text); http_error(conn, 400, "cannot tokenize"); return; }

    uint32_t *ids = malloc((size_t)(n ? n : 1) * sizeof *ids);
    if (!ids) { free(text); http_error(conn, 500, "out of memory"); return; }
    mynah_slm_tokenize(c->tok, text, 0, ids, (size_t)n);

    size_t cap = (size_t)n * 8 + 64;
    char *out = malloc(cap);
    size_t o = (size_t)snprintf(out, cap, "{\"count\":%ld,\"tokens\":[", n);
    for (long i = 0; i < n; i++)
        o += (size_t)snprintf(out + o, cap - o, "%s%u", i ? "," : "", ids[i]);
    o += (size_t)snprintf(out + o, cap - o, "]}\n");

    http_respond(conn, 200, "application/json", out, o);
    free(out); free(ids); free(text);
}

static void handle_chat(server_ctx *c, http_conn *conn,
                        const char *body, size_t blen) {
    json_val root, msgs;
    if (json_parse(body, body + blen, &root) != 0 || root.kind != JSON_OBJECT) {
        http_error(conn, 400, "malformed JSON body");
        return;
    }
    if (json_object_get(&root, "messages", &msgs) != 0 || msgs.kind != JSON_ARRAY) {
        http_error(conn, 400, "expected an array field \"messages\"");
        return;
    }

    /* Render the turn. Contents are copied out of the body, so nothing here
     * borrows a span that a later realloc could move. */
    mynah_slm_message rendered[32];
    char *owned[32];
    size_t n_msg = 0;

    for (size_t i = 0; i < 32; i++) {
        json_val m, role, content;
        if (json_array_at(&msgs, i, &m) != 0) break;
        if (m.kind != JSON_OBJECT) continue;
        if (json_object_get(&m, "content", &content) != 0 || content.kind != JSON_STRING) continue;

        char r[32] = "user";
        if (json_object_get(&m, "role", &role) == 0 && role.kind == JSON_STRING)
            json_string_copy(&role, r, sizeof r);

        owned[n_msg] = json_string_dup(&content);
        if (!owned[n_msg]) break;
        rendered[n_msg].content = owned[n_msg];
        rendered[n_msg].role =
            !strcmp(r, "system")    ? MYNAH_SLM_ROLE_SYSTEM :
            !strcmp(r, "assistant") ? MYNAH_SLM_ROLE_ASSISTANT :
            !strcmp(r, "tool")      ? MYNAH_SLM_ROLE_TOOL : MYNAH_SLM_ROLE_USER;
        n_msg++;
    }
    if (n_msg == 0) { http_error(conn, 400, "no usable messages"); return; }

    const int stream  = json_get_bool(&root, "stream", 0);
    const int max_new = (int)json_get_number(&root, "max_tokens", 256);

    /* Thinking is off unless asked for, and even then it never reaches the
     * content field: OpenAI clients render "content" verbatim, and a TTS
     * bridge speaks it. */
    json_val think_v;
    mynah_slm_think think = MYNAH_SLM_THINK_OFF;
    if (json_object_get(&root, "think", &think_v) == 0 && think_v.kind == JSON_STRING) {
        char t[16] = "off";
        json_string_copy(&think_v, t, sizeof t);
        if      (!strcmp(t, "on"))  think = MYNAH_SLM_THINK_ON;
        else if (!strcmp(t, "low")) think = MYNAH_SLM_THINK_LOW;
    }

    mynah_slm_sampler_params sp;
    mynah_slm_sampler_defaults(&sp);
    sp.temp  = (float)json_get_number(&root, "temperature", sp.temp);
    sp.top_p = (float)json_get_number(&root, "top_p", sp.top_p);
    sp.top_k = (uint32_t)json_get_number(&root, "top_k", sp.top_k);
    sp.seed  = (uint64_t)json_get_number(&root, "seed", 0);

    const long need = mynah_slm_render_chat(rendered, n_msg, think, NULL, 0);
    char *text = malloc((size_t)need + 1);
    if (!text) { http_error(conn, 500, "out of memory"); goto cleanup_msgs; }
    mynah_slm_render_chat(rendered, n_msg, think, text, (size_t)need + 1);

    const long n_prompt = mynah_slm_tokenize(c->tok, text, 1, NULL, 0);
    uint32_t *ids = malloc((size_t)(n_prompt > 0 ? n_prompt : 1) * sizeof *ids);
    if (n_prompt <= 0 || !ids) {
        http_error(conn, 400, "empty prompt");
        free(text);
        goto cleanup_msgs;
    }
    mynah_slm_tokenize(c->tok, text, 1, ids, (size_t)n_prompt);

    pthread_mutex_lock(&c->stat_mu);
    const unsigned long seq = ++c->next_id;
    pthread_mutex_unlock(&c->stat_mu);

    char req_id[64];
    snprintf(req_id, sizeof req_id, "chatcmpl-%lu", seq);

    char model_json[128];
    json_escape(c->model_name, strlen(c->model_name), model_json, sizeof model_json);

    if (stream) http_begin_sse(conn);

    emit_ctx e = { .conn = conn, .id = req_id, .model_name = model_json, .stream = stream };

    mynah_slm_timing tm;
    mynah_slm_timing_reset(&tm);
    tm.n_threads = mynah_slm_threads_count();

    /* ── the serialization point ── */
    pthread_mutex_lock(&c->infer_mu);

    mynah_slm_timing_start(&tm);
    mynah_slm_state st;
    char err[256];
    const uint32_t want = (uint32_t)n_prompt + (uint32_t)max_new + 8;
    if (mynah_slm_state_init(&st, c->model, want < c->n_ctx ? want : c->n_ctx,
                             err, sizeof err) != 0) {
        pthread_mutex_unlock(&c->infer_mu);
        http_error(conn, 500, err);
        free(ids); free(text);
        goto cleanup_msgs;
    }
    mynah_slm_timing_end_load(&tm);

    mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, mynah_slm_vocab_size(c->model));
    const uint32_t eos[] = { mynah_slm_tokenizer_eos(c->tok), 151643u };

    mynah_slm_gen_params gp = {
        .prompt = ids, .n_prompt = (size_t)n_prompt,
        .max_new = (uint32_t)max_new,
        .eos = eos, .n_eos = 2,
        .cb = answer_cb, .cb_ctx = &e,
        .think_open  = mynah_slm_token_find(c->tok, "<think>"),
        .think_close = mynah_slm_token_find(c->tok, "</think>"),
        .cb_think = NULL,          /* discarded: reasoning is not content */
    };
    mynah_slm_generate(&st, c->tok, sam, &gp, &tm);

    mynah_slm_sampler_free(sam);
    mynah_slm_state_free(&st);
    pthread_mutex_unlock(&c->infer_mu);
    /* ── end serialization ── */

    record_tok_s(c, mynah_slm_decode_tok_s(&tm));

    /* Timings ride in the response so a client never has to time the socket
     * itself. On a stream they arrive in the final event. */
    char usage[512];
    snprintf(usage, sizeof usage,
        "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,"
        "\"total_tokens\":%u,\"load_ms\":%.1f,\"ttft_ms\":%.1f,"
        "\"prefill_tok_s\":%.2f,\"decode_tok_s\":%.2f,\"threads\":%d}",
        tm.n_prompt, tm.n_gen, tm.n_prompt + tm.n_gen,
        tm.load_s * 1000.0, tm.ttft_s * 1000.0,
        mynah_slm_prefill_tok_s(&tm), mynah_slm_decode_tok_s(&tm), tm.n_threads);

    if (stream) {
        char last[1024];
        const int n = snprintf(last, sizeof last,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"model\":%s,\"choices\":[{\"index\":0,\"delta\":{},"
            "\"finish_reason\":\"stop\"}],%s}\n\ndata: [DONE]\n\n",
            req_id, model_json, usage);
        http_write(conn, last, (size_t)n);
    } else {
        char esc_stack[4096];
        char *esc = esc_stack;
        const size_t need_esc = json_escape(e.buf ? e.buf : "", e.used, NULL, 0) + 8;
        if (need_esc > sizeof esc_stack) esc = malloc(need_esc);
        if (esc) {
            json_escape(e.buf ? e.buf : "", e.used, esc, need_esc);
            size_t cap = need_esc + 1024;
            char *out = malloc(cap);
            if (out) {
                const int n = snprintf(out, cap,
                    "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":%s,"
                    "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                    "\"content\":%s},\"finish_reason\":\"stop\"}],%s}\n",
                    req_id, model_json, esc, usage);
                http_respond(conn, 200, "application/json", out, (size_t)n);
                free(out);
            }
            if (esc != esc_stack) free(esc);
        }
    }

    free(e.buf);
    free(ids);
    free(text);

cleanup_msgs:
    for (size_t i = 0; i < n_msg; i++) free(owned[i]);
}

/* ── routing ──────────────────────────────────────────────────────────────── */

static void on_request(void *user, http_conn *conn, const http_request *req) {
    server_ctx *c = user;

    if (!strcmp(req->method, "GET") && !strcmp(req->path, "/health")) {
        handle_health(c, conn);
    } else if (!strcmp(req->method, "GET") && !strcmp(req->path, "/v1/models")) {
        handle_models(c, conn);
    } else if (!strcmp(req->method, "POST") && !strcmp(req->path, "/v1/tokenize")) {
        handle_tokenize(c, conn, req->body, req->body_len);
    } else if (!strcmp(req->method, "POST") &&
               (!strcmp(req->path, "/v1/chat/completions"))) {
        handle_chat(c, conn, req->body, req->body_len);
    } else {
        http_error(conn, 404, "no such endpoint");
    }
}

static void usage_text(FILE *f) {
    fprintf(f,
        "mynah-slm-server %s\n"
        "\n"
        "usage: mynah-slm-server -m <model.gguf> [--port 8080] [--host 127.0.0.1]\n"
        "                       [--ctx N] [-t N]\n"
        "\n"
        "  GET  /health              model, threads, recent decode tok/s\n"
        "  GET  /v1/models\n"
        "  POST /v1/chat/completions  (+ \"stream\": true for SSE)\n"
        "  POST /v1/tokenize\n"
        "\n"
        "Inference is serialized: one request at a time, all threads. Queueing\n"
        "shows up as latency, never as an error.\n",
        mynah_slm_version());
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *host = "127.0.0.1";
    int port = 8080, n_ctx = 0, threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(a, "-m") && v)        { model_path = v; i++; }
        else if (!strcmp(a, "--port") && v)    { port = atoi(v); i++; }
        else if (!strcmp(a, "--host") && v)    { host = v; i++; }
        else if (!strcmp(a, "--ctx") && v)     { n_ctx = atoi(v); i++; }
        else if ((!strcmp(a, "-t") || !strcmp(a, "--threads")) && v) { threads = atoi(v); i++; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage_text(stdout); return 0; }
        else { fprintf(stderr, "mynah-slm-server: unknown option '%s'\n", a); return 2; }
    }
    if (!model_path) { usage_text(stderr); return 2; }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    /* A client that disconnects mid-stream must not take the server with it. */
    signal(SIGPIPE, SIG_IGN);

    char err[256];
    server_ctx ctx;
    memset(&ctx, 0, sizeof ctx);
    pthread_mutex_init(&ctx.infer_mu, NULL);
    pthread_mutex_init(&ctx.stat_mu, NULL);

    ctx.model = mynah_slm_load(model_path, err, sizeof err);
    if (!ctx.model) { fprintf(stderr, "mynah-slm-server: %s\n", err); return 1; }

    ctx.tok = mynah_slm_tokenizer_load(ctx.model->gguf, err, sizeof err);
    if (!ctx.tok) { fprintf(stderr, "mynah-slm-server: %s\n", err); return 1; }

    const char *slash = strrchr(model_path, '/');
    ctx.model_name = slash ? slash + 1 : model_path;
    ctx.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : mynah_slm_n_ctx(ctx.model);

    const int nth = mynah_slm_threads_init(threads);
    fprintf(stderr, "mynah-slm-server %s | %s | ctx %u | %d threads | http://%s:%d\n",
            mynah_slm_version(), ctx.model_name, ctx.n_ctx, nth, host, port);

    const int rc = http_serve(host, port, on_request, &ctx, &g_stop, err, sizeof err);
    if (rc != 0) fprintf(stderr, "mynah-slm-server: %s\n", err);

    mynah_slm_tokenizer_free(ctx.tok);
    mynah_slm_free(ctx.model);
    mynah_slm_threads_shutdown();
    return rc;
}
