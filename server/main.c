/* mynah-slm-server — OpenAI-compatible HTTP, no framework.
 *
 * Two serving modes, chosen at start:
 *
 *   --slots 1 (default)  the serialized path described below, unchanged.
 *   --slots N > 1        continuous batching (server/slots.h): one scheduler
 *                        thread owns the model and advances up to N requests
 *                        per decode step; prompts are prefilled in slices
 *                        between steps; a full pending queue is a 503.
 *                        Opt-in until a real checkpoint has measured it
 *                        (.work/serving-continuous-batching.md).
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
 * the model, up to --max-conns connections; one more is a 503 with
 * Retry-After at accept.
 *
 * NO ZOMBIE WORK. A client that leaves stops costing CPU at the next step
 * boundary: the peer-gone probe (http_peer_gone) is asked while a request
 * waits for the model, again once it has it, before every prefill batch and
 * before every decode step (generate.h's cancel hook) — not only when a token
 * happens to be written, which never happens during a prefill, on the
 * thinking channel, while a tool call accumulates, or in a non-streaming
 * request. Every socket has a send timeout, so a client that stops reading
 * cannot pin the model inside send() either.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "generate.h"
#include "http.h"
#include "json.h"
#include "model.h"
#include "mynah_slm.h"
#include "sampler.h"
#include "slots.h"
#include "template.h"
#include "threads.h"
#include "kvcache.h"
#include "timing.h"
#include "tokenizer.h"
#include "tools.h"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    mynah_slm_model_t   *model;
    mynah_slm_tokenizer *tok;
    const char          *model_name;
    uint32_t             n_ctx;
    uint32_t             slots;      /* > 1: continuous batching (slots.h) */
    unsigned long        rejected_queue;   /* 503: every slot busy, queue full */

    /* The serialization point: one request inside the model at a time.
     * A flag under a mutex rather than the mutex itself, so a waiter can
     * wake up every few ms, see that its client left, and leave the queue
     * without ever running. */
    pthread_mutex_t gate_mu;
    pthread_cond_t  gate_cv;
    int             gate_busy;
    unsigned        gate_waiting;

    /* Requests whose client left: before the model (queued) and inside it. */
    unsigned long   cancelled_queued, cancelled_running;

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

/* ── the serialization gate ──────────────────────────────────────────────── */

/* Enter the model, or give up because the client is gone. Polls the client
 * every 20 ms while queued: a disconnect while waiting costs no inference. */
static int gate_enter(server_ctx *c, http_conn *conn) {
    pthread_mutex_lock(&c->gate_mu);
    c->gate_waiting++;
    while (c->gate_busy) {
        if (http_peer_gone(conn)) {
            c->gate_waiting--;
            c->cancelled_queued++;
            pthread_mutex_unlock(&c->gate_mu);
            return -1;
        }
        struct timespec dl;
        clock_gettime(CLOCK_REALTIME, &dl);
        dl.tv_nsec += 20 * 1000000L;
        if (dl.tv_nsec >= 1000000000L) { dl.tv_nsec -= 1000000000L; dl.tv_sec++; }
        pthread_cond_timedwait(&c->gate_cv, &c->gate_mu, &dl);
    }
    c->gate_busy = 1;
    c->gate_waiting--;
    pthread_mutex_unlock(&c->gate_mu);
    return 0;
}

static void gate_leave(server_ctx *c) {
    pthread_mutex_lock(&c->gate_mu);
    c->gate_busy = 0;
    pthread_cond_broadcast(&c->gate_cv);
    pthread_mutex_unlock(&c->gate_mu);
}

static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

/* Set by main once the accept loop has returned (g_stop is the signal
 * handler's flag; this is the one other threads read). From then on new
 * work and /health answer 503. */
static atomic_int g_shutting_down;

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

/* The generation's cancel hook (generate.h): asked before every prefill batch
 * and every decode step. A failed write counts as gone too. */
typedef struct {
    http_conn *conn;
    emit_ctx  *answer;
} cancel_ctx;

static int chat_cancel(void *ctx) {
    cancel_ctx *cc = ctx;
    return cc->answer->failed || http_peer_gone(cc->conn);
}

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

/* The tool channel is accumulated, never streamed frame by frame: a function
 * call is only usable once the JSON is complete, and half of an object is not
 * a partial answer, it is unparseable. */
static int tool_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    emit_ctx *e = ctx;
    return len ? emit_append(e, text, len) : 0;
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

/* The end of a chat completion, shared by both serving modes: the tool call
 * parsed out of its channel, the finish reason, and the final SSE frames or the
 * whole JSON response. `buf` is the answer text (non-stream; may be NULL) and
 * is trimmed in place; `usage` is the rendered "usage":{...} member. */
static void send_completion(http_conn *conn, int stream, const char *req_id,
                            const char *model_json, char *buf, size_t used,
                            const char *tool, size_t tool_used, const char *usage) {
    /* A function call is data, not prose: it leaves the answer channel empty
     * and changes finish_reason. A client that ignores tool_calls must not be
     * handed the raw JSON in `content` — that is what the channel split in
     * generate.c is for. */
    mynah_slm_tool_call parsed[16];
    size_t n_parsed = 0;
    char  *calls_json = NULL;
    if (tool_used) {
        size_t blocks = 0;
        const long got = mynah_slm_tool_calls_parse(tool, tool_used, parsed,
                                                    sizeof parsed / sizeof *parsed, &blocks);
        if (got > 0) {
            n_parsed = (size_t)(got < 16 ? got : 16);
            const size_t nj = mynah_slm_tool_calls_to_json(parsed, n_parsed, NULL, 0) + 1;
            calls_json = malloc(nj);
            if (calls_json) mynah_slm_tool_calls_to_json(parsed, n_parsed, calls_json, nj);
        }
    }
    const char *finish = calls_json ? "tool_calls" : "stop";

    /* The newline the template puts between an answer and a call is glue, not
     * content. Only trimmed when a call actually followed. */
    if (calls_json && buf)
        while (used && (buf[used - 1] == '\n' || buf[used - 1] == ' ' ||
                          buf[used - 1] == '\t' || buf[used - 1] == '\r'))
            buf[--used] = '\0';

    if (stream) {
        /* One frame carrying the whole call. Splitting a JSON object across
         * deltas would let a client act on half an argument list. */
        if (calls_json) {
            const size_t cap = strlen(calls_json) + 512;
            char *frame = malloc(cap);
            if (frame) {
                const int n = snprintf(frame, cap,
                    "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                    "\"model\":%s,\"choices\":[{\"index\":0,"
                    "\"delta\":{\"tool_calls\":%s},\"finish_reason\":null}]}\n\n",
                    req_id, model_json, calls_json);
                http_write(conn, frame, (size_t)n);
                free(frame);
            }
        }
        char last[1024];
        const int n = snprintf(last, sizeof last,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"model\":%s,\"choices\":[{\"index\":0,\"delta\":{},"
            "\"finish_reason\":\"%s\"}],%s}\n\ndata: [DONE]\n\n",
            req_id, model_json, finish, usage);
        http_write(conn, last, (size_t)n);
    } else {
        char esc_stack[4096];
        char *esc = esc_stack;
        const size_t need_esc = json_escape(buf ? buf : "", used, NULL, 0) + 8;
        if (need_esc > sizeof esc_stack) esc = malloc(need_esc);
        if (esc) {
            json_escape(buf ? buf : "", used, esc, need_esc);
            size_t cap = need_esc + 1024 + (calls_json ? strlen(calls_json) : 0);
            char *out = malloc(cap);
            if (out) {
                /* OpenAI sends content: null when the turn was only a call. */
                const int n = snprintf(out, cap,
                    "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":%s,"
                    "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                    "\"content\":%s%s%s},\"finish_reason\":\"%s\"}],%s}\n",
                    req_id, model_json,
                    (calls_json && used == 0) ? "null" : esc,
                    calls_json ? ",\"tool_calls\":" : "",
                    calls_json ? calls_json : "",
                    finish, usage);
                http_respond(conn, 200, "application/json", out, (size_t)n);
                free(out);
            }
            if (esc != esc_stack) free(esc);
        }
    }

    mynah_slm_tool_calls_free(parsed, n_parsed);
    free(calls_json);
}

/* ── handlers ─────────────────────────────────────────────────────────────── */

static void handle_health(server_ctx *c, http_conn *conn) {
    if (atomic_load(&g_shutting_down)) {
        http_busy(conn, "the server is shutting down", 1);
        return;
    }
    double avg = 0.0;
    pthread_mutex_lock(&c->stat_mu);
    for (unsigned i = 0; i < c->n_stat; i++) avg += c->recent_tok_s[i];
    if (c->n_stat) avg /= c->n_stat;
    const unsigned n = c->n_stat;
    pthread_mutex_unlock(&c->stat_mu);

    pthread_mutex_lock(&c->gate_mu);
    const unsigned waiting = c->gate_waiting;
    const int busy = c->gate_busy;
    const unsigned long cq = c->cancelled_queued, cr = c->cancelled_running;
    pthread_mutex_unlock(&c->gate_mu);

    char body[2048];
    int len = snprintf(body, sizeof body,
        "{\"status\":\"ok\",\"model\":\"%s\",\"n_ctx\":%u,\"threads\":%d,"
        "\"recent_requests\":%u,\"recent_decode_tok_s\":%.2f,"
        "\"running\":%d,\"waiting\":%u,\"connections\":%d,\"rejected\":%lu,"
        "\"cancelled\":%lu,\"cancelled_queued\":%lu,\"cancelled_running\":%lu}\n",
        c->model_name, c->n_ctx, mynah_slm_threads_count(), n, avg,
        busy, waiting, http_live_connections(), http_rejected_connections(),
        cq + cr, cq, cr);
    if (c->slots > 1 && len > 2 && (size_t)len < sizeof body) {
        /* Replace the closing "}\n" with the scheduler's members. */
        len -= 2;
        pthread_mutex_lock(&c->stat_mu);
        const unsigned long rq = c->rejected_queue;
        pthread_mutex_unlock(&c->stat_mu);
        len += snprintf(body + len, sizeof body - (size_t)len, ",\"rejected_queue\":%lu,", rq);
        if ((size_t)len < sizeof body) {
            const int k = slots_health(body + len, sizeof body - (size_t)len);
            if (k < 0) {                 /* stopping: the scheduler is going away */
                http_busy(conn, "the server is shutting down", 1);
                return;
            }
            len += k;
        }
        if ((size_t)len + 3 < sizeof body) len += snprintf(body + len, sizeof body - (size_t)len, "}\n");
    }
    if ((size_t)len >= sizeof body) len = (int)sizeof body - 1;
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

/* ── the continuous-batching path (--slots N > 1) ─────────────────────────── */

static void chat_slots(server_ctx *c, http_conn *conn, const uint32_t *ids, size_t n_prompt,
                       int max_new, const mynah_slm_sampler_params *sp, int stream,
                       const char *req_id, const char *model_json,
                       long tool_open, long tool_close) {
    const uint32_t eos[] = { mynah_slm_tokenizer_eos(c->tok), 151643u };
    slots_params p;
    memset(&p, 0, sizeof p);
    p.ids = ids;
    p.n_prompt = n_prompt;
    p.max_new = max_new > 0 ? (uint32_t)max_new : 0;
    p.sp = *sp;
    p.eos = eos;
    p.n_eos = 2;
    p.think_open  = mynah_slm_token_find(c->tok, "<think>");
    p.think_close = mynah_slm_token_find(c->tok, "</think>");
    p.tool_open = tool_open;
    p.tool_close = tool_close;
    p.stream = stream;
    p.req_id = req_id;
    p.model_json = model_json;

    if (http_peer_gone(conn)) {                       /* left while we parsed */
        pthread_mutex_lock(&c->gate_mu);
        c->cancelled_queued++;
        pthread_mutex_unlock(&c->gate_mu);
        return;
    }
    slots_result r;
    const int run = slots_run(conn, &p, &r);
    if (run == SLOTS_STOPPING) {
        http_busy(conn, "the server is shutting down", 1);
        return;
    }
    if (run != 0) {
        /* Every slot busy and the queue full: refuse NOW, visibly, rather
         * than park the client where no metric sees the wait. */
        pthread_mutex_lock(&c->stat_mu);
        c->rejected_queue++;
        pthread_mutex_unlock(&c->stat_mu);
        http_busy(conn, "every slot is busy and the queue is full, retry shortly", 1);
        return;
    }

    if (r.outcome == MYNAH_SLM_JOB_CANCELLED || r.client_gone) {
        pthread_mutex_lock(&c->gate_mu);
        if (r.tm.t0_ == 0.0) c->cancelled_queued++; else c->cancelled_running++;
        pthread_mutex_unlock(&c->gate_mu);
        fprintf(stderr, "[%s cancelled: client gone during %s, %u tokens generated of "
                        "max_tokens %d, prompt %zu, queued %.1f ms]\n",
                req_id, r.tm.t0_ == 0.0 ? "the queue" : r.tm.n_prompt ? "decode" : "prefill",
                r.tm.n_gen, max_new, n_prompt, r.queue_ms);
    } else if (r.outcome != MYNAH_SLM_JOB_DONE) {
        const int status = r.outcome == MYNAH_SLM_JOB_REFUSED ? 400 : 500;
        const char *msg = r.error[0] ? r.error : "generation failed";
        if (r.header_sent) {
            char esc[256], frame[512];
            json_escape(msg, strlen(msg), esc, sizeof esc);
            const int n = snprintf(frame, sizeof frame,
                                   "data: {\"error\":{\"message\":%s}}\n\ndata: [DONE]\n\n", esc);
            http_write(conn, frame, (size_t)n);
        } else {
            http_error(conn, status, msg);
        }
    } else {
        record_tok_s(c, mynah_slm_decode_tok_s(&r.tm));
        /* The same members as the serialized path, plus where the time went
         * before admission: TTFT is measured from ADMISSION (as it is from
         * the lock there), so the queue is reported separately, never
         * hidden inside it or left out. */
        char usage[640];
        snprintf(usage, sizeof usage,
            "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,"
            "\"total_tokens\":%u,\"load_ms\":%.1f,\"ttft_ms\":%.1f,"
            "\"prefill_tok_s\":%.2f,\"decode_tok_s\":%.2f,\"threads\":%d,"
            "\"queue_ms\":%.1f,\"slots\":%u}",
            r.tm.n_prompt, r.tm.n_gen, r.tm.n_prompt + r.tm.n_gen,
            r.tm.load_s * 1000.0, r.tm.ttft_s * 1000.0,
            mynah_slm_prefill_tok_s(&r.tm), mynah_slm_decode_tok_s(&r.tm), r.tm.n_threads,
            r.queue_ms, c->slots);
        send_completion(conn, stream, req_id, model_json, r.content, r.content_used,
                        r.tool, r.tool_used, usage);
    }
    free(r.content);
    free(r.tool);
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
    mynah_slm_tool_call *replay[32];
    size_t n_replay[32];
    size_t n_msg = 0;
    mynah_slm_tool_set *tools = NULL;

    memset(rendered, 0, sizeof rendered);
    memset(owned, 0, sizeof owned);
    memset(replay, 0, sizeof replay);
    memset(n_replay, 0, sizeof n_replay);

    for (size_t i = 0; i < 32; i++) {
        json_val m, role, content, tc;
        if (json_array_at(&msgs, i, &m) != 0) break;
        if (m.kind != JSON_OBJECT) continue;

        const int has_content =
            (json_object_get(&m, "content", &content) == 0 && content.kind == JSON_STRING);

        /* An assistant turn that only called a function carries no content.
         * Dropping it would leave the tool result answering nothing, and the
         * model would see a response to a question it never asked. */
        if (json_object_get(&m, "tool_calls", &tc) == 0 && tc.kind == JSON_ARRAY) {
            const size_t span = (size_t)(tc.end - tc.start);
            const long k = mynah_slm_tool_calls_from_json(tc.start, span, NULL, 0);
            if (k > 0) {
                replay[n_msg] = calloc((size_t)k, sizeof **replay);
                if (replay[n_msg]) {
                    mynah_slm_tool_calls_from_json(tc.start, span, replay[n_msg], (size_t)k);
                    n_replay[n_msg] = (size_t)k;
                }
            }
        }
        if (!has_content && !n_replay[n_msg]) continue;

        char r[32] = "user";
        if (json_object_get(&m, "role", &role) == 0 && role.kind == JSON_STRING)
            json_string_copy(&role, r, sizeof r);

        owned[n_msg] = has_content ? json_string_dup(&content) : NULL;
        if (has_content && !owned[n_msg]) break;
        rendered[n_msg].content      = owned[n_msg];
        rendered[n_msg].tool_calls   = replay[n_msg];
        rendered[n_msg].n_tool_calls = n_replay[n_msg];
        rendered[n_msg].role =
            !strcmp(r, "system")    ? MYNAH_SLM_ROLE_SYSTEM :
            !strcmp(r, "assistant") ? MYNAH_SLM_ROLE_ASSISTANT :
            !strcmp(r, "tool")      ? MYNAH_SLM_ROLE_TOOL : MYNAH_SLM_ROLE_USER;
        n_msg++;
    }
    if (n_msg == 0) { http_error(conn, 400, "no usable messages"); goto cleanup_msgs; }

    /* Tools go in the system turn, so they are part of the prompt and not a
     * mode: `tool_choice: "none"` is simply not sending them. */
    json_val choice;
    int offer_tools = 1;
    if (json_object_get(&root, "tool_choice", &choice) == 0 && choice.kind == JSON_STRING) {
        char tc_s[32] = "auto";
        json_string_copy(&choice, tc_s, sizeof tc_s);
        if (!strcmp(tc_s, "none")) offer_tools = 0;
        else if (strcmp(tc_s, "auto") != 0 && strcmp(tc_s, "required") != 0) {
            http_error(conn, 400, "tool_choice: only \"auto\" and \"none\" are supported");
            goto cleanup_msgs;
        }
        /* "required" is accepted but not enforced: forcing a call needs
         * constrained decoding, and pretending otherwise would be a lie the
         * client only discovers in production. */
    }

    json_val tools_v;
    if (offer_tools &&
        json_object_get(&root, "tools", &tools_v) == 0 && tools_v.kind == JSON_ARRAY) {
        char terr[160];
        tools = mynah_slm_tools_parse(tools_v.start, (size_t)(tools_v.end - tools_v.start),
                                      terr, sizeof terr);
        if (!tools) { http_error(conn, 400, terr); goto cleanup_msgs; }
    }

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

    const mynah_slm_tool *tool_items = mynah_slm_tools_items(tools);
    const size_t n_tools = mynah_slm_tools_count(tools);

    const mynah_slm_chat_family *fam = mynah_slm_chat_family_for(mynah_slm_arch(c->model));
    const long need = mynah_slm_render_chat_tools(fam, rendered, n_msg, tool_items,
                                                  n_tools, think, NULL, 0);
    if (need < 0) { http_error(conn, 400, "cannot render these messages"); goto cleanup_msgs; }
    char *text = malloc((size_t)need + 1);
    if (!text) { http_error(conn, 500, "out of memory"); goto cleanup_msgs; }
    mynah_slm_render_chat_tools(fam, rendered, n_msg, tool_items, n_tools, think,
                                text, (size_t)need + 1);

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

    if (c->slots > 1) {
        const mynah_slm_chat_family *cf = mynah_slm_chat_family_for(mynah_slm_arch(c->model));
        chat_slots(c, conn, ids, (size_t)n_prompt, max_new, &sp, stream, req_id, model_json,
                   tools ? mynah_slm_token_find(c->tok, cf->call_open) : -1,
                   tools ? mynah_slm_token_find(c->tok, cf->call_close) : -1);
        free(ids);
        free(text);
        goto cleanup_msgs;
    }

    if (stream) http_begin_sse(conn);

    emit_ctx e = { .conn = conn, .id = req_id, .model_name = model_json, .stream = stream };
    /* Never streamed: a tool call is only usable whole. */
    emit_ctx e_tool = { .conn = conn, .id = req_id, .model_name = model_json, .stream = 0 };

    mynah_slm_timing tm;
    mynah_slm_timing_reset(&tm);
    tm.n_threads = mynah_slm_threads_count();

    /* ── the serialization point ──
     * Asked before queueing and again once inside: a client that left while
     * the request was parsed or queued costs no inference at all. */
    const int entered = gate_enter(c, conn) == 0;
    if (!entered || http_peer_gone(conn)) {
        if (entered) {
            pthread_mutex_lock(&c->gate_mu);
            c->cancelled_queued++;
            pthread_mutex_unlock(&c->gate_mu);
            gate_leave(c);
        }
        fprintf(stderr, "[%s cancelled while queued: client gone, 0 tokens computed]\n",
                req_id);
        free(ids); free(text);
        goto cleanup_msgs;
    }

    mynah_slm_timing_start(&tm);
    mynah_slm_state st;
    char err[256];
    const uint32_t want = (uint32_t)n_prompt + (uint32_t)max_new + 8;
    /* bf16 KV: half the memory of f32 and faster with it, at a perplexity
     * difference of nothing (docs/perf.md). A server holds a cache per request
     * and is where the footprint matters most. */
    if (mynah_slm_state_init_kv(&st, c->model, want < c->n_ctx ? want : c->n_ctx,
                                MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16,
                                err, sizeof err) != 0) {
        gate_leave(c);
        http_error(conn, 500, err);
        free(ids); free(text);
        goto cleanup_msgs;
    }
    mynah_slm_timing_end_load(&tm);

    mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, mynah_slm_vocab_size(c->model));
    const uint32_t eos[] = { mynah_slm_tokenizer_eos(c->tok), 151643u };

    mynah_slm_gen_params gp;
    mynah_slm_gen_params_init(&gp);
    gp.prompt = ids; gp.n_prompt = (size_t)n_prompt;
    gp.max_new = (uint32_t)max_new;
    gp.eos = eos; gp.n_eos = 2;
    gp.cb = answer_cb; gp.cb_ctx = &e;
    gp.think_open  = mynah_slm_token_find(c->tok, "<think>");
    gp.think_close = mynah_slm_token_find(c->tok, "</think>");
    gp.cb_think = NULL;            /* discarded: reasoning is not content */
    if (tools) {
        /* By family: LFM2's delimiters are not Qwen3's (src/template.h). */
        const mynah_slm_chat_family *cf =
            mynah_slm_chat_family_for(mynah_slm_arch(c->model));
        gp.tool_open  = mynah_slm_token_find(c->tok, cf->call_open);
        gp.tool_close = mynah_slm_token_find(c->tok, cf->call_close);
        gp.cb_tool = tool_cb; gp.cb_tool_ctx = &e_tool;
    }
    cancel_ctx cc = { .conn = conn, .answer = &e };
    gp.cancel = chat_cancel; gp.cancel_ctx = &cc;
    mynah_slm_generate(&st, c->tok, sam, &gp, &tm);

    mynah_slm_sampler_free(sam);
    mynah_slm_state_free(&st);
    gate_leave(c);
    /* ── end serialization ── */

    /* Gone mid-request: the model was released at the step boundary above.
     * Nobody is left to answer; say so in the log, with how far it got. */
    if (tm.cancelled || e.failed) {
        pthread_mutex_lock(&c->gate_mu);
        c->cancelled_running++;
        pthread_mutex_unlock(&c->gate_mu);
        fprintf(stderr, "[%s cancelled: client gone during %s, %u tokens generated "
                        "of max_tokens %d, prompt %ld]\n",
                req_id, tm.n_prompt ? "decode" : "prefill", tm.n_gen, max_new, n_prompt);
        free(e_tool.buf);
        free(e.buf);
        free(ids);
        free(text);
        goto cleanup_msgs;
    }

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

    send_completion(conn, stream, req_id, model_json, e.buf, e.used,
                    e_tool.buf, e_tool.used, usage);
    free(e_tool.buf);
    free(e.buf);
    free(ids);
    free(text);

cleanup_msgs:
    mynah_slm_tools_free(tools);
    for (size_t i = 0; i < 32; i++) {
        free(owned[i]);
        mynah_slm_tool_calls_free(replay[i], n_replay[i]);
        free(replay[i]);
    }
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
        "  --max-conns N        connections at once; one more gets 503 (default 64)\n"
        "  --slots N            continuous batching: N requests per decode step,\n"
        "                       one scheduler thread (default 1 = serialized)\n"
        "  --queue N            requests waiting for a slot before a 503 (default 2N)\n"
        "  --send-timeout-ms N  a client that stops reading is dropped (default 5000)\n"
        "\n"
        "By default inference is serialized: one request at a time, all threads.\n"
        "With --slots N one scheduler thread steps up to N requests together.\n"
        "Either way a client that disconnects stops costing CPU at the next step.\n",
        mynah_slm_version());
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *host = "127.0.0.1";
    int port = 8080, n_ctx = 0, threads = 0;
    http_limits limits = { 0, 0, 0 };
    int slots = 1, queue = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(a, "-m") && v)        { model_path = v; i++; }
        else if (!strcmp(a, "--port") && v)    { port = atoi(v); i++; }
        else if (!strcmp(a, "--host") && v)    { host = v; i++; }
        else if (!strcmp(a, "--ctx") && v)     { n_ctx = atoi(v); i++; }
        else if ((!strcmp(a, "-t") || !strcmp(a, "--threads")) && v) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--max-conns") && v)       { limits.max_conns = atoi(v); i++; }
        else if (!strcmp(a, "--slots") && v)           { slots = atoi(v); i++; }
        else if (!strcmp(a, "--queue") && v)           { queue = atoi(v); i++; }
        else if (!strcmp(a, "--send-timeout-ms") && v) { limits.send_timeout_ms = atoi(v); i++; }
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
    pthread_mutex_init(&ctx.gate_mu, NULL);
    pthread_cond_init(&ctx.gate_cv, NULL);
    pthread_mutex_init(&ctx.stat_mu, NULL);

    ctx.model = mynah_slm_load(model_path, err, sizeof err);
    if (!ctx.model) { fprintf(stderr, "mynah-slm-server: %s\n", err); return 1; }

    ctx.tok = mynah_slm_tokenizer_load(ctx.model->gguf, err, sizeof err);
    if (!ctx.tok) { fprintf(stderr, "mynah-slm-server: %s\n", err); return 1; }

    const char *slash = strrchr(model_path, '/');
    ctx.model_name = slash ? slash + 1 : model_path;
    ctx.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : mynah_slm_n_ctx(ctx.model);

    const int nth = mynah_slm_threads_init(threads);
    ctx.slots = slots > 1 ? (uint32_t)slots : 1;
    if (ctx.slots > 1 &&
        slots_start(ctx.model, ctx.tok, ctx.slots, ctx.n_ctx,
                    queue > 0 ? (uint32_t)queue : 0, err, sizeof err) != 0) {
        fprintf(stderr, "mynah-slm-server: --slots %u: %s\n", ctx.slots, err);
        return 1;
    }
    fprintf(stderr, "mynah-slm-server %s | %s | ctx %u | %d threads | http://%s:%d\n",
            mynah_slm_version(), ctx.model_name, ctx.n_ctx, nth, host, port);

    const int rc = http_serve(host, port, on_request, &ctx, &limits, &g_stop, err, sizeof err);
    if (rc != 0) fprintf(stderr, "mynah-slm-server: %s\n", err);
    /* The listener is closed: no new connection. Connection threads already
     * accepted are detached and may still be reading a request, so the
     * engine is first marked stopping (later requests and /health answer
     * 503 without touching it), then its scheduler is joined, and only once
     * every connection has left is anything freed. */
    atomic_store(&g_shutting_down, 1);
    if (ctx.slots > 1) slots_shutdown();

    /* Connection threads are detached and use the model and the tokenizer:
     * freeing those under a live one is a use-after-free, and even after the
     * last one returned nothing orders its accesses before our free (TSan
     * flagged exactly that). The live counter is decremented as each thread's
     * last act, so waiting for zero is both the drain and the ordering. If
     * they do not finish in time, exit without freeing: the OS reclaims it,
     * which is safe; a free under a running thread is not. */
    for (int i = 0; i < 300 && http_live_connections() > 0; i++) {
        struct timespec ts = { 0, 100 * 1000000L };
        nanosleep(&ts, NULL);
    }
    if (http_live_connections() > 0 || (ctx.slots > 1 && slots_free(1000) != 0)) {
        fprintf(stderr, "mynah-slm-server: %d connection(s) still running at exit; "
                        "not freeing the model under them\n", http_live_connections());
        return rc;
    }

    mynah_slm_tokenizer_free(ctx.tok);
    mynah_slm_free(ctx.model);
    mynah_slm_threads_shutdown();
    return rc;
}
