/* http.c — see http.h.
 * SPDX-License-Identifier: MIT */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE          /* POLLRDHUP on glibc */
#endif
#include "http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAX_BODY (4u * 1024u * 1024u)

struct http_conn {
    int        fd;
    /* Set by a failed or timed-out send, or by the probe. Atomic because the
     * probe may run on a thread that is not the writer's. */
    atomic_int dead;
    /* The read side reached EOF (set by the probe, any thread). */
    atomic_int eof;
    /* The writer thread's alone: the response has started, and when it
     * last wrote (monotonic seconds). */
    int        head_sent;
    double     last_write;
};

static int g_probe_ms = 250;

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static atomic_int   g_live;
static atomic_ulong g_rejected;

int           http_live_connections(void)     { return atomic_load(&g_live); }
unsigned long http_rejected_connections(void) { return atomic_load(&g_rejected); }

int http_write(http_conn *c, const char *data, size_t len) {
    if (atomic_load(&c->dead)) return -1;
    c->last_write = mono_s();
    while (len > 0) {
        /* MSG_NOSIGNAL where it exists; SIGPIPE is ignored process-wide too. */
#ifdef MSG_NOSIGNAL
        const ssize_t n = send(c->fd, data, len, MSG_NOSIGNAL);
#else
        const ssize_t n = send(c->fd, data, len, 0);
#endif
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            /* Peer gone, or SO_SNDTIMEO expired on a client that stopped
             * reading: either way nobody is receiving. Tell the caller once,
             * stay quiet after. */
            atomic_store(&c->dead, 1);
            return -1;
        }
        data += n;
        len  -= (size_t)n;
    }
    return 0;
}

int http_fd_probe(int fd) {
    if (fd < 0) return 2;
    struct pollfd p;
    p.fd = fd;
    p.events = POLLIN;
#ifdef POLLRDHUP
    p.events |= POLLRDHUP;        /* Linux: the peer's close, without a read */
#endif
    p.revents = 0;
    const int ready = poll(&p, 1, 0);       /* zero timeout: never blocks */
    if (ready < 0) return (errno != EINTR && errno != EAGAIN) ? 2 : 0;
    if (ready == 0) return 0;
    /* A reset (or our own write after the peer closed, answered with one)
     * sets POLLERR and POLLHUP; a FIN alone sets neither. */
    if (p.revents & (POLLHUP | POLLERR | POLLNVAL)) return 2;
#ifdef POLLRDHUP
    if (p.revents & POLLRDHUP) return 1;     /* FIN: half-close or close */
#endif
    if (p.revents & POLLIN) {
        /* Readable is either a pipelined byte or EOF; only a zero-length
         * peek tells them apart, and it cannot block: poll said readable. */
        char b;
        const ssize_t n = recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return 1;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return 2;
    }
    return 0;
}

int http_peer_gone(http_conn *c) {
    if (atomic_load(&c->dead)) return 1;
    const int st = http_fd_probe(c->fd);
    if (st == 1) atomic_store(&c->eof, 1);
    if (st != 2) return 0;
    atomic_store(&c->dead, 1);
    return 1;
}

int http_peer_eof(http_conn *c) { return atomic_load(&c->eof); }

int http_head_sent(const http_conn *c) { return c->head_sent; }

static void begin_body(http_conn *c) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n";
    c->head_sent = 1;
    http_write(c, head, sizeof head - 1);
}

void http_keepalive(http_conn *c, int sse) {
    if (atomic_load(&c->dead)) return;
    if (!atomic_load(&c->eof) && (http_peer_gone(c) || !atomic_load(&c->eof))) return;
    if (!c->head_sent) {
        /* The only way to tell a half-close from a close is to write, and
         * the first thing written must be the response's start. A client
         * that only half-closed gets its answer with a 200 committed now;
         * an error after this point is carried in the body. */
        if (sse) http_begin_sse(c);
        else begin_body(c);
        return;
    }
    if (mono_s() - c->last_write < g_probe_ms / 1000.0) return;
    if (sse) http_write(c, ":\n\n", 3);      /* an SSE comment: ignored */
    else     http_write(c, " ", 1);           /* JSON allows leading whitespace */
}

static const char *status_text(int s) {
    switch (s) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Error";
    }
}

void http_respond(http_conn *c, int status, const char *content_type,
                  const char *body, size_t len) {
    if (c->head_sent) {                       /* committed: the body is all that is left */
        if (len) http_write(c, body, len);
        return;
    }
    c->head_sent = 1;
    char head[512];
    const int n = snprintf(head, sizeof head,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n",
        status, status_text(status), content_type, len);
    http_write(c, head, (size_t)n);
    if (len) http_write(c, body, len);
}

void http_error(http_conn *c, int status, const char *message) {
    /* The message is ours, never echoed user input, so a plain format is safe
     * here — but it still goes through a bounded buffer. */
    char body[512];
    const int n = snprintf(body, sizeof body,
        "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}\n",
        message ? message : "error");
    http_respond(c, status, "application/json", body, (size_t)n);
}

static int busy_body(char *body, size_t cap, const char *message) {
    return snprintf(body, cap,
        "{\"error\":{\"message\":\"%s\",\"type\":\"server_busy\"}}\n",
        message ? message : "busy");
}

static int busy_head(char *head, size_t cap, int body_len, int retry_after_s) {
    return snprintf(head, cap,
        "HTTP/1.1 503 Service Unavailable\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Retry-After: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n",
        body_len, retry_after_s > 0 ? retry_after_s : 1);
}

void http_busy(http_conn *c, const char *message, int retry_after_s) {
    char body[256], head[384];
    const int nb = busy_body(body, sizeof body, message);
    const int nh = busy_head(head, sizeof head, nb, retry_after_s);
    if (!c->head_sent) http_write(c, head, (size_t)nh);
    c->head_sent = 1;
    http_write(c, body, (size_t)nb);
}

void http_begin_sse(http_conn *c) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        /* Without this a reverse proxy buffers the stream and the whole point
         * of streaming is lost between here and the client. */
        "X-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n";
    if (c->head_sent) return;
    c->head_sent = 1;
    http_write(c, head, sizeof head - 1);
}

/* ── request reading ──────────────────────────────────────────────────────── */

typedef struct {
    int          fd;
    http_handler fn;
    void        *user;
    int          send_timeout_ms, recv_timeout_ms;
    int          header_timeout_ms, body_timeout_ms;
} conn_arg;

static void set_timeout(int fd, int opt, int ms) {
    struct timeval tv = { .tv_sec = ms / 1000, .tv_usec = (ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, opt, &tv, sizeof tv);
}

/* Wait until fd is readable or the deadline (monotonic seconds) passes.
 * 1 readable, 0 deadline, -1 error. */
static int wait_readable(int fd, double deadline) {
    for (;;) {
        const double left = deadline - mono_s();
        if (left <= 0.0) return 0;
        struct pollfd p = { .fd = fd, .events = POLLIN, .revents = 0 };
        const int r = poll(&p, 1, (int)(left * 1000.0) + 1);
        if (r > 0) return 1;
        if (r == 0) continue;               /* re-check the clock */
        if (errno != EINTR) return -1;
    }
}

/* Read until the header terminator, then exactly Content-Length more. A
 * fixed-size read would truncate a long prompt; a read-until-EOF would hang on
 * a keep-alive client. Every recv waits at most until the request's absolute
 * deadline (headers, then body): *timed_out says that is why it stopped. */
static char *read_request(int fd, int header_ms, int body_ms, size_t *out_len,
                          size_t *out_head, int *timed_out) {
    size_t cap = 8192, used = 0;
    *timed_out = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    const double t0 = mono_s();
    double deadline = t0 + header_ms / 1000.0;
    size_t head_end = 0;
    for (;;) {
        if (used + 1 >= cap) {
            cap *= 2;
            if (cap > MAX_BODY + 65536) { free(buf); return NULL; }
            char *g = realloc(buf, cap);
            if (!g) { free(buf); return NULL; }
            buf = g;
        }
        const int w = wait_readable(fd, deadline);
        if (w == 0) { *timed_out = 1; free(buf); return NULL; }
        if (w < 0) break;
        const ssize_t n = recv(fd, buf + used, cap - used - 1, 0);
        if (n <= 0) break;
        used += (size_t)n;
        buf[used] = '\0';

        if (!head_end) {
            char *p = strstr(buf, "\r\n\r\n");
            if (p) {
                head_end = (size_t)(p - buf) + 4;
                deadline = mono_s() + body_ms / 1000.0;     /* the body's own budget */
            }
        }
        if (head_end) {
            size_t want = 0;
            const char *cl = strcasestr(buf, "content-length:");
            if (cl && cl < buf + head_end) want = strtoul(cl + 15, NULL, 10);
            if (want > MAX_BODY) { free(buf); return NULL; }
            if (used >= head_end + want) break;
        }
    }
    if (!head_end) { free(buf); return NULL; }
    *out_len  = used;
    *out_head = head_end;
    return buf;
}

static void *serve_conn(void *arg) {
    conn_arg ca = *(conn_arg *)arg;
    free(arg);

    /* Nagle would sit on a 40-byte SSE frame waiting for company, which is
     * exactly the delay streaming exists to avoid. */
    const int one = 1;
    setsockopt(ca.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    set_timeout(ca.fd, SO_SNDTIMEO, ca.send_timeout_ms);
    set_timeout(ca.fd, SO_RCVTIMEO, ca.recv_timeout_ms);

    size_t len = 0, head_end = 0;
    int timed_out = 0;
    char *raw = read_request(ca.fd, ca.header_timeout_ms, ca.body_timeout_ms, &len, &head_end,
                             &timed_out);

    http_conn conn;
    memset(&conn, 0, sizeof conn);
    conn.fd = ca.fd;
    atomic_init(&conn.dead, 0);
    atomic_init(&conn.eof, 0);
    if (!raw) {
        if (timed_out) http_error(&conn, 408, "the request was not received in time");
        else           http_error(&conn, 400, "malformed request");
        close(ca.fd);
        atomic_fetch_sub(&g_live, 1);
        return NULL;
    }

    http_request req;
    memset(&req, 0, sizeof req);

    /* "METHOD path HTTP/1.1" */
    const char *p = raw;
    size_t i = 0;
    while (*p && *p != ' ' && i + 1 < sizeof req.method) req.method[i++] = *p++;
    req.method[i] = '\0';
    while (*p == ' ') p++;
    i = 0;
    while (*p && *p != ' ' && *p != '?' && i + 1 < sizeof req.path) req.path[i++] = *p++;
    req.path[i] = '\0';

    req.body     = raw + head_end;
    req.body_len = len > head_end ? len - head_end : 0;

    ca.fn(ca.user, &conn, &req);

    free(raw);
    close(ca.fd);
    atomic_fetch_sub(&g_live, 1);
    return NULL;
}

/* Over the cap: answer 503 + Retry-After at once and close. Whatever request
 * bytes have already arrived are drained without waiting first, so the close
 * is a FIN and the client reads the 503 instead of a reset (the hole mynah-tts
 * names in its own fail-fast path). Never blocks the accept loop for long:
 * a non-blocking drain and a short send timeout. */
static void reject_busy(int fd) {
    set_timeout(fd, SO_SNDTIMEO, 200);
    char body[256], head[384];
    const int nb = busy_body(body, sizeof body, "too many connections, retry shortly");
    const int nh = busy_head(head, sizeof head, nb, 1);
    char sink[4096];
    while (recv(fd, sink, sizeof sink, MSG_DONTWAIT) > 0) {}
#ifdef MSG_NOSIGNAL
    send(fd, head, (size_t)nh, MSG_NOSIGNAL);
    send(fd, body, (size_t)nb, MSG_NOSIGNAL);
#else
    send(fd, head, (size_t)nh, 0);
    send(fd, body, (size_t)nb, 0);
#endif
    shutdown(fd, SHUT_WR);
    while (recv(fd, sink, sizeof sink, MSG_DONTWAIT) > 0) {}
    close(fd);
    atomic_fetch_add(&g_rejected, 1);
}

int http_serve(const char *host, int port, http_handler fn, void *user,
               const http_limits *limits, volatile sig_atomic_t *stop,
               char *err, size_t errsz) {
    const int max_conns = (limits && limits->max_conns > 0) ? limits->max_conns : 64;
    const int send_ms = (limits && limits->send_timeout_ms > 0) ? limits->send_timeout_ms : 5000;
    const int recv_ms = (limits && limits->recv_timeout_ms > 0) ? limits->recv_timeout_ms : 30000;
    if (limits && limits->probe_interval_ms > 0) g_probe_ms = limits->probe_interval_ms;
    const int head_ms = (limits && limits->header_timeout_ms > 0) ? limits->header_timeout_ms : 10000;
    const int body_ms = (limits && limits->body_timeout_ms > 0) ? limits->body_timeout_ms : 30000;

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { snprintf(err, errsz, "socket: %s", strerror(errno)); return 1; }

    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        snprintf(err, errsz, "bad host address: %s", host);
        close(fd);
        return 1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        snprintf(err, errsz, "bind %s:%d: %s", host, port, strerror(errno));
        close(fd);
        return 1;
    }
    if (listen(fd, 64) != 0) {
        snprintf(err, errsz, "listen: %s", strerror(errno));
        close(fd);
        return 1;
    }

    /* A 1s accept timeout so the loop notices *stop without needing the signal
     * handler to touch the socket. */
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    while (!*stop) {
        const int cfd = accept(fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            break;
        }
        /* The cap is checked at accept and the listener keeps being
         * polled, so a full server refuses visibly instead of leaving
         * clients in the kernel backlog where no metric sees the wait. */
        if (atomic_load(&g_live) >= max_conns) { reject_busy(cfd); continue; }

        conn_arg *ca = malloc(sizeof *ca);
        if (!ca) { close(cfd); continue; }
        *ca = (conn_arg){ .fd = cfd, .fn = fn, .user = user,
                          .send_timeout_ms = send_ms, .recv_timeout_ms = recv_ms,
                          .header_timeout_ms = head_ms, .body_timeout_ms = body_ms };

        atomic_fetch_add(&g_live, 1);
        pthread_t th;
        if (pthread_create(&th, NULL, serve_conn, ca) != 0) {
            atomic_fetch_sub(&g_live, 1);
            close(cfd);
            free(ca);
            continue;
        }
        pthread_detach(th);
    }
    close(fd);
    return 0;
}
