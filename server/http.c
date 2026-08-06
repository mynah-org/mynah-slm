/* http.c — see http.h.
 * SPDX-License-Identifier: MIT */
#include "http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_BODY (4u * 1024u * 1024u)

struct http_conn {
    int  fd;
    int  dead;
};

int http_write(http_conn *c, const char *data, size_t len) {
    if (c->dead) return -1;
    while (len > 0) {
        const ssize_t n = send(c->fd, data, len, 0);
        if (n <= 0) {
            if (errno == EINTR) continue;
            c->dead = 1;          /* peer gone: tell the caller once, stay quiet after */
            return -1;
        }
        data += n;
        len  -= (size_t)n;
    }
    return 0;
}

static const char *status_text(int s) {
    switch (s) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        default:  return "Error";
    }
}

void http_respond(http_conn *c, int status, const char *content_type,
                  const char *body, size_t len) {
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
    http_write(c, head, sizeof head - 1);
}

/* ── request reading ──────────────────────────────────────────────────────── */

typedef struct {
    int          fd;
    http_handler fn;
    void        *user;
} conn_arg;

/* Read until the header terminator, then exactly Content-Length more. A
 * fixed-size read would truncate a long prompt; a read-until-EOF would hang on
 * a keep-alive client. */
static char *read_request(int fd, size_t *out_len, size_t *out_head) {
    size_t cap = 8192, used = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    size_t head_end = 0;
    for (;;) {
        if (used + 1 >= cap) {
            cap *= 2;
            if (cap > MAX_BODY + 65536) { free(buf); return NULL; }
            char *g = realloc(buf, cap);
            if (!g) { free(buf); return NULL; }
            buf = g;
        }
        const ssize_t n = recv(fd, buf + used, cap - used - 1, 0);
        if (n <= 0) break;
        used += (size_t)n;
        buf[used] = '\0';

        if (!head_end) {
            char *p = strstr(buf, "\r\n\r\n");
            if (p) head_end = (size_t)(p - buf) + 4;
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

    size_t len = 0, head_end = 0;
    char *raw = read_request(ca.fd, &len, &head_end);

    http_conn conn = { .fd = ca.fd, .dead = 0 };
    if (!raw) {
        http_error(&conn, 400, "malformed request");
        close(ca.fd);
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
    return NULL;
}

int http_serve(const char *host, int port, http_handler fn, void *user,
               volatile sig_atomic_t *stop, char *err, size_t errsz) {
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
        conn_arg *ca = malloc(sizeof *ca);
        if (!ca) { close(cfd); continue; }
        *ca = (conn_arg){ .fd = cfd, .fn = fn, .user = user };

        pthread_t th;
        if (pthread_create(&th, NULL, serve_conn, ca) != 0) {
            close(cfd);
            free(ca);
            continue;
        }
        pthread_detach(th);
    }
    close(fd);
    return 0;
}
