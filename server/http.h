/* http.h — the HTTP/1.1 subset this server needs, on plain sockets.
 *
 * No framework, no libcurl, no TLS. A reverse proxy does TLS; adding it here
 * would be the second-largest dependency in the project for something nginx
 * already does better.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_HTTP_H
#define MYNAH_SLM_HTTP_H

#include <signal.h>
#include <stddef.h>

typedef struct http_conn http_conn;

typedef struct {
    char        method[8];
    char        path[512];
    const char *body;
    size_t      body_len;
} http_request;

typedef void (*http_handler)(void *user, http_conn *conn, const http_request *req);

typedef struct {
    /* Connections served at once; one more is answered 503 + Retry-After at
     * accept and closed, never parked. 0 = 64. */
    int max_conns;
    /* SO_SNDTIMEO on every accepted socket. A client that stops reading
     * without closing fills its TCP window, and an unbounded send() would
     * then hold whatever the writer holds — the model, in the serialized
     * server — forever. A send that times out marks the connection dead.
     * 0 = 5000 ms. */
    int send_timeout_ms;
    /* SO_RCVTIMEO while the request is read: a client that opens a socket
     * and sends nothing must not keep a thread. 0 = 30000 ms. */
    int recv_timeout_ms;
} http_limits;

/* Accept loop. Returns when *stop becomes non-zero (a signal handler sets it)
 * or on a fatal error. One thread per connection, at most `max_conns` of
 * them; the threads keep parsing and socket writes off the model's thread.
 * `limits` may be NULL for the defaults. */
int http_serve(const char *host, int port, http_handler fn, void *user,
               const http_limits *limits, volatile sig_atomic_t *stop,
               char *err, size_t errsz);

/* Live connections and how many were refused at the cap, for /health. */
int           http_live_connections(void);
unsigned long http_rejected_connections(void);

/* Complete response with a Content-Length. */
void http_respond(http_conn *c, int status, const char *content_type,
                  const char *body, size_t len);

void http_error(http_conn *c, int status, const char *message);

/* 503 with a Retry-After header: the overload answer, sent at once. */
void http_busy(http_conn *c, const char *message, int retry_after_s);

/* Switch to an SSE stream: headers now, frames as they come. */
void http_begin_sse(http_conn *c);

/* Raw write on an open connection. Returns -1 once the peer is gone (or a
 * send timed out), and every call after that too. */
int http_write(http_conn *c, const char *data, size_t len);

/* Has the peer gone? Non-blocking and cheap — one poll(2) with a zero
 * timeout, plus a one-byte MSG_PEEK when the socket reads as readable — so it
 * can be asked before every decode step. Gone = a write already failed, the
 * socket reports hang-up / error / read-side shutdown, or a peek reads EOF.
 * A pipelined byte from a live client is NOT gone. Sticky once true.
 * Safe to call from a thread other than the one writing. */
int http_peer_gone(http_conn *c);

/* The same probe on a bare descriptor (what http_peer_gone runs), exposed
 * for tests/test_http.c. 1 = gone, 0 = alive. */
int http_fd_peer_gone(int fd);

#endif /* MYNAH_SLM_HTTP_H */
