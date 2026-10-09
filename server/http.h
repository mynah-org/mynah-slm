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
    /* Liveness probe period once the client's read side has reached EOF
     * (see http_keepalive). 0 = 250 ms. */
    int probe_interval_ms;
    /* ABSOLUTE deadlines for reading a request, from accept: the headers
     * must be complete within header_timeout_ms, the body within
     * body_timeout_ms after them; past either the answer is 408 and the
     * connection is closed. SO_RCVTIMEO bounds one recv(), not a request,
     * so a client dripping a byte every few seconds could otherwise hold a
     * connection — and, --max-conns of them, every legitimate client out.
     * 0 = 10000 / 30000 ms. */
    int header_timeout_ms;
    int body_timeout_ms;
    /* SO_SNDBUF for a streaming response (set by http_begin_sse), so the
     * bytes a client that stopped reading can park in the kernel are
     * bounded. 0 = 64 KiB; -1 = leave the system's autotuning. */
    int stream_sndbuf_bytes;
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

/* Complete response with a Content-Length. If the response has already
 * been started (http_begin_sse, or http_keepalive committed it), only the
 * body is written: the status line is gone, so an error after that point
 * travels in-band, as the body. */
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
 * can be asked before every decode step. Sticky once true. Safe to call from
 * a thread other than the one writing.
 *
 * Gone = a write failed (EPIPE, ECONNRESET, a send timeout), or the socket
 * reports a reset or hang-up (POLLERR / POLLHUP). An EOF on the read side is
 * NOT gone: a client that half-closes after its request (shutdown(SHUT_WR):
 * nc -N, socat, some proxies) still reads the answer, and from here it looks
 * exactly like one that closed the socket. The EOF is recorded
 * (http_peer_eof) and http_keepalive resolves it with a write: a closed
 * socket answers it with a reset, which the next probe sees.
 * A pipelined byte from a live client is neither. */
int http_peer_gone(http_conn *c);

/* The read side reached EOF (half-close or close — not distinguishable
 * without writing). Set by http_peer_gone. */
int http_peer_eof(http_conn *c);

/* Resolve an EOF: once the read side has reached EOF, make sure the client
 * was written to within the last probe interval, so a client that really
 * closed answers with a reset. Nothing happens before EOF. The first write
 * starts the response if it has not started — the SSE header (`sse`), or a
 * 200 with a close-delimited JSON body — and the later ones are an SSE
 * comment line (":\n\n") or one byte of JSON whitespace, both ignored by a
 * client. Call it from the thread that writes the response. */
void http_keepalive(http_conn *c, int sse);

/* Is the client still taking what we send? Linux: the bytes the peer has
 * acknowledged (sent - SIOCOUTQ) must advance while anything is unacked; a
 * client whose acknowledged count has not moved for the send timeout has
 * stopped reading, and the connection is marked dead (1). A send() blocked
 * on a full buffer is bounded by SO_SNDTIMEO already; this catches the
 * client whose kernel buffers are still absorbing the stream, which with
 * an autotuned send buffer can be minutes. Elsewhere: 0 (the capped send
 * buffer and SO_SNDTIMEO are the bound). Writer thread only. */
int http_send_stalled(http_conn *c);

/* Has the response been started (status line written)? */
int http_head_sent(const http_conn *c);

/* The probe on a bare descriptor, exposed for tests/test_http.c:
 * 0 alive, 1 EOF on the read side (half-closed or closed), 2 gone (reset,
 * hang-up, a closed descriptor). */
int http_fd_probe(int fd);

#endif /* MYNAH_SLM_HTTP_H */
