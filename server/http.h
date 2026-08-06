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

/* Accept loop. Returns when *stop becomes non-zero (a signal handler sets it)
 * or on a fatal error. One thread per connection: the work is serialized at
 * the model anyway, so the threads exist to keep parsing and socket writes off
 * the critical path, not to run inferences in parallel. */
int http_serve(const char *host, int port, http_handler fn, void *user,
               volatile sig_atomic_t *stop, char *err, size_t errsz);

/* Complete response with a Content-Length. */
void http_respond(http_conn *c, int status, const char *content_type,
                  const char *body, size_t len);

void http_error(http_conn *c, int status, const char *message);

/* Switch to an SSE stream: headers now, frames as they come. */
void http_begin_sse(http_conn *c);

/* Raw write on an open connection. Returns -1 once the peer is gone, which is
 * how a generation loop learns to stop early. */
int http_write(http_conn *c, const char *data, size_t len);

#endif /* MYNAH_SLM_HTTP_H */
