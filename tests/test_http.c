/* test_http.c — the peer-gone probe, on real loopback TCP sockets.
 *
 * The no-zombie-work rule rests on one question asked before every decode
 * step: "is the client still there?". Wrong one way, a live client gets its
 * request killed; wrong the other way, a departed one keeps the model busy to
 * max_tokens. So both directions are pinned here, with no model:
 *
 *   open, idle               alive
 *   open, a pipelined byte   alive (readable is not the same as gone)
 *   peer half-closed         EOF, NOT gone — it still reads our answer
 *   peer closed (FIN)        EOF, not gone until written to; gone after
 *                            one write (the closed socket resets it)
 *   peer reset (SO_LINGER 0) gone
 *
 * SPDX-License-Identifier: MIT */
#include "../server/http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int failures;

static void check(const char *what, int ok) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

/* A connected loopback pair: *srv is the accepted end (what the server
 * probes), *cli the client's. */
static int tcp_pair(int *srv, int *cli) {
    const int l = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t al = sizeof a;
    if (l < 0 || bind(l, (struct sockaddr *)&a, sizeof a) != 0 || listen(l, 1) != 0 ||
        getsockname(l, (struct sockaddr *)&a, &al) != 0) return -1;
    *cli = socket(AF_INET, SOCK_STREAM, 0);
    if (*cli < 0 || connect(*cli, (struct sockaddr *)&a, sizeof a) != 0) return -1;
    *srv = accept(l, NULL, NULL);
    close(l);
    return *srv < 0 ? -1 : 0;
}

/* A FIN or RST takes a moment to cross loopback; poll up to 1 s for state
 * `want`. Returns the number of probes it took, or -1 if never. */
static int probes_until(int fd, int want) {
    for (int i = 1; i <= 1000; i++) {
        if (http_fd_probe(fd) == want) return i;
        struct timespec ts = { 0, 1000000 };
        nanosleep(&ts, NULL);
    }
    return -1;
}
#define probes_until_gone(fd) probes_until((fd), 2)

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    int s, c;

    if (tcp_pair(&s, &c) != 0) { printf("FAIL no loopback TCP\n"); return 1; }
    check("an open, idle client is alive", http_fd_probe(s) == 0);
    check("asking again does not change the answer", http_fd_probe(s) == 0);
    send(c, "x", 1, 0);
    {
        struct timespec ts = { 0, 20000000 };
        nanosleep(&ts, NULL);
    }
    check("a pipelined byte is not a disconnect", http_fd_probe(s) == 0);
    char b;
    check("... and the probe did not consume it", recv(s, &b, 1, 0) == 1 && b == 'x');
    close(c);
    const int e = probes_until(s, 1);
    check("a closed client shows EOF first (indistinguishable from a half-close)", e > 0);
    check("... and is not called gone before anything is written", http_fd_probe(s) == 1);
    send(s, "x", 1, MSG_NOSIGNAL);              /* what http_keepalive does */
    const int n = probes_until_gone(s);
    printf("     gone %d probe(s) after one write\n", n);
    check("a closed client is gone once written to (it answers with a reset)", n > 0);
    close(s);

    /* Half-close (shutdown(SHUT_WR) after the request): EOF, never gone,
     * and it still receives what is written — the review's R2 case, where
     * such a client used to get no answer at all. */
    if (tcp_pair(&s, &c) != 0) return 1;
    shutdown(c, SHUT_WR);
    check("a half-closed client shows EOF", probes_until(s, 1) > 0);
    for (int i = 0; i < 3; i++) send(s, "y", 1, MSG_NOSIGNAL);
    {
        struct timespec ts = { 0, 50000000 };
        nanosleep(&ts, NULL);
    }
    check("... is not gone after writes", http_fd_probe(s) == 1);
    char got[8];
    check("... and reads them", recv(c, got, sizeof got, 0) == 3);
    close(c);
    close(s);

    if (tcp_pair(&s, &c) != 0) return 1;
    struct linger lg = { 1, 0 };
    setsockopt(c, SOL_SOCKET, SO_LINGER, &lg, sizeof lg);
    close(c);                                   /* RST, not FIN */
    check("a reset client is gone", probes_until_gone(s) > 0);
    close(s);

    check("a closed descriptor is gone", http_fd_probe(-1) == 2);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
