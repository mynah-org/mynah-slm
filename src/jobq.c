/* jobq.c — see jobq.h.
 * SPDX-License-Identifier: MIT */
#include "jobq.h"

#include <pthread.h>
#include <stdlib.h>

struct mynah_slm_jobq {
    pthread_mutex_t mu;
    pthread_cond_t  arrived;
    void  **ring;
    size_t  cap, head, len;
    int     closed;
};

mynah_slm_jobq *mynah_slm_jobq_new(size_t cap) {
    if (cap == 0) return NULL;
    mynah_slm_jobq *q = calloc(1, sizeof *q);
    if (!q) return NULL;
    q->ring = calloc(cap, sizeof *q->ring);
    if (!q->ring) { free(q); return NULL; }
    q->cap = cap;
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->arrived, NULL);
    return q;
}

void mynah_slm_jobq_free(mynah_slm_jobq *q) {
    if (!q) return;
    pthread_cond_destroy(&q->arrived);
    pthread_mutex_destroy(&q->mu);
    free(q->ring);
    free(q);
}

int mynah_slm_jobq_push(mynah_slm_jobq *q, void *job) {
    pthread_mutex_lock(&q->mu);
    if (q->closed || q->len == q->cap) {
        pthread_mutex_unlock(&q->mu);
        return -1;
    }
    q->ring[(q->head + q->len) % q->cap] = job;
    q->len++;
    pthread_cond_signal(&q->arrived);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

void *mynah_slm_jobq_pop(mynah_slm_jobq *q, int block) {
    pthread_mutex_lock(&q->mu);
    while (block && q->len == 0 && !q->closed)
        pthread_cond_wait(&q->arrived, &q->mu);
    void *job = NULL;
    if (q->len) {
        job = q->ring[q->head];
        q->ring[q->head] = NULL;
        q->head = (q->head + 1) % q->cap;
        q->len--;
    }
    pthread_mutex_unlock(&q->mu);
    return job;
}

void mynah_slm_jobq_close(mynah_slm_jobq *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = 1;
    pthread_cond_broadcast(&q->arrived);
    pthread_mutex_unlock(&q->mu);
}

size_t mynah_slm_jobq_depth(mynah_slm_jobq *q) {
    pthread_mutex_lock(&q->mu);
    const size_t n = q->len;
    pthread_mutex_unlock(&q->mu);
    return n;
}

size_t mynah_slm_jobq_cap(const mynah_slm_jobq *q) { return q->cap; }
