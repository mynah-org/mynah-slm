/* jobq.h — the bounded pending queue between the HTTP threads and the
 * scheduler.
 *
 * Bounded, and a push NEVER waits: a full queue is an immediate refusal that
 * the caller turns into a 503 with Retry-After. The sibling that measured it
 * (mynah-tts .work/serving-design.md §6) found that a client parked where it
 * cannot be seen — in the kernel backlog, or in an unbounded queue — is a
 * multi-second tail no metric reports: "a wait you cannot see is worse than a
 * refusal you can".
 *
 * Jobs are opaque pointers; the queue never owns or frees one. The ring is
 * allocated once at creation.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_JOBQ_H
#define MYNAH_SLM_JOBQ_H

#include <stddef.h>

typedef struct mynah_slm_jobq mynah_slm_jobq;

/* cap >= 1. NULL on allocation failure. */
mynah_slm_jobq *mynah_slm_jobq_new(size_t cap);
void            mynah_slm_jobq_free(mynah_slm_jobq *q);

/* 0 when queued; -1 when full or closed — answer 503, never wait. */
int    mynah_slm_jobq_push(mynah_slm_jobq *q, void *job);

/* The oldest job, or NULL. With `block`, waits until there is one or the
 * queue is closed; NULL then means closed and empty. */
void  *mynah_slm_jobq_pop(mynah_slm_jobq *q, int block);

/* Refuse every later push and wake every blocked pop. Jobs already queued are
 * still handed out, so they can be answered. */
void   mynah_slm_jobq_close(mynah_slm_jobq *q);

/* Take every queued job for which pred(ud, job) is non-zero out of the
 * queue, oldest first, at most `cap` of them into `out`; the others keep
 * their order. Returns how many were taken. pred runs under the queue's
 * lock: keep it cheap and never call back into the queue. This is how
 * jobs whose client left while queued stop holding a place. */
size_t mynah_slm_jobq_remove_if(mynah_slm_jobq *q, int (*pred)(void *ud, void *job),
                                void *ud, void **out, size_t cap);

size_t mynah_slm_jobq_depth(mynah_slm_jobq *q);
size_t mynah_slm_jobq_cap(const mynah_slm_jobq *q);

#endif /* MYNAH_SLM_JOBQ_H */
