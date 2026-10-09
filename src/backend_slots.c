/* backend_slots.c — the per-request slot pool, for every backend.
 *
 * All the allocation happens in slots_create; acquire and release only move a
 * slot between three states and touch a fence:
 *
 *   FREE   --acquire-->  HELD  --release (record fence)-->  PARKED
 *   PARKED --acquire sees the fence passed--> handed out as HELD again
 *
 * PARKED exists because a release happens at a step boundary while the
 * device may still be running the step that was already submitted for that
 * row. Reusing the slot before that step ends would let the old request's
 * last kernel write into the new request's KV. Waiting for it would stall
 * every other row. Parking behind a fence does neither. On a synchronous
 * backend (the CPU) there are no fence ops and PARKED is passed at once.
 *
 * Contract and rationale: src/backend.h, .work/cuda-backend.md.
 *
 * SPDX-License-Identifier: MIT */
#include "backend_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SLOT_FREE = 0, SLOT_HELD, SLOT_PARKED };

typedef struct {
    int            state;
    uint64_t       generation;   /* bumped at every release */
    mynah_slm_bkv *kv;
    float         *scratch;
    void          *fence;        /* NULL on a synchronous backend */
} slot;

struct mynah_slm_bslots {
    uint32_t n, held;
    slot    *s;
};

static void set_err(char *err, size_t errsz, const char *msg) {
    if (err && errsz > 0) snprintf(err, errsz, "%s", msg);
}

void mynah_slm_backend_slots_destroy(mynah_slm_backend *b, mynah_slm_bslots *p) {
    if (!b || !p) return;
    void *st = NULL;
    const mynah_slm_backend_ops *ops = mynah_slm_backend_ops_of(b, &st);
    /* Shutdown: a sync is acceptable here and is what makes the frees safe. */
    mynah_slm_backend_sync(b, NULL, 0);
    for (uint32_t i = 0; i < p->n; i++) {
        mynah_slm_backend_kv_free(b, p->s[i].kv);
        mynah_slm_backend_free(b, p->s[i].scratch);
        if (p->s[i].fence && ops->fence_destroy) ops->fence_destroy(st, p->s[i].fence);
    }
    free(p->s);
    free(p);
}

int mynah_slm_backend_slots_create(mynah_slm_backend *b, const mynah_slm_bslots_desc *d,
                                   mynah_slm_bslots **out, char *err, size_t errsz) {
    if (out) *out = NULL;
    if (!b || !d || !out || d->n_slots == 0) {
        set_err(err, errsz, "invalid slot pool description");
        return -1;
    }
    void *st = NULL;
    const mynah_slm_backend_ops *ops = mynah_slm_backend_ops_of(b, &st);
    mynah_slm_bslots *p = calloc(1, sizeof *p);
    if (p) p->s = calloc(d->n_slots, sizeof *p->s);
    if (!p || !p->s) {
        free(p);
        set_err(err, errsz, "out of memory for the slot pool");
        return -1;
    }
    p->n = d->n_slots;
    for (uint32_t i = 0; i < p->n; i++) {
        int rc = mynah_slm_backend_kv_create(b, &d->kv, &p->s[i].kv, err, errsz);
        if (rc == 0 && d->scratch &&
            !(p->s[i].scratch = mynah_slm_backend_alloc(b, d->scratch, err, errsz)))
            rc = -1;
        if (rc == 0 && ops->fence_create)
            rc = ops->fence_create(st, &p->s[i].fence, err, errsz);
        if (rc != 0) {
            mynah_slm_backend_slots_destroy(b, p);
            return rc;               /* 1 passes through: the KV desc was refused */
        }
    }
    *out = p;
    return 0;
}

int mynah_slm_backend_slot_acquire(mynah_slm_backend *b, mynah_slm_bslots *p,
                                   uint32_t *slot_out, uint64_t *generation,
                                   char *err, size_t errsz) {
    if (!b || !p || !slot_out) { set_err(err, errsz, "invalid slot acquire"); return -1; }
    void *st = NULL;
    const mynah_slm_backend_ops *ops = mynah_slm_backend_ops_of(b, &st);
    /* Lowest index first: deterministic, and it keeps the live rows dense at
     * the low end, which is what lets a smaller width bucket take them. */
    for (uint32_t i = 0; i < p->n; i++) {
        slot *s = &p->s[i];
        if (s->state == SLOT_HELD) continue;
        if (s->state == SLOT_PARKED && s->fence && ops->fence_query) {
            const int q = ops->fence_query(st, s->fence, err, errsz);
            if (q < 0) return -1;
            if (q == 0) continue;    /* its last step is still running */
        }
        s->state = SLOT_HELD;
        p->held++;
        *slot_out = i;
        if (generation) *generation = s->generation;
        return 0;
    }
    return 1;
}

int mynah_slm_backend_slot_release(mynah_slm_backend *b, mynah_slm_bslots *p,
                                   uint32_t idx, char *err, size_t errsz) {
    if (!b || !p || idx >= p->n || p->s[idx].state != SLOT_HELD) {
        set_err(err, errsz, "slot release: slot is not held");
        return -1;
    }
    void *st = NULL;
    const mynah_slm_backend_ops *ops = mynah_slm_backend_ops_of(b, &st);
    slot *s = &p->s[idx];
    if (s->fence && ops->fence_record &&
        ops->fence_record(st, s->fence, err, errsz) != 0) {
        /* No fence, no proof the old step is done: keep the slot out of
         * circulation rather than risk a late write into the next request. */
        return -1;
    }
    s->generation++;
    s->state = SLOT_PARKED;
    p->held--;
    return 0;
}

mynah_slm_bkv *mynah_slm_backend_slot_kv(mynah_slm_bslots *p, uint32_t i) {
    return (p && i < p->n) ? p->s[i].kv : NULL;
}

float *mynah_slm_backend_slot_scratch(mynah_slm_bslots *p, uint32_t i) {
    return (p && i < p->n) ? p->s[i].scratch : NULL;
}

uint64_t mynah_slm_backend_slot_generation(const mynah_slm_bslots *p, uint32_t i) {
    return (p && i < p->n) ? p->s[i].generation : 0;
}

uint32_t mynah_slm_backend_slots_held(const mynah_slm_bslots *p) {
    return p ? p->held : 0;
}
