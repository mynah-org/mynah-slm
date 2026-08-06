/* mynah_slm.h — the public C API of the Mynah SLM engine.
 *
 * Contract, everywhere in this header:
 *   - errors are returned as a code plus a message in the caller's buffer;
 *     the library NEVER writes to stderr
 *   - nothing here is model-specific: no Qwen, no Gemma, no architecture enum
 *   - on a failed open, *out is set to NULL, so an ignored return code fails
 *     safely. Do not pass a variable that still owns a live handle.
 *
 * This header parses as C++ too.
 *
 * STATUS: scaffolding. Only the introspection half below exists today; the
 * session/generate API in PLAN.md lands with the engine. Functions are added
 * here when they work, never before — an empty declaration that lies is worse
 * than a missing one.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_H
#define MYNAH_SLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build string from `git describe`, informational. */
const char *mynah_slm_version(void);

/* ── checkpoint introspection ───────────────────────────────────────────────
 * What is inside a checkpoint, before deciding whether we can run it. Backs
 * `mynah-slm inspect` and, later, the config extraction the loader needs. */

/* One row of the per-type census: how much of a checkpoint is stored in a
 * given ggml block type, and whether this build can actually use it. */
typedef struct {
    int         type;          /* INGOT_TYPE_* */
    const char *type_name;     /* "Q4_K", "F32", ... — static storage */
    size_t      tensors;       /* how many tensors use it */
    uint64_t    elements;      /* total elements stored in it */
    uint64_t    bytes;         /* total on-disk bytes */
    int         can_dequant;   /* 0 = ingot knows the geometry but cannot decode */
} mynah_slm_type_use;

/* Everything `inspect` reports. `types` is owned by the struct; release with
 * mynah_slm_report_free(). */
typedef struct {
    char                 path[1024];
    /* Copied, not borrowed: ingot's arch string lives in the handle, and the
     * handle is closed before this returns. `type_name` below is fine to
     * borrow — those are static. */
    char                 arch[64];     /* GGUF "general.architecture", "" if absent */
    uint32_t             gguf_version;
    uint32_t             shards;
    size_t               n_tensors;
    size_t               n_kv;
    uint64_t             total_bytes;
    double               bits_per_weight;  /* total_bytes*8 / total elements */
    int                  runnable;     /* 0 when some tensor cannot be decoded */
    mynah_slm_type_use  *types;
    size_t               n_types;
} mynah_slm_report;

/* Open a GGUF (single file or `-00001-of-000NN` split) and fill `out`.
 * Returns 0 on success, -1 on failure with the reason in `err`. */
int  mynah_slm_inspect_gguf(mynah_slm_report *out, const char *path,
                            char *err, size_t errsz);

void mynah_slm_report_free(mynah_slm_report *r);

#ifdef __cplusplus
}
#endif
#endif /* MYNAH_SLM_H */
