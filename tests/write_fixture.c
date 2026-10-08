/* write_fixture.c — write one of tests/fixture_model.h's synthetic
 * checkpoints to a path, for scripts and benches that want a model file and
 * cannot download one.
 *
 *   write_fixture <tiny|tiny-f32|06b> <out.gguf>
 *
 * Noise for weights: these files exercise the engine, never the quality.
 *
 * SPDX-License-Identifier: MIT */
#include "fixture_model.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: write_fixture <tiny|tiny-f32|06b> <out.gguf>\n");
        return 2;
    }
    fixture_spec s;
    if      (!strcmp(argv[1], "tiny"))     fixture_spec_tiny(&s, 1);
    else if (!strcmp(argv[1], "tiny-f32")) fixture_spec_tiny(&s, 0);
    else if (!strcmp(argv[1], "06b"))      fixture_spec_06b_shape(&s);
    else { fprintf(stderr, "write_fixture: unknown spec '%s'\n", argv[1]); return 2; }

    char err[256];
    if (fixture_write(argv[2], &s, err, sizeof err) != 0) {
        fprintf(stderr, "write_fixture: %s\n", err);
        return 1;
    }
    return 0;
}
