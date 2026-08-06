# mynah-slm — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
# NOTE: deliberately NO -ffast-math (mynah-asr uses it, we don't). A decoder
# runs expf over logits and softmax over attention scores; under -ffast-math an
# inf is UB, and gcc on x86 vectorizes expf through libmvec and turns it into a
# NaN. That exact bug hit mynah-asr's Linux CI on 2026-07-18 while clang/ARM
# survived by luck. Not worth the few percent here.
CFLAGS  ?= -std=c11 -O3 -march=native -Wall -Wextra -iquote src -D_DEFAULT_SOURCE
LDFLAGS ?=

CFLAGS += -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate
  BLAS_DEF := MYNAH_SLM_BLAS_ACCELERATE
  CFLAGS  += -DMYNAH_SLM_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK
else
  LDFLAGS += -lopenblas
  BLAS_DEF := MYNAH_SLM_BLAS_OPENBLAS
  CFLAGS  += -DMYNAH_SLM_BLAS_OPENBLAS
  # fail early with a clear hint instead of "cblas.h: No such file or directory"
  ifeq ($(filter clean help,$(MAKECMDGOALS)),)
    ifeq ($(shell printf '\043include <cblas.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo ok),)
      $(error OpenBLAS headers not found. Install them first: `sudo apt install libopenblas-dev` (Debian/Ubuntu) or `sudo dnf install openblas-devel` (Fedora))
    endif
  endif
endif
LDFLAGS += -lpthread -lm

# hook for recursive variant builds: these ADD to the flags this Makefile
# computed instead of overriding CFLAGS
CFLAGS  += $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

# ingot: the GGUF/safetensors reader, vendored as a subtree and built by its own
# Makefile so this one never learns how it is compiled.
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
CFLAGS  += -I$(INGOT_DIR)/include
LDFLAGS += $(INGOT_LIB)

SRC := $(wildcard src/*.c)
OBJ := $(SRC:%.c=build/%.o)
HDR := $(wildcard src/*.h) $(wildcard include/*.h)

CFLAGS += -Iinclude

# version injected from git (informational string in `mynah-slm --version`)
MYNAH_SLM_BUILD := $(shell git describe --always --dirty 2>/dev/null || echo dev)
CFLAGS += -DMYNAH_SLM_BUILD='"$(MYNAH_SLM_BUILD)"'

# Weights live on the NAS (see CLAUDE.md): models/ is a symlink to
# /Volumes/shared/mynah-slm/models. For anything that MEASURES, stage a copy
# locally first with scripts/use_model.sh — reading weights over SMB is ~20x
# slower on the I/O and, worse, unreproducible once page-cache pressure starts
# re-faulting them mid-run. models-local/ wins automatically when populated.
MODEL_NAME ?= Qwen3-0.6B-Q4_K_M.gguf
MODEL      ?= $(firstword $(wildcard models-local/$(MODEL_NAME)) models/$(MODEL_NAME))

all: mynah-slm

help:
	@echo "mynah-slm targets:"
	@echo "  all          mynah-slm CLI (default)"
	@echo "  lib          libmynah_slm.a"
	@echo "  shared       libmynah_slm.{dylib,so}"
	@echo "  test         unit tests (exit 77 = skipped, model missing)"
	@echo "  debug        -O0 -g rebuild"
	@echo "  ubsan        UBSan rebuild + test, then clean"
	@echo "  asan         ASan+UBSan rebuild + test (LINUX CI ONLY, see below)"
	@echo "  leaks        macOS native leak check (no rebuild)"
	@echo "  update-ingot refresh the vendored ingot subtree"
	@echo "  install      PREFIX=$(PREFIX)"

mynah-slm: $(OBJ) build/cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# objects in build/ (never next to the sources: the variant builds — ubsan,
# asan — must not pollute the normal one)
build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib

$(OBJ): | $(INGOT_LIB)

# ── tests ──────────────────────────────────────────────────────────────────
# test_ingot needs no model: it pins the container-layer contract (block
# geometry, dequant coverage) so a bad subtree update fails here and not
# three modules later.
TESTS := tests/test_ingot tests/test_inspect tests/test_kernels tests/test_model

tests/%: build/tests/%.o $(OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

test: $(TESTS) mynah-slm
	@for t in $(TESTS); do \
	  $$t; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@./mynah-slm --version >/dev/null || exit 1
	@if [ -e "$(MODEL)" ]; then ./mynah-slm inspect "$(MODEL)" >/dev/null || exit 1; \
	 else echo "SKIP inspect: $(MODEL) not found (weights live on the NAS, see CLAUDE.md)"; fi

# ── libraries ──────────────────────────────────────────────────────────────
lib: libmynah_slm.a
libmynah_slm.a: $(OBJ)
	ar rcs $@ $^

ifeq ($(UNAME_S),Darwin)
  SOEXT := .dylib
else
  SOEXT := .so
endif
shared: libmynah_slm$(SOEXT)
libmynah_slm$(SOEXT): $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

# ── alternative builds ─────────────────────────────────────────────────────
# Memory/UB policy on macOS: `make leaks` (native, fast) + `make ubsan` (low
# overhead). ASan is VERY SLOW on a Mac and tends to hang with a large model:
# Linux CI only. Same rule as mynah-asr and qwen-tts.
#
# These go through EXTRA_CFLAGS, never CFLAGS=. Overriding CFLAGS drops
# everything this Makefile computed — the include paths and, less obviously,
# -DMYNAH_SLM_BUILD='"..."' with its quoting, which then fails to compile.
# EXTRA_CFLAGS is appended last, so a later -O0/-O1 also wins over -O3.
SAN_ADD := -g -fno-omit-frame-pointer

debug:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="$(SAN_ADD) -O0"

# clean at the end too: the sanitized objects must NOT be left behind to
# pollute the normal build
ubsan:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="$(SAN_ADD) -O2 -fsanitize=undefined" \
	  EXTRA_LDFLAGS="-fsanitize=undefined" all test && $(MAKE) clean
asan:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="$(SAN_ADD) -O1 -fsanitize=address,undefined" \
	  EXTRA_LDFLAGS="-fsanitize=address,undefined" all test && $(MAKE) clean

leaks: mynah-slm $(TESTS)
	@# test_inspect is the one that allocates (the census grows by realloc and
	@# the fixture writer holds buffers): it is the real subject here.
	leaks --atExit -- tests/test_inspect 2>&1 | tail -2
	@if [ -e "$(MODEL)" ]; then \
	   leaks --atExit -- ./mynah-slm inspect "$(MODEL)" 2>&1 | tail -3; \
	 else echo "SKIP leaks/inspect: $(MODEL) not found"; fi

clean:
	rm -rf build mynah-slm libmynah_slm.a libmynah_slm$(SOEXT) $(TESTS) dist
	@# Without this, libingot.a survives a clean: update the subtree and the
	@# next build silently links the previous library.
	@test -d $(INGOT_DIR) && $(MAKE) -C $(INGOT_DIR) clean || true

# Refresh the vendored ingot subtree from upstream. A plain clone already
# contains ingot (subtree = real files in-tree, nothing to init); this is only
# needed to pick up new upstream commits. Requires a clean working tree.
update-ingot:
	git subtree pull --prefix $(INGOT_DIR) https://github.com/mynah-org/ingot.git main --squash
	@$(MAKE) -C $(INGOT_DIR) clean

PREFIX ?= /usr/local
install: mynah-slm libmynah_slm.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 755 mynah-slm $(DESTDIR)$(PREFIX)/bin/
	install -m 644 libmynah_slm.a $(DESTDIR)$(PREFIX)/lib/
	install -m 644 include/mynah_slm.h $(DESTDIR)$(PREFIX)/include/

.PHONY: all help lib shared test debug ubsan asan leaks clean install update-ingot
