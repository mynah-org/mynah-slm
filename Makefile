# mynah-slm — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
# NOTE: deliberately NO -ffast-math (mynah-asr uses it, we don't). A decoder
# runs expf over logits and softmax over attention scores; under -ffast-math an
# inf is UB, and gcc on x86 vectorizes expf through libmvec and turns it into a
# NaN. That exact bug hit mynah-asr's Linux CI on 2026-07-18 while clang/ARM
# survived by luck. Not worth the few percent here.
# Split out so a cross build can replace it without overriding CFLAGS entirely
# — which would drop every computed flag below, including the quoted
# -DMYNAH_SLM_BUILD. See test-x86-rosetta.
ARCH_FLAGS ?= -march=native
CFLAGS  ?= -std=c11 -O3 $(ARCH_FLAGS) -Wall -Wextra -iquote src -D_DEFAULT_SOURCE
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

# Checkpoints live in models/, which is often not local storage — a network
# share, an external drive. For anything that MEASURES, stage a copy on the
# local disk first with scripts/use_model.sh: remote weights are ~20x slower on
# the I/O and, worse, unreproducible once page-cache pressure starts re-faulting
# them mid-run. models-local/ wins automatically when populated.
MODEL_NAME ?= Qwen3-0.6B-Q4_K_M.gguf
MODEL      ?= $(firstword $(wildcard models-local/$(MODEL_NAME)) models/$(MODEL_NAME))

all: mynah-slm mynah-slm-server

help:
	@echo "mynah-slm targets:"
	@echo "  all          mynah-slm CLI + mynah-slm-server (default)"
	@echo "  lib          libmynah_slm.a"
	@echo "  shared       libmynah_slm.{dylib,so}"
	@echo "  test         unit tests + parity (exit 77 = skipped, model missing)"
	@echo "  test-parity  C forward pass vs the numpy oracle, stage by stage"
	@echo "  test-server  end-to-end HTTP checks (needs a minute of generation)"
	@echo "  bench        per-tensor matvec throughput"
	@echo "  check-x86    cross-compile the AVX2 paths"
	@echo "  test-x86-rosetta  build x86_64 and RUN the suite under Rosetta"
	@echo "  golden-dump  regenerate the oracle's reference activations"
	@echo "  debug        -O0 -g rebuild"
	@echo "  ubsan        UBSan rebuild + test, then clean"
	@echo "  asan         ASan+UBSan rebuild + test (LINUX CI ONLY, see below)"
	@echo "  leaks        macOS native leak check (no rebuild)"
	@echo "  update-ingot refresh the vendored ingot subtree"
	@echo "  install      PREFIX=$(PREFIX)"

# $(INGOT_LIB) is a real prerequisite, not just order-only. Without it a
# `make update-ingot` (or any local ingot change) rebuilds the archive and
# leaves every binary linked against the PREVIOUS one — silently. Cost an
# entire benchmark that showed a 2.9x kernel win producing no end-to-end
# change, which looked like the kernel being irrelevant rather than absent.
mynah-slm: $(OBJ) build/cli/main.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

SERVER_OBJ := build/server/main.o build/server/http.o
mynah-slm-server: $(OBJ) $(SERVER_OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

build/server/%.o: server/%.c $(HDR) $(wildcard server/*.h)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -iquote server -c $< -o $@

# objects in build/ (never next to the sources: the variant builds — ubsan,
# asan — must not pollute the normal one)
build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib CC="$(CC)" CFLAGS="-O2 $(ARCH_FLAGS)"

$(OBJ): | $(INGOT_LIB)

# ── tests ──────────────────────────────────────────────────────────────────
# test_ingot needs no model: it pins the container-layer contract (block
# geometry, dequant coverage) so a bad subtree update fails here and not
# three modules later.
TESTS := tests/test_batch tests/test_ingot tests/test_inspect tests/test_kernels tests/test_model tests/test_think tests/test_tokenizer tests/test_tools

# The parity harness is built like the others but driven separately: it dumps
# activations, and tools/eval/compare.py is what judges them.
PARITY     := tests/test_parity
GOLDEN_DIR := tests/golden/it_hello
DUMP_DIR   := build/dump/it_hello
PROMPT     ?= Ciao! Come stai?

tests/%: build/tests/%.o build/tests/npy.o $(OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

test: $(TESTS) mynah-slm
	@for t in $(TESTS); do \
	  $$t; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@./mynah-slm --version >/dev/null || exit 1
	@if [ -e "$(MODEL)" ]; then ./mynah-slm inspect "$(MODEL)" >/dev/null || exit 1; \
	 else echo "SKIP inspect: $(MODEL) not found (scripts/download_model.sh --list)"; fi
	@$(MAKE) --no-print-directory test-parity

# C forward pass vs the numpy oracle, stage by stage. Skips (77) rather than
# failing when the model or the golden dumps are absent: weights are not a
# build dependency. Regenerate the golden side with `make golden-dump`.
test-parity: $(PARITY)
	@mkdir -p $(DUMP_DIR)
	@$(PARITY) "$(MODEL)" $(GOLDEN_DIR) $(DUMP_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP parity: model or golden dumps missing (make golden-dump)"; exit 0; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	  cd tools && uv run python -m eval.compare ../$(GOLDEN_DIR) ../$(DUMP_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP parity: nothing to compare"; exit 0; else exit $$rc; fi

# Per-tensor matvec throughput — where a decode step goes. Not part of `test`:
# it measures, and a measurement that runs on every build gets ignored.
BENCH := tests/bench_matvec
bench: $(BENCH)
	@$(BENCH) "$(MODEL)"; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP bench: model missing (scripts/use_model.sh)"; exit 0; \
	  else exit $$rc; fi

# End-to-end server checks: shape, determinism (sequential and concurrent),
# SSE framing, and that reasoning never reaches content. Separate from `test`
# because it spends a minute of real generation.
test-server: mynah-slm-server
	@sh tests/test_server.sh "$(MODEL)"; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server: model missing"; exit 0; \
	  else exit $$rc; fi

# Regenerate the oracle's reference activations. Slow (no KV cache, on purpose)
# and only needed when the prompt or the dumped stages change.
golden-dump:
	@test -e "$(MODEL)" || { echo "no model at $(MODEL) — scripts/use_model.sh"; exit 1; }
	@mkdir -p $(GOLDEN_DIR)
	cd tools && uv run python -m oracle.generate ../$(MODEL) \
	  --prompt "$(PROMPT)" --dump-dir ../$(GOLDEN_DIR) -n 1

# ── x86, from an arm64 laptop ──────────────────────────────────────────────
# The AVX2 paths in src/qmat.c, src/kvcache.c and src/kernels.c would otherwise
# only ever meet a compiler on someone else's machine. Two targets, because
# they answer different questions:
#
#   check-x86         does it COMPILE at AVX2+F16C and at AVX-512? Catches
#                     #ifdef rot and intrinsic misuse, nothing else.
#   test-x86-rosetta  does it RUN? Builds the whole suite as x86_64 and runs it
#                     under Rosetta, which translates AVX2. This is the one
#                     that would catch a wrong shuffle. AVX-512 is not
#                     translated and stays compile-only.
X86_TARGET ?= x86_64-apple-macos13.3
X86_SRC := $(SRC) $(wildcard tests/*.c)

check-x86:
	@mkdir -p build/x86
	@for f in $(X86_SRC); do \
	  $(CC) -target $(X86_TARGET) -std=c11 -O2 -Wall -Wextra -iquote src -Iinclude \
	    -I$(INGOT_DIR)/include -DMYNAH_SLM_BUILD='"x86check"' -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK \
	    -mavx2 -mfma -mf16c -c $$f -o build/x86/$$(basename $$f .c).avx2.o || exit 1; \
	done
	@echo "x86-64 cross-compile OK (avx2 + fma + f16c)"

# INGOT_CAPS_ASSUME is not optional here, it is what makes this target mean
# something. Rosetta EXECUTES AVX2 but does not advertise it in CPUID, and
# ingot dispatches on CPUID at runtime — so without it every ingot kernel under
# this target quietly runs its scalar path and the gate tests nothing x86 about
# ingot. (Our own kernels are gated at compile time and did run, which is how
# the gap stayed hidden.) Added upstream for exactly this reason.
test-x86-rosetta:
	$(MAKE) clean
	INGOT_CAPS_ASSUME=avx2 $(MAKE) CC="$(CC) -arch x86_64" ARCH_FLAGS="-mavx2 -mfma -mf16c" test
	$(MAKE) clean

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
	@# test_tools allocates on every path too — tool sets, parsed calls, the
	@# rendered prompt — and unlike the others it needs no checkpoint, so this
	@# stays a fast check.
	leaks --atExit -- tests/test_tools 2>&1 | tail -2
	@if [ -e "$(MODEL)" ]; then \
	   leaks --atExit -- ./mynah-slm inspect "$(MODEL)" 2>&1 | tail -3; \
	 else echo "SKIP leaks/inspect: $(MODEL) not found"; fi

clean:
	rm -rf build mynah-slm mynah-slm-server libmynah_slm.a libmynah_slm$(SOEXT) $(TESTS) $(PARITY) $(BENCH) dist
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
install: mynah-slm mynah-slm-server libmynah_slm.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 755 mynah-slm mynah-slm-server $(DESTDIR)$(PREFIX)/bin/
	install -m 644 libmynah_slm.a $(DESTDIR)$(PREFIX)/lib/
	install -m 644 include/mynah_slm.h $(DESTDIR)$(PREFIX)/include/

.PHONY: all help lib shared test test-parity test-server bench check-x86 test-x86-rosetta golden-dump debug ubsan asan leaks clean install update-ingot
