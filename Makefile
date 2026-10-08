# mynah-slm — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux),
# or our own GEMM with BLAS=none (no vendor dependency, see below).
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

# BLAS: which f32 GEMM prefill and batched attention run on. Four values:
#
#   auto        `openblas` on Linux, `accelerate` on macOS — the default, and
#               unchanged from before src/sgemm.c existed.
#   none        OUR src/sgemm.c, no vendor BLAS in the process and no -dev
#               package to install. Deterministic (bit-identical across
#               thread counts) and on our own thread pool — the reasons
#               mynah-tts made the same move (its .work/no-blas.md). OPT-IN,
#               NOT THE DEFAULT: on the prefill shapes that dominate
#               (T=256 batches) it measured 0.5-0.65x OpenBLAS on a 4-vCPU
#               Cascade Lake VM. The flip is gated in .work/no-blas.md.
#   openblas    Linux vendor BLAS.
#   accelerate  macOS.
#
# With a vendor linked, MYNAH_SLM_SGEMM=own routes to ours at run time, so the
# A/B runs interleaved in one process (`tests/test_sgemm bench`). Whatever
# links, src/sgemm.c owns the one entry point (mynah_slm_sgemm).
BLAS ?= auto
ifeq ($(UNAME_S),Darwin)
  BLAS_RESOLVED := $(if $(filter auto,$(BLAS)),accelerate,$(BLAS))
else
  BLAS_RESOLVED := $(if $(filter auto,$(BLAS)),openblas,$(BLAS))
endif

ifeq ($(BLAS_RESOLVED),none)
  BLAS_DEF := MYNAH_SLM_BLAS_NONE
else ifeq ($(BLAS_RESOLVED),accelerate)
  ifneq ($(UNAME_S),Darwin)
    $(error BLAS=accelerate is macOS-only; this host is $(UNAME_S))
  endif
  LDFLAGS += -framework Accelerate
  BLAS_DEF := MYNAH_SLM_BLAS_ACCELERATE
  CFLAGS  += -DACCELERATE_NEW_LAPACK
else ifeq ($(BLAS_RESOLVED),openblas)
  LDFLAGS += -lopenblas
  BLAS_DEF := MYNAH_SLM_BLAS_OPENBLAS
  # fail early with a clear hint instead of "cblas.h: No such file or directory"
  ifeq ($(filter clean help,$(MAKECMDGOALS)),)
    ifeq ($(shell printf '\043include <cblas.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo ok),)
      $(error OpenBLAS headers not found. Install them first (`sudo apt install libopenblas-dev`), or build without a vendor BLAS: `make BLAS=none`)
    endif
  endif
else
  $(error BLAS=$(BLAS) is not a profile. Use auto, none, openblas or accelerate)
endif
CFLAGS += -D$(BLAS_DEF)
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

# ── kernel TUs: runtime ISA dispatch (src/kern.h, .work/isa-runtime-dispatch.md)
# Every SIMD kernel we own is compiled ONCE PER ISA into its own object, and
# src/isa.c picks one table per family at run time (CPUID / getauxval /
# sysctl, narrowed by MYNAH_SLM_ISA, verified against the scalar table). So a
# portable build (ARCH_FLAGS=-march=x86-64-v2 or armv8-a) still runs the
# AVX2 / AVX-512 VNNI / dotprod kernels where the CPU has them, and
# `mynah-slm --dispatch` proves which ones resolved.
#
# The architecture comes from the COMPILER, not the host: a cross build with
# CC=aarch64-linux-gnu-gcc gets the arm64 set.
KERN_MACHINE := $(shell $(CC) $(ARCH_FLAGS) -dumpmachine 2>/dev/null)
ifneq ($(filter x86_64% amd64%,$(KERN_MACHINE)),)
  KERN_ARCH := x86
else ifneq ($(filter aarch64% arm64%,$(KERN_MACHINE)),)
  KERN_ARCH := arm64
else
  KERN_ARCH := generic
endif

# which TUs each family has, per architecture (src/isa.c lists the same sets)
KERN_QMAT_x86      := scalar avx2 avx512vnni
KERN_ATTN_x86      := scalar avx2
KERN_SGEMM_x86     := scalar avx2 avx512
KERN_QMAT_arm64    := scalar neon neon_dotprod
KERN_ATTN_arm64    := scalar neon
KERN_SGEMM_arm64   := scalar neon
KERN_QMAT_generic  := scalar
KERN_ATTN_generic  := scalar
KERN_SGEMM_generic := scalar

# per-TU ISA flags, added AFTER ARCH_FLAGS so they win. The x86 "scalar" and
# "avx2" TUs also switch the wider ISAs OFF, so a -march=native build's
# MYNAH_SLM_ISA=scalar really is a pre-AVX2 run and not one the compiler
# quietly auto-vectorized with AVX-512.
KF_x86_scalar      := -mno-avx2 -mno-fma
KF_x86_avx2        := -mavx2 -mfma -mf16c -mno-avx512f
KF_x86_avx512      := -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq
KF_x86_avx512vnni  := $(KF_x86_avx512) -mavx512vnni
KF_arm64_scalar    :=
KF_arm64_neon      :=
# only when the baseline lacks it: a second -march would override -mcpu
KF_arm64_neon_dotprod := $(if $(findstring __ARM_FEATURE_DOTPROD,$(shell $(CC) $(ARCH_FLAGS) -dM -E -xc /dev/null 2>/dev/null)),,-march=armv8.2-a+dotprod)
KF_generic_scalar  :=

KID_scalar       := 0
KID_neon         := 1
KID_neon_dotprod := 2
KID_avx2         := 3
KID_avx512       := 4
KID_avx512vnni   := 5

KERN_SRC := src/qmat_kern.c src/attn_kern.c
KERN_OBJ := $(foreach t,$(KERN_QMAT_$(KERN_ARCH)),build/kern/qmat_$(t).o) \
            $(foreach t,$(KERN_ATTN_$(KERN_ARCH)),build/kern/attn_$(t).o) \
            $(foreach t,$(KERN_SGEMM_$(KERN_ARCH)),build/kern/sgemm_$(t).o)
KERN_TU_FLAGS = $(KF_$(KERN_ARCH)_$*) -DMYNAH_SLM_KERN_TU=$* -DMYNAH_SLM_KERN_ID=$(KID_$*)

SRC := $(filter-out $(KERN_SRC),$(wildcard src/*.c))
OBJ := $(SRC:%.c=build/%.o) $(KERN_OBJ)
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
	@echo "  test-server-cancel  disconnects cost no CPU, 503 at the cap (no model needed)"
	@echo "  test-server-slots   --slots 4 continuous batching end to end (no model needed)"
	@echo "  bench        per-tensor matvec throughput"
	@echo "  cuda         opt-in CUDA build in build/cuda/ (CUDA_ARCH=sm_89; needs nvcc)"
	@echo "  cuda-test    build and run the CUDA self-test (skips without a device)"
	@echo "  bench-qmat   kernel A/B on synthetic matrices (no model)"
	@echo "  dispatch     which kernels resolved on this CPU (mynah-slm --dispatch)"
	@echo "  bench-decode one decode step for B streams: solo vs batched (S1-c)"
	@echo "  check-x86    cross-compile the AVX2 paths"
	@echo "  test-x86-rosetta  build x86_64 and RUN the suite under Rosetta"
	@echo "  golden-dump  regenerate the oracle's reference activations"
	@echo "  debug        -O0 -g rebuild"
	@echo "  ubsan        UBSan rebuild + test, then clean"
	@echo "  asan         ASan+UBSan rebuild + test (LINUX CI ONLY, see below)"
	@echo "  tsan         ThreadSanitizer: scheduler, queue, pool, batched decode"
	@echo "  leaks        macOS native leak check (no rebuild)"
	@echo "  warnings     the CI -Werror gate, same flags (your CC: see caveat)"
	@echo "  update-ingot refresh the vendored ingot subtree"
	@echo "  install      PREFIX=$(PREFIX)"
	@echo "  dist         relocatable tarball in dist/ (see ARCH_FLAGS first)"

# $(INGOT_LIB) is a real prerequisite, not just order-only. Without it a
# `make update-ingot` (or any local ingot change) rebuilds the archive and
# leaves every binary linked against the PREVIOUS one — silently. Cost an
# entire benchmark that showed a 2.9x kernel win producing no end-to-end
# change, which looked like the kernel being irrelevant rather than absent.
mynah-slm: $(OBJ) build/cli/main.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

SERVER_OBJ := build/server/main.o build/server/http.o build/server/slots.o
mynah-slm-server: $(OBJ) $(SERVER_OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

build/server/%.o: server/%.c $(HDR) $(wildcard server/*.h)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -iquote server -c $< -o $@

# objects in build/ (never next to the sources: the variant builds — ubsan,
# asan — must not pollute the normal one). Test objects also depend on the
# test-side headers (tests/fixture_model.h); listed first so make 3.81, which
# takes the first matching pattern, picks it for them.
build/tests/%.o: tests/%.c $(HDR) $(wildcard tests/*.h)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# the kernel TUs: one source, one object per ISA (see KERN_* above)
build/kern/qmat_%.o: src/qmat_kern.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(KERN_TU_FLAGS) -c $< -o $@
build/kern/attn_%.o: src/attn_kern.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(KERN_TU_FLAGS) -c $< -o $@
build/kern/sgemm_%.o: src/sgemm.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(KERN_TU_FLAGS) -c $< -o $@

# What dispatch resolved on this machine, and why. No model needed.
dispatch: mynah-slm
	@./mynah-slm --dispatch

$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib CC="$(CC)" CFLAGS="-O2 $(ARCH_FLAGS)"

$(OBJ): | $(INGOT_LIB)

# ── tests ──────────────────────────────────────────────────────────────────
# test_ingot needs no model: it pins the container-layer contract (block
# geometry, dequant coverage) so a bad subtree update fails here and not
# three modules later.
TESTS := tests/test_backend tests/test_forward_backend tests/test_batch tests/test_ingot tests/test_sgemm tests/test_threads tests/test_inspect tests/test_kernels tests/test_model tests/test_think tests/test_tokenizer tests/test_tools tests/test_isa tests/test_synth tests/test_http tests/test_sched

# The parity harness is built like the others but driven separately: it dumps
# activations, and tools/eval/compare.py is what judges them.
PARITY     := tests/test_parity
GOLDEN_DIR := tests/golden/it_hello
DUMP_DIR   := build/dump/it_hello
PROMPT     ?= Ciao! Come stai?

# fixture_model.o is the synthetic checkpoint writer (tests/fixture_model.h):
# linked into every test so any of them can build a model with no download.
TEST_SUPPORT := build/tests/npy.o build/tests/fixture_model.o

# The peer-gone probe lives in the server's HTTP layer; its test links it.
tests/test_http: build/tests/test_http.o build/server/http.o $(TEST_SUPPORT) $(OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)
tests/%: build/tests/%.o $(TEST_SUPPORT) $(OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

test: $(TESTS) mynah-slm
	@for t in $(TESTS); do \
	  $$t; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@./mynah-slm --version >/dev/null || exit 1
	@python3 tools/check_plan.py || exit 1
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

# The same discipline on SYNTHETIC matrices of the real shapes, no checkpoint:
# kernel before/after (K3, K4), interleaved in one process with a control row.
# It answers "is the kernel faster here", never "is decode faster".
BENCH_QMAT := tests/bench_qmat
bench-qmat: $(BENCH_QMAT)
	@$(BENCH_QMAT) all
# One decode step for B streams, three ways (B solo steps / one forward_multi
# step with one qmatmat per weight / one with B matvecs per weight), interleaved
# in one process. With no MODEL it writes a 0.6B-GEOMETRY fixture with noise
# weights (~400 MB in TMPDIR) — real shapes, so real costs, no quality claim.
# Not part of `test`: it measures. .work/serving-continuous-batching.md S1-c.
BENCH_DECODE := tests/bench_decode
bench-decode: $(BENCH_DECODE)
	@if [ -e "$(MODEL)" ]; then $(BENCH_DECODE) "$(MODEL)"; else $(BENCH_DECODE); fi

# End-to-end server checks: shape, determinism (sequential and concurrent),
# SSE framing, and that reasoning never reaches content. Separate from `test`
# because it spends a minute of real generation.
test-server: mynah-slm-server
	@bash tests/test_server.sh "$(MODEL)"; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server: model missing"; exit 0; \
	  else exit $$rc; fi

# No zombie work, model-free: a client that leaves (streaming, non-streaming,
# during its prompt, while queued) stops costing CPU at the next step and the
# next request is served at once; the connection cap answers 503 +
# Retry-After. Writes tests/fixture_model.c's "slow" fixture (~25 MB, noise
# weights). About a minute; Linux adds a CPU-idle check from /proc.
test-server-cancel: mynah-slm-server tests/write_fixture
	@bash tests/test_server_cancel.sh; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server-cancel"; exit 0; else exit $$rc; fi

# Continuous batching end to end (--slots 4), model-free on the "slow"
# fixture: concurrent answers == the serialized server's, byte for byte;
# steps really batched; first token beside 3 long streams; 503 + Retry-After
# when slots and queue are full; leavers free their slots. ~2 minutes.
test-server-slots: mynah-slm-server tests/write_fixture
	@bash tests/test_server_slots.sh; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server-slots"; exit 0; else exit $$rc; fi

# Regenerate the oracle's reference activations. Slow (no KV cache, on purpose)
# and only needed when the prompt or the dumped stages change.
golden-dump:
	@test -e "$(MODEL)" || { echo "no model at $(MODEL) — scripts/use_model.sh"; exit 1; }
	@mkdir -p $(GOLDEN_DIR)
	cd tools && uv run python -m oracle.generate ../$(MODEL) \
	  --prompt "$(PROMPT)" --dump-dir ../$(GOLDEN_DIR) -n 1

# ── CUDA, opt-in ───────────────────────────────────────────────────────────
# The default build never needs nvcc and never sees gpu/. `make cuda` builds a
# SEPARATE tree, build/cuda/: every C source again with -DMYNAH_SLM_ENABLE_CUDA
# (which is what lets src/backend.c reach the CUDA backend at all), plus the
# .cu files through nvcc, linked by nvcc so the CUDA runtime comes from the
# toolkit and no -lcuda driver library is named. Structure from mynah-tts
# Makefile:836-922.
#
#   make cuda CUDA_ARCH=sm_89     the CLI and the self-test binary
#   make cuda-test                build and run the self-test (77 = no device)
#
# CUDA_ARCH defaults to `native`, which asks the installed GPU — so on a
# machine without one (every CI runner) it must be named. It is part of the
# object ABI and goes through a stamp file: switching sm_80 -> sm_89 rebuilds
# the .cu objects instead of silently relinking the old cubin.
NVCC      ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O2 -std=c++17
CUDA_BUILD := build/cuda
ifeq ($(CUDA_ARCH),native)
  CUDA_ARCH_FLAGS := -arch=native
else
  CUDA_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
CUDA_HDR        := $(HDR) $(wildcard gpu/cuda/*.h)
CUDA_CU_OBJ     := $(patsubst %.cu,$(CUDA_BUILD)/%.o,$(wildcard gpu/cuda/*.cu))
# + the per-ISA kernel TUs (src/kern.h). They carry no CUDA code, so they are
# shared with the CPU build rather than rebuilt under build/cuda/.
CUDA_SRC_OBJ    := $(SRC:%.c=$(CUDA_BUILD)/%.o) $(KERN_OBJ)
CUDA_TEST_OBJ   := $(CUDA_BUILD)/gpu/cuda/self_test.o $(CUDA_BUILD)/gpu/cuda/test_cuda.o
CUDA_ARCH_STAMP := $(CUDA_BUILD)/.cuda-arch

$(CUDA_BUILD)/%.o: %.c $(CUDA_HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYNAH_SLM_ENABLE_CUDA -iquote gpu/cuda -c $< -o $@

# -ffp-contract=off for the HOST half of each .cu: g++ contracts a*b - c into
# an FMA even in -std=c++17 (GCC only turns contraction off for ISO *C*), so
# on any target with FMA (aarch64, x86 with -mfma) the host check would run a
# different rounding than the device's __fmul_rn/__fsub_rn and the CPU's
# gcc -std=c11 code, and stop being bitwise.
$(CUDA_BUILD)/gpu/cuda/%.o: gpu/cuda/%.cu $(CUDA_HDR) $(CUDA_ARCH_STAMP)
	@mkdir -p $(@D)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc is required for make cuda; install the NVIDIA CUDA toolkit" >&2; exit 2; }
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Isrc -Igpu/cuda -I$(INGOT_DIR)/include -Iinclude \
	  -Xcompiler -Wall,-Wextra,-fPIC,-ffp-contract=off -c $< -o $@

.PHONY: cuda-arch-stamp-force
cuda-arch-stamp-force:
$(CUDA_ARCH_STAMP): cuda-arch-stamp-force
	@mkdir -p $(@D)
	@if test ! -f "$@" || ! grep -Fqx '$(CUDA_ARCH)' "$@"; then printf '%s\n' '$(CUDA_ARCH)' > "$@"; fi

$(CUDA_SRC_OBJ) $(CUDA_TEST_OBJ) $(CUDA_BUILD)/cli/main.o: | $(INGOT_LIB)

$(CUDA_BUILD)/mynah-slm: $(CUDA_SRC_OBJ) $(CUDA_BUILD)/cli/main.o $(CUDA_CU_OBJ) $(INGOT_LIB)
	$(NVCC) $(CUDA_ARCH_FLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

$(CUDA_BUILD)/test_cuda: $(CUDA_SRC_OBJ) $(CUDA_TEST_OBJ) $(CUDA_CU_OBJ) $(INGOT_LIB)
	$(NVCC) $(CUDA_ARCH_FLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

cuda: $(CUDA_BUILD)/mynah-slm $(CUDA_BUILD)/test_cuda
	@echo "CUDA build ready ($(CUDA_ARCH)): $(CUDA_BUILD)/mynah-slm $(CUDA_BUILD)/test_cuda"

# Every CUDA kernel against the CPU backend (gpu/cuda/self_test.c). Without a
# device it says so and skips, like the model-backed tests do.
cuda-test: $(CUDA_BUILD)/test_cuda
	@$(CUDA_BUILD)/test_cuda; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP cuda-test: no CUDA device"; exit 0; else exit $$rc; fi

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

X86_CHECK = $(CC) -target $(X86_TARGET) -std=c11 -O2 -Wall -Wextra -D_DEFAULT_SOURCE -iquote src -Iinclude \
	    -I$(INGOT_DIR)/include -DMYNAH_SLM_BUILD='"x86check"' -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK
check-x86:
	@mkdir -p build/x86
	@for f in $(filter-out $(KERN_SRC),$(X86_SRC)); do \
	  $(X86_CHECK) -mavx2 -mfma -mf16c -c $$f -o build/x86/$$(basename $$f .c).avx2.o || exit 1; \
	done
	@# every x86 kernel TU, each with its own flags: AVX-512 included, which
	@# Rosetta cannot run but a compiler can still check
	@$(foreach t,$(KERN_QMAT_x86),$(X86_CHECK) $(KF_x86_$(t)) -DMYNAH_SLM_KERN_TU=$(t) \
	  -DMYNAH_SLM_KERN_ID=$(KID_$(t)) -c src/qmat_kern.c -o build/x86/qmat_$(t).o &&) true
	@$(foreach t,$(KERN_ATTN_x86),$(X86_CHECK) $(KF_x86_$(t)) -DMYNAH_SLM_KERN_TU=$(t) \
	  -DMYNAH_SLM_KERN_ID=$(KID_$(t)) -c src/attn_kern.c -o build/x86/attn_$(t).o &&) true
	@$(foreach t,$(KERN_SGEMM_x86),$(X86_CHECK) $(KF_x86_$(t)) -DMYNAH_SLM_KERN_TU=$(t) \
	  -DMYNAH_SLM_KERN_ID=$(KID_$(t)) -c src/sgemm.c -o build/x86/sgemm_$(t).o &&) true
	@echo "x86-64 cross-compile OK (baseline avx2 + every x86 kernel TU)"

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

# The exact command safety.yml's `warnings` job runs, so a red gate can be
# reproduced before pushing instead of after. ARCH_FLAGS is emptied and -O2
# appended for the reason that workflow spells out: -march=native and -O3 make
# gcc's interprocedural warnings depend on which runner the job landed on
# rather than on the code. Clean on both sides — these objects are built with
# different flags than the normal build and must not be left to be relinked.
#
# Caveat, and it is precisely why those pushes went red: this runs YOUR
# compiler. On this Mac that is clang, while every warning that has actually
# broken this repo so far (-Wformat-truncation, glibc's fortified snprintf, a
# header glibc needs and libSystem does not) is gcc-on-glibc. Green here is
# necessary, not sufficient — `make warnings CC=gcc-14` if you have it from
# Homebrew, but only Linux CI covers the glibc half.
warnings:
	$(MAKE) clean && $(MAKE) lib ARCH_FLAGS= EXTRA_CFLAGS="-O2 -Werror" && $(MAKE) clean

# clean at the end too: the sanitized objects must NOT be left behind to
# pollute the normal build
ubsan:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="$(SAN_ADD) -O2 -fsanitize=undefined" \
	  EXTRA_LDFLAGS="-fsanitize=undefined" all test && $(MAKE) clean
asan:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="$(SAN_ADD) -O1 -fsanitize=address,undefined" \
	  EXTRA_LDFLAGS="-fsanitize=address,undefined" all test && $(MAKE) clean

# The concurrency the server runs on — the pending queue, the scheduler loop
# with producer threads, and the pool — under ThreadSanitizer. BLAS=none so no
# uninstrumented vendor threads are in the process. Linux/clang or gcc.
tsan:
	$(MAKE) clean && $(MAKE) BLAS=none EXTRA_CFLAGS="$(SAN_ADD) -O1 -fsanitize=thread" \
	  EXTRA_LDFLAGS="-fsanitize=thread" tests/test_sched tests/test_threads tests/test_synth
	TSAN_OPTIONS=halt_on_error=1 tests/test_sched && TSAN_OPTIONS=halt_on_error=1 tests/test_threads \
	  && TSAN_OPTIONS=halt_on_error=1 tests/test_synth; rc=$$?; $(MAKE) clean; exit $$rc

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
	rm -rf build mynah-slm mynah-slm-server libmynah_slm.a libmynah_slm$(SOEXT) $(TESTS) $(PARITY) $(BENCH) $(BENCH_QMAT) $(BENCH_DECODE) tests/write_fixture dist
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

# A relocatable tarball of the two binaries, the static library, the public
# header and the model scripts — what .github/workflows/release.yml uploads to
# a tag, and what anyone can build by hand. Everything lands in dist/, which is
# gitignored. Usage: make dist  (or `make dist MYNAH_SLM_BUILD=v0.1.0` to name
# it after the tag rather than after `git describe`).
#
# Beware ARCH_FLAGS: the default -march=native is right for a local build and
# wrong for a binary somebody else runs. The release workflow overrides it, and
# so should anyone shipping this by hand.
DIST_OS   := $(shell uname -s | tr '[:upper:]' '[:lower:]')
DIST_ARCH := $(shell uname -m)
DIST_NAME := mynah-slm-$(MYNAH_SLM_BUILD)-$(DIST_OS)-$(DIST_ARCH)
DIST_DIR  := dist/$(DIST_NAME)
dist: mynah-slm mynah-slm-server libmynah_slm.a
	@rm -rf $(DIST_DIR)
	@mkdir -p $(DIST_DIR)/bin $(DIST_DIR)/lib $(DIST_DIR)/include $(DIST_DIR)/scripts
	install -m 755 mynah-slm mynah-slm-server $(DIST_DIR)/bin/
	install -m 644 libmynah_slm.a $(DIST_DIR)/lib/
	install -m 644 include/mynah_slm.h $(DIST_DIR)/include/
	install -m 644 LICENSE README.md $(DIST_DIR)/
	install -m 755 scripts/download_model.sh scripts/use_model.sh $(DIST_DIR)/scripts/
	@strip $(DIST_DIR)/bin/mynah-slm $(DIST_DIR)/bin/mynah-slm-server 2>/dev/null || true
	cd dist && tar czf $(DIST_NAME).tar.gz $(DIST_NAME)
	@rm -rf $(DIST_DIR)
	@echo "" && echo "-> dist/$(DIST_NAME).tar.gz"
	@cd dist && shasum -a 256 $(DIST_NAME).tar.gz 2>/dev/null || (cd dist && sha256sum $(DIST_NAME).tar.gz)

.PHONY: all help lib shared cuda cuda-test test test-parity test-server bench check-x86 test-x86-rosetta golden-dump debug ubsan asan leaks warnings clean install dist update-ingot bench-qmat dispatch bench-decode test-server-cancel tsan test-server-slots
