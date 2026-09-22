# Standalone. Links libmynah_slm only for the production baselines.
#
# Three targets, one source tree:
#   make              -- host: ARM sdot or x86 AVX-512 VNNI, auto-detected
#   make axion        -- Neoverse V2: sdot AND smmla arms, both compiled in
#   make x86          -- Zen 4 / Ice Lake+: AVX-512 VNNI arm
#
# An arm is compiled in by a -D and selected at RUN TIME from the OS-reported
# capabilities. The smmla arm cannot run on an M1 (FEAT_I8MM=0 -> SIGILL, not a
# slow fallback) and the AVX-512 arm cannot run on a pre-Ice-Lake x86, so
# neither is ever chosen from an #ifdef -- see caps.c.
ROOT   := ../..
CC     ?= cc
UNAME  := $(shell uname -s)
MACH   := $(shell uname -m)

CFLAGS := -std=c11 -O3 -Wall -Wextra -I. -I$(ROOT)/src \
          -I$(ROOT)/third_party/ingot/include
LDLIBS := $(ROOT)/libmynah_slm.a $(ROOT)/third_party/ingot/libingot.a -lpthread -lm
ifeq ($(UNAME),Darwin)
  LDLIBS += -framework Accelerate
else
  LDLIBS += $(shell pkg-config --libs openblas 2>/dev/null)
endif

SRC    := main.c kernels.c formats.c gguf_bench.c caps.c kernels_i8mm.c \
          kernels_avx512.c
ARCH   ?= -march=native
ifeq ($(MACH),x86_64)
  DEFS ?= -DTERNARY_BUILD_AVX512
else
  DEFS ?= -DTERNARY_BUILD_DOTPROD
endif

ternary_gemv: $(SRC)
	$(MAKE) -C $(ROOT) lib
	$(CC) $(CFLAGS) $(ARCH) $(DEFS) -o $@ $(SRC) $(LDLIBS)

# Neoverse V2 / Google Axion. -mcpu rather than -march=native so the binary is
# reproducible and the ISA it was built for is stated, not inferred.
axion:
	$(MAKE) ARCH="-mcpu=neoverse-v2" \
	        DEFS="-DTERNARY_BUILD_DOTPROD -DTERNARY_BUILD_I8MM" ternary_gemv

# AMD Zen 4 / Intel Ice Lake and later: AVX-512 with VNNI.
x86:
	$(MAKE) ARCH="-march=native -mavx512f -mavx512bw -mavx512vl -mavx512vnni" \
	        DEFS="-DTERNARY_BUILD_AVX512" ternary_gemv

# Runs everywhere, including on a CPU with no i8mm: validates the row-pair
# LAYOUT the smmla kernel reads, bit-for-bit against the T0 oracle.
test_pair: test_pair.c kernels.c kernels_i8mm.c formats.c caps.c
	$(CC) $(CFLAGS) $(ARCH) $(DEFS) -o $@ $^ -lm

clean:
	rm -f ternary_gemv test_pair
.PHONY: clean axion x86
