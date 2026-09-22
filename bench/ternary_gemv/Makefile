# Standalone. Links libmynah_slm only for the production baselines.
#
# Two targets, one source tree:
#   make              -- host (Apple Silicon or Linux/aarch64), sdot arm
#   make axion        -- Neoverse V2: sdot AND smmla arms, both compiled in
#
# The smmla arm is compiled for armv8.2+i8mm and CANNOT run on an M1
# (FEAT_I8MM=0 -> SIGILL, not a slow fallback), so the host build leaves it
# out and `test_pair` gates its LAYOUT instead -- see kernels_i8mm.c.
ROOT   := ../..
CC     ?= cc
UNAME  := $(shell uname -s)

CFLAGS := -std=c11 -O3 -Wall -Wextra -I. -I$(ROOT)/src \
          -I$(ROOT)/third_party/ingot/include
LDLIBS := $(ROOT)/libmynah_slm.a $(ROOT)/third_party/ingot/libingot.a -lpthread -lm
ifeq ($(UNAME),Darwin)
  LDLIBS += -framework Accelerate
endif

SRC    := main.c kernels.c formats.c gguf_bench.c caps.c
ARCH   ?= -march=native
DEFS   ?= -DTERNARY_BUILD_DOTPROD

ternary_gemv: $(SRC) kernels_i8mm.c
	$(MAKE) -C $(ROOT) lib
	$(CC) $(CFLAGS) $(ARCH) $(DEFS) -o $@ $(SRC) kernels_i8mm.c $(LDLIBS)

# Neoverse V2 / Google Axion. -mcpu rather than -march=native so the binary is
# reproducible and the ISA it was built for is stated, not inferred.
axion:
	$(MAKE) ARCH="-mcpu=neoverse-v2" \
	        DEFS="-DTERNARY_BUILD_DOTPROD -DTERNARY_BUILD_I8MM" ternary_gemv

# Runs everywhere, including on a CPU with no i8mm: validates the row-pair
# LAYOUT the smmla kernel reads, bit-for-bit against the T0 oracle.
test_pair: test_pair.c kernels.c kernels_i8mm.c formats.c caps.c
	$(CC) $(CFLAGS) $(ARCH) $(DEFS) -o $@ $^ -lm

clean:
	rm -f ternary_gemv test_pair
.PHONY: clean axion
