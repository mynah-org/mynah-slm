# Standalone. Links libmynah_slm only for the production baselines.
ROOT   := ../..
CC     ?= cc
CFLAGS := -std=c11 -O3 -march=native -Wall -Wextra -I. -I$(ROOT)/src \
          -I$(ROOT)/third_party/ingot/include
LDLIBS := $(ROOT)/libmynah_slm.a $(ROOT)/third_party/ingot/libingot.a \
          -framework Accelerate -lpthread -lm

ternary_gemv: main.c kernels.c formats.c gguf_bench.c
	$(MAKE) -C $(ROOT) lib
	$(CC) $(CFLAGS) -o $@ main.c kernels.c formats.c gguf_bench.c $(LDLIBS)

clean:
	rm -f ternary_gemv
.PHONY: clean
