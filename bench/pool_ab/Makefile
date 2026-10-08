# Old pool vs current pool, same benchmark source, two binaries.
#
#   make            build both (the old pool is extracted from git, untouched)
#   ./run.sh [thr]  ABAB over 10 rounds, prints the evidence table
#
# OLD_REV is the last commit with the mutex+condvar pool.
ROOT    := ../..
OLD_REV ?= cde1599
CC      ?= cc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -D_DEFAULT_SOURCE
LDLIBS  := -lpthread

all: region_cost_old region_cost_new burst_cpu_old burst_cpu_new

old/threads.c old/threads.h:
	@mkdir -p old
	git -C $(ROOT) show $(OLD_REV):src/threads.c > old/threads.c
	git -C $(ROOT) show $(OLD_REV):src/threads.h > old/threads.h

%_old: %.c old/threads.c old/threads.h
	$(CC) $(CFLAGS) -Iold -o $@ $< old/threads.c $(LDLIBS)

%_new: %.c $(ROOT)/src/threads.c $(ROOT)/src/threads.h
	$(CC) $(CFLAGS) -I$(ROOT)/src -o $@ $< $(ROOT)/src/threads.c $(LDLIBS)

clean:
	rm -rf old region_cost_old region_cost_new burst_cpu_old burst_cpu_new

.PHONY: all clean
