#!/bin/sh
# ABAB: old pool, new pool (spin default), new pool with spin 0, alternating
# order every round, 10 rounds. Prints a markdown table: median, min..max and
# CV per side, and in how many rounds the new pool won.
#
#   ./run.sh [threads]      default 4
#
# State the machine, the governor and whether anything else was running next
# to the table; a number without that is not a measurement.
set -eu
cd "$(dirname "$0")"
THR=${1:-4}
make -s all
python3 -I - "$THR" <<'PY'
import os, statistics as st, subprocess, sys
thr = sys.argv[1]
cfg = [("old", "./region_cost_old", {}), ("new", "./region_cost_new", {}),
       ("new_spin0", "./region_cost_new", {"MYNAH_SLM_POOL_SPIN_US": "0"})]
cpu = [("old", "./burst_cpu_old", {}), ("new", "./burst_cpu_new", {})]
res = {}
for rnd in range(10):
    for name, exe, env in (cfg if rnd % 2 == 0 else cfg[::-1]) + (cpu if rnd % 2 == 0 else cpu[::-1]):
        e = dict(os.environ); e.update(env)
        out = subprocess.run([exe, thr], capture_output=True, text=True, env=e, check=True).stdout
        tag = "cpu_" + name if "burst" in exe else name
        for line in out.splitlines():
            k, v = line.split()
            res.setdefault((tag, k), []).append(float(v))
f = lambda v: "%.1f (%.1f..%.1f, CV %.0f%%)" % (st.median(v), min(v), max(v), 100 * st.pstdev(v) / st.mean(v))
print("| region | thr | serial us | old us | new us | new spin=0 us | old/new | new wins |")
print("|---|---|---|---|---|---|---|---|")
for k in ("empty", "tiny", "small", "medium"):
    o, n, z = res[("old", k)], res[("new", k)], res[("new_spin0", k)]
    ser = "%.1f" % st.median(res[("old", "serial_" + k)]) if k != "empty" else "-"
    wins = sum(b < a for a, b in zip(o, n))
    print("| %s | %s | %s | %s | %s | %s | %.2fx | %d/10 |" % (k, thr, ser, f(o), f(n), f(z), st.median(o) / st.median(n), wins))
print()
print("| burst (225 regions + 1 ms gap) | old | new |")
print("|---|---|---|")
for k in ("wall_ms_per_step", "cpu_ms_per_step"):
    print("| %s | %s | %s |" % (k, f(res[("cpu_old", k)]), f(res[("cpu_new", k)])))
PY
