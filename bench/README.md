# Decoder benchmarks

`bench.py` compares the ways LogGOPSim can read a GOAL binary trace:

| config | what runs |
|---|---|
| `baseline` | an unmodified LogGOPSim build on the legacy file (optional) |
| `fixed` | the original hand-written decoder (`--decoder fixed`, legacy files only) |
| `interp-legacy` | the schema-agnostic interpreter on the legacy file |
| `interp-goalx` | the interpreter on a GOALX file with the default schema |
| `interp-narrow` | the interpreter on a GOALX file with narrow fields (`--schema name=spec` for others) |

For each trace the harness measures:
- decoder throughput (`LogGOPSim --bench-decode`): every record decoded, nothing simulated
- end-to-end wall time, split into load and simulation (`--timing`)
- peak RSS
- file size

It checks that all decoders produce the same checksum and that every configuration finishes at the same simulated time. Configurations run round-robin, and results are medians with nonparametric 95% confidence intervals. The output goes to `results.csv` (raw runs) and `summary.md`.

## Running

Build both trees optimized; the Makefile defaults to `-O0 -g`, and the summary warns about debug builds:

```sh
git worktree add ../LogGOPSim-baseline <upstream commit>
for d in src/LogGOPSim ../LogGOPSim-baseline/src/LogGOPSim; do
  make -C $d CXXFLAGS="-std=c++11 -O3 -DNDEBUG -Wno-deprecated -I/opt/homebrew/include/"
done

bench/bench.py --sim src/LogGOPSim/LogGOPSim --txt2bin src/LogGOPSim/txt2bin \
               --baseline-sim ../LogGOPSim-baseline/src/LogGOPSim/LogGOPSim \
               --generate medium --reps 10 --out bench-results
```

- `--generate small|medium|large` creates traces with Schedgen2; `--goal file.goal` adds your own.
- `--skip-e2e` measures decoder throughput only.

Use an idle machine, and report the summary's environment block with the numbers. On Linux, pin the run (`taskset -c 2 bench/bench.py ...`) and fix the CPU frequency if you can.
