#!/usr/bin/env python3
"""
Benchmarks the GOAL trace decoders of LogGOPSim.

For every trace it compares
  baseline         an unmodified LogGOPSim build on the legacy file (optional)
  fixed            the original hand-written decoder on the legacy file
  interp-legacy    the schema-agnostic interpreter on the legacy file
  interp-goalx     the interpreter on a GOALX file with the default schema
  interp-<name>    the interpreter on a GOALX file with a custom schema

and measures decoder throughput (--bench-decode), end-to-end time split into
load and simulation (--timing), peak RSS and file size. Configurations are run
round-robin, and every result is reported as a median with a nonparametric
95% confidence interval.

Example:
  bench/bench.py --sim new/LogGOPSim --txt2bin new/txt2bin \\
                 --baseline-sim base/LogGOPSim --generate medium --reps 10
"""

import argparse
import csv
import datetime
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCHEDGEN2 = os.path.join(REPO, "src", "Schedgen2", "schedgen.py")

NARROW = "peer:14,tag:22,size:24,cpu:2,nic:1,ndep:12,depix:20,nsdep:4"

# (name, schedgen2 arguments) per preset
PRESETS = {
    "small": [
        ("ring_allreduce_128", ["allreduce", "--ptrn", "ring", "--comm_size", "128", "--datasize", "1048576"]),
        ("alltoall_64", ["alltoall", "--comm_size", "64", "--datasize", "4096"]),
    ],
    "medium": [
        ("ring_allreduce_512", ["allreduce", "--ptrn", "ring", "--comm_size", "512", "--datasize", "1048576"]),
        ("alltoall_256", ["alltoall", "--comm_size", "256", "--datasize", "4096"]),
        ("recdoub_allreduce_4096", ["allreduce", "--ptrn", "recdoub", "--comm_size", "4096", "--datasize", "65536"]),
    ],
    "large": [
        ("ring_allreduce_1024", ["allreduce", "--ptrn", "ring", "--comm_size", "1024", "--datasize", "1048576"]),
        ("alltoall_512", ["alltoall", "--comm_size", "512", "--datasize", "4096"]),
        ("recdoub_allreduce_8192", ["allreduce", "--ptrn", "recdoub", "--comm_size", "8192", "--datasize", "65536"]),
    ],
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(cmd):
    """Runs cmd and returns (exit code, output, wall seconds, peak RSS in bytes)."""
    start = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.read().decode(errors="replace")
    _, status, usage = os.wait4(p.pid, 0)
    wall = time.perf_counter() - start
    p.returncode = os.waitstatus_to_exitcode(status)
    p.stdout.close()
    # ru_maxrss is in bytes on macOS and in KiB on Linux
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return p.returncode, out, wall, rss


def check(cmd):
    code, out, _, _ = run(cmd)
    if code != 0:
        sys.exit("command failed (%d): %s\n%s" % (code, " ".join(cmd), out))
    return out


def parse_kv(line_prefix, out):
    for line in out.splitlines():
        if line.startswith(line_prefix):
            return dict(re.findall(r"(\w+)=(\S+)", line))
    return None


def finish_lines(out):
    return [l for l in out.splitlines() if l.startswith("Maximum finishing time") or l.startswith("Host ")]


def median_ci(values):
    """Median and a nonparametric 95% CI from order statistics (Le Boudec)."""
    xs = sorted(values)
    n = len(xs)
    med = statistics.median(xs)
    if n < 6:
        return med, xs[0], xs[-1]  # too few samples, report the full range
    lo = int(math.floor((n - 1.96 * math.sqrt(n)) / 2))
    hi = int(math.ceil(1 + (n + 1.96 * math.sqrt(n)) / 2))
    return med, xs[max(lo - 1, 0)], xs[min(hi - 1, n - 1)]


def environment(args):
    def first_line(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]
        except Exception:
            return "unknown"

    cpu = platform.processor() or "unknown"
    if sys.platform == "darwin":
        cpu = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.*)", f.read())
            cpu = m.group(1) if m else cpu
    return [
        ("date", datetime.datetime.now().isoformat(timespec="seconds")),
        ("host", platform.node()),
        ("os", "%s %s" % (platform.system(), platform.release())),
        ("cpu", cpu),
        ("compiler", first_line([os.environ.get("CXX", "c++"), "--version"])),
        ("commit", first_line(["git", "-C", REPO, "rev-parse", "--short", "HEAD"])),
        ("uncommitted changes", "yes" if subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                                                        capture_output=True, text=True).stdout.strip() else "no"),
        ("sim", args.sim),
        ("baseline", args.baseline_sim or "-"),
        ("reps", "%d (+%d warmup), decode reps per run %d" % (args.reps, args.warmup, args.decode_reps)),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="LogGOPSim with GOALX support")
    ap.add_argument("--txt2bin", required=True, help="txt2bin with GOALX support")
    ap.add_argument("--baseline-sim", help="unmodified LogGOPSim build to compare against")
    ap.add_argument("--goal", action="append", default=[], help="GOAL text trace (repeatable)")
    ap.add_argument("--generate", choices=sorted(PRESETS), help="generate traces with Schedgen2")
    ap.add_argument("--schema", action="append", default=[],
                    help="extra GOALX schema as name=spec (repeatable); default: narrow=" + NARROW)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--decode-reps", type=int, default=5, help="passes over the trace per --bench-decode run")
    ap.add_argument("--skip-e2e", action="store_true", help="only measure decoder throughput")
    ap.add_argument("--out", default="bench-results")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
    schemas = [tuple(s.split("=", 1)) for s in args.schema] or [("narrow", NARROW)]

    traces = [(os.path.splitext(os.path.basename(g))[0], os.path.abspath(g)) for g in args.goal]
    for name, gen_args in PRESETS.get(args.generate, []):
        path = os.path.join(args.out, "traces", name + ".goal")
        if not os.path.exists(path):
            log("generating %s" % name)
            check([sys.executable, SCHEDGEN2] + gen_args + ["--output", os.path.abspath(path)])
        traces.append((name, path))
    if not traces:
        sys.exit("no traces: use --goal or --generate")

    rows = []
    summary = {}
    debug_build = False

    for tname, goal in traces:
        base = os.path.join(args.out, "traces", tname)
        files = {"legacy": base + ".legacy.bin", "goalx": base + ".goalx.bin"}
        log("converting %s" % tname)
        check([args.txt2bin, "-i", goal, "-o", files["legacy"]])
        check([args.txt2bin, "-i", goal, "-o", files["goalx"], "--format", "goalx"])
        trace_schemas = []
        for sname, spec in schemas:
            path = "%s.%s.bin" % (base, sname)
            code, out, _, _ = run([args.txt2bin, "-i", goal, "-o", path, "--format", "goalx", "--schema", spec])
            if code != 0:
                # the schema is too narrow for this trace: txt2bin refuses instead of truncating
                log("skipping schema %s for %s: %s" % (sname, tname, out.strip().splitlines()[-1]))
                continue
            files[sname] = path
            trace_schemas.append(sname)

        configs = []
        if args.baseline_sim:
            configs.append(("baseline", [args.baseline_sim, "-f", files["legacy"]], "legacy", False))
        configs.append(("fixed", [args.sim, "-f", files["legacy"], "--decoder", "fixed"], "legacy", True))
        configs.append(("interp-legacy", [args.sim, "-f", files["legacy"], "--decoder", "interp"], "legacy", True))
        configs.append(("interp-goalx", [args.sim, "-f", files["goalx"], "--decoder", "interp"], "goalx", True))
        for sname in trace_schemas:
            configs.append(("interp-" + sname, [args.sim, "-f", files[sname], "--decoder", "interp"], sname, True))

        sizes = {k: os.path.getsize(v) for k, v in files.items()}
        res = {c[0]: {"decode": [], "wall": [], "load": [], "sim": [], "rss": []} for c in configs}
        checksums, finishes, records = {}, {}, None

        # decoder throughput, round-robin over configurations
        for rep in range(args.warmup + args.reps):
            for cname, cmd, fmt, new in configs:
                if not new:
                    continue
                code, out, _, _ = run(cmd + ["--bench-decode", str(args.decode_reps)])
                d = parse_kv("DECODE:", out)
                if code != 0 or d is None:
                    sys.exit("decode benchmark failed for %s/%s:\n%s" % (tname, cname, out))
                debug_build |= d["build"] != "optimized"
                checksums[cname] = d["checksum"]
                records = int(d["records"])
                if rep >= args.warmup:
                    res[cname]["decode"].append(float(d["records_per_s"]))
                    rows.append([tname, cname, "decode", rep - args.warmup, "", "", "", "",
                                 d["records_per_s"], sizes[fmt]])
        if len(set(checksums.values())) != 1:
            sys.exit("decoders disagree on %s: %s" % (tname, checksums))

        # end-to-end simulation, round-robin over configurations
        if not args.skip_e2e:
            for rep in range(args.warmup + args.reps):
                for cname, cmd, fmt, new in configs:
                    code, out, wall, rss = run(cmd + ["-b"] + (["--timing"] if new else []))
                    if code != 0:
                        sys.exit("simulation failed for %s/%s:\n%s" % (tname, cname, out))
                    finishes.setdefault(cname, finish_lines(out))
                    t = parse_kv("TIMING:", out) if new else None
                    if rep < args.warmup:
                        continue
                    r = res[cname]
                    r["wall"].append(wall)
                    r["rss"].append(rss)
                    if t:
                        r["load"].append(float(t["load_s"]))
                        r["sim"].append(float(t["sim_s"]))
                    rows.append([tname, cname, "e2e", rep - args.warmup, "%.6f" % wall,
                                 t["load_s"] if t else "", t["sim_s"] if t else "", rss, "", sizes[fmt]])
            if len(set(tuple(v) for v in finishes.values())) != 1:
                sys.exit("simulation results differ on %s: %s" % (tname, finishes))

        summary[tname] = (configs, res, sizes, records)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "config", "kind", "rep", "wall_s", "load_s", "sim_s", "peak_rss_bytes",
                    "records_per_s", "file_bytes"])
        w.writerows(rows)

    md = render(summary, environment(args), args, debug_build)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    log("wrote %s and %s" % (os.path.join(args.out, "results.csv"), os.path.join(args.out, "summary.md")))


def fmt_ci(values, scale=1.0, digits=3):
    med, lo, hi = median_ci(values)
    return "%.*f [%.*f, %.*f]" % (digits, med * scale, digits, lo * scale, digits, hi * scale)


def render(summary, env, args, debug_build):
    out = ["# LogGOPSim decoder benchmark", ""]
    if debug_build:
        out += ["> **Warning:** the measured LogGOPSim was built without optimization. "
                "Rebuild with `-O3 -DNDEBUG` before drawing conclusions.", ""]
    out += ["Medians with nonparametric 95% confidence intervals in brackets "
            "(with fewer than 6 repetitions the brackets show min and max).", ""]
    out += ["| | |", "|---|---|"] + ["| %s | %s |" % kv for kv in env] + [""]

    for tname, (configs, res, sizes, records) in summary.items():
        out += ["## %s" % tname, ""]
        out += ["%s records. File sizes: %s." % (
            "{:,}".format(records) if records else "?",
            ", ".join("%s %s B" % (k, "{:,}".format(v)) for k, v in sizes.items())), ""]

        ref = "fixed"
        out += ["### Decoder throughput (`--bench-decode`)", "",
                "| config | Mrecords/s | vs fixed |", "|---|---|---|"]
        ref_med = median_ci(res[ref]["decode"])[0]
        for cname, _, _, new in configs:
            if not new:
                continue
            vals = res[cname]["decode"]
            out.append("| %s | %s | %.2fx |" % (cname, fmt_ci(vals, 1e-6, 1), median_ci(vals)[0] / ref_med))
        out.append("")

        if args.skip_e2e:
            continue
        ref = "baseline" if "baseline" in res else "fixed"
        ref_med = median_ci(res[ref]["wall"])[0]
        out += ["### End to end (`-b --timing`)", "",
                "| config | wall s | load s | simulate s | peak RSS MiB | speedup vs %s |" % ref,
                "|---|---|---|---|---|---|"]
        for cname, _, _, _ in configs:
            r = res[cname]
            out.append("| %s | %s | %s | %s | %s | %.3fx |" % (
                cname, fmt_ci(r["wall"]),
                fmt_ci(r["load"], digits=4) if r["load"] else "-",
                fmt_ci(r["sim"]) if r["sim"] else "-",
                fmt_ci(r["rss"], 1.0 / 2**20, 1),
                ref_med / median_ci(r["wall"])[0]))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
