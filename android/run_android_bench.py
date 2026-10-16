"""
where-teacher-acts: on-device latency of the deployed graphs on a USB-connected Android phone, from the laptop.

Same protocol as the desktop benchmark (pipeline/27_bench_latency.py): every graph is timed round robin for
--rounds rounds, each round giving, per graph and thread setting (1 and 4 intra-op threads), the median
of 200 timed runs after 30 warm-up runs; the reported latency is the median over rounds of the per-round
medians, with the minimum and maximum over rounds, and the 95th percentile is the median over rounds of
the per-round 95th percentiles. The benchmark program (qj_bench, built by build.sh against the ONNX
Runtime Android package of the desktop version) runs on the phone's fastest cores (taskset); the battery
temperature and the thermal status are logged before and after every round, and the phone rests
--cooldown seconds between rounds. After the rounds, each graph is loaded and run once more in a process of its
own (5 warm-up and 20 runs, one thread) to read the process's peak resident memory (VmHWM), next to the same
reading for a process that loads no graph.

Inputs: a folder of graphs with a manifest.csv (pipeline/38_export_latency_graphs.py writes both). Outputs,
refused if present unless --force: latency_android.csv (one row per graph, the desktop file's columns plus
the device), latency_android_rounds.csv (every round) and latency_android_device.json.

    python android/run_android_bench.py --graphs <dir with manifest.csv> [--rounds 5] [--tag main]

Nothing on the phone is touched outside /data/local/tmp/qj_bench, which is removed at the end unless
--keep-on-device.
"""
import os
import re
import sys
import csv
import json
import time
import shutil
import argparse
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE = "/data/local/tmp/qj_bench"
ARM_PARTS = {"0xd03": "Cortex-A53", "0xd04": "Cortex-A35", "0xd05": "Cortex-A55", "0xd07": "Cortex-A57",
             "0xd08": "Cortex-A72", "0xd09": "Cortex-A73", "0xd0a": "Cortex-A75", "0xd0b": "Cortex-A76",
             "0xd0d": "Cortex-A77", "0xd41": "Cortex-A78", "0xd44": "Cortex-X1", "0xd46": "Cortex-A510",
             "0xd47": "Cortex-A710", "0xd48": "Cortex-X2", "0xd4d": "Cortex-A715", "0xd4e": "Cortex-X3",
             "0xd80": "Cortex-A520", "0xd81": "Cortex-A720", "0xd82": "Cortex-X4"}
THREADS = (1, 4)
FIELDS = ["student", "network", "precision", "graph", "params_backbone_M", "params_deployed_M", "size_MB"]
for _t in THREADS:
    FIELDS += [f"lat_t{_t}_median_ms", f"lat_t{_t}_min_ms", f"lat_t{_t}_max_ms", f"lat_t{_t}_p95_ms"]
FIELDS += ["peak_rss_MB", "rounds", "iters", "device", "soc", "android", "cores", "temp_min_C", "temp_max_C",
           "status", "error"]


def default_adb():
    sdk = os.environ.get("ANDROID_HOME") or os.path.expanduser("~/Library/Android/sdk")
    p = os.path.join(sdk, "platform-tools", "adb")
    return p if os.path.exists(p) else (shutil.which("adb") or "adb")


class Device:
    def __init__(self, adb, serial):
        self.base = [adb] + (["-s", serial] if serial else [])

    def run(self, *args, check=True, timeout=None):
        r = subprocess.run(self.base + list(args), capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"adb {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout

    def sh(self, cmd, check=True, timeout=None):
        return self.run("shell", cmd, check=check, timeout=timeout)

    def prop(self, name):
        return self.sh(f"getprop {name}", check=False).strip()


def pick_device(adb, serial):
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    devs = [ln.split("\t") for ln in out.splitlines()[1:] if "\t" in ln]
    ready = [d[0] for d in devs if d[1] == "device"]
    if serial:
        if serial not in ready:
            sys.exit(f"device {serial} is not ready: {devs}")
        return serial
    if not ready:
        hint = " (unauthorized: accept the USB debugging prompt on the phone)" if any(
            d[1] == "unauthorized" for d in devs) else ""
        sys.exit(f"no Android device ready{hint}; adb devices printed:\n{out}")
    if len(ready) > 1:
        sys.exit(f"several devices are connected, choose one with --serial: {ready}")
    return ready[0]


def device_info(dev):
    info = {k: dev.prop(p) for k, p in (
        ("manufacturer", "ro.product.manufacturer"), ("model", "ro.product.model"),
        ("soc_manufacturer", "ro.soc.manufacturer"), ("soc_model", "ro.soc.model"),
        ("hardware", "ro.hardware"), ("board", "ro.product.board"), ("android", "ro.build.version.release"),
        ("sdk", "ro.build.version.sdk"), ("abi", "ro.product.cpu.abi"), ("fingerprint", "ro.build.fingerprint"))}
    freqs = {}
    for ln in dev.sh("for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do "
                     "echo $f $(cat $f); done", check=False).splitlines():
        m = re.match(r".*/cpu(\d+)/cpufreq/cpuinfo_max_freq (\d+)", ln.strip())
        if m:
            freqs[int(m.group(1))] = int(m.group(2))
    info["cpu_max_khz"] = freqs
    # the core type of every core, from /proc/cpuinfo (ARM part numbers)
    parts, cur = {}, None
    for ln in dev.sh("cat /proc/cpuinfo", check=False).splitlines():
        k, _, v = ln.partition(":")
        if k.strip() == "processor" and v.strip().isdigit():
            cur = int(v.strip())
        elif k.strip() == "CPU part" and cur is not None:
            parts[cur] = ARM_PARTS.get(v.strip().lower(), v.strip())
    info["cpu_part"] = parts
    info["page_size"] = dev.sh("getconf PAGE_SIZE", check=False).strip()
    return info


def fastest_cores(freqs, n=4):
    """The n cores with the highest maximum frequency (ties broken by the higher core number, where phones
    put their large cores), as a sorted list and a taskset mask."""
    if not freqs:
        return [], None
    cores = sorted(sorted(freqs, key=lambda c: (freqs[c], c), reverse=True)[:n])
    return cores, format(sum(1 << c for c in cores), "x")


def battery_temp_c(dev):
    m = re.search(r"temperature:\s*(-?\d+)", dev.sh("dumpsys battery", check=False))
    return int(m.group(1)) / 10 if m else float("nan")


def thermal_status(dev):
    m = re.search(r"Thermal Status:\s*(\d+)", dev.sh("dumpsys thermalservice", check=False))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="folder with the .onnx graphs and manifest.csv")
    ap.add_argument("--bin", default=os.path.join(HERE, "build"), help="folder with qj_bench and libonnxruntime.so")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--cooldown", type=int, default=90, help="seconds of rest between rounds")
    ap.add_argument("--max-temp", type=float, default=40.0,
                    help="wait (up to 10 min) before a round until the battery is at or below this temperature")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default=None, help="default reports_qj/bench/<tag>")
    ap.add_argument("--adb", default=default_adb())
    ap.add_argument("--serial", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-on-device", action="store_true")
    ap.add_argument("--no-memory", action="store_true", help="skip the peak-memory pass")
    a = ap.parse_args()

    out_dir = a.out_dir or os.path.join(ROOT, "reports_qj", "bench", a.tag)
    out_csv = os.path.join(out_dir, "latency_android.csv")
    rounds_csv = os.path.join(out_dir, "latency_android_rounds.csv")
    dev_json = os.path.join(out_dir, "latency_android_device.json")
    for p in (out_csv, rounds_csv, dev_json):
        if os.path.exists(p) and not a.force:
            sys.exit(f"{p} exists; pass --force to overwrite it")
    manifest = list(csv.DictReader(open(os.path.join(a.graphs, "manifest.csv"))))
    files = [m["file"] for m in manifest]
    missing = [f for f in files if not os.path.exists(os.path.join(a.graphs, f))]
    for f in ("qj_bench", "libonnxruntime.so"):
        if not os.path.exists(os.path.join(a.bin, f)):
            missing.append(os.path.join(a.bin, f))
    if missing:
        sys.exit(f"missing inputs: {missing[:5]}")

    dev = Device(a.adb, pick_device(a.adb, a.serial))
    info = device_info(dev)
    if info["abi"] != "arm64-v8a":
        sys.exit(f"the benchmark is built for arm64-v8a, the device reports {info['abi']}")
    cores, mask = fastest_cores(info["cpu_max_khz"])
    info.update(cores=cores, taskset_mask=mask, onnxruntime="1.26.0 (Android package, CPU execution provider)",
                protocol={"rounds": a.rounds, "warmup": a.warmup, "iters": a.iters, "threads": THREADS,
                          "size": a.size, "cooldown_s": a.cooldown, "max_temp_C": a.max_temp})
    print(f"[android] {info['manufacturer']} {info['model']} ({info['soc_manufacturer']} {info['soc_model']}), "
          f"Android {info['android']} (SDK {info['sdk']}), page size {info['page_size']}, "
          f"cores {cores} (mask {mask}), max kHz {info['cpu_max_khz']}", flush=True)

    dev.sh(f"rm -rf {REMOTE} && mkdir -p {REMOTE}/graphs")
    dev.run("push", os.path.join(a.bin, "qj_bench"), f"{REMOTE}/qj_bench")
    dev.run("push", os.path.join(a.bin, "libonnxruntime.so"), f"{REMOTE}/libonnxruntime.so")
    for f in files:
        dev.run("push", os.path.join(a.graphs, f), f"{REMOTE}/graphs/{f}")
    dev.sh(f"chmod 755 {REMOTE}/qj_bench")
    print(f"[android] pushed the benchmark and {len(files)} graphs to {REMOTE}", flush=True)

    pin = f"taskset {mask} " if mask else ""
    cmd = (f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE} {pin}./qj_bench --threads {','.join(map(str, THREADS))} "
           f"--warmup {a.warmup} --iters {a.iters} --size {a.size} --search "
           + " ".join(f"graphs/{f}" for f in files))
    per_round, temps, peak_mb = [], [], {}
    t_start = time.time()
    try:
        for rnd in range(1, a.rounds + 1):
            waited = 0
            while battery_temp_c(dev) > a.max_temp and waited < 600:
                time.sleep(30)
                waited += 30
            before, status_before = battery_temp_c(dev), thermal_status(dev)
            t0 = time.time()
            out = dev.sh(cmd, timeout=6 * 3600)
            after, status_after = battery_temp_c(dev), thermal_status(dev)
            temps += [before, after]
            rows = list(csv.DictReader(out.splitlines()))
            for r in rows:
                r.update(round=rnd, temp_before_C=before, temp_after_C=after, thermal_before=status_before,
                         thermal_after=status_after)
            per_round += rows
            bad = [r for r in rows if r["status"] != "OK"]
            print(f"   round {rnd}/{a.rounds}: {len(rows)} rows, {len(bad)} failed, {time.time() - t0:.0f} s, "
                  f"battery {before:.1f} -> {after:.1f} C, thermal status {status_before} -> {status_after}"
                  + (f"; first failure: {bad[0]['graph']} {bad[0]['error']}" if bad else ""), flush=True)
            if rnd < a.rounds:
                time.sleep(a.cooldown)
        if not a.no_memory:
            def peak(args):
                rows_m = list(csv.DictReader(dev.sh(f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE} {pin}./qj_bench "
                                                    f"--threads 1 --warmup 5 --iters 20 --size {a.size} --peak-memory "
                                                    + args, timeout=1800).splitlines()))
                vals = [r for r in rows_m if r["graph"] == "peak_rss_kb"]
                return float(vals[0]["median_ms"]) / 1024 if vals else float("nan")
            info["peak_rss_MB_no_graph"] = round(peak(""), 1)
            for f in files:
                peak_mb[f] = round(peak(f"graphs/{f}"), 1)
            print(f"   peak memory: no graph {info['peak_rss_MB_no_graph']} MB, graphs "
                  f"{min(peak_mb.values()):.1f} to {max(peak_mb.values()):.1f} MB", flush=True)
    finally:
        if not a.keep_on_device:
            dev.sh(f"rm -rf {REMOTE}", check=False)
    info["minutes"] = round((time.time() - t_start) / 60, 1)

    by = {}
    for r in per_round:
        by.setdefault((r["graph"], int(r["threads"])), []).append(r)
    out = []
    dev_name = f"{info['manufacturer']} {info['model']}".strip()
    soc = f"{info['soc_manufacturer']} {info['soc_model']}".strip() or info["hardware"]
    for m in manifest:
        row = {k: m.get(k, "") for k in FIELDS if k in m}
        row.update(peak_rss_MB=peak_mb.get(m["file"], ""),
                   rounds=a.rounds, iters=a.iters, device=dev_name, soc=soc, android=info["android"],
                   cores=" ".join(map(str, cores)), temp_min_C=min(temps), temp_max_C=max(temps),
                   status="SUCCESS", error="")
        for t in THREADS:
            rs = by.get((f"graphs/{m['file']}", t), [])
            ok = [r for r in rs if r["status"] == "OK"]
            if len(ok) != a.rounds:
                row["status"] = "FAILED"
                row["error"] = (rs[0]["error"] if rs and rs[0]["error"] else f"{len(ok)} of {a.rounds} rounds")
                continue
            med = [float(r["median_ms"]) for r in ok]
            row[f"lat_t{t}_median_ms"] = round(st.median(med), 2)
            row[f"lat_t{t}_min_ms"] = round(min(med), 2)
            row[f"lat_t{t}_max_ms"] = round(max(med), 2)
            row[f"lat_t{t}_p95_ms"] = round(st.median(float(r["p95_ms"]) for r in ok), 2)
        out.append(row)
    srch = [float(r["median_ms"]) for r in per_round if r["graph"] == "gallery_search_9804x512" and r["status"] == "OK"]
    if srch:
        out.append({"student": "-", "network": "gallery search, 9804 x 512, C++ loop", "precision": "FP32",
                    "lat_t1_median_ms": round(st.median(srch), 3), "lat_t1_min_ms": round(min(srch), 3),
                    "lat_t1_max_ms": round(max(srch), 3), "rounds": a.rounds, "iters": a.iters,
                    "device": dev_name, "soc": soc, "android": info["android"], "status": "SUCCESS"})

    os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    with open(rounds_csv, "w", newline="") as f:
        keys = list(per_round[0].keys()) if per_round else []
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_round)
    with open(dev_json, "w") as f:
        json.dump(info, f, indent=2, default=str)
    failed = [r for r in out if r["status"] != "SUCCESS"]
    for r in out:
        if r.get("network"):
            print(f"   {r['status']:7s} {r.get('student', ''):24s} {r.get('network', ''):8s} {r.get('precision', '')}  "
                  f"t1 {r.get('lat_t1_median_ms', '')} ms  t4 {r.get('lat_t4_median_ms', '')} ms {r.get('error', '')}")
    print(f"\n{len(out)} rows, {len(failed)} failed, {info['minutes']} min, battery {min(temps):.1f} to "
          f"{max(temps):.1f} C -> {out_csv}\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
