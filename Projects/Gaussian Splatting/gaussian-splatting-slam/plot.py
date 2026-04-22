import argparse
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def read_iter_time_csv(path):
    # Expected header: wall_time_s,iter,iter_ms
    # Example row: 1763809107.388539,1,266.363647
    wall_s = []
    iters = []
    iter_ms = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                wall_s.append(float(parts[0]))
                iters.append(int(float(parts[1])))
                iter_ms.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(wall_s), np.array(iters), np.array(iter_ms)


def parse_nvidia_timestamp(ts_str):
    # Format: "YYYY/MM/DD HH:MM:SS.mmm"
    # Example: "2025/11/22 01:06:10.563"
    return datetime.strptime(ts_str.strip(), "%Y/%m/%d %H:%M:%S.%f")


def read_util_csv(path):
    # Expected rows like:
    # 2025/11/22 01:06:10.563, 94
    # or sometimes: "... , 94 %"
    t = []
    u = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            ts = parts[0]
            util_str = parts[1]
            # keep digits only
            digits = "".join(ch for ch in util_str if ch.isdigit())
            if digits == "":
                continue
            try:
                dt = parse_nvidia_timestamp(ts)
                util = int(digits)
            except Exception:
                continue
            t.append(dt)
            u.append(util)

    # Convert datetimes to relative seconds from start
    if not t:
        raise RuntimeError("No valid rows parsed from util_log.csv")
    t0 = t[0]
    t_rel = np.array([(ti - t0).total_seconds() for ti in t], dtype=np.float64)
    u = np.array(u, dtype=np.float64)
    return t_rel, u


def moving_average(x, win):
    if win <= 1:
        return x
    win = int(win)
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, kernel, mode="same")


def nearest_util_at_wall_times(util_t_rel, util_u, iter_wall_s):
    # Align util samples to iteration wall times using nearest neighbor.
    # util_t_rel is relative seconds from util start. iter_wall_s is unix seconds.
    # We'll convert iter times to relative seconds from util start by:
    #   iter_rel = iter_wall_s - iter_wall_s[0] + offset
    # But better: align using absolute time. We don't have util absolute unix time here.
    # So we align approximately by shifting iter times so both start at 0.
    iter_rel = iter_wall_s - iter_wall_s[0]
    # Now nearest neighbor match
    idx = np.searchsorted(util_t_rel, iter_rel, side="left")
    idx = np.clip(idx, 0, len(util_t_rel) - 1)
    # choose closer of idx and idx-1
    idx2 = np.clip(idx - 1, 0, len(util_t_rel) - 1)
    choose_idx = np.where(
        np.abs(util_t_rel[idx] - iter_rel) < np.abs(util_t_rel[idx2] - iter_rel),
        idx,
        idx2,
    )
    return iter_rel, util_u[choose_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter_csv", default="iter_time.csv")
    ap.add_argument("--util_csv", default="util_log.csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--iter_smooth", type=int, default=200, help="rolling window (iters) for iter_ms smoothing")
    ap.add_argument("--util_smooth", type=int, default=25, help="rolling window (samples) for util smoothing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    wall_s, iters, iter_ms = read_iter_time_csv(args.iter_csv)
    util_t, util_u = read_util_csv(args.util_csv)

    # ---------- Plot 1: iter_ms vs iteration ----------
    plt.figure()
    plt.plot(iters, iter_ms)
    if len(iter_ms) >= args.iter_smooth:
        sm = moving_average(iter_ms, args.iter_smooth)
        plt.plot(iters, sm)
    plt.xlabel("Iteration")
    plt.ylabel("GPU time per iteration (ms)")
    plt.title("GS training: per-iteration GPU time")
    plt.tight_layout()
    p1 = os.path.join(args.outdir, "gs_iter_time_ms.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # ---------- Plot 2: utilization vs time ----------
    plt.figure()
    plt.plot(util_t, util_u)
    if len(util_u) >= args.util_smooth:
        smu = moving_average(util_u, args.util_smooth)
        plt.plot(util_t, smu)
    plt.xlabel("Time since util log start (s)")
    plt.ylabel("GPU utilization (%)")
    plt.ylim(0, 100)
    plt.title("GS training: GPU utilization over time")
    plt.tight_layout()
    p2 = os.path.join(args.outdir, "gs_utilization.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    # ---------- Plot 3: iter_ms and aligned util (approx) ----------
    iter_rel_s, util_at_iter = nearest_util_at_wall_times(util_t, util_u, wall_s)

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(iters, iter_ms)
    if len(iter_ms) >= args.iter_smooth:
        ax1.plot(iters, moving_average(iter_ms, args.iter_smooth))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("GPU time per iteration (ms)")

    ax2 = ax1.twinx()
    ax2.plot(iters, util_at_iter)
    ax2.set_ylabel("GPU utilization (%)")
    ax2.set_ylim(0, 100)

    plt.title("GS training: iteration time + utilization (aligned approx)")
    plt.tight_layout()
    p3 = os.path.join(args.outdir, "gs_iter_time_vs_util.png")
    plt.savefig(p3, dpi=200)
    plt.close()

    # ---------- Print summary ----------
    def stats(name, x):
        x = x[np.isfinite(x)]
        print(f"\n{name}:")
        print(f"  count = {len(x)}")
        print(f"  mean  = {np.mean(x):.3f}")
        print(f"  std   = {np.std(x):.3f}")
        print(f"  min   = {np.min(x):.3f}")
        print(f"  p50   = {np.percentile(x, 50):.3f}")
        print(f"  p90   = {np.percentile(x, 90):.3f}")
        print(f"  p99   = {np.percentile(x, 99):.3f}")
        print(f"  max   = {np.max(x):.3f}")

    stats("iter_ms (ms)", iter_ms)
    stats("util (%)", util_u)

    print("\nSaved plots:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print("\nIf Thor has no GUI, copy plots to your machine:")
    print(f"  scp -r {os.path.abspath(args.outdir)} <your_machine>:.")


if __name__ == "__main__":
    main()
