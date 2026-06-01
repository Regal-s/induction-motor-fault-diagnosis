r"""
Drop-in Python replacement for yroru.m: replays a 6-channel CSV (V1,V2,V3,I1,I2,I3)
out a serial COM port as 14-byte UART frames, paced at Fs Hz, looping continuously.

Wire format (matches the MATLAB script byte-for-byte):
  +------+----+----+----+----+----+----+----+----+----+----+----+----+------+
  | 0xAA | C1hi C1lo C2hi C2lo C3hi C3lo C4hi C4lo C5hi C5lo C6hi C6lo | 0x55 |
  +------+-------------------------------------------------------------+------+
Each 16-bit channel is Q15 int16 big-endian. The Jetson receiver
(realtime/uart_receiver.py) discards V1..V3 and keeps I1..I3.

Run (PowerShell on the laptop):
  pip install pyserial pandas numpy
  python realtime\uart_sender.py --port COM3 --baud 921600 --fs 2400 --csv "<path>"

Add --once to play through and exit, --max-rows N to cap, --speed K to overspeed.
"""
from __future__ import annotations
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import serial  # pyserial
except ImportError:
    serial = None  # checked in main()

SYNC_BYTE = 0xAA
CHECK_BYTE = 0x55
PACKET_LEN = 14
PACKET_FMT = ">BhhhhhhB"
Q15_INT16_MAX = 32767


def to_q15(arr: np.ndarray) -> np.ndarray:
    """Map an arbitrary 1-D numeric array to int16 in [-32768, 32767], mirroring the
    branch logic in yroru.m's data-type detection."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer):
        if a.dtype == np.uint16 or (a.min() >= 0 and a.max() <= 65535 and a.min() >= 32768):
            return (a.astype(np.int32) - 32768).clip(-32768, 32767).astype(np.int16)
        return a.clip(-32768, 32767).astype(np.int16)
    a = a.astype(np.float64)
    if np.all(np.isclose(a, np.round(a))) and a.min() >= -32768 and a.max() <= 32767:
        return np.round(a).astype(np.int16)
    if np.all(np.isclose(a, np.round(a))) and a.min() >= 0 and a.max() <= 65535:
        return (np.round(a).astype(np.int32) - 32768).clip(-32768, 32767).astype(np.int16)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.int16)
    norm = 2.0 * (a - lo) / (hi - lo) - 1.0
    return np.round(norm * Q15_INT16_MAX).clip(-32768, 32767).astype(np.int16)


def build_packets(ch: np.ndarray) -> bytes:
    """ch shape (N, 6) int16 -> contiguous bytes, N * 14 long."""
    N = ch.shape[0]
    pkt = np.empty((N, PACKET_LEN), dtype=np.uint8)
    pkt[:, 0] = SYNC_BYTE
    pkt[:, 13] = CHECK_BYTE
    # int16 -> big-endian (hi, lo) per channel via .view trick (NumPy is little-endian on x86,
    # so view as ('>i2') and frombuffer)
    be = ch.astype('>i2').tobytes()              # N*6 int16 big-endian
    pkt[:, 1:13] = np.frombuffer(be, dtype=np.uint8).reshape(N, 12)
    return pkt.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs", type=float, default=2400.0, help="sample (packet) rate in Hz")
    ap.add_argument("--csv", required=True, help="path to the 6-channel CSV (V1,V2,V3,I1,I2,I3)")
    ap.add_argument("--once", action="store_true", help="play through and exit (default: loop forever)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap to first N rows of the CSV (0 = use all)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (1.0 = real-time)")
    ap.add_argument("--batch", type=int, default=100, help="packets per write")
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed. Run: pip install pyserial")

    csv = Path(args.csv)
    if not csv.is_file():
        sys.exit(f"CSV not found: {csv}")

    print(f"loading {csv}")
    df = pd.read_csv(csv)
    if df.shape[1] < 6:
        sys.exit(f"need at least 6 columns, got {df.shape[1]}")
    if args.max_rows:
        df = df.head(args.max_rows)
    n = len(df)
    cols = list(df.columns[:6])
    print(f"  rows: {n}  columns used: {cols}")

    # convert each column to Q15 int16
    ch = np.stack([to_q15(df[c].to_numpy()) for c in cols], axis=1)   # (N, 6) int16
    print(f"  Q15 range: [{ch.min()}, {ch.max()}]")

    # pre-build the full byte blob once, then slice per batch in the loop
    print("building packets ...")
    blob = build_packets(ch)
    print(f"  {n} packets = {len(blob)} bytes")

    fs_eff = args.fs * args.speed
    pkt_period = 1.0 / fs_eff
    min_baud = int(np.ceil(PACKET_LEN * 10 * fs_eff))
    print(f"effective rate: {fs_eff:.1f} pkt/s (need >= {min_baud} baud, you have {args.baud})")
    if args.baud < min_baud:
        print(f"  WARNING: baud is below minimum -- data will be dropped")

    print(f"opening {args.port} @ {args.baud} baud ...")
    ser = serial.Serial(args.port, args.baud, timeout=2.0, write_timeout=5.0)
    ser.reset_output_buffer()
    time.sleep(0.2)
    print("transmitting (Ctrl+C to stop)")

    total_pkts = 0
    total_bytes = 0
    last_report = time.time()
    start = time.time()

    try:
        loop_count = 0
        while True:
            loop_count += 1
            i = 0
            while i < n:
                t_batch_start = time.time()
                j = min(i + args.batch, n)
                ser.write(blob[i * PACKET_LEN: j * PACKET_LEN])
                count = j - i
                total_pkts += count
                total_bytes += count * PACKET_LEN

                # pace
                target = count * pkt_period
                spent = time.time() - t_batch_start
                if target > spent:
                    time.sleep(target - spent)

                i = j

                now = time.time()
                if now - last_report >= 1.0:
                    el = now - start
                    rate = total_pkts / el if el > 0 else 0.0
                    mbps = total_bytes * 8 / el / 1e6 if el > 0 else 0.0
                    print(f"loop {loop_count:>5d}  pkts {total_pkts:>10d}  "
                          f"rate {rate:8.1f} pkt/s  {mbps:.2f} Mbps")
                    last_report = now

            if args.once:
                break
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try:
            ser.flush()
            ser.close()
        except Exception:
            pass

    el = time.time() - start
    print(f"\nsummary: loops={loop_count}  packets={total_pkts}  bytes={total_bytes}  "
          f"duration={el:.2f}s  actual={total_pkts / el if el > 0 else 0:.1f} pkt/s  "
          f"target={args.fs} pkt/s")


if __name__ == "__main__":
    main()
