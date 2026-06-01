r"""
UART sender for ibarram recordings -- reuses the same 14-byte wire protocol as
uart_sender.py / yroru.m, but reads the signal from the existing ibarram windows.npy +
window_labels.csv (via realtime.streamio.reconstruct_recording) instead of a CSV.

Voltage channels (V1/V2/V3) are zeroed (the ibarram set is current-only); the receiver
keeps only channels 4..6 (I1/I2/I3) anyway. Wire rate defaults to fs=1000 (ibarram native).

Run from the laptop:
  python realtime\uart_sender_ibarram.py --port COM3 --baud 921600 \
      --case ibarram_ITSC_B30_Repetition05 --speed 1.0
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from math import gcd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from realtime.streamio import reconstruct_recording   # uses experimental/ibarram/
from realtime.uart_sender import build_packets, PACKET_LEN, to_q15

try:
    import serial
except ImportError:
    serial = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs", type=float, default=5000.0,
                    help="wire sample rate (ibarram native = 1000; default 5000 simulates a real embedded ADC)")
    ap.add_argument("--fs-native", type=float, default=1000.0,
                    help="ibarram source rate; data is resampled from this to --fs before transmitting")
    ap.add_argument("--case", required=True, help="e.g. ibarram_ITSC_B30_Repetition05")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--loop", action="store_true", help="replay forever (default: once)")
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed. Run: pip install pyserial")

    print(f"loading {args.case} ...")
    sig, truth = reconstruct_recording(args.case)                 # (3, L), {label, severity_pct, phase}
    print(f"  reconstructed at native {args.fs_native:.0f} Hz: shape={sig.shape}, truth={truth}")

    # Resample from native rate (1000 Hz) to wire rate (e.g. 5000 Hz)
    if abs(args.fs - args.fs_native) > 1e-6:
        g = gcd(int(round(args.fs)), int(round(args.fs_native)))
        up = int(round(args.fs)) // g
        down = int(round(args.fs_native)) // g
        sig = resample_poly(sig, up=up, down=down, axis=-1).astype(np.float32)
        print(f"  resampled {args.fs_native:.0f} -> {args.fs:.0f} Hz (poly {up}/{down}); new shape={sig.shape}")
    L = sig.shape[1]

    # Build 6-channel int16 packet matrix: V1..V3 zero, I1..I3 = sig (Q15)
    # ibarram windows are normalised to roughly [-1, +1], so map to int16 Q15.
    I = np.clip(sig.T, -1.0, 1.0)                                  # (L, 3) float32
    I_q15 = np.round(I * 32767).astype(np.int16)                   # (L, 3) int16
    V_q15 = np.zeros((L, 3), dtype=np.int16)                       # V1..V3 = 0
    ch6 = np.hstack([V_q15, I_q15])                                # (L, 6)
    print(f"  Q15 currents range: [{I_q15.min()}, {I_q15.max()}]")

    blob = build_packets(ch6)
    print(f"  packets: {L}  bytes: {len(blob)}")

    fs_eff = args.fs * args.speed
    pkt_period = 1.0 / fs_eff
    print(f"opening {args.port} @ {args.baud} baud  (eff rate {fs_eff:.1f} pkt/s)")
    s = serial.Serial(args.port, args.baud, timeout=2.0, write_timeout=5.0)
    s.reset_output_buffer(); time.sleep(0.2)
    print("transmitting ...")

    total = 0
    start = time.time(); last_rep = start
    try:
        loop = 0
        while True:
            loop += 1
            i = 0
            while i < L:
                t0 = time.time()
                j = min(i + args.batch, L)
                s.write(blob[i * PACKET_LEN: j * PACKET_LEN])
                count = j - i
                total += count
                target = count * pkt_period
                spent = time.time() - t0
                if target > spent:
                    time.sleep(target - spent)
                i = j
                now = time.time()
                if now - last_rep >= 1.0:
                    el = now - start
                    print(f"  loop {loop} pkts {total} ({total/el:.1f}/s)")
                    last_rep = now
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try: s.flush(); s.close()
        except Exception: pass

    el = time.time() - start
    print(f"sent {total} packets in {el:.2f}s ({total/el:.1f} pkt/s)  truth={truth}")


if __name__ == "__main__":
    main()
