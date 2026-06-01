r"""
UART receiver for the 6-channel `yroru.m` sender (V1,V2,V3,I1,I2,I3 Q15 big-endian).

Companion to realtime/receiver.py but speaks the MATLAB script's framed UART protocol
instead of TCP. Discards the three voltage channels, keeps the three stator currents,
de-quantises Q15 -> float, resamples from the wire rate (e.g. 2400 Hz) to the deployed
model's training rate (read from realtime/deploy/meta.json, typically 1000 Hz) using a
polyphase FIR (scipy.signal.resample_poly), windows the resampled stream, and runs the
deployed cascade (onset -> detect -> severity -> phase) with a rolling per-recording
aggregator (majority-vote phase, median severity).

Packet (14 bytes, big-endian per channel):
  +------+----+----+----+----+----+----+----+----+----+----+----+----+------+
  | 0xAA | V1hi V1lo V2hi V2lo V3hi V3lo I1hi I1lo I2hi I2lo I3hi I3lo | 0x55 |
  +------+-------------------------------------------------------------+------+

Run on the Jetson:
  pip3 install --user pyserial scipy
  python3 -u realtime/uart_receiver.py --port /dev/ttyUSB0 --baud 921600 --fs 2400 --f0 50
"""
from __future__ import annotations
import argparse
import json
import struct
import sys
import time
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np

try:
    import serial  # pyserial
except ImportError:
    serial = None  # checked in main(); the parser still works without it

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None  # checked in main()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from realtime.streamio import CascadeInfer

SYNC_BYTE = 0xAA
CHECK_BYTE = 0x55
PACKET_LEN = 14
PACKET_FMT = ">BhhhhhhB"   # SYNC + 6 int16 (big-endian) + CHECK
assert struct.calcsize(PACKET_FMT) == PACKET_LEN

Q15_SCALE = 32768.0          # int16 -> float in [-1, +1)


def parse_packets(buf: bytearray):
    """Walk `buf`, emit (i_a, i_b, i_c) for every well-formed packet, return the
    unconsumed tail. We only keep currents (channels 4,5,6); voltages are discarded."""
    out = []
    i, n, dropped = 0, len(buf), 0
    while i + PACKET_LEN <= n:
        if buf[i] != SYNC_BYTE:
            i += 1; dropped += 1
            continue
        if buf[i + PACKET_LEN - 1] != CHECK_BYTE:
            i += 1; dropped += 1
            continue
        _, _v1, _v2, _v3, c4, c5, c6, _ = struct.unpack_from(PACKET_FMT, buf, i)
        out.append((c4 / Q15_SCALE, c5 / Q15_SCALE, c6 / Q15_SCALE))
        i += PACKET_LEN
    return out, buf[i:], dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyUSB0 or /dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs", type=float, default=2400.0,
                    help="wire sample rate in Hz (must match yroru.m's Fs)")
    ap.add_argument("--f0", type=float, default=50.0,
                    help="line frequency of your data in Hz (overrides bundle meta.json)")
    ap.add_argument("--window", type=int, default=500,
                    help="window length at the model rate (default 500 = 0.5 s at 1 kHz)")
    ap.add_argument("--stride", type=int, default=250,
                    help="stride at the model rate")
    ap.add_argument("--aggregate-windows", type=int, default=20,
                    help="rolling window count for per-recording-style aggregation")
    ap.add_argument("--deploy", default=str(ROOT / "realtime" / "deploy"),
                    help="path to the deployed model bundle")
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed. Run: pip3 install --user pyserial")
    if resample_poly is None:
        sys.exit("scipy not installed. Run: pip3 install --user scipy")

    print(f"loading deployed cascade from {args.deploy} ...")
    cascade = CascadeInfer(Path(args.deploy))
    target_fs = float(cascade.fs)                              # model's training rate (e.g. 1000)
    bundle_f0 = float(cascade.f0)
    if args.f0 != bundle_f0:
        print(f"overriding bundle f0={bundle_f0} -> {args.f0}")
        cascade.f0 = float(args.f0)

    wire_fs = float(args.fs)
    if abs(wire_fs - target_fs) < 1e-6:
        up, down = 1, 1
    else:
        g = gcd(int(round(wire_fs)), int(round(target_fs)))
        up = int(round(target_fs)) // g
        down = int(round(wire_fs)) // g

    raw_window = int(round(args.window * down / up))          # raw samples per model window
    raw_stride = int(round(args.stride * down / up))          # raw samples per stride
    print(f"ready. wire={wire_fs} Hz, model={target_fs} Hz, ratio up/down = {up}/{down}, "
          f"raw_window={raw_window}, raw_stride={raw_stride}, f0={cascade.f0} Hz")
    print(f"opening serial {args.port} @ {args.baud} baud ...")

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    print("listening for UART frames (Ctrl+C to stop)")

    raw_buf = bytearray()
    sa, sb, sc = [], [], []          # raw current samples at wire rate
    samples_seen = 0
    bytes_dropped = 0
    win_idx = 0
    t_start = time.time()
    history = deque(maxlen=args.aggregate_windows)

    try:
        while True:
            # Poll-based read: ser.in_waiting + a tiny sleep when idle.
            # The blocking ser.read() on CDC-ACM gadgets raises a spurious
            # SerialException ("readiness with no data") whenever the host
            # opens/reconfigures the port; polling avoids the false alarm
            # entirely and is robust across sender start/stop cycles.
            try:
                n = ser.in_waiting
            except (serial.SerialException, OSError):
                time.sleep(0.05)
                continue
            if n <= 0:
                time.sleep(0.002)
                chunk = b""
            else:
                try:
                    chunk = ser.read(n)
                except (serial.SerialException, OSError):
                    time.sleep(0.05)
                    continue
            if chunk:
                raw_buf.extend(chunk)
                samples, raw_buf, dropped = parse_packets(raw_buf)
                bytes_dropped += dropped
                for ia, ib, ic in samples:
                    sa.append(ia); sb.append(ib); sc.append(ic)
                samples_seen += len(samples)

            while len(sa) >= raw_window:
                raw = np.stack([
                    np.asarray(sa[:raw_window], dtype=np.float32),
                    np.asarray(sb[:raw_window], dtype=np.float32),
                    np.asarray(sc[:raw_window], dtype=np.float32),
                ], axis=0)                                     # (3, raw_window)

                # 2400 -> 1000 Hz polyphase FIR resample (5/12 ratio for this case)
                if (up, down) == (1, 1):
                    window = raw
                else:
                    window = resample_poly(raw, up=up, down=down, axis=-1).astype(np.float32)
                # resample_poly's output length is ceil(raw_window * up / down) which is
                # exactly args.window for the (5,12,1200,500) case.
                if window.shape[-1] != args.window:
                    window = window[..., :args.window]         # safety clamp

                pred = cascade.classify(window)
                win_idx += 1
                history.append(pred)

                t = samples_seen / wire_fs
                print(f"t={t:6.2f}s win#{win_idx:04d}  "
                      f"{pred.get('state', '?'):>7s}  "
                      f"sev={pred.get('severity_pct', '-')}  "
                      f"phase={pred.get('phase', '-')}  "
                      f"onset_idx={pred.get('onset_index', 0.0):.3f}")

                # slide raw buffer
                sa = sa[raw_stride:]
                sb = sb[raw_stride:]
                sc = sc[raw_stride:]

                if win_idx % args.aggregate_windows == 0:
                    states = [h.get("state") for h in history if h.get("state")]
                    sevs = [h.get("severity_pct") for h in history if h.get("severity_pct") is not None]
                    phases = [h.get("phase") for h in history if h.get("phase")]
                    state_vote = max(set(states), key=states.count) if states else "?"
                    sev_median = float(np.median(sevs)) if sevs else None
                    phase_vote = max(set(phases), key=phases.count) if phases else None
                    el = time.time() - t_start
                    mbps = samples_seen * PACKET_LEN * 8 / el / 1e6 if el > 0 else 0.0
                    print(f"--- AGG over {len(history)} windows: "
                          f"state={state_vote.upper()} severity_pct={sev_median} phase={phase_vote} "
                          f"| {mbps:.2f} Mbps  bytes_dropped={bytes_dropped}")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try: ser.close()
        except Exception: pass
        print(json.dumps({
            "wire_samples": samples_seen, "windows": win_idx,
            "bytes_dropped": bytes_dropped, "seconds": round(time.time() - t_start, 3)
        }, indent=2))


if __name__ == "__main__":
    main()
