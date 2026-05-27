r"""
SENDER (runs on the laptop). Replays a saved REAL-machine recording's three-phase current over
TCP/Ethernet to the Jetson receiver, at the real sampling rate, to emulate a live feed.

Protocol: connect -> send one newline-terminated JSON header
  {"case_id","fs","n","channels":3,"truth":{...}}
then stream little-endian float32 in packets of `packet` samples (packet*3 floats each), pacing at fs.

Run (laptop):  python realtime/sender.py --host <jetson-ip> --port 9009 --case ibarram_ITSC_B30_Repetition05
               python realtime/sender.py --host <jetson-ip> --all-rep 05      # stream every rep-05 recording
"""
from __future__ import annotations
import sys, json, time, socket, argparse
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from realtime.streamio import reconstruct_recording, list_recordings


def stream_one(sock_args, case_id, fs, packet, speed):
    sig, truth = reconstruct_recording(case_id)
    n = sig.shape[1]
    s = socket.create_connection(sock_args, timeout=10)
    header = json.dumps({"case_id": case_id, "fs": fs, "n": n, "channels": 3, "truth": truth}) + "\n"
    s.sendall(header.encode())
    dt = packet / fs / max(speed, 1e-6)
    t0 = time.time()
    for i in range(0, n, packet):
        chunk = sig[:, i:i + packet].T.reshape(-1).astype("<f4").tobytes()  # interleaved [t0:abc, t1:abc,...]
        s.sendall(chunk)
        time.sleep(max(0, dt))
    s.shutdown(socket.SHUT_WR);
    try: s.recv(1)  # let receiver finish
    except Exception: pass
    s.close()
    print(f"sent {case_id}  ({n} samples, {n/fs:.1f}s of signal in {time.time()-t0:.1f}s, "
          f"truth={truth['label']}/{truth.get('phase')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--case", default=None); ap.add_argument("--all-rep", default=None)
    ap.add_argument("--fs", type=float, default=1000.0); ap.add_argument("--packet", type=int, default=50)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (1=real time)")
    a = ap.parse_args()
    cases = ([a.case] if a.case else
             list_recordings().query("rep == @a.all_rep")["case_id"].tolist() if a.all_rep else [])
    if not cases:
        print("specify --case <id> or --all-rep <NN>"); return
    for c in cases:
        stream_one((a.host, a.port), c, a.fs, a.packet, a.speed)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
