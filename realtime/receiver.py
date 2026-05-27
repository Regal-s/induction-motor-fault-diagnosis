r"""
RECEIVER (runs on the NVIDIA Jetson). Listens on TCP/Ethernet, ingests the streamed three-phase
current, windows it, extracts features, and runs the deployed cascade:
    Stage-0 onset  ->  detect (healthy/faulty)  ->  severity (%)  ->  faulted phase
emitting a per-window diagnosis and a final per-recording aggregated verdict (median severity,
majority-vote phase) -- the realistic decision granularity on hardware.

Run (Jetson):  python realtime/receiver.py --port 9009
One connection == one recording (the sender opens a fresh connection per recording).
"""
from __future__ import annotations
import sys, json, socket, argparse, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\Naveen"); sys.path.insert(0, str(ROOT))
from realtime.streamio import CascadeInfer


def recv_header(conn):
    buf = b""
    while b"\n" not in buf:
        b = conn.recv(1)
        if not b:
            return None
        buf += b
    return json.loads(buf.decode().strip())


def handle(conn, infer, W, stride):
    hdr = recv_header(conn)
    if hdr is None:
        return
    case = hdr["case_id"]; truth = hdr.get("truth", {})
    print(f"\n>>> stream START {case}  (fs={hdr['fs']}, n={hdr['n']}, truth={truth.get('label')}/{truth.get('phase')})")
    ring = np.zeros((3, 0), np.float32); next_at = W
    sev_votes, phase_votes, n_faulty, n_win = [], [], 0, 0
    onset_t = None; total = 0
    while True:
        data = conn.recv(65536)
        if not data:
            break
        arr = np.frombuffer(data, dtype="<f4")
        k = (len(arr) // 3) * 3
        if k == 0:
            continue
        samp = arr[:k].reshape(-1, 3).T            # (3, K)
        ring = np.concatenate([ring, samp], axis=1); total += samp.shape[1]
        while ring.shape[1] >= next_at:
            win = ring[:, next_at - W:next_at]
            res = infer.classify(win); n_win += 1
            if onset_t is None and res.get("onset"):
                onset_t = (next_at - W) / hdr["fs"]
            tag = res["state"]
            if res["state"] == "faulty":
                n_faulty += 1; sev_votes.append(res["severity_pct"]); phase_votes.append(res["phase"])
                tag += f"  sev={res['severity_pct']}%  phase={res['phase']}"
            print(f"   t={(next_at)/hdr['fs']:5.2f}s  win#{n_win:02d}  {tag}"
                  f"  (onset_idx={res.get('onset_index',0):.3f})")
            next_at += stride
    # ---- per-recording verdict ----
    faulty = n_faulty > n_win / 2 if n_win else False
    verdict = {"case_id": case, "state": "FAULTY" if faulty else "HEALTHY",
               "windows": n_win, "faulty_windows": n_faulty, "onset_s": onset_t}
    if faulty and sev_votes:
        verdict["severity_pct"] = float(np.median(sev_votes))
        verdict["phase"] = Counter(phase_votes).most_common(1)[0][0]
    ok = ("OK" if (faulty == (truth.get("label") == 1) and
                   (not faulty or verdict.get("phase") == truth.get("phase"))) else "MISMATCH")
    print(f"<<< VERDICT {case}: {verdict}  | truth={truth}  [{ok}]")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--once", action="store_true", help="exit after one connection (testing)")
    a = ap.parse_args()
    print("loading deployed cascade ...")
    infer = CascadeInfer(); W = infer.meta["W"]; stride = infer.meta["stride"]
    print(f"ready. cascade metrics (held-out rep): {json.dumps(infer.meta['metrics'])}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port)); srv.listen(4)
    print(f"listening on {a.host}:{a.port}  (W={W}, stride={stride})")
    try:
        while True:
            conn, addr = srv.accept()
            try:
                handle(conn, infer, W, stride)
            except Exception as e:
                import traceback; print("stream error:", e); traceback.print_exc()
            finally:
                conn.close()
            if a.once:
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
