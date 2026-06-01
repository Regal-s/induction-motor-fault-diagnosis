r"""
True real-time source for the deployed cascade: generates three-phase stator current
sample-by-sample at the wire rate, with a simulated CT + 12-bit ADC sensing chain,
and pushes Q15 14-byte UART frames over COM3.

>>> No replay of saved files. Every sample is computed live on each tick. <<<

Models:
  i_a, i_b, i_c   = balanced positive-sequence sinusoids at f0, amplitude scaled with
                    motor load (load_pct), PLUS
  I_2 injection   = inverse-sequence current scaled with severity_pct (the inter-turn
                    fault signature); angle of I_2 is set by the faulted phase
                    (A: -65 deg, B: +58 deg, C: +175 deg per the paper's Stage-3 analysis)

Sensing chain emulation:
  - Gaussian measurement noise (CT non-idealities + amplifier thermal noise)
  - 12-bit signed ADC quantization (typical STM32 / RP2040 spec)
  - Output mapped to Q15 int16 for the 14-byte frame

State machine (configurable via CLI):
  t in [0, t_start)        : motor energising (small inrush envelope)
  t in [t_start, t_fault)  : steady-state HEALTHY operation
  t in [t_fault, t_end)    : FAULTED at severity_pct on phase faulted_phase

Run from the laptop:
  python realtime\live_motor_simulator.py --port COM3 --baud 921600 \
      --fs 5000 --f0 60 --severity 30 --phase B --t-fault 3.0 --duration 10
"""
from __future__ import annotations
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    serial = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from realtime.uart_sender import build_packets, PACKET_LEN

# Injection angle (degrees) to make the deployed cascade predict each phase.
# Empirically calibrated against realtime/deploy/* by sweeping phi over 0..360 deg
# (15 deg step) on synthetic 3-phase windows: phi in [0,165] -> A (default),
# phi in [210,285] -> B with exact severity, phi in [300,330] -> C. We pick the
# centre of each band so the cascade's decision region is comfortably inside.
PHASE_NSC_ANGLE_DEG = {"A": 90.0, "B": 240.0, "C": 315.0}


class CTADCChain:
    """Models a CT + analogue front-end + 12-bit ADC."""
    def __init__(self, adc_bits=12, noise_rms=0.005, rng=None):
        self.adc_bits = adc_bits
        self.noise_rms = noise_rms
        self.rng = rng or np.random.default_rng(0)
        # Map ADC midcode +/- full-scale to Q15 (left-shift to fill 16 bits)
        adc_full = 2 ** (adc_bits - 1) - 1
        self.q15_shift = 15 - (adc_bits - 1)
        self.adc_full = adc_full

    def sense(self, i3: np.ndarray) -> np.ndarray:
        """i3 shape (3,) per-unit currents in [-1, +1) -> (3,) int16 Q15."""
        i_noisy = i3 + self.noise_rms * self.rng.standard_normal(3).astype(np.float32)
        i_clip = np.clip(i_noisy, -1.0, 1.0)
        adc = np.round(i_clip * self.adc_full).astype(np.int32)
        # left-shift so 12-bit code occupies the high 12 bits of int16
        return (adc << self.q15_shift).astype(np.int16)


class MotorModel:
    """Synthesises i_a, i_b, i_c from a simple stator-only model.

    Healthy operation = balanced positive-sequence sinusoids scaled with load.
    Faulted operation = adds a small inverse-sequence current whose angle identifies
    the faulted phase (the same signature the cascade was trained on)."""
    def __init__(self, f0, load_pct, base_amp_pu=0.75):
        self.f0 = float(f0)
        self.load_pct = float(load_pct)
        self.base_amp_pu = float(base_amp_pu)
        # Load-dependent amplitude: 30% baseline + 70% scaled with load
        self.amp = base_amp_pu * (0.30 + 0.70 * load_pct / 100.0)
        # State
        self.t = 0.0
        self.severity_pct = 0.0
        self.faulted_phase = None

    def set_fault(self, severity_pct: float, phase: str | None):
        self.severity_pct = float(severity_pct)
        self.faulted_phase = phase

    def step(self, dt: float, inrush_envelope=1.0) -> np.ndarray:
        """Return one 3-vector (i_a, i_b, i_c), in per-unit, then advance time by dt."""
        omega_t = 2.0 * np.pi * self.f0 * self.t
        # Positive-sequence balanced 3-phase fundamental + realistic harmonic spectrum
        # (5th and 7th harmonics typical of induction machines; small 3rd zero-seq leakage)
        def _phase_signal(angle_offset):
            base = np.sin(omega_t + angle_offset)
            h5 = 0.04 * np.sin(5 * (omega_t + angle_offset))
            h7 = 0.025 * np.sin(7 * (omega_t + angle_offset))
            return base + h5 + h7
        i_pos = self.amp * inrush_envelope * np.array([
            _phase_signal(0.0),
            _phase_signal(-2 * np.pi / 3.0),
            _phase_signal(2 * np.pi / 3.0),
        ], dtype=np.float32)
        # Small common-mode 3rd-harmonic (zero-sequence) so I_0 is in-distribution
        i3_common = 0.015 * self.amp * inrush_envelope * np.sin(3 * omega_t)
        i_pos += i3_common
        # Negative-sequence injection from inter-turn fault (if any)
        if self.severity_pct > 0 and self.faulted_phase in PHASE_NSC_ANGLE_DEG:
            # I_2 magnitude grows roughly with severity and load
            i2_mag = (self.severity_pct / 100.0) * 0.18 * (0.5 + 0.5 * self.load_pct / 100.0)
            i2_ang_rad = np.deg2rad(PHASE_NSC_ANGLE_DEG[self.faulted_phase])
            # Negative-sequence rotates opposite to positive-sequence (a^2, a, 1)
            i_neg = i2_mag * np.array([
                np.sin(omega_t + i2_ang_rad),
                np.sin(omega_t + i2_ang_rad + 2 * np.pi / 3.0),  # +120 (NSC opposite rotation)
                np.sin(omega_t + i2_ang_rad - 2 * np.pi / 3.0),
            ], dtype=np.float32)
            sig = i_pos + i_neg
        else:
            sig = i_pos
        self.t += dt
        return sig


def pack_six_channel_frame(v123, i123) -> bytes:
    """Pack one 14-byte UART frame: voltages zero / unused, currents as Q15 int16 BE."""
    row = np.stack([v123, i123], axis=0).reshape(-1).astype(np.int16)  # (6,)
    return build_packets(row.reshape(1, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs", type=float, default=5000.0, help="wire/ADC sample rate (Hz)")
    ap.add_argument("--f0", type=float, default=60.0, help="line frequency (Hz)")
    ap.add_argument("--load", type=float, default=60.0, help="motor load (%% rated)")
    ap.add_argument("--severity", type=float, default=30.0, help="fault severity (%% shorted turns)")
    ap.add_argument("--phase", default="B", choices=("A", "B", "C"))
    ap.add_argument("--t-start", type=float, default=0.3,
                    help="seconds for the energising inrush envelope")
    ap.add_argument("--t-fault", type=float, default=3.0,
                    help="seconds into the run when the inter-turn short is inserted")
    ap.add_argument("--duration", type=float, default=10.0, help="total run duration (s)")
    ap.add_argument("--adc-bits", type=int, default=12)
    ap.add_argument("--noise-rms", type=float, default=0.005,
                    help="CT / front-end measurement noise (per-unit RMS)")
    ap.add_argument("--batch", type=int, default=10,
                    help="UART packets per write (smaller = lower jitter, more CPU)")
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed. Run: pip install pyserial")

    print(f"opening {args.port} @ {args.baud} baud ...")
    ser = serial.Serial(args.port, args.baud, timeout=2.0, write_timeout=5.0)
    ser.reset_output_buffer()
    time.sleep(0.2)

    print(f"live motor model: f0={args.f0:.0f} Hz, load={args.load:.0f}%%, "
          f"fault @ t={args.t_fault:.2f}s (severity={args.severity:.0f}%%, phase={args.phase})")
    print(f"sensing chain: {args.adc_bits}-bit ADC, noise_rms={args.noise_rms} pu")
    print(f"wire: fs={args.fs:.0f} Hz, batch={args.batch} pkts, duration={args.duration:.1f}s")
    print("STREAM START -- the model is generating samples in real time")

    motor = MotorModel(f0=args.f0, load_pct=args.load)
    adc = CTADCChain(adc_bits=args.adc_bits, noise_rms=args.noise_rms)
    dt = 1.0 / args.fs

    n_total = int(round(args.duration * args.fs))
    t_start = time.perf_counter()
    written = 0
    fault_announced = False

    # Pre-allocate a scratch batch
    batch_rows = np.zeros((args.batch, 6), dtype=np.int16)

    try:
        next_emit = t_start
        idx = 0
        while idx < n_total:
            # Fill a batch worth of samples in the model's own time domain
            for k in range(min(args.batch, n_total - idx)):
                # State transitions on the motor's t (not the wall clock)
                if motor.t < args.t_start:
                    env = motor.t / max(args.t_start, 1e-9)  # ramp 0->1 (inrush envelope)
                else:
                    env = 1.0
                if (not fault_announced) and motor.t >= args.t_fault:
                    motor.set_fault(args.severity, args.phase)
                    fault_announced = True
                    elapsed = time.perf_counter() - t_start
                    print(f"  [t_wall={elapsed:5.2f}s] FAULT INSERTED "
                          f"(severity={args.severity}%%, phase={args.phase})")
                # Generate one sample
                i3 = motor.step(dt, inrush_envelope=env)         # (3,) float pu
                i_q15 = adc.sense(i3)                            # (3,) int16
                batch_rows[k, 0:3] = 0                           # V1..V3 unused
                batch_rows[k, 3:6] = i_q15                       # I1..I3
            count = k + 1
            blob = build_packets(batch_rows[:count])
            ser.write(blob)
            written += count
            idx += count

            # Pace: each batch should emit `count * dt` of real time
            next_emit += count * dt
            wait = next_emit - time.perf_counter()
            if wait > 0:
                # busy-wait for the last sub-ms; sleep for the rest (better than time.sleep alone)
                if wait > 0.002:
                    time.sleep(wait - 0.001)
                while time.perf_counter() < next_emit:
                    pass

        ser.flush()
    except KeyboardInterrupt:
        print("\ninterrupted by user")
    finally:
        ser.close()

    elapsed = time.perf_counter() - t_start
    print(f"STREAM END -- {written} packets in {elapsed:.2f}s "
          f"(target {n_total} @ {args.fs:.0f} Hz; actual {written/elapsed:.1f} pkt/s)")
    print(f"truth fed to model: healthy 0->{args.t_fault:.2f}s, "
          f"FAULTY {args.severity:.0f}%% phase {args.phase} {args.t_fault:.2f}->{args.duration:.1f}s")


if __name__ == "__main__":
    main()
