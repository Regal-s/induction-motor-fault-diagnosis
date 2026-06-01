r"""
One-shot restructurer for manuscript/main.tex Section IV (Performance Evaluation).

IEEE-Transactions-style flow:
  A. Implementation and Evaluation Protocol
  B. Main Cascade Performance (in-distribution and cross-load)
     [absorbs old "Headline Results" + "Why the Tasks Differ"]
  C. Comparative Model Evaluation
     [renamed from "Eight-Model Benchmark"]
  D. Mechanism Ablations: Load-Robustness Components
     [renamed; ALL ablation-style subsections grouped right after]
  E. Biomedical Anomaly Features            (MOVED UP from after Hardware)
  F. Feature Importance and Load-Invariant Selection
  G. Hyperparameter Tuning and Physics-Consistent Augmentation
  H. Robustness, Calibration, and Onset Detection
     [renamed from "Calibration, Onset, Noise, ..."]
  I. Within-Hardware Validation on a Real Motor
  J. Real-Time Edge Deployment on a Jetson Orin Nano
  K. Discussion: Positioning, Limitations, and Future Work

The Discussion stays at the end. We also insert a one-paragraph roadmap right after
\section{Performance Evaluation}.
"""
from pathlib import Path
import re

MS = Path(__file__).parent / "manuscript" / "main.tex"
text = MS.read_text(encoding="utf-8")

# Locate Section IV start (\section{Performance Evaluation}) and end (\section{Conclusion})
m_iv = re.search(r"\\section\{Performance Evaluation\}", text)
m_concl = re.search(r"\\section\{Conclusion\}", text)
assert m_iv and m_concl and m_concl.start() > m_iv.start()
prefix = text[: m_iv.end()]
section_iv = text[m_iv.end() : m_concl.start()]
suffix = text[m_concl.start() :]

# Split section_iv by \subsection lines
pattern = re.compile(r"(\\subsection\{[^}]+\})", flags=re.MULTILINE)
parts = pattern.split(section_iv)
# parts is [leading_text_before_first_subsection, "\\subsection{...}", body, "\\subsection{...}", body, ...]
leading = parts[0]
blocks = []  # list of (heading_text, body)
for i in range(1, len(parts), 2):
    head = parts[i]
    body = parts[i + 1] if (i + 1) < len(parts) else ""
    name = re.match(r"\\subsection\{(.+)\}", head).group(1).strip()
    blocks.append((name, body))

# Build a lookup by current name
by_name = {name: body for name, body in blocks}
print("Current subsections in section IV (in source order):")
for name, _ in blocks:
    print("  -", name)

# Required current names (assert presence so we fail loud if user already edited)
required = [
    "Implementation and Evaluation Protocol",
    "Headline Results",
    "Why the Tasks Differ: Cross-Load Analysis and the Load-Aware Fix",
    "Eight-Model Benchmark",
    "Load-Robustness Mechanisms (Ablations)",
    "Calibration, Onset, Noise, and Training Behaviour",
    "Hardware Validation (Within-Domain)",
    "Biomedical Anomaly Features (Ablation)",
    "Feature Importance and Reduction",
    "Hyperparameter Tuning and Data Augmentation",
    "Real-Time Edge Testing on NVIDIA Jetson",
    "Discussion: Positioning and Deployment",
]
missing = [n for n in required if n not in by_name]
if missing:
    raise SystemExit(f"missing subsections: {missing}")

# Desired (new) order with renames: (old_name, new_name)
new_order = [
    ("Implementation and Evaluation Protocol",
     "Implementation and Evaluation Protocol"),
    ("Headline Results",
     "Main Cascade Performance: In-Distribution and Cross-Load"),
    ("Why the Tasks Differ: Cross-Load Analysis and the Load-Aware Fix",
     "Cross-Load Analysis and the Load-Aware Severity Fix"),
    ("Eight-Model Benchmark",
     "Comparative Model Evaluation"),
    ("Load-Robustness Mechanisms (Ablations)",
     "Mechanism Ablations: Load-Robustness Components"),
    # --- ablations grouped here ---
    ("Biomedical Anomaly Features (Ablation)",
     "Biomedical Anomaly Features"),
    ("Feature Importance and Reduction",
     "Feature Importance and Load-Invariant Selection"),
    ("Hyperparameter Tuning and Data Augmentation",
     "Hyperparameter Tuning and Physics-Consistent Augmentation"),
    # --- robustness / hardware / deployment ---
    ("Calibration, Onset, Noise, and Training Behaviour",
     "Robustness, Calibration, and Onset Detection"),
    ("Hardware Validation (Within-Domain)",
     "Within-Hardware Validation on a Real Motor"),
    ("Real-Time Edge Testing on NVIDIA Jetson",
     "Real-Time Edge Deployment on a Jetson Orin Nano"),
    # --- discussion last ---
    ("Discussion: Positioning and Deployment",
     "Discussion: Positioning, Limitations, and Future Work"),
]

# Roadmap paragraph to insert right after \section{Performance Evaluation}
roadmap = (
    "\n"
    "This section evaluates the proposed framework end-to-end. \\S IV-A specifies the\n"
    "implementation, splits, and metrics; \\S IV-B reports the main cascade performance\n"
    "in-distribution and under the primary cross-load (LOLO) protocol; \\S IV-C compares\n"
    "eight model classes on identical splits and tests cross-model significance;\n"
    "\\S IV-D--G isolate the contribution of every novel mechanism (load-robustness\n"
    "components, biomedical features, feature selection, augmentation/tuning) through\n"
    "controlled ablations; \\S IV-H quantifies probabilistic calibration, noise\n"
    "robustness, and online fault-onset detection; \\S IV-I validates the cascade\n"
    "within-hardware on a publicly released real-motor dataset; \\S IV-J reports the\n"
    "measured end-to-end real-time validation on a Jetson Orin Nano edge module; and\n"
    "\\S IV-K positions the framework relative to the literature and lists the residual\n"
    "limitations.\n\n"
)

# Reassemble Section IV
new_iv = roadmap + leading
for old_name, new_name in new_order:
    body = by_name[old_name]
    new_iv += f"\\subsection{{{new_name}}}" + body

# Stitch back into the manuscript
new_text = prefix + new_iv + suffix

# Sanity check: SAME bodies are reused (just rearranged), nothing dropped or duplicated
old_bodies = sorted(body for _, body in blocks)
new_bodies = sorted(by_name[old] for old, _ in new_order)
assert old_bodies == new_bodies, "content was lost or duplicated during reassembly"

MS.write_text(new_text, encoding="utf-8")

print("\nNew subsection order:")
for _, new_name in new_order:
    print("  -", new_name)
print("\nDone. main.tex restructured.")
