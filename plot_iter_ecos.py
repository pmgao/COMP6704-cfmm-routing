import os
import re
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Raw solver logs (MOSEK/ECOS-like) for multiple problem sizes
text_8x12 = r"""
It     pcost       dcost      gap   pres   dres    k/t    mu     step   sigma     IR    |   BT
 0  +0.000e+00  -1.315e+02  +4e+02  6e-01  6e-01  1e+00  1e+00    ---    ---    0  0  - |  -  - 
 1  -1.551e-01  -5.779e+01  +2e+02  2e-01  3e-01  4e-01  4e-01  0.6432  1e-01   0  0  0 |  1  1
 2  -2.604e-01  -1.337e+01  +4e+01  5e-02  8e-02  1e-01  1e-01  0.7833  3e-02   0  0  0 |  1  1
 3  -7.733e-01  -7.229e+00  +2e+01  2e-02  4e-02  8e-02  6e-02  0.5310  2e-01   0  0  0 |  0  2
 4  -7.571e-01  -5.541e+00  +1e+01  2e-02  3e-02  5e-02  4e-02  0.9791  7e-01   0  0  0 |  9  0
 5  -6.990e-01  -3.398e+00  +8e+00  9e-03  2e-02  2e-02  2e-02  0.9791  6e-01   0  0  0 |  7  0
 6  -6.708e-01  -2.680e+00  +6e+00  7e-03  2e-02  1e-02  2e-02  0.9791  8e-01   0  0  0 | 10  0
 7  -7.191e-01  -1.883e+00  +4e+00  4e-03  9e-03  8e-03  1e-02  0.5013  2e-01   0  0  0 |  3  3
 8  -7.557e-01  -1.533e+00  +2e+00  2e-03  6e-03  5e-03  7e-03  0.6266  5e-01   0  0  0 |  6  2
 9  -7.488e-01  -1.448e+00  +2e+00  2e-03  6e-03  4e-03  6e-03  0.9791  9e-01   0  0  0 | 14  0
10  -7.836e-01  -1.193e+00  +1e+00  1e-03  3e-03  2e-03  4e-03  0.5013  2e-01   0  0  0 |  3  3
11  -8.100e-01  -1.082e+00  +9e-01  8e-04  2e-03  1e-03  2e-03  0.7833  6e-01   0  0  0 |  7  1
12  -8.073e-01  -1.064e+00  +8e-01  8e-04  2e-03  1e-03  2e-03  0.9791  9e-01   0  0  0 | 15  0
13  -8.291e-01  -9.792e-01  +4e-01  5e-04  1e-03  7e-04  1e-03  0.5013  2e-01   0  0  0 |  3  3
14  -8.410e-01  -9.492e-01  +3e-01  3e-04  1e-03  5e-04  8e-04  0.9791  7e-01   0  1  1 |  9  0
15  -8.502e-01  -9.216e-01  +2e-01  2e-04  7e-04  3e-04  5e-04  0.7833  6e-01   1  1  0 |  7  1
16  -8.519e-01  -9.164e-01  +2e-01  2e-04  6e-04  3e-04  4e-04  0.9791  9e-01   1  0  0 | 13  0
17  -8.603e-01  -8.981e-01  +9e-02  1e-04  4e-04  2e-04  2e-04  0.5013  2e-01   1  0  0 |  3  3
18  -8.661e-01  -8.881e-01  +5e-02  7e-05  2e-04  9e-05  1e-04  0.7833  5e-01   1  0  1 |  6  1
19  -8.683e-01  -8.839e-01  +3e-02  5e-05  1e-04  6e-05  9e-05  0.9791  7e-01   0  1  0 |  9  0
20  -8.715e-01  -8.789e-01  +2e-02  2e-05  7e-05  3e-05  4e-05  0.6266  2e-01   0  0  0 |  3  2
21  -8.733e-01  -8.762e-01  +6e-03  9e-06  3e-05  1e-05  2e-05  0.9791  4e-01   0  0  0 |  5  0
22  -8.740e-01  -8.752e-01  +3e-03  4e-06  1e-05  4e-06  7e-06  0.6266  1e-01   1  1  1 |  2  2
23  -8.744e-01  -8.746e-01  +3e-04  5e-07  1e-06  5e-07  8e-07  0.9791  1e-01   0  0  0 |  2  0
24  -8.745e-01  -8.745e-01  +1e-04  2e-07  6e-07  2e-07  4e-07  0.6250  7e-02   1  1  0 |  1  2
25  -8.745e-01  -8.745e-01  +2e-05  3e-08  8e-08  3e-08  4e-08  0.8770  5e-03   0  0  0 |  0  0
26  -8.745e-01  -8.745e-01  +1e-05  2e-08  5e-08  2e-08  3e-08  0.4010  2e-01   0  1  1 |  3  4
27  -8.745e-01  -8.745e-01  +3e-06  4e-09  1e-08  4e-09  7e-09  0.7833  3e-03   1  1  1 |  0  1
28  -8.745e-01  -8.745e-01  +1e-06  2e-09  5e-09  2e-09  3e-09  0.6266  6e-02   0  0  0 |  2  2
29  -8.745e-01  -8.745e-01  +2e-07  4e-10  1e-09  4e-10  6e-10  0.7833  1e-02   0  0  0 |  1  1
30  -8.745e-01  -8.745e-01  +9e-08  1e-10  4e-10  1e-10  2e-10  0.6266  5e-02   1  0  0 |  2  2
31  -8.745e-01  -8.745e-01  +2e-08  3e-11  1e-10  3e-11  6e-11  0.7833  9e-03   1  1  0 |  1  1
32  -8.745e-01  -8.745e-01  +8e-09  1e-11  4e-11  1e-11  2e-11  0.6266  5e-02   0  0  1 |  2  2
"""

# More logs for other scales (truncated here for brevity in this comment-only task)
text_32x100 = r""" ... """
text_32x200 = r""" ... """
text_64x800 = r""" ... """
text_128x800 = r""" ... """
text_128x4000 = r""" ... """

def parse_iters_mosek_like(text):
    """
    Parse MOSEK/ECOS-style iteration logs.

    Header example:
      It     pcost       dcost      gap   pres   dres    k/t    mu     step   sigma     IR    |   BT

    Data row example:
      0  +0.000e+00  -1.315e+02  +4e+02  6e-01  6e-01  1e+00  1e+00    ---    ---    0  0  - |  -  -

    Extract only: iteration index, pcost, dcost, gap, pres, dres, k/t, mu, and step.
    Ignore other columns.
    """
    rows = []
    in_table = False

    # Detect header line to start parsing table
    header_re = re.compile(
        r"^\s*It\s+pcost\s+dcost\s+gap\s+pres\s+dres\s+k/t\s+mu\b", re.IGNORECASE
    )
    # Skip separator or trailing '|' lines if any appear
    sep_re = re.compile(r"^\s*[-=]{3,}|\|\s*$")

    # Numeric patterns (supporting scientific notation)
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    # Step may be numeric or placeholder like '---'
    maybe_num = rf"(?:{num}|[-]+|---)"

    # Compile a regex to capture the required columns
    line_re = re.compile(
        rf"^\s*(?P<iter>-?\d+)\s+"
        rf"(?P<pcost>{num})\s+"
        rf"(?P<dcost>{num})\s+"
        rf"(?P<gap>{num})\s+"
        rf"(?P<pres>{num})\s+"
        rf"(?P<dres>{num})\s+"
        rf"(?P<kt>{num})\s+"
        rf"(?P<mu>{num})\s+"
        rf"(?P<step>{maybe_num})"
        r"(?:\s+.*)?$"
    )

    # Iterate through lines, start parsing after the header is found
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not in_table:
            if header_re.search(line):
                in_table = True
            continue
        if sep_re.search(line):
            continue
        m = line_re.match(line)
        if not m:
            continue
        g = m.groupdict()
        # Convert step to float, allow NaN for '---'
        step_val = float("nan")
        step_str = g["step"].strip()
        if step_str not in ("---", "------"):
            try:
                step_val = float(step_str)
            except ValueError:
                step_val = float("nan")
        try:
            # Append parsed row with standardized keys
            rows.append({
                "ite": int(g["iter"]),
                "POBJ": float(g["pcost"]),
                "DOBJ": float(g["dcost"]),
                "PFEAS": float("nan"),   # Not available in this log; keep NaN for compatibility
                "DFEAS": float("nan"),
                "GFEAS": float("nan"),
                "PRSTATUS": float("nan"),
                "MU": float(g["mu"]),
                "TIME": float("nan"),
                "GAP_raw": float(g["gap"]),
                "PRES_raw": float(g["pres"]),
                "DRES_raw": float(g["dres"]),
                "KT_raw": float(g["kt"]),
                "STEP_raw": step_val,
                "SCALE": float("nan"),
            })
        except ValueError:
            # Skip malformed rows
            continue

    # Ensure rows are sorted by iteration number
    rows.sort(key=lambda r: r["ite"])
    return rows


# Map each scale name to the parsed sequence of iterations
datasets = {
    "8x12": parse_iters_mosek_like(text_8x12),
    "32x100": parse_iters_mosek_like(text_32x100),
    "32x200": parse_iters_mosek_like(text_32x200),
    "64x800": parse_iters_mosek_like(text_64x800),
    "128x800": parse_iters_mosek_like(text_128x800),
    "128x4000": parse_iters_mosek_like(text_128x4000),
}

# Fixed color palette per scale for consistent plotting
colors = {
    "8x12": "tab:blue",
    "32x100": "tab:orange",
    "32x200": "tab:green",
    "64x800": "tab:purple",
    "128x800": "tab:brown",
    "128x4000": "tab:red",
}

# Create the figure; a 2x2 GridSpec with top row for titles/space and bottom for main plots
fig = plt.figure(figsize=(18, 8))
# fig.suptitle("CVXPY Objectives and Primal-Dual Gap over Iterations", fontsize=14, y=0.96)

# Main grid: two rows (top placeholders), two columns
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 3.0], hspace=0.35, wspace=0.25)

# Empty axes to reserve layout space or future annotations
ax_empty1 = fig.add_subplot(gs[0, 0]); ax_empty1.axis("off")
ax_empty2 = fig.add_subplot(gs[0, 1]); ax_empty2.axis("off")

# Left bottom: Primal objective plot
ax_pobj = fig.add_subplot(gs[1, 0])
for name, data in datasets.items():
    if not data:
        continue
    x = [d["ite"] for d in data]
    y = [d["POBJ"] for d in data]
    ax_pobj.plot(x, y, marker='o', linewidth=1.5, label=name, alpha=0.9, color=colors.get(name))
ax_pobj.set_title("Primal Objective (POBJ)")
ax_pobj.set_xlabel("Iteration")
ax_pobj.set_ylabel("POBJ (others)")
ax_pobj.grid(True, linestyle="--", alpha=0.4)
ax_pobj.legend(ncol=1, fontsize=8, loc="lower right")

# Right composite grid: two side-by-side axes occupying the right half
gs_right = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.55, right=0.95, bottom=0.10, top=0.86)

# Top-right (actually right-left): Dual objective plot
ax_dobj = fig.add_subplot(gs_right[0, 0])
for name, data in datasets.items():
    if not data:
        continue
    x = [d["ite"] for d in data]
    y = [d["DOBJ"] for d in data]
    ax_dobj.plot(x, y, marker='o', linewidth=1.5, label=name, alpha=0.9, color=colors.get(name))
ax_dobj.set_title("Dual Objective (DOBJ)")
ax_dobj.set_xlabel("Iteration")
ax_dobj.set_ylabel("Dual Objective")
ax_dobj.grid(True, linestyle="--", alpha=0.4)
ax_dobj.legend(ncol=1, fontsize=8, loc="best")

# Top-right (actually right-right): Convergence curve based on relative primal-dual gap
ax_gap = fig.add_subplot(gs_right[0, 1])

def _positive(vals):
    """
    Replace non-positive or non-finite values with NaN so semilogy can handle them.
    """
    out = []
    for v in vals:
        if v is None or not math.isfinite(v) or v <= 0:
            out.append(float("nan"))
        else:
            out.append(v)
    return out

# Plot relative gap: |POBJ-DOBJ| / max(1, |DOBJ|) on a log scale
for name, data in datasets.items():
    if not data:
        continue
    x = [d["ite"] for d in data]
    num = [abs(d["POBJ"] - d["DOBJ"]) for d in data]
    denom = [max(1.0, abs(d["DOBJ"])) for d in data]
    y = [n / m if m != 0 else float("nan") for n, m in zip(num, denom)]
    ax_gap.semilogy(x, _positive(y), marker='o', linewidth=1.5, label=name, alpha=0.9, color=colors.get(name))

ax_gap.set_title("Convergence Rate")
ax_gap.set_xlabel("Iteration")
ax_gap.set_ylabel("Convergence Rate")
ax_gap.grid(True, linestyle="--", alpha=0.4)
ax_gap.legend(ncol=1, fontsize=8, loc="best")

# Export plot to PDF
out_dir = "."
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "iterative_objectives_ecos.pdf")
plt.savefig(out_path, dpi=2000, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.close()