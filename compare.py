import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from collections import OrderedDict
import math
import os

# Data input
scales = [
    "8x12",
    "32x100",
    "32x200",
    "64x800",
    "128x800",
    "128x4000",
]

# Per-solver runtime data (seconds) for each problem scale
data_time = {
    "Clarabel": {
        "8x12":   0.003,
        "32x100": 0.041,
        "32x200": 0.063,
        "64x800": None,       # None means missing or not applicable (break the line)
        "128x800": None,
        "128x4000": None,
    },
    "SCS": {
        "8x12":   0.023,
        "32x100": 0.616,
        "32x200": 1.07,
        "64x800": 38.90,
        "128x800": 42.44,
        "128x4000": 212.19,
    },
    "ECOS": {
        "8x12":   0.002,
        "32x100": 0.021,
        "32x200": 0.036,
        "64x800": 0.298,
        "128x800": 1.451,
        "128x4000": 2.130,
    },
    "MOSEK": {
        "8x12":   0.013,
        "32x100": 0.029,
        "32x200": 0.045,
        "64x800": 0.168,
        "128x800": 0.188,
        "128x4000": 1.895,
    },
}

# Order in which solvers are plotted
solver_order = ["Clarabel", "SCS", "ECOS", "MOSEK"]
# solver_order = ["Clarabel", "ECOS", "MOSEK"]

# Color map per solver
colors = {
    "Clarabel": "tab:purple",
    "SCS": "tab:orange",
    "ECOS": "tab:green",
    "MOSEK": "tab:blue",
}
# Marker style per solver
markers = {
    "Clarabel": "o",
    "SCS": "s",
    "ECOS": "D",
    "MOSEK": "^",
}

# X-axis tick positions and labels
x_idx = list(range(len(scales)))
x_labels = scales

# Create main figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Plot each solver's time vs. problem size, breaking lines on None/inf
for solver in solver_order:
    seg_x, seg_y = [], []
    for i, sc in enumerate(scales):
        t = data_time[solver].get(sc, None)
        if t is None or (isinstance(t, float) and (not math.isfinite(t))):
            # If we encounter a gap and have a segment buffered, plot it and start a new segment
            if seg_x:
                ax.plot(seg_x, seg_y, marker=markers[solver], linewidth=2,
                        color=colors[solver],
                        label=solver if solver not in ax.get_legend_handles_labels()[1] else None)
                seg_x, seg_y = [], []
        else:
            seg_x.append(i)
            seg_y.append(t)
    # Plot any remaining segment at the end
    if seg_x:
        ax.plot(seg_x, seg_y, marker=markers[solver], linewidth=2,
                color=colors[solver],
                label=solver if solver not in ax.get_legend_handles_labels()[1] else None)

# Axis formatting
ax.set_xticks(x_idx)
ax.set_xticklabels(x_labels, rotation=0)
ax.set_xlabel("Problem size (#Tokens x #Pools)")
ax.set_ylabel("Solving time (s)")
ax.grid(True, linestyle="--", alpha=0.4)

# Reorder legend entries to the desired display order
desired_order = ["ECOS", "Clarabel", "MOSEK", "SCS"]
handles, labels = ax.get_legend_handles_labels()
label_to_handle = {lab: h for h, lab in zip(handles, labels)}
ordered_handles = [label_to_handle[lab] for lab in desired_order if lab in label_to_handle]
ordered_labels = [lab for lab in desired_order if lab in label_to_handle]
ax.legend(ordered_handles, ordered_labels, title="Solver", ncol=2, loc="upper left")

# Create an inset axis to zoom into the small-scale regimes
axins = inset_axes(ax, width="45%", height="55%", loc="upper right", borderpad=1.0)
for solver in solver_order:
    seg_x, seg_y = [], []
    for i, sc in enumerate(scales):
        t = data_time[solver].get(sc, None)
        if t is None or (isinstance(t, float) and (not math.isfinite(t))):
            if seg_x:
                axins.plot(seg_x, seg_y, marker=markers[solver], linewidth=2,
                           color=colors[solver])
                seg_x, seg_y = [], []
        else:
            seg_x.append(i)
            seg_y.append(t)
    if seg_x:
        axins.plot(seg_x, seg_y, marker=markers[solver], linewidth=2,
                   color=colors[solver])

# Inset x-limits to show first three scales only
axins.set_xlim(-0.3, 2.3)

# Compute y-limits for inset based on available data in the first three scales
y_vals_small = []
for solver in solver_order:
    for i in range(3):  # 0,1,2
        t = data_time[solver].get(scales[i], None)
        if (t is not None) and math.isfinite(t):
            y_vals_small.append(t)
if y_vals_small:
    ymin, ymax = min(y_vals_small), max(y_vals_small)
    pad = (ymax - ymin) * 0.2 if ymax > ymin else ymax * 0.2 + 1e-3
    axins.set_ylim(max(0, ymin - pad), ymax + pad)

# Inset axis ticks/labels
axins.set_xticks([0, 1, 2])
axins.set_xticklabels(scales[:3], rotation=0, fontsize=9)
axins.grid(True, linestyle="--", alpha=0.4)

# Draw connectors between main axes and inset to indicate zoomed area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", linestyle="--")

# Tight layout for better spacing
plt.tight_layout()

# Save figure to PDF
out_dir = "."
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "solver_runtime_vs_size_0.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
print(f"Saved figure to: {pdf_path}")
plt.show()