import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np
from pathlib import Path
from adjustText import adjust_text

textscaling = 1.5
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'xtick.labelsize': 10 * textscaling,
    'ytick.labelsize': 10 * textscaling,
    'axes.labelsize': 12 * textscaling,
    'axes.titlesize': 14 * textscaling,
    'legend.fontsize': 10 * textscaling,
    'axes.grid': True,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
    'figure.dpi': 600
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / 'data_processed'
PLOTS = ROOT / 'plots'
FILENAME = DATA_PROCESSED / 'motion_data.csv'

PLOTS.mkdir(parents=True, exist_ok=True)

# --- Annotations: list of (timestamp_in_seconds, "label text") ---
ANNOTATIONS = [
    (0, "Start"),
    (17, "Launch"),
    (19, "Dive loop"),
    (29, "Heart-line roll"),
    (33, "Overbanked curve"),
    (38, "Braking"),
]

df = pd.read_csv(FILENAME)

required_cols = ['Time', 'Pos_x', 'Pos_y', 'Pos_z']
if not all(col in df.columns for col in required_cols):
    raise ValueError(
        f"The file {FILENAME} is missing one of these columns: {required_cols}"
    )

# Compute data ranges for proportional subplot heights
x_range = df['Pos_x'].max() - df['Pos_x'].min()
y_range = df['Pos_y'].max() - df['Pos_y'].min()
z_range = df['Pos_z'].max() - df['Pos_z'].min()

# Height/width ratio for each subplot (with set_aspect('equal'))
ratio_yz = z_range / y_range  # Subplot 1: Y vs Z
ratio_xz = z_range / x_range  # Subplot 2: X vs Z
ratio_xy = y_range / x_range  # Subplot 3: X vs Y

fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[ratio_xy, ratio_yz, ratio_xz])
axes = [fig.add_subplot(gs[i]) for i in range(3)]


def add_depth_line(ax, h, v, depth, min_lw=1.5, max_lw=3):
    """Draw a trajectory line with thickness and color mapped from depth values."""
    pts = np.column_stack([h, v])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    # Normalize depth to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        d_norm = (depth[:-1] - d_min) / (d_max - d_min)
    else:
        d_norm = np.full_like(depth[:-1], 0.5)
    # Map to linewidth
    lw = min_lw + (max_lw - min_lw) * d_norm
    # Map to color: light gray (0.85) → black (0.0)
    gray = 0.85 - 0.85 * d_norm  # 0.85 down to 0.0
    colors = np.column_stack([gray, gray, gray])
    lc = LineCollection(segments, linewidths=lw, colors=colors, alpha=1.0, rasterized=True)
    ax.add_collection(lc)
    ax.autoscale()

# --- Subplot 1: View from the top (XY-plane, looking along Z, thickness = Z) ---
ax1 = axes[0]
add_depth_line(ax1, df['Pos_x'].values, df['Pos_y'].values, df['Pos_z'].values)
ax1.set_xlabel(r'$x\;\mathrm{[m]}$ (East)')
ax1.set_ylabel(r'$y\;\mathrm{[m]}$ (North)')
ax1.set_title('View from Top')
ax1.set_aspect('equal')

# --- Subplot 2: View from west (YZ-plane, looking along X, thickness = X) ---
ax2 = axes[1]
add_depth_line(ax2, df['Pos_y'].values, df['Pos_z'].values, -df['Pos_x'].values)
ax2.set_xlabel(r'$y\;\mathrm{[m]}$ (North)')
ax2.set_ylabel(r'$z\;\mathrm{[m]}$ (Up)')
ax2.set_title('View from West')
ax2.set_aspect('equal')
ax2.invert_xaxis()

# --- Subplot 3: View from south (XZ-plane, looking along Y, thickness = Y) ---
ax3 = axes[2]
add_depth_line(ax3, df['Pos_x'].values, df['Pos_z'].values, -df['Pos_y'].values)
ax3.set_xlabel(r'$x\;\mathrm{[m]}$ (East)')
ax3.set_ylabel(r'$z\;\mathrm{[m]}$ (Up)')
ax3.set_title('View from South')
ax3.set_aspect('equal')

# --- Draw annotations on all three subplots ---
cmap = plt.cm.tab10
n = max(1, len(ANNOTATIONS) - 1)

texts_per_ax = [[] for _ in axes]

for i, (t_stamp, label) in enumerate(ANNOTATIONS):
    color = cmap(i / n) if n > 0 else cmap(0)
    idx = (df['Time'] - t_stamp).abs().idxmin()
    px, py, pz = df.loc[idx, 'Pos_x'], df.loc[idx, 'Pos_y'], df.loc[idx, 'Pos_z']

    for ax_idx, (ax, (h, v)) in enumerate(zip(
        axes,
        [(px, py), (py, pz), (px, pz)]
    )):
        ax.scatter(h, v, color=color, s=40, zorder=5, alpha=1.0, edgecolors='black', linewidth=0)
        txt = ax.annotate(
            label, (h, v),
            fontsize=8 * textscaling, color='black',
            ha='center', va='bottom',
        )
        texts_per_ax[ax_idx].append(txt)

plt.tight_layout()

# Let matplotlib compute initial text positions, then adjust
fig.canvas.draw()

for ax, texts in zip(axes, texts_per_ax):
    if len(texts) > 1:
        adjust_text(
            texts, ax=ax,
            force_text=0.5,
            expand=(1.05, 1.05),
            only_move={'points': 'xy', 'text': 'xy'},
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.5),
        )

plt.tight_layout()

OUTPUT_PDF = PLOTS / 'trajectory_views.pdf'
plt.savefig(OUTPUT_PDF)
print(f"Successfully saved {OUTPUT_PDF}")
plt.close()
