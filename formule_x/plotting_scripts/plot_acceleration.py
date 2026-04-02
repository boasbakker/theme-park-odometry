import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
DATA_CUT = ROOT / 'data_raw_cut'
DATA_PROCESSED = ROOT / 'data_processed'
PLOTS = ROOT / 'plots'

tasks = [
    {
        "file": DATA_PROCESSED / 'accelerometer_earth.csv',
        "cols": ['Time', 'Accel_x_earth', 'Accel_y_earth', 'Accel_z_earth'],
        "output": PLOTS / "acceleration_earth.pdf"
    },
    {
        "file": DATA_CUT / 'Accelerometer.csv',
        "cols": ['Time (s)', 'X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
        "output": PLOTS / "acceleration_raw.pdf"
    },
    {
        "file": DATA_PROCESSED / 'accelerometer_cart.csv',
        "cols": ['Time', 'Accel_x_cart', 'Accel_y_cart', 'Accel_z_cart'],
        "output": PLOTS / "acceleration_cart.pdf"
    }
]

PLOTS.mkdir(parents=True, exist_ok=True)

for task in tasks:
    try:
        df = pd.read_csv(task["file"])
        t_col, x_col, y_col, z_col = task["cols"]

        if not all(col in df.columns for col in task["cols"]):
            print(f"Skipping {task['file']}: Missing columns.")
            continue

        plt.figure()
        plt.scatter(df[t_col], df[x_col], label='x', color='blue', alpha=0.5, rasterized=True, linewidth=0)
        plt.scatter(df[t_col], df[y_col], label='y', color='orange', alpha=0.5, rasterized=True, linewidth=0)
        plt.scatter(df[t_col], df[z_col], label='z', color='green', alpha=0.5, rasterized=True, linewidth=0)

        plt.xlabel(r'$t\;[\mathrm{s}]$')
        plt.ylabel(r'$a\;[\mathrm{m}/\mathrm{s}^{2}]$')

        legend = plt.legend(markerscale=3)

        for handle in legend.legend_handles:
            handle.set_alpha(1)

        plt.savefig(task["output"])
        print(f"Successfully saved {task['output']}")
        plt.close()

    except FileNotFoundError:
        print(f"Error: Could not find '{task['file']}'")
    except Exception as e:
        print(f"An error occurred processing {task['file']}: {e}")