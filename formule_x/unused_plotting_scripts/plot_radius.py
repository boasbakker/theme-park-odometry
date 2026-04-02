import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

textscaling = 1.5
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 600,
    'xtick.labelsize': 10 * textscaling,
    'ytick.labelsize': 10 * textscaling,
    'axes.labelsize': 12 * textscaling,
    'axes.titlesize': 14 * textscaling,
    'legend.fontsize': 10 * textscaling,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'lines.linewidth': 2,
    'lines.markersize': 3,
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / 'data_processed'
PLOTS = ROOT / 'plots'

FILENAME_VEL = DATA_PROCESSED / 'motion_data.csv'
FILENAME_ACC = DATA_PROCESSED / 'accelerometer_earth.csv'
T_MIN, T_MAX = 18.0, 21.5

PLOTS.mkdir(parents=True, exist_ok=True)

try:
    df_v = pd.read_csv(FILENAME_VEL)
    required_vel_cols = ['Time', 'Vel_x', 'Vel_y', 'Vel_z']
    if not all(col in df_v.columns for col in required_vel_cols):
        raise ValueError(f"{FILENAME_VEL} is missing required columns: {required_vel_cols}")

    df_v['Time'] = pd.to_numeric(df_v['Time'], errors='coerce')
    df_v = df_v.dropna(subset=['Time']).sort_values('Time')

    df_a = pd.read_csv(FILENAME_ACC)
    required_acc_cols = ['Time', 'Accel_x_earth', 'Accel_y_earth', 'Accel_z_earth']
    if not all(col in df_a.columns for col in required_acc_cols):
        raise ValueError(f"{FILENAME_ACC} is missing required columns: {required_acc_cols}")

    df_a['Time'] = pd.to_numeric(df_a['Time'], errors='coerce')
    df_a = df_a.dropna(subset=['Time']).sort_values('Time')

    df = pd.merge_asof(df_v, df_a, on='Time', direction='nearest')

    v = np.sqrt(df['Vel_x']**2 + df['Vel_y']**2 + df['Vel_z']**2)

    dot_prod = (df['Accel_x_earth'] * df['Vel_x'] +
                df['Accel_y_earth'] * df['Vel_y'] +
                df['Accel_z_earth'] * df['Vel_z'])

    v_safe = v.copy()
    v_safe[v_safe == 0] = np.nan
    a_t = dot_prod / v_safe

    a_sq = df['Accel_x_earth']**2 + df['Accel_y_earth']**2 + df['Accel_z_earth']**2

    a_n_sq = a_sq - a_t**2
    a_n_sq = a_n_sq.clip(lower=0)
    a_n = np.sqrt(a_n_sq)

    df = df.assign(Vel_mag=v, Accel_mag=a_n)

    df_win = df[(df['Time'] >= T_MIN) & (df['Time'] <= T_MAX)].copy()

    if df_win.empty:
        print(f"No data found in the time window [{T_MIN}, {T_MAX}] seconds.")
        raise SystemExit(0)

    accel_tol = 1e-8
    small_accel_mask = df_win['Accel_mag'].abs() <= accel_tol
    if small_accel_mask.any():
        print(f"Warning: {small_accel_mask.sum()} sample(s) in the window have near-zero acceleration; they will become NaN in R.")

    R = (df_win['Vel_mag'] ** 2) / df_win['Accel_mag']
    R = R.replace([np.inf, -np.inf], np.nan)

    df_win = df_win.assign(TurnRadius=R)

    plt.figure()
    plt.scatter(df_win['Time'], df_win['TurnRadius'], label=r'Turn radius $R = \frac{v^2}{a_n}$', alpha=1, color='blue', rasterized=True, linewidth=0)
    plt.ylim(0, 20)
    plt.xlabel(r'$t\;[\mathrm{s}]$')
    plt.ylabel(r'$R\;[\mathrm{m}]$')
    plt.legend()
    outname = PLOTS / f"turn_radius_{int(T_MIN)}-{int(T_MAX)}s.pdf"
    plt.savefig(outname)
    print(f"Saved plot to '{outname}'")

    csv_out = DATA_PROCESSED / f"turn_radius_{int(T_MIN)}-{int(T_MAX)}s.csv"
    df_win[['Time', 'Vel_mag', 'Accel_mag', 'TurnRadius']].to_csv(csv_out, index=False)
    print(f"Saved computed table to '{csv_out}'")

except FileNotFoundError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
