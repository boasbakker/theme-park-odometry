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
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'lines.linewidth': 2,
    'figure.dpi': 600
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / 'data_processed'
PLOTS = ROOT / 'plots'
FILENAME = DATA_PROCESSED / 'motion_data.csv'
OUTPUT_PDF = PLOTS / 'energy_profile.pdf'
G = 9.81

PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILENAME)

required_cols = ['Time', 'Vel_x', 'Vel_y', 'Vel_z', 'Pos_z']
if not all(col in df.columns for col in required_cols):
    raise ValueError(
        f"The file {FILENAME} is missing one of these columns: {required_cols}"
    )

v_squared = df['Vel_x']**2 + df['Vel_y']**2 + df['Vel_z']**2
kinetic_energy = 0.5 * v_squared
potential_energy =  G * df['Pos_z']
total_energy = kinetic_energy + potential_energy

plt.figure()

plt.plot(df['Time'], kinetic_energy, label='Kinetic', color='blue', rasterized=True)
plt.plot(df['Time'], potential_energy, label='Gravitational', color='orange', rasterized=True)
plt.plot(df['Time'], total_energy, label='Total', linestyle='--', color='green', rasterized=True)

plt.xlabel(r'$t\;[\mathrm{s}]$')
plt.ylabel(r'$E/m\;[\mathrm{m}^2/\mathrm{s}^{2}]$')

plt.legend()
plt.savefig(OUTPUT_PDF)