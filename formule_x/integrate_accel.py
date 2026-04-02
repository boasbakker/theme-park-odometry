#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data_processed"
INPUT_CSV = DATA_PROCESSED / "accelerometer_earth.csv"
OUTPUT_CSV = DATA_PROCESSED / "motion_data.csv"

# --- COASTER SPECIFIC SETTINGS ---
# Adjust these values when applying to a different roller coaster
STATIONARY_WINDOW = 2.0        # Seconds of stationary data at start/end to establish gravity subtraction
G = 9.80665                    # Default gravity constant, if stationary window cannot be found
# ---------------------------------

def read_accel_csv(path):
    df = pd.read_csv(path)
    time_col = [c for c in df.columns if 'Time' in c or 'time' in c.lower()][0]
    acc_cols = [c for c in df.columns if c.strip().lower().startswith('acc')]
    
    if len(acc_cols) < 3:
        raise ValueError("Couldn't find 3 acceleration columns starting with 'Acc'.")
        
    acc_x, acc_y, acc_z = acc_cols[0], acc_cols[1], acc_cols[2]
    t = df[time_col].to_numpy(dtype=float)
    acc = df[[acc_x, acc_y, acc_z]].to_numpy(dtype=float)
    return t, acc

def trapezoidal_integrate_vector(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    N, M = y.shape
    out = np.zeros_like(y)
    for i in range(1, N):
        dt = t[i] - t[i-1]
        out[i] = out[i-1] + 0.5 * (y[i-1] + y[i]) * dt
    return out

def remove_linear_ramp_by_endpoint(arr, t):
    T = t[-1] - t[0]
    if T <= 0:
        return arr.copy()
    tau = (t - t[0]) / T
    ramp = np.outer(tau, arr[-1])
    return arr - ramp

def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    print(f"Reading {INPUT_CSV}...")
    t, acc = read_accel_csv(INPUT_CSV)
    
    acc_corr = acc.copy()
    mask_static = (t <= (t[0] + STATIONARY_WINDOW)) | (t >= (t[-1] - STATIONARY_WINDOW))
    g_est = np.mean(acc[mask_static, 2]) if np.any(mask_static) else G
    acc_corr[:, 2] -= g_est
    
    print("Integrating Acceleration -> Velocity...")
    vel = trapezoidal_integrate_vector(t, acc_corr)

    print(f"Applying linear drift correction (Velocity). {np.linalg.norm(vel[-1])} m/s")
    vel_corr = remove_linear_ramp_by_endpoint(vel, t)

    vel_norm = np.linalg.norm(vel_corr, axis=1)
    i_vmax = int(np.nanargmax(vel_norm))
    v_max = vel_corr[i_vmax]
    print(
        "Maximum velocity: "
        f"vector=[{v_max[0]:.6f}, {v_max[1]:.6f}, {v_max[2]:.6f}] m/s, "
        f"norm={vel_norm[i_vmax]:.6f} m/s, time={t[i_vmax]:.6f} s"
    )

    print("Integrating Velocity -> Position...")
    pos = trapezoidal_integrate_vector(t, vel_corr)
    
    print(f"Applying linear drift correction (Position). {np.linalg.norm(pos[-1])} m")
    pos_corr = remove_linear_ramp_by_endpoint(pos, t)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {OUTPUT_CSV}...")
    df_out = pd.DataFrame({
        'Time': t,
        'Vel_x': vel_corr[:, 0],
        'Vel_y': vel_corr[:, 1],
        'Vel_z': vel_corr[:, 2],
        'Pos_x': pos_corr[:, 0],
        'Pos_y': pos_corr[:, 1],
        'Pos_z': pos_corr[:, 2]
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()