#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from rotation_common import (
    find_best_xy_interval_for_heading,
    read_csv_guess,
    vec_to_rotation_from_two_vectors,
)

ROOT = Path(__file__).resolve().parent
DATA_CUT = ROOT / "data_raw_cut"
DATA_PROCESSED = ROOT / "data_processed"
ACCEL_CSV = DATA_CUT / "Accelerometer.csv"
OUT_CSV = DATA_PROCESSED / "accelerometer_cart.csv"

def main():
    print(f"Reading {ACCEL_CSV}...")
    adf = read_csv_guess(ACCEL_CSV)

    accel_cols = [c for c in adf.columns if 'accel' in c.lower() or 'acceleration' in c.lower() or 'acc' in c.lower()]
    if len(accel_cols) == 0:
        non_time = [c for c in adf.columns if c != 'Time']
        accel_cols = non_time[-3:]

    times = np.asarray(adf['Time'], dtype=float)
    accel = np.vstack([
        adf[accel_cols[0]].astype(float), 
        adf[accel_cols[1]].astype(float), 
        adf[accel_cols[2]].astype(float)
    ]).T

    if len(times) < 2:
        raise ValueError('Not enough accelerometer samples')

    t0 = times[0]
    tN = times[-1]

    start_mask = (times <= (t0 + 2.0 + 1e-12))
    end_mask = (times >= (tN - 2.0 - 1e-12))
    
    if np.sum(start_mask) < 3 or np.sum(end_mask) < 3:
        print('Warning: fewer than 3 samples found in 2s gravity windows.')

    g_start_measured = np.mean(accel[start_mask], axis=0)
    g_end_measured = np.mean(accel[end_mask], axis=0)
    
    target_gravity = np.array([0.0, 0.0, 9.81])

    R_start = vec_to_rotation_from_two_vectors(g_start_measured, target_gravity)
    R_end = vec_to_rotation_from_two_vectors(g_end_measured, target_gravity)

    print("Computing drift correction (SLERP between start and end gravity alignments)...")
    
    key_times = [t0, tN]
    key_rots = Rotation.concatenate([R_start, R_end])
    slerp = Slerp(key_times, key_rots)

    rotations_over_time = slerp(times)

    accel_leveled = rotations_over_time.apply(accel)

    print("Detecting launch event to align Y-axis...")
    best_mean_xy, best_idx = find_best_xy_interval_for_heading(
        times, accel_leveled, window_duration=2.0, az_min=9.0, az_max=11.0
    )

    if best_mean_xy is None:
        print('Warning: Could not detect clear launch event. Y-axis alignment might be arbitrary.')
        accel_final = accel_leveled
    else:
        v_des = np.array([0.0, 1.0])

        ang_cur = np.arctan2(best_mean_xy[1], best_mean_xy[0])
        ang_des = np.arctan2(v_des[1], v_des[0])
        theta = ang_des - ang_cur

        R_heading = Rotation.from_euler('z', theta)

        accel_final = R_heading.apply(accel_leveled)
        
        start_idx, end_idx = best_idx
        print(f"Launch detected at {times[start_idx]:.2f}s - {times[end_idx]:.2f}s.")
        print(f"Applied Z-rotation of {np.rad2deg(theta):.2f} deg to align launch to Y+.")

    out_df = pd.DataFrame({
        'Time': times,
        'Accel_x_cart': accel_final[:, 0],
        'Accel_y_cart': accel_final[:, 1],
        'Accel_z_cart': accel_final[:, 2],
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'Successfully wrote rotated data to {OUT_CSV}')
    print("Frame definition: X+=Right, Y+=Forward, Z+=Up")

if __name__ == '__main__':
    main()