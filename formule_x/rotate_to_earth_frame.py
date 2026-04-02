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
GYRO_CSV = DATA_CUT / "Gyroscope.csv"
ACCEL_CSV = DATA_CUT / "Accelerometer.csv"
OUT_ACCEL_CSV = DATA_PROCESSED / "accelerometer_earth.csv"
OUT_ORIENTATION_CSV = DATA_PROCESSED / "orientation.csv"

launch_slope = -0.35


def integrate_gyro_to_rotations(times, gyro_xyz):
    """
    Integrate gyroscope (rad/s) given at times to produce Rotation objects mapping body->Earth.
    Assumes gyro_xyz is Nx3 and times is length N, or gyro has been interpolated to the desired time grid.

    Returns list (or array) of Rotation objects of same length as times where r[0] should be set by caller to initial orientation.
    This function returns incremental rotations (dR) between successive samples as Rotation objects.
    """
    n = len(times)
    dRs = [None] * n
    dRs[0] = Rotation.identity()
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            dt = 1e-12
        omega = gyro_xyz[i - 1]
        rotvec = omega * dt
        dR = Rotation.from_rotvec(rotvec)
        dRs[i] = dR
    return dRs


def quaternion_integration_body_frame(initial_rotation, dRs):
    """
    Given an initial Rotation (scipy Rotation) and per-sample body-frame incremental rotations dRs
    (Rotation objects representing rotation produced by body angular velocity over dt), compute the
    orientation sequence r_k where r_{k+1} = r_k * dR_{k+1} (right-multiplication), which corresponds to
    integrating body-frame angular velocities.
    Returns list of Rotation objects.
    """
    rots = [None] * len(dRs)
    r = initial_rotation
    rots[0] = r
    for i in range(1, len(dRs)):
        r = r * dRs[i]
        rots[i] = r
    return rots


def slerp_correction(times, target_rot):
    """
    Create a sequence of correction rotations that interpolates from identity (no correction)
    at the start time to the target_rot (the full correction) at the end time.

    target_rot is a Rotation object representing the full correction to be applied in the Earth frame.
    Returns Rotation objects of same length as times.
    """
    # build two key rotations: identity at t0, target_rot at tN
    key_times = [times[0], times[-1]]
    key_rots = Rotation.concatenate([Rotation.identity(), target_rot])
    slerp = Slerp(key_times, key_rots)
    return slerp(times)


def main():
    gdf = read_csv_guess(GYRO_CSV)
    adf = read_csv_guess(ACCEL_CSV)

    # Heuristics to find columns
    gyro_cols = [c for c in gdf.columns if 'gyro' in c.lower() or 'gyroscope' in c.lower() or 'gyr' in c.lower()]
    if len(gyro_cols) == 0:
        non_time = [c for c in gdf.columns if c != 'Time']
        gyro_cols = non_time[-3:]
    accel_cols = [c for c in adf.columns if 'accel' in c.lower() or 'acceleration' in c.lower() or 'acc' in c.lower()]
    if len(accel_cols) == 0:
        non_time = [c for c in adf.columns if c != 'Time']
        accel_cols = non_time[-3:]

    g_times = np.asarray(gdf['Time'], dtype=float)
    gyro = np.vstack([gdf[gyro_cols[0]].astype(float), gdf[gyro_cols[1]].astype(float), gdf[gyro_cols[2]].astype(float)]).T

    a_times = np.asarray(adf['Time'], dtype=float)
    accel = np.vstack([adf[accel_cols[0]].astype(float), adf[accel_cols[1]].astype(float), adf[accel_cols[2]].astype(float)]).T

    times = a_times.copy()

    if len(times) < 2:
        raise ValueError('Not enough accelerometer samples')

    t0 = times[0]
    tN = times[-1]

    # find first 2s and last 2s windows (based on accelerometer times)
    start_mask = (times <= (t0 + 2.0 + 1e-12))
    end_mask = (times >= (tN - 2.0 - 1e-12))
    if np.sum(start_mask) < 3 or np.sum(end_mask) < 3:
        print('Warning: fewer than 3 samples found in 2s gravity windows. Results may be noisy.')

    g_start = np.mean(accel[start_mask], axis=0)
    g_end = np.mean(accel[end_mask], axis=0)

    # target gravity vector in Earth frame
    g_target = np.array([0.0, 0.0, 9.81])

    # initial rotation that maps start gravity to Earth gravity direction
    R_init = vec_to_rotation_from_two_vectors(g_start, g_target)

    # Interpolate gyro to accel times
    gyro_interp = np.zeros((len(times), 3), dtype=float)
    for i in range(3):
        gyro_interp[:, i] = np.interp(times, g_times, gyro[:, i])
    omega_norm = np.linalg.norm(gyro_interp, axis=1)
    i_omax = int(np.nanargmax(omega_norm))
    omega_max = gyro_interp[i_omax]
    print(
        "Maximum angular velocity: "
        f"vector=[{omega_max[0]:.6f}, {omega_max[1]:.6f}, {omega_max[2]:.6f}] rad/s, "
        f"norm={omega_norm[i_omax]:.6f} rad/s, time={times[i_omax]:.6f} s"
    )

    # integrate gyro: compute per-sample incremental rotations
    dRs = integrate_gyro_to_rotations(times, gyro_interp)

    # integrate quaternions starting from R_init (maps body->Earth at start)
    rots = quaternion_integration_body_frame(R_init, dRs)

    # rotate accelerations at each sample into Earth frame using current orientation
    accel_rotated = np.vstack([r.apply(v) for r, v in zip(rots, accel)])

    # compute what the average of the last 2s acceleration looks like after rotation
    accel_rotated_end_mean = np.mean(accel_rotated[end_mask], axis=0)

    # desired final average is g_target. compute correction rotation in Earth frame that maps accel_rotated_end_mean to g_target
    R_drift = vec_to_rotation_from_two_vectors(accel_rotated_end_mean, g_target)

    # Print angular velocity drift
    T_total = tN - t0
    drift_rotvec = R_drift.as_rotvec()
    angular_velocity_drift = np.rad2deg(np.linalg.norm(drift_rotvec) / T_total)
    print(
        "Total angular drift (rad): "
        f"vector=[{drift_rotvec[0]:.6f}, {drift_rotvec[1]:.6f}, {drift_rotvec[2]:.6f}]"
    )
    print(f"Angular velocity drift correction: {np.linalg.norm(drift_rotvec)} / {T_total} = {angular_velocity_drift:.6f} deg/s")

    # create correction rotations applied smoothly from start to end (in Earth frame)
    correction_rots = slerp_correction(times, R_drift)

    # apply correction rots to accel_rotated
    accel_rotated_corrected = np.vstack([corr.apply(v) for corr, v in zip(correction_rots, accel_rotated)])

    best_mean_xy, best_idx = find_best_xy_interval_for_heading(times, accel_rotated_corrected, window_duration=2.0, az_min=9.0, az_max=11.0)

    if best_mean_xy is None:
        raise RuntimeError(
            'Error: no 2-second interval found with 9 < a_z_avg < 11 and '
            '5 < a_z_min and a_z_max < 15; automatic Z rotation is required, exiting.'
        )
    else:
        # desired direction in XY plane
        v_des = np.array([1, launch_slope])
        # compute angles
        ang_cur = np.arctan2(best_mean_xy[1], best_mean_xy[0])
        ang_des = np.arctan2(v_des[1], v_des[0])
        theta = ang_des - ang_cur
        # build rotation about Earth Z by theta
        R_heading = Rotation.from_euler('z', theta)
        accel_final = np.vstack([R_heading.apply(v) for v in accel_rotated_corrected])
        start_idx, end_idx = best_idx
        start_time = times[start_idx]
        end_time = times[end_idx]
        print(
            f"Applied automatic Z-rotation of {np.rad2deg(theta):.3f} deg "
            f"to align best interval (indices {start_idx}..{end_idx}, "
            f"times {start_time:.6f}s..{end_time:.6f}s) to direction {v_des}."
        )

    final_rots = [R_heading * corr_rot * rot for corr_rot, rot in zip(correction_rots, rots)]

    final_rots_stack = Rotation.concatenate(final_rots)

    quats_xyzw = final_rots_stack.as_quat()
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]

    eulers_deg = final_rots_stack.as_euler('zyx', degrees=True)

    orientation_df = pd.DataFrame({
        'Time': times,
        'q_w': quats_wxyz[:, 0],
        'q_x': quats_wxyz[:, 1],
        'q_y': quats_wxyz[:, 2],
        'q_z': quats_wxyz[:, 3],
        'yaw_deg': eulers_deg[:, 0],
        'pitch_deg': eulers_deg[:, 1],
        'roll_deg': eulers_deg[:, 2],
    })
    OUT_ORIENTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    orientation_df.to_csv(OUT_ORIENTATION_CSV, index=False, float_format='%.15g')
    print(f'Wrote orientation to {OUT_ORIENTATION_CSV}')

    out_df = pd.DataFrame({
        'Time': times,
        'Accel_x_earth': accel_final[:, 0],
        'Accel_y_earth': accel_final[:, 1],
        'Accel_z_earth': accel_final[:, 2],
    })

    accel_norm = np.linalg.norm(accel_final, axis=1)
    i_amax = int(np.nanargmax(accel_norm))
    a_max = accel_final[i_amax]
    print(
        "Maximum acceleration: "
        f"vector=[{a_max[0]:.6f}, {a_max[1]:.6f}, {a_max[2]:.6f}] m/s^2, "
        f"norm={accel_norm[i_amax]:.6f} m/s^2, time={times[i_amax]:.6f} s"
    )

    out_df.to_csv(OUT_ACCEL_CSV, index=False)

    print(f'Wrote rotated accelerometer to {OUT_ACCEL_CSV}')


if __name__ == '__main__':
    main()