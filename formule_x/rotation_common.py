from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def read_csv_guess(file_path, time_col_prefix='Time'):
    """Read CSV and normalize the first matching time column to `Time`."""
    file_path = Path(file_path)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        raise

    time_col = None
    for c in df.columns:
        if c.strip().lower().startswith(time_col_prefix.lower()):
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column starting with '{time_col_prefix}' found in {file_path}")
    return df.rename(columns={time_col: 'Time'})


def vec_to_rotation_from_two_vectors(a, b, eps=1e-8):
    """Return a scipy Rotation that rotates vector `a` to vector `b`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        raise ValueError("Zero-length vector passed to vec_to_rotation_from_two_vectors")

    ua = a / na
    ub = b / nb
    dot = np.dot(ua, ub)

    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0

    if abs(dot - 1.0) < 1e-8:
        return Rotation.identity()
    if abs(dot + 1.0) < 1e-8:
        axis = np.cross(ua, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(ua, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)

    axis = np.cross(ua, ub)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle)


def find_best_xy_interval_for_heading(times, accel_xyz_earth, window_duration=2.0, az_min=9.0, az_max=11.0):
    """
    Find the launch-like interval with near-1g vertical acceleration and strongest mean XY acceleration.

    Returns:
        tuple[np.ndarray | None, tuple[int | None, int | None]]
    """
    n = len(times)
    best_mag = -1.0
    best_mean_xy = None
    best_idx = (None, None)

    for i in range(n):
        t_end = times[i] + window_duration
        j = np.searchsorted(times, t_end, side='right') - 1
        if j <= i:
            continue

        window = accel_xyz_earth[i:j + 1]
        az_mean = np.mean(window[:, 2])
        az_min_window = np.min(window[:, 2])
        az_max_window = np.max(window[:, 2])

        if not (az_min < az_mean < az_max):
            continue
        if not (az_min_window > 5.0 and az_max_window < 15.0):
            continue

        mean_xy = np.mean(window[:, 0:2], axis=0)
        mag_xy = np.linalg.norm(mean_xy)
        if mag_xy > best_mag:
            best_mag = mag_xy
            best_mean_xy = mean_xy
            best_idx = (i, j)

    return best_mean_xy, best_idx
