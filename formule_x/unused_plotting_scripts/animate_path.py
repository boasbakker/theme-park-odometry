#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
INPUT_CSV = DATA_PROCESSED / "motion_data.csv"
ACCEL_CSV = DATA_PROCESSED / "accelerometer_earth.csv"

def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    print(f"Loading {INPUT_CSV} for visualization...")
    df = pd.read_csv(INPUT_CSV)

    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.sort_values('Time')

    has_accel = False
    if ACCEL_CSV.exists():
        print(f"Loading {ACCEL_CSV} for tangent circle calculation...")
        df_accel = pd.read_csv(ACCEL_CSV)
        df_accel['Time'] = pd.to_numeric(df_accel['Time'], errors='coerce')
        df_accel = df_accel.dropna(subset=['Time']).sort_values('Time')

        df = pd.merge_asof(df, df_accel, on='Time', direction='nearest')
        has_accel = True
    else:
        print(f"Warning: {ACCEL_CSV} not found. Tangent circle will not be shown.")

    t = df['Time'].to_numpy()
    x = df['Pos_x'].to_numpy()
    y = df['Pos_y'].to_numpy()
    z = df['Pos_z'].to_numpy()

    circle_data = {'R': None, 'N': None, 'T': None}
    if has_accel and all(c in df.columns for c in ['Vel_x', 'Accel_x_earth']):
        v_vec = np.column_stack((df['Vel_x'], df['Vel_y'], df['Vel_z']))
        a_vec = np.column_stack((df['Accel_x_earth'], df['Accel_y_earth'], df['Accel_z_earth']))

        v_norm = np.linalg.norm(v_vec, axis=1)
        v_norm_safe = np.where(v_norm == 0, 1e-9, v_norm)
        T_hat = v_vec / v_norm_safe[:, None]

        at_mag = np.einsum('ij,ij->i', a_vec, T_hat)
        at_vec = T_hat * at_mag[:, None]

        an_vec = a_vec - at_vec
        an_mag = np.linalg.norm(an_vec, axis=1)

        an_mag_safe = np.where(an_mag == 0, 1e-9, an_mag)
        N_hat = an_vec / an_mag_safe[:, None]

        with np.errstate(divide='ignore', invalid='ignore'):
            R = (v_norm**2) / an_mag
            R[an_mag < 0.05] = np.nan

        circle_data['R'] = R
        circle_data['N'] = N_hat
        circle_data['T'] = T_hat

    pos = np.column_stack((x, y, z))

    fig = plt.figure(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection='3d')

    last_elev = [30]

    def on_mouse_move(event):
        if event.inaxes == ax and event.button == 1:
            elev = np.clip(ax.elev, 0.1, 89.9)
            ax.view_init(elev=elev, azim=ax.azim)
            last_elev[0] = elev
        elif event.inaxes == ax and (ax.elev < 0 or ax.elev > 90):
            ax.view_init(elev=last_elev[0], azim=ax.azim)

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    ax.view_init(elev=last_elev[0], azim=-60)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('Reconstructed 3D Path')

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    dx, dy, dz = (np.ptp(c) if np.ptp(c) > 0 else 1.0 for c in (x, y, z))
    ax.set_box_aspect((dx, dy, dz))

    ax.scatter(x[0], y[0], z[0], marker='o', color='g', s=50, label='Start')
    ax.legend()

    line, = ax.plot([], [], [], lw=2, color='blue', label='Path')
    point, = ax.plot([], [], [], 'o', color='orange', markersize=5, label='Current Pos')
    circle_line, = ax.plot([], [], [], lw=1, color='red', alpha=0.6, label='Tangent Circle')
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    class AnimationState:
        def __init__(self, timestamps):
            self.timestamps = timestamps
            self.start_timestamp = self.timestamps[0]
            self.total_duration = self.timestamps[-1] - self.start_timestamp
            if self.total_duration == 0:
                self.total_duration = 1.0
            self.start_time = time.time()
            self.speed_multiplier = 1.0

    anim_state = AnimationState(t)

    def animate(frame):
        elapsed_time = time.time() - anim_state.start_time
        current_animation_time = (elapsed_time * anim_state.speed_multiplier) % anim_state.total_duration
        current_timestamp = anim_state.start_timestamp + current_animation_time

        frame_index = np.searchsorted(anim_state.timestamps, current_timestamp)
        if frame_index >= len(anim_state.timestamps):
            frame_index = len(anim_state.timestamps) - 1

        line.set_data(x[:frame_index + 1], y[:frame_index + 1])
        line.set_3d_properties(z[:frame_index + 1])

        point.set_data([x[frame_index]], [y[frame_index]])
        point.set_3d_properties([z[frame_index]])

        if circle_data['R'] is not None and frame_index < len(circle_data['R']):
            R_val = circle_data['R'][frame_index]
            if not np.isnan(R_val) and R_val < 500:
                P_curr = np.array([x[frame_index], y[frame_index], z[frame_index]])
                N_hz = circle_data['N'][frame_index]
                T_hz = circle_data['T'][frame_index]

                C = P_curr + R_val * N_hz

                th = np.linspace(0, 2 * np.pi, 100)
                circle_pts = C + R_val * (np.outer(np.cos(th), -N_hz) + np.outer(np.sin(th), T_hz))

                circle_line.set_data(circle_pts[:, 0], circle_pts[:, 1])
                circle_line.set_3d_properties(circle_pts[:, 2])
            else:
                circle_line.set_data([], [])
                circle_line.set_3d_properties([])

        time_text.set_text(f'Time: {anim_state.timestamps[frame_index]:.2f}s')
        return line, point, circle_line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=None, interval=15, blit=False)

    ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
    speed_slider = Slider(
        ax=ax_slider,
        label='Speed',
        valmin=0.1,
        valmax=10.0,
        valinit=1.0,
        valfmt='%1.1fx'
    )

    def update_speed(val):
        anim_state.speed_multiplier = val

    speed_slider.on_changed(update_speed)

    plt.show()

if __name__ == "__main__":
    main()