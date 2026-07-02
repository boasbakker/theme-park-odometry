import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Frank de Kogel, 19/09/2025

begin_index = 7000 # From which index does the ride start?
end_index = 17500 # At which index does the ride end? (0 to include every measurement)

# A nice excerpt from Finn's measurements:
begin_index, end_index = 12623, 13387

# Period pendulum ship = 7.61

accelerometer_path = "data_raw/Accelerometer.csv"
orientation_path = "data_raw/Orientation.csv"

g = 9.81 # set to 0 to not subtract gravity
rounding_decimals = 8

use_rotation = True # Should the inverse phone rotation be used?
use_correction = True # Should corrections be applied so that the start position equals the end position?
use_extra_correction = False # Should the sixth-degree polynomial be used for extra correction of the potential energy drift?
plot_calculated_positions = False # Should a graph of the positions be created?
plot_calculated_velocities = False # Should the velocity be in a separate graph?
plot_acceleration = False # Should the acceleration be plotted? (for raw rotation: set rotation=False. use_correction=False, and g=0)
plot_abs_velocity = False # Should the calculated absolute velocity be plotted?
plot_abs_acceleration = False # Should the absolute acceleration be plotted?
plot_energies = False # Should the total energy be plotted?
plot_orientation = False # Should the phone orientation (according to Phyphox) be shown?
plot_2d_path = True
plot_calculated_radii = False


accel_correction = np.array([0., 0., 0.]) # Acceleration correction in absolute plane
vel_correction = np.array([0., 0., 0.])



vel0 = np.array((0., 0., 0.)) # Initial velocity

pos0 = np.array((0., 0., 0.)) # Initial position

def transform_rotation(phone_accel, orientation):

    # Data shape:
    # phone_accel = [phone_accelX, phone_accelY, phone_accelZ]
    # orientation = [wQuaternion, xQuaternion, yQuaternion, zQuaternion]

    # I spent a good twenty (correction: forty) hours on this, because quaternions and relative angles are an abomination.

    # This function serves to map the XYZ axes (rotating with the phone) to stationary axes (fixed to the earth).
    # I tried to achieve this with orientation angles, yaw, pitch, roll, and at one point had twenty-nine trigonometric functions in a single vector.
    # Do not attempt to understand this, it is difficult.

    # Thanks to the numerous heroes on math stackoverflow, who helped me with quaternions through questions from eight years ago


    # W = cos (0.5 × totale hoek)
    # X = x × sin (0.5 × xHoek)
    # Y = y × sin (0.5 × yHoek)
    # Z = z × sin (0.5 × zHoek)

    rotation_matrix = np.array([[2 * (orientation[0] * orientation[0] + orientation[1] * orientation[1]) - 1,
                                     2 * (orientation[1] * orientation[2] - orientation[0] * orientation[3]),
                                     2 * (orientation[1] * orientation[3] + orientation[0] * orientation[2])],
                                    [2 * (orientation[1] * orientation[2] + orientation[0] * orientation[3]),
                                     2 * (orientation[0] * orientation[0] + orientation[2] * orientation[2]) - 1,
                                     2 * (orientation[2] * orientation[3] - orientation[0] * orientation[1])],
                                    [2 * (orientation[1] * orientation[3] - orientation[0] * orientation[2]),
                                     2 * (orientation[2] * orientation[3] + orientation[0] * orientation[1]),
                                     2 * (orientation[0] * orientation[0] + orientation[3] * orientation[3]) - 1]]) # Yep, looks nice doesn't it!

    rotation_matrix = np.transpose(rotation_matrix) 
    converted_vector = np.array([0.0, 0.0, 0.0])
   
    np.vecmat(phone_accel, rotation_matrix, out=converted_vector) # Here the phone's matrix is rotated so that the Z-axis is pointing up.

    return converted_vector

def parse_raw_data(accelerometer_path, orientation_path, rounding_decimals, g, accel_correction, vel_correction, vel0, begin_pos, use_rotation):

    global end_index
    global begin_index

    with open(accelerometer_path, 'r') as acceleration_raw_data_file: # Opens the accelerometer file from phyphox
        acceleration_raw_data = acceleration_raw_data_file.read() # Reads the data file
   
    with open(orientation_path, 'r') as orientation_raw_data_file:
        orientation_raw_data = orientation_raw_data_file.read()
   
    # Data acquired

    time_list = [0.0] # Lists to store data in

    phone_accel_list = [[0.0, 0.0, -9.81]] # Acceleration is needed to measure velocity and position

    orientation_list = [[0.0, 0.0, 0.0, 0.0]] # Lists for orientation quaternions (that is a new concept, I hope it works) (it worked!)

    net_accel_list = [np.copy(accel_correction)] # Keep track of all accelerations

    abs_accel_list = [0.0]

    velocity = np.array([0., 0., 0.])
   
    for i in range(len(vel0)):
        velocity[i] = vel0[i] # Variable to determine current velocity

    velocity_list = [np.copy(velocity)] # List to keep track of velocities

    abs_velocity_list = [0.0]

    pos = np.array([0., 0., 0.])

    for i in range(len(begin_pos)):
        pos[i] = begin_pos[i] # Current position

    pos_list = [np.copy(begin_pos)] # List to keep track of position

    # Extract the acceleration measurements:

    for line in acceleration_raw_data.split("\n"): # Splits the raw data into measurement points
        try: # A 'try' must be used, because there is text at the top of the data.
            # Format:
            # "Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"

            measurement_time = round(float(line.split(",")[0]), rounding_decimals)
            x_accel = round(float(line.split(",")[1]), rounding_decimals)
            y_accel = round(float(line.split(",")[2]), rounding_decimals)
            z_accel = round(float(line.split(",")[3]), rounding_decimals)

            time_list.append(measurement_time) # Save the measurement
            phone_accel_list.append([x_accel, y_accel, z_accel])


        except (ValueError, IndexError):
            pass
   
    for line in orientation_raw_data.split("\n"):
        try:
           
            orientation_list.append([float(line.split(",")[1]),
                                    float(line.split(",")[2]),
                                    float(line.split(",")[3]),
                                    float(line.split(",")[4])]) # Adding the orientation
           
        except (ValueError, IndexError): # In case there is text (ew) in the data
            pass
   
    # All measurements are now saved

    time_list = np.array(time_list)
    phone_accel_list = np.array(phone_accel_list)
    orientation_list = np.array(orientation_list)

    # Now the measurements need to be 'cut' to the amount of data that will actually be used



    if end_index == 0: # Use all measurements
        end_index = min(len(time_list), len(phone_accel_list), len(orientation_list)) - 1 # shortest list

    phone_accel_list = phone_accel_list[begin_index:end_index]
    orientation_list = orientation_list[begin_index:end_index]
    time_list = time_list[begin_index:end_index + 1] # Logic, because an initial position has been added to the position list
   

    # Calculate the time between measurement points
    dT = (time_list[len(time_list)-1] - time_list[0]) / len(time_list)  # Last time - first time / number of times

    for indx in range(len(time_list)-1): # Goes over every measurement point

        if use_rotation:
            # Convert the relative acceleration into stable acceleration
            stable_acceleration_vector = transform_rotation(phone_accel_list[indx], orientation_list[indx])
           
        else:
            stable_acceleration_vector = phone_accel_list[indx]

        g_vector = np.array([0.0, 0.0, float(g)])

        # Subtract gravity from the acceleration
        stable_acceleration_vector -= g_vector

        # Use correction in fixed space (still needs adaptation)
        stable_acceleration_vector += accel_correction

        net_accel_list.append(np.copy(stable_acceleration_vector))

        velocity += stable_acceleration_vector * dT

        velocity_list.append(np.copy(velocity))

        abs_velocity_list.append(np.linalg.norm(velocity))
        abs_accel_list.append(np.linalg.norm(stable_acceleration_vector))

        pos += velocity * dT + vel_correction * dT

        pos_list.append(np.copy(pos))

    pos_list = np.array(pos_list)
    velocity_list = np.array(velocity_list)
    net_accel_list = np.array(net_accel_list)
    time_list = np.array(time_list)
    abs_velocity_list = np.array(abs_velocity_list).reshape(-1, 1) # reshape slightly to match vstack format when extracted later
    abs_accel_list = np.array(abs_accel_list).reshape(-1, 1)

    return pos_list, velocity_list, net_accel_list, time_list, abs_velocity_list, abs_accel_list, orientation_list


def correct(use_double_correction=True):
    global accel_correction
    global vel_correction

    # First do a run without any corrections:
    pos_list, velocity_list, accelList, time_list, abs_velocity_list, abs_accel_list, orientation_list = parse_raw_data(accelerometer_path, orientation_path, rounding_decimals, g, accel_correction, vel_correction, vel0, pos0, use_rotation)
    # First performs an acceleration correction (vEnd = vBegin)
    # Then performs a 
    begin_vel = velocity_list[0]
    begin_time = time_list[0]

    print("Initial velocity:", begin_vel)
    
    end_vel = velocity_list[len(velocity_list)-1]
    end_time = time_list[len(velocity_list)-1]

    print("End velocity:", end_vel)

    delta_vel = end_vel - begin_vel
    delta_t = end_time - begin_time

    if not use_double_correction:
        accel_correction = -(1/2) * accel_correction

    # Correction in speed = delta V / delta T

    accel_correction = -delta_vel / delta_t

    print("Acceleration correction:")
    print(accel_correction)

    # Then do a run with only the acceleration correction

    pos_list, velocity_list, accelList, time_list, abs_velocity_list, abs_accel_list, orientation_list = parse_raw_data(accelerometer_path, orientation_path, rounding_decimals, g, accel_correction, vel_correction, vel0, pos0, use_rotation)

    # Performs the same correction as for velocity, but then for position

    begin_pos = pos_list[0]
    begin_time = time_list[0]
    end_pos = pos_list[len(pos_list)-1]
    end_time = time_list[len(pos_list)-1]

    delta_t = end_time - begin_time
    delta_pos = end_pos - begin_pos

    vel_correction = -delta_pos / delta_t

    print("Initial position:", begin_pos)
    print("End position:", end_pos)
    print("Velocity correction:", vel_correction)

if use_correction:
    correct()


pos_list, velocity_list, accelList, time_list, abs_velocity_list, abs_accel_list, orientation_list = parse_raw_data(accelerometer_path, orientation_path, rounding_decimals, g, accel_correction, vel_correction, vel0, pos0, use_rotation)

# You now have data in pos_list, velocity_list, accelList, abs_velocity_list, abs_accel_list with corresponding times time_list

# Calculation of energies:

def calc_potential_energy(pos_list):
    # Given that the z-axis is now equal to the vertical axis, it can be stated that this is the axis on which gravity purely acts
    # E_pot = m g h
    return pos_list.transpose()[2] * g - (min(pos_list.transpose()[2])) * g# End result in Joule / kg, where zero potential energy equals the bottom of the movement

def calc_kinetic_energy(abs_velocity_list):
    # It is assumed that there is no rotational energy, only kinetic energy in the (linear) movement of the ship

    # E_kin = 0.5 * m * v^2

    return 0.5 * (abs_velocity_list.transpose()[0].transpose())**2 # Result in Joule / kg

def e_pot_min_curve_fit(t, a, b, c, d, e, f, g): # Simple curvefit to absorb the drift in the potential energy minima
    return (a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5 + g * t ** 6)

def find_e_pot_minima(time_list, E_pot):
    flip_points = []
    n = 10
    
    # Require a (2n+1)-point "V" shape
    for i in range(n, len(E_pot) - n):
        is_v_shape = True
        
        # Check falling left side and rising right side
        for j in range(1, n + 1):
            if not (E_pot[i - j + 1] < E_pot[i - j]) or not (E_pot[i + j - 1] < E_pot[i + j]):
                is_v_shape = False
                break
                    
        if is_v_shape:
            flip_points.append(i)
            
    return time_list[flip_points], E_pot[flip_points]

def half_circle_fit(x, xCenter, yCenter, radius):
    return -1 * np.sqrt(radius**2 - (x-xCenter)**2) + yCenter

def horizontal_line_fit(x, y):
    return y

def find_circle_params(print_params=False):
    """Returns circle d_center, z_center and radius"""
    x_list = np.transpose(pos_list)[0]
    y_list = np.transpose(pos_list)[1]
    z_list = np.transpose(pos_list)[2]
    d_list = np.sqrt(y_list**2 + x_list**2)
    val, cov = curve_fit(half_circle_fit, d_list, z_list, p0=[10, 5, 12])

    if print_params: print("Radius:", round(val[2], 4), "+-", round(np.sqrt(cov[2][2]), 4), "m")
    return val[0], val[1], val[2], np.sqrt(cov[2][2])

def update_plot_style(text_scaling):
    # Code by Boas Bakker for universal plots
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'xtick.labelsize': 10 * text_scaling,
        'ytick.labelsize': 10 * text_scaling,
        'axes.labelsize': 12 * text_scaling,
        'axes.titlesize': 14 * text_scaling,
        'legend.fontsize': 10 * text_scaling,
        'axes.grid': True,
        'lines.linewidth': 2,
        'lines.markersize': 2,
        'figure.dpi': 600
    })

update_plot_style(1.5)

if plot_calculated_positions:
    x_list = np.transpose(pos_list)[0]
    y_list = np.transpose(pos_list)[1]
    z_list = np.transpose(pos_list)[2]

    fig3d = plt.figure()
    ax = fig3d.add_subplot(projection='3d')
    ax.scatter(x_list, y_list, z_list)
    ax.axis('equal')
    ax.set_xlabel("X (Noord-Zuid)")
    ax.set_ylabel("Y (Oost-West)")
    ax.set_zlabel("Hoogte")

if plot_calculated_velocities:
    velplot = plt.figure(figsize=(10, 6))
    plt.plot(time_list, velocity_list.transpose()[0], 'b.', label='V$_x$')
    plt.plot(time_list, velocity_list.transpose()[1], '.', color='orange', label='V$_y$')
    plt.plot(time_list, velocity_list.transpose()[2], 'g.', label='V$_z$')
    plt.xlabel("$t$ [s]")
    plt.ylabel("$v$ [m/s]")
    plt.legend()

if plot_acceleration:
    accelPlot = plt.figure(figsize=(10, 6))
    plt.scatter(time_list, np.transpose(np.transpose(accelList)[0]), color='blue', label='x', alpha=0.5, rasterized=True, linewidth=0)
    plt.scatter(time_list, np.transpose(np.transpose(accelList)[1]), color='orange', label='y', alpha=0.5, rasterized=True, linewidth=0)
    plt.scatter(time_list, np.transpose(np.transpose(accelList)[2]), color='green', label='z', alpha=0.5, rasterized=True, linewidth=0)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$a$ [m/s$^2$]")
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.savefig("acceleration_raw.pdf", format='pdf')

if plot_2d_path:

    x_list = np.transpose(pos_list)[0]
    y_list = np.transpose(pos_list)[1]
    z_list = np.transpose(pos_list)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    d_list = np.sqrt(y_list**2 + x_list**2)

    d_center, z_center, radius, unc = find_circle_params(True)    
    
    d_test = np.linspace(min(d_list)-1, max(d_list)+1, 2000) # d_list is cyclic, maximum must be used
    fit_heights = half_circle_fit(d_test, d_center, z_center, radius)

    path2Dplot = plt.figure(figsize=(10, 6))
    plt.scatter(d_list, z_list, label='Calculated Positions', color='green', alpha=0.5, rasterized=True, linewidth=0)
    # plt.plot(d_list, z_list, label='Calculated Positions', color='green', rasterized=True)
    plt.plot(d_test, fit_heights, color='black', linestyle='--', label=('Circular Fit, $r = ' + str(round(radius, 1)) + '$ m'), rasterized=True)

    plt.xlabel("$d$ [m]")
    plt.ylabel("Height [m]")
    plt.axis('equal')
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.savefig("flattened_path.pdf", format='pdf')

if plot_calculated_radii:
    velocities = abs_velocity_list.transpose()[0] # The velocity is always tangential to the center, by definition

    x_list = np.transpose(pos_list)[0]
    y_list = np.transpose(pos_list)[1]
    z_list = np.transpose(pos_list)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    d_list = np.sqrt(y_list**2 + x_list**2)

    x_accels = np.transpose(np.transpose(accelList)[0])
    y_accels = np.transpose(np.transpose(accelList)[1])
    z_accels = np.transpose(np.transpose(accelList)[2])

    d_accels = np.sqrt(x_accels ** 2 + y_accels ** 2)

    d_center, z_center, *args = find_circle_params()

    vector_point_center = np.array([(d_center - d_list), (z_center - z_list)]).transpose()
    vector_accel = np.array([d_accels, z_accels]).transpose()

    radial_accel = []
    for indx, vec in enumerate(vector_point_center):
        radial_accel.append(abs(np.dot(vector_accel[indx], vec) / np.linalg.norm(vec)))

    radii = velocities**2 / radial_accel

    s1 = 70
    s2 = 210

    calculated_radius, *args = curve_fit(horizontal_line_fit, time_list[s1:s2], radii[s1:s2])

    print(calculated_radius)

    radiiPlot = plt.figure(figsize=(10, 6))
    plt.scatter(time_list, radii, label='Calculated Radii', color='blue', linewidth=0, alpha=0.5, rasterized=True)
    plt.vlines(time_list[s1], 0, 50)
    plt.vlines(time_list[s2], 0, 50)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$R$ [m]")
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)

if plot_abs_velocity:
    absVelPlot = plt.figure()
    plt.plot(time_list, abs_velocity_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute velocity (x, y, z) (m/s)")

if plot_abs_acceleration:
    absAccelPlot = plt.figure()
    plt.plot(time_list, abs_accel_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute acceleration (x, y, z) (m/s^2)")

if plot_orientation:
    orientationPlot = plt.figure(figsize=(10, 6))

    w = orientation_list.transpose()[0]
    x = orientation_list.transpose()[1]
    y = orientation_list.transpose()[2]
    z = orientation_list.transpose()[3]

    roll = np.atan2(2*(x * w + y * z), 1-2*(x**2 + y**2))
    pitch = np.asin(2*(y * w - z * x))
    yaw = np.atan2(2*(z * w + x * y), 1-2*(y**2 + z**2))
    #plt.plot(time_list[:len(orientation_list.transpose()[0])], orientation_list.transpose()[0], marker='.', label='w', color='black', rasterized=True)
    #plt.plot(time_list[:len(orientation_list.transpose()[1])], orientation_list.transpose()[1], marker='.', label='x', color='blue', rasterized=True)
    #plt.plot(time_list[:len(orientation_list.transpose()[2])], orientation_list.transpose()[2], marker='.', label='y', color='orange', rasterized=True)
    #plt.plot(time_list[:len(orientation_list.transpose()[3])], orientation_list.transpose()[3], marker='.', label='z', color='green', rasterized=True)

    plt.plot(time_list[:len(orientation_list.transpose()[1])], roll, label='Roll', color='blue', rasterized=True)
    plt.plot(time_list[:len(orientation_list.transpose()[2])], pitch, label='Pitch', color='orange', rasterized=True)
    plt.plot(time_list[:len(orientation_list.transpose()[3])], yaw, label='Yaw', color='green', rasterized=True)

    plt.xlabel("$t$ [s]")
    plt.ylabel("Angle [rad]")
   
    plt.legend()
    plt.savefig("orientation_raw.pdf", format='pdf')

if plot_energies:
    energiesPlot = plt.figure(figsize=(10, 6))

    E_pot = calc_potential_energy(pos_list)
    E_kin = calc_kinetic_energy(abs_velocity_list) * (13/13) # This correction follows from the fact that the ship's center of mass will have a higher velocity, because it has a larger radius than the measuring phone
    
    ts, ps = find_e_pot_minima(time_list, E_pot)

    print("Testing polynomial degrees for RMSE:")
    try:
        for degree in range(2, 11):
            coeffs = np.polyfit(ts, ps, degree)
            p_val = np.polyval(coeffs, ts)
            rmse = np.sqrt(np.mean((ps - p_val)**2))
            print(f"Degree {degree} RMSE: {rmse}")
        
        # Keep degree 6 for the actual plot as it provides a good balance between fitting the minima and avoiding overfitting
        val = np.polyfit(ts, ps, 6)
        yTest = np.polyval(val, time_list)
    except Exception as e:
        print("Minimums could not be fitted, too little data points or error:", e)
        yTest = np.zeros_like(time_list)

    plt.rcParams['lines.linewidth'] = 2 # Overwrite the line width for the energy plot
    
    if (use_extra_correction):
        plt.plot(time_list, np.add(E_pot, -1*yTest), color='orange', label='Corrected gravitational', rasterized=True)
        plt.plot(time_list, E_kin, color='blue', label='Kinetic', rasterized=True)
        plt.plot(time_list, np.add(np.add(E_kin, E_pot), -1 * yTest), color='green', linestyle='--', label='Total', rasterized=True)
        if(end_index- begin_index > 10000): 
            legend = plt.legend(loc="lower right") # Move the legend for Figure 10
    else:
        plt.plot(time_list, E_pot, color='blue', label='Uncorrected gravitational', rasterized=True)
        plt.plot(ts, ps, 'r.', markersize=15, label="Gravitational minima")
        plt.plot(time_list, yTest, 'k--', label='Curve Fit through minima', rasterized=True)
        legend = plt.legend() 

    for handle in legend.legend_handles:
        handle.set_alpha(1)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$E/m$ [m$^2$/s$^2$]")
    plt.savefig("energy_profile.pdf", format='pdf')