import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Frank de Kogel, 19/09/2025

beginIndex = 7000 # From which index does the ride start?
eindIndex = 17500 # At which index does the ride end? (0 to include every measurement)

# A nice excerpt from Finn's measurements:
# beginIndex, eindIndex = 12623, 13387

# Period pendulum ship = 7.61

accelerometerPath = "data_raw/Accelerometer.csv"
orientationPath = "data_raw/Orientation.csv"

g = 9.81 # set to 0 to not subtract gravity
roundingDecimals = 8

useRotation = True # Should the inverse phone rotation be used?
useCorrection = True # Should corrections be applied so that the start position equals the end position?
useExtraCorrection = True # Should the sixth-degree polynomial be used for extra correction of the potential energy drift?
plotCalculatedPositions = False # Should a graph of the positions be created?
plotCalculatedVelocities = False # Should the velocity be in a separate graph?
plotAcceleration = False # Should the acceleration be plotted? (for raw rotation: set rotation=False. useCorrection=False, and g=0)
plotAbsVelocity = False # Should the calculated absolute velocity be plotted?
plotAbsAcceleration = False # Should the absolute acceleration be plotted?
plotEnergies = True # Should the total energy be plotted?
plotOrientation = False # Should the phone orientation (according to Phyphox) be shown?
plot2DPath = False
plotCalculatedRadii = False


accelCorrection = np.array([0., 0., 0.]) # Acceleration correction in absolute plane
velCorrectie = np.array([0., 0., 0.])



vel0 = np.array((0., 0., 0.)) # Initial velocity

pos0 = np.array((0., 0., 0.)) # Initial position

def transformRotation(phoneAccel, orientation):

    # Data shape:
    # phoneAccel = [phoneAccelX, phoneAccelY, phoneAccelZ]
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

    rotatieMatrix = np.array([[2 * (orientation[0] * orientation[0] + orientation[1] * orientation[1]) - 1,
                                     2 * (orientation[1] * orientation[2] - orientation[0] * orientation[3]),
                                     2 * (orientation[1] * orientation[3] + orientation[0] * orientation[2])],
                                    [2 * (orientation[1] * orientation[2] + orientation[0] * orientation[3]),
                                     2 * (orientation[0] * orientation[0] + orientation[2] * orientation[2]) - 1,
                                     2 * (orientation[2] * orientation[3] - orientation[0] * orientation[1])],
                                    [2 * (orientation[1] * orientation[3] - orientation[0] * orientation[2]),
                                     2 * (orientation[2] * orientation[3] + orientation[0] * orientation[1]),
                                     2 * (orientation[0] * orientation[0] + orientation[3] * orientation[3]) - 1]]) # Yep, looks nice doesn't it!

    rotatieMatrix = np.transpose(rotatieMatrix) 
    convertedVector = np.array([0.0, 0.0, 0.0])
   
    np.vecmat(phoneAccel, rotatieMatrix, out=convertedVector) # Here the phone's matrix is rotated so that the Z-axis is pointing up.

    return convertedVector

def parseRawData(accelerometerPath, orientationPath, afronding, g, accelCorrectie, velCorrectie, vel0, beginPos, useRotation):

    global eindIndex
    global beginIndex

    with open(accelerometerPath, 'r') as accelerationRawDataFile: # Opens the accelerometer file from phyphox
        accelerationRawData = accelerationRawDataFile.read() # Reads the data file
   
    with open(orientationPath, 'r') as orientationRawDataFile:
        orientationRawData = orientationRawDataFile.read()
   
    # Data acquired

    timeList = [0.0] # Lists to store data in

    phoneAccelList = [[0.0, 0.0, -9.81]] # Acceleration is needed to measure velocity and position

    orientationList = [[0.0, 0.0, 0.0, 0.0]] # Lists for orientation quaternions (that is a new concept, I hope it works) (it worked!)

    netAccelList = [np.copy(accelCorrectie)] # Keep track of all accelerations

    absAccelList = [0.0]

    velocity = np.array([0., 0., 0.])
   
    for i in range(len(vel0)):
        velocity[i] = vel0[i] # Variable to determine current velocity

    velocityList = [np.copy(velocity)] # List to keep track of velocities

    absVelocityList = [0.0]

    pos = np.array([0., 0., 0.])

    for i in range(len(beginPos)):
        pos[i] = beginPos[i] # Current position

    posList = [np.copy(beginPos)] # List to keep track of position

    # Extract the acceleration measurements:

    for line in accelerationRawData.split("\n"): # Splits the raw data into measurement points
        try: # A 'try' must be used, because there is text at the top of the data.
            # Format:
            # "Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"

            metingTijd = round(float(line.split(",")[0]), afronding)
            xAcceleratie = round(float(line.split(",")[1]), afronding)
            yAcceleratie = round(float(line.split(",")[2]), afronding)
            zAcceleratie = round(float(line.split(",")[3]), afronding)

            timeList.append(metingTijd) # Save the measurement
            phoneAccelList.append([xAcceleratie, yAcceleratie, zAcceleratie])


        except:
            pass
   
    for line in orientationRawData.split("\n"):
        try:
           
            orientationList.append([float(line.split(",")[1]),
                                    float(line.split(",")[2]),
                                    float(line.split(",")[3]),
                                    float(line.split(",")[4])]) # Adding the orientation
           
        except: # In case there is text (ew) in the data
            pass
   
    # All measurements are now saved

    timeList = np.array(timeList)
    phoneAccelList = np.array(phoneAccelList)
    orientationList = np.array(orientationList)

    # Now the measurements need to be 'cut' to the amount of data that will actually be used



    if eindIndex == 0: # Use all measurements
        eindIndex = min(len(timeList), len(phoneAccelList), len(orientationList)) - 1 # shortest list

    phoneAccelList = phoneAccelList[beginIndex:eindIndex]
    orientationList = orientationList[beginIndex:eindIndex]
    timeList = timeList[beginIndex:eindIndex + 1] # Logic, because an initial position has been added to the position list
   

    # Calculate the time between measurement points
    dT = (timeList[len(timeList)-1] - timeList[0]) / len(timeList)  # Last time - first time / number of times

    for indx in range(len(timeList)-1): # Goes over every measurement point

        if useRotation:
            # Convert the relative acceleration into stable acceleration
            stableAccelerationVector = transformRotation(phoneAccelList[indx], orientationList[indx])
           
        else:
            stableAccelerationVector = phoneAccelList[indx]

        gVector = np.array([0.0, 0.0, float(g)])

        # Subtract gravity from the acceleration
        stableAccelerationVector -= gVector

        # Use correction in fixed space (still needs adaptation)
        stableAccelerationVector += accelCorrectie

        netAccelList.append(np.copy(stableAccelerationVector))

        velocity += stableAccelerationVector * dT

        velocityList.append(np.copy(velocity))

        absVelocityList.append(np.linalg.norm(velocity))
        absAccelList.append(np.linalg.norm(stableAccelerationVector))

        pos += velocity * dT + velCorrectie * dT

        posList.append(np.copy(pos))

    posList = np.array(posList)
    velocityList = np.array(velocityList)
    netAccelList = np.array(netAccelList)
    timeList = np.array(timeList)
    absVelocityList = np.array(absVelocityList).reshape(-1, 1) # reshape slightly to match vstack format when extracted later
    absAccelList = np.array(absAccelList).reshape(-1, 1)

    return posList, velocityList, netAccelList, timeList, absVelocityList, absAccelList, orientationList


def correct(useDoubleCorrection=True):
    global accelCorrection
    global velCorrectie

    # First do a run without any corrections:
    posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)
    # First performs an acceleration correction (vEnd = vBegin)
    # Then performs a 
    beginVel = velocityList[0]
    beginTime = timeList[0]

    print("Initial velocity:", beginVel)
    
    endVel = velocityList[len(velocityList)-1]
    endTime = timeList[len(velocityList)-1]

    print("End velocity:", endVel)

    deltaVel = endVel - beginVel
    deltaT = endTime - beginTime

    if not useDoubleCorrection:
        accelCorrection = -(1/2) * accelCorrection

    # Correction in speed = delta V / delta T

    accelCorrection = -deltaVel / deltaT

    print("Acceleration correction:")
    print(accelCorrection)

    # Then do a run with only the acceleration correction

    posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)

    # Performs the same correction as for velocity, but then for position

    beginPos = posList[0]
    beginTime = timeList[0]
    endPos = posList[len(posList)-1]
    endTime = timeList[len(posList)-1]

    deltaT = endTime - beginTime
    deltaPos = endPos - beginPos

    velCorrectie = -deltaPos / deltaT

    print("Initial position:", beginPos)
    print("End position:", endPos)
    print("Velocity correction:", velCorrectie)

if useCorrection:
    correct()


posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)

# You now have data in posList, velocityList, accelList, absVelocityList, absAccelList with corresponding times timeList

# Calculation of energies:

def calcPotentialEnergy(posList):
    # Given that the z-axis is now equal to the vertical axis, it can be stated that this is the axis on which gravity purely acts
    # E_pot = m g h
    return posList.transpose()[2] * g - (min(posList.transpose()[2])) * g# End result in Joule / kg, where zero potential energy equals the bottom of the movement

def calcKineticEnergy(absVelocityList):
    # It is assumed that there is no rotational energy, only kinetic energy in the (linear) movement of the ship

    # E_kin = 0.5 * m * v^2

    return 0.5 * (absVelocityList.transpose()[0].transpose())**2 # Result in Joule / kg

def EpotMinCurveFit(t, a, b, c, d, e, f, g): # Simple curvefit to absorb the drift in the potential energy minima
    return (a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5 + g * t ** 6)

def findEpotMinima(timeList, E_pot):
    flipPoints = []
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
            flipPoints.append(i)
            
    return timeList[flipPoints], E_pot[flipPoints]

def halfCircleFit(x, xCenter, yCenter, radius):
    return -1 * np.sqrt(radius**2 - (x-xCenter)**2) + yCenter

def horizontalLineFit(x, y):
    return y

def findCircleParams(printParams=False):
    """Returns circle dCenter, zCenter and radius"""
    xList = np.transpose(posList)[0]
    yList = np.transpose(posList)[1]
    zList = np.transpose(posList)[2]
    dList = np.sqrt(yList**2 + xList**2)
    val, cov = curve_fit(halfCircleFit, dList, zList, p0=[10, 5, 12])

    if printParams: print("Radius:", round(val[2], 4), "+-", round(np.sqrt(cov[2][2]), 4), "m")
    return val[0], val[1], val[2], np.sqrt(cov[2][2])

def updatePlotStyle(textScaling):
    # Code by Boas Bakker for universal plots
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'xtick.labelsize': 10 * textScaling,
        'ytick.labelsize': 10 * textScaling,
        'axes.labelsize': 12 * textScaling,
        'axes.titlesize': 14 * textScaling,
        'legend.fontsize': 10 * textScaling,
        'axes.grid': True,
        'lines.linewidth': 2,
        'lines.markersize': 2,
        'figure.dpi': 600
    })

updatePlotStyle(1.5)

if plotCalculatedPositions:
    xList = np.transpose(posList)[0]
    yList = np.transpose(posList)[1]
    zList = np.transpose(posList)[2]

    fig3d = plt.figure()
    ax = fig3d.add_subplot(projection='3d')
    ax.scatter(xList, yList, zList)
    ax.axis('equal')
    ax.set_xlabel("X (Noord-Zuid)")
    ax.set_ylabel("Y (Oost-West)")
    ax.set_zlabel("Hoogte")

if plotCalculatedVelocities:
    velplot = plt.figure(figsize=(10, 6))
    plt.plot(timeList, velocityList.transpose()[0], 'b.', label='V$_x$')
    plt.plot(timeList, velocityList.transpose()[1], '.', color='orange', label='V$_y$')
    plt.plot(timeList, velocityList.transpose()[2], 'g.', label='V$_z$')
    plt.xlabel("$t$ [s]")
    plt.ylabel("$v$ [m/s]")
    plt.legend()

if plotAcceleration:
    accelPlot = plt.figure(figsize=(10, 6))
    plt.scatter(timeList, np.transpose(np.transpose(accelList)[0]), color='blue', label='x', alpha=0.5, rasterized=True, linewidth=0)
    plt.scatter(timeList, np.transpose(np.transpose(accelList)[1]), color='orange', label='y', alpha=0.5, rasterized=True, linewidth=0)
    plt.scatter(timeList, np.transpose(np.transpose(accelList)[2]), color='green', label='z', alpha=0.5, rasterized=True, linewidth=0)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$a$ [m/s$^2$]")
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.savefig("acceleration_raw.pdf", format='pdf')

if plot2DPath:

    xList = np.transpose(posList)[0]
    yList = np.transpose(posList)[1]
    zList = np.transpose(posList)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    dList = np.sqrt(yList**2 + xList**2)

    dCenter, zCenter, radius, unc = findCircleParams(True)    
    
    dTest = np.linspace(min(dList)-1, max(dList)+1, 2000) # dList is cyclic, maximum must be used
    fitHeights = halfCircleFit(dTest, dCenter, zCenter, radius)

    path2Dplot = plt.figure(figsize=(10, 6))
    plt.scatter(dList, zList, label='Calculated Positions', color='green', alpha=0.5, rasterized=True, linewidth=0)
    # plt.plot(dList, zList, label='Calculated Positions', color='green', rasterized=True)
    plt.plot(dTest, fitHeights, color='black', linestyle='--', label=('Circular Fit, r = ' + str(round(radius, 1)) + ' m'), rasterized=True)

    plt.xlabel("$x$ [m]")
    plt.ylabel("Height [m]")
    plt.axis('equal')
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.savefig("flattened_path.pdf", format='pdf')

if plotCalculatedRadii:
    velocities = absVelocityList.transpose()[0] # The velocity is always tangential to the center, by definition

    xList = np.transpose(posList)[0]
    yList = np.transpose(posList)[1]
    zList = np.transpose(posList)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    dList = np.sqrt(yList**2 + xList**2)

    xAccels = np.transpose(np.transpose(accelList)[0])
    yAccels = np.transpose(np.transpose(accelList)[1])
    zAccels = np.transpose(np.transpose(accelList)[2])

    dAccels = np.sqrt(xAccels ** 2 + yAccels ** 2)

    dCenter, zCenter, *args = findCircleParams()

    vectorPointCenter = np.array([(dCenter - dList), (zCenter - zList)]).transpose()
    vectorAccel = np.array([dAccels, zAccels]).transpose()

    radialAccel = []
    for indx, vec in enumerate(vectorPointCenter):
        radialAccel.append(abs(np.dot(vectorAccel[indx], vec) / np.linalg.norm(vec)))

    radii = velocities**2 / radialAccel

    s1 = 70
    s2 = 210

    calculatedRadius, *args = curve_fit(horizontalLineFit, timeList[s1:s2], radii[s1:s2])

    print(calculatedRadius)

    radiiPlot = plt.figure(figsize=(10, 6))
    plt.scatter(timeList, radii, label='Calculated Radii', color='blue', linewidth=0, alpha=0.5, rasterized=True)
    plt.vlines(timeList[s1], 0, 50)
    plt.vlines(timeList[s2], 0, 50)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$R$ [m]")
    legend = plt.legend(markerscale=3)

    for handle in legend.legend_handles:
        handle.set_alpha(1)

if plotAbsVelocity:
    absVelPlot = plt.figure()
    plt.plot(timeList, absVelocityList)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute velocity (x, y, z) (m/s)")

if plotAbsAcceleration:
    absAccelPlot = plt.figure()
    plt.plot(timeList, absAccelList)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute acceleration (x, y, z) (m/s^2)")

if plotOrientation:
    orientationPlot = plt.figure(figsize=(10, 6))

    w = orientationList.transpose()[0]
    x = orientationList.transpose()[1]
    y = orientationList.transpose()[2]
    z = orientationList.transpose()[3]

    roll = np.atan2(2*(x * w + y * z), 1-2*(x**2 + y**2))
    pitch = np.asin(2*(y * w - z * x))
    yaw = np.atan2(2*(z * w + x * y), 1-2*(y**2 + z**2))
    #plt.plot(timeList[:len(orientationList.transpose()[0])], orientationList.transpose()[0], marker='.', label='w', color='black', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[1])], orientationList.transpose()[1], marker='.', label='x', color='blue', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[2])], orientationList.transpose()[2], marker='.', label='y', color='orange', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[3])], orientationList.transpose()[3], marker='.', label='z', color='green', rasterized=True)

    plt.plot(timeList[:len(orientationList.transpose()[1])], roll, label='Roll', color='blue', rasterized=True)
    plt.plot(timeList[:len(orientationList.transpose()[2])], pitch, label='Pitch', color='orange', rasterized=True)
    plt.plot(timeList[:len(orientationList.transpose()[3])], yaw, label='Yaw', color='green', rasterized=True)

    plt.xlabel("$t$ [s]")
    plt.ylabel("Angle [rad]")
   
    plt.legend()
    plt.savefig("orientation_raw.pdf", format='pdf')

if plotEnergies:
    energiesPlot = plt.figure(figsize=(10, 6))

    E_pot = calcPotentialEnergy(posList)
    E_kin = calcKineticEnergy(absVelocityList) * (13/13) # This correction follows from the fact that the ship's center of mass will have a higher velocity, because it has a larger radius than the measuring phone
    
    ts, ps = findEpotMinima(timeList, E_pot)

    print("Testing polynomial degrees for RMSE:")
    try:
        for degree in range(2, 11):
            coeffs = np.polyfit(ts, ps, degree)
            p_val = np.polyval(coeffs, ts)
            rmse = np.sqrt(np.mean((ps - p_val)**2))
            print(f"Degree {degree} RMSE: {rmse}")
        
        # Keep degree 6 for the actual plot as it provides a good balance between fitting the minima and avoiding overfitting
        val = np.polyfit(ts, ps, 6)
        yTest = np.polyval(val, timeList)
    except Exception as e:
        print("Minimums could not be fitted, too little data points or error:", e)
        yTest = np.zeros_like(timeList)

    plt.rcParams['lines.linewidth'] = 2 # Overwrite the line width for the energy plot
    
    if (useExtraCorrection):
        plt.plot(timeList, np.add(E_pot, -1*yTest), color='orange', label='Corrected gravitational', rasterized=True)
        plt.plot(timeList, E_kin, color='blue', label='Kinetic', rasterized=True)
        plt.plot(timeList, np.add(np.add(E_kin, E_pot), -1 * yTest), color='green', linestyle='--', label='Total', rasterized=True)
        if(eindIndex- beginIndex > 10000): 
            legend = plt.legend(loc="lower right") # Move the legend for Figure 10
    else:
        plt.plot(timeList, E_pot, color='blue', label='Uncorrected gravitational', rasterized=True)
        plt.plot(ts, ps, 'r.', markersize=15, label="Gravitational minima")
        plt.plot(timeList, yTest, 'k--', label='Curve Fit through minima', rasterized=True)
        legend = plt.legend() 

    for handle in legend.legend_handles:
        handle.set_alpha(1)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$E/m$ [m$^2$/s$^2$]")
    plt.savefig("energy_profile.pdf", format='pdf')