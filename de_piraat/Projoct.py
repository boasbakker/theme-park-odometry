import matplotlib.pyplot as plt
import numpy
import math
from scipy.optimize import curve_fit



# Frank de Kogel, 19/09/2025

beginIndex = 7000 # Vanaf welke index beginnen de metingen pas? 12623
eindIndex = 17500 # Vanaf waar eindigen de metingen? (0 voor meenemen elke meting) 13387

# Een mooi boogje van finn's metingen:
# beginIndex 12626
# eindIndex 13386

# Voor metingen Formule X:
# begin: 3000
# eind: 7200

# Periode piraat = 7.61
# beginpunten piraat emma:
# 5225, 5598, 5985

accelerometerPath = "/home/frank/Documents/TN_Jaar1/Inleidend_Practicum_1/Finn-Piraat1/Accelerometer.csv"
orientationPath = "/home/frank/Documents/TN_Jaar1/Inleidend_Practicum_1/Finn-Piraat1/Orientation.csv"

g = 9.81
roundingDecimals = 8

useRotation = True # Moet de inverse telefoonrotatie gebruikt worden?
useCorrection = True # Moet er gecorrigeerd worden, zodat beginpositie gelijk is aan eindpositie?
plotCalculatedPositions = False # Moet er een grafiek gemaakt worden van de posities?
plotCalculatedVelocities = False # Moet de snelheid in een aparte grafiek?
plotAcceleration = False # Moet de (geroteerde) acceleratie geplot worden?
plotAbsVelocity = False # Moet de berekende absolute snelheid geplot worden?
plotAbsAcceleration = False # Moet de absolute acceleratie geplot worden?
plotEnergies = True # Moet de totale energie geplot worden?
plotOrientation = True # Moet de telefoonorientatie (volgens Phyphox) getoond worden?
plot2DPath = False
plotCalculatedRadii = False


accelCorrection = numpy.array([0., 0., 0.]) # Acceleratiecorrectie in absoluut vlak
velCorrectie = numpy.array([0., 0., 0.])
phoneAccelCorrectie = numpy.array([0., 0., 0.]) # Theoretisch. Zou zeer hip zijn als het werkt. Gaat komen.


# Script van Boas voor gelijke plots -----

textscaling = 1.5
plt.rcParams.update({
    # Figure
    'figure.figsize': (10, 6),
    'figure.dpi': 600,

    # Tick labels
    'xtick.labelsize': 10 * textscaling,
    'ytick.labelsize': 10 * textscaling,

    # Axis labels & title
    'axes.labelsize': 12 * textscaling,
    'axes.titlesize': 14 * textscaling,

    # Legend
    'legend.fontsize': 10 * textscaling,

    # Grid
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,

    # Line defaults
    'lines.linewidth': 2,
})

# ------------


vel0 = numpy.array((0., 0., 0.)) # Beginsnelheid

pos0 = numpy.array((0., 0., 0.)) # Beginpositie

def transformRotation(phoneAccel, orientation):

    # Vorm data:
    # phoneAccel = [phoneAccelX, phoneAccelY, phoneAccelZ]
    # orientation = [wQuaternion, xQuaternion, yQuaternion, zQuaternion]

    # Ik heb hier een goede twintig (correctie veertig) uur aan besteed, omdat quaternionen en relatieve hoeken een gruwel zijn.

    # Deze functie dient om de (met de telefoon meedraaiende) XYZ-assen te mappen naar stationaire (aan de aarde vastgekoppelde) assen.
    # Ik heb getracht dit te bereiken met orientatiehoeken, yaw, pitch, roll, ik had op een gegeven moment negenentwintig trigonometriefuncties in een enkele vector.
    # Poog niet dit te begrijpen, het is lastig.

    # Dank aan de tientallen helden op de wiskundestackoverflow, die mij met quaternionen hebben geholpen door vragen van acht jaar geleden


    # W = cos (0.5 × totale hoek)
    # X = x × sin (0.5 × xHoek)
    # Y = y × sin (0.5 × yHoek)
    # Z = z × sin (0.5 × zHoek)

    rotatieMatrix = numpy.array([[2 * (orientation[0] * orientation[0] + orientation[1] * orientation[1]) - 1,
                                     2 * (orientation[1] * orientation[2] - orientation[0] * orientation[3]),
                                     2 * (orientation[1] * orientation[3] + orientation[0] * orientation[2])],
                                    [2 * (orientation[1] * orientation[2] + orientation[0] * orientation[3]),
                                     2 * (orientation[0] * orientation[0] + orientation[2] * orientation[2]) - 1,
                                     2 * (orientation[2] * orientation[3] - orientation[0] * orientation[1])],
                                    [2 * (orientation[1] * orientation[3] - orientation[0] * orientation[2]),
                                     2 * (orientation[2] * orientation[3] + orientation[0] * orientation[1]),
                                     2 * (orientation[0] * orientation[0] + orientation[3] * orientation[3]) - 1]]) # Yep, ziet er leuk uit he!

    rotatieMatrix = numpy.transpose(rotatieMatrix) 
    convertedVector = numpy.array([0.0, 0.0, 0.0])
   
    numpy.vecmat(phoneAccel, rotatieMatrix, out=convertedVector) # Hier wordt de matrix van de telefoon gedraaid zodat de Z-as boven staat.

    return convertedVector

def parseRawData(accelerometerPath, orientationPath, afronding, g, accelCorrectie, velCorrectie, vel0, beginPos, useRotation):

    global eindIndex
    global beginIndex

    with open(accelerometerPath, 'r') as accelerationRawDataFile: # Opent het accelerometerbestand van phyphox
        accelerationRawData = accelerationRawDataFile.read() # Leest het databestand
   
    with open(orientationPath, 'r') as orientationRawDataFile:
        orientationRawData = orientationRawDataFile.read()
   
    # Data geacquisitieerd

    timeList = numpy.array([0]) # Lijsten om data in te stoppen

    phoneAccelList = numpy.array([[0, 0, -9.81]]) # Acceleratie is nodig om snelheid en positie te meten

    orientationList = numpy.array([[0, 0, 0, 0]]) # Lijsten voor orientatiequaternionen (dat is een nieuw begrip, ik hoop dat het werkt) (het werkte!)

    netAccelList = numpy.array([accelCorrectie]) # Bijhouden alle acceleraties

    absAccelList = numpy.array([0])

    velocity = numpy.array([0., 0., 0.])
   
    for i in range(len(vel0)):
        velocity[i] = vel0[i] # Variabele om de huidige snelheid te bepalen

    velocityList = numpy.array([velocity]) # Lijst om de snelheden bij te houden

    absVelocityList = numpy.array([0])

    pos = numpy.array([0., 0., 0.])

    for i in range(len(beginPos)):
        pos[i] = beginPos[i] # Huidige positie

    posList = numpy.array([beginPos]) # Lijst om positie bij te houden

    # Extraheer de acceleratiemetingen:

    for line in accelerationRawData.split("\n"): # Splitst de ruwe data in meetpunten
        try: # Er moet een 'try' in gegooid worden, want bovenaan de data staat tekst.
            # Formaat:
            # "Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"

            metingTijd = round(float(line.split(",")[0]), afronding)
            xAcceleratie = round(float(line.split(",")[1]), afronding)
            yAcceleratie = round(float(line.split(",")[2]), afronding)
            zAcceleratie = round(float(line.split(",")[3]), afronding)

            timeList = numpy.append(timeList, metingTijd) # Sla de meting op
            phoneAccelList = numpy.vstack([phoneAccelList, numpy.array([xAcceleratie, yAcceleratie, zAcceleratie])])


        except:
            pass
   
    for line in orientationRawData.split("\n"):
        try:
           
            orientationList = numpy.vstack([orientationList, numpy.array([float(line.split(",")[1]),
                                                                        float(line.split(",")[2]),
                                                                        float(line.split(",")[3]),
                                                                        float(line.split(",")[4])])]) # Toevoegen van de orientatie
           
        except: # Voor als er tekst (ieuw) in de data staat
            pass
   
    # Alle metingen zijn nu opgeslagen

    # Nu moeten de metingen worden 'geknipt' naar de hoeveelheid data die ook echt gebruikt gaat worden



    if eindIndex == 0: # Gebruik alle metingen
        eindIndex = min(len(timeList), len(phoneAccelList), len(orientationList)) - 1 # kortste lijst

    phoneAccelList = phoneAccelList[beginIndex:eindIndex]
    orientationList = orientationList[beginIndex:eindIndex]
    timeList = timeList[beginIndex:eindIndex + 1] # Logica, want er is een beginpositie toegevoegd aan de positielijst
   

    # Bereken de tijd tussen meetpunten
    dT = (timeList[len(timeList)-1] - timeList[0]) / len(timeList)  # Laatste tijdstip - eerste tijdstip / aantal tijdstippen

    for indx in range(len(timeList)-1): # Gaat elk meetpunt langs

        if useRotation:
            # Bouw de relatieve accelratie om in stabiele accelratie
            stableAccelerationVector = transformRotation(phoneAccelList[indx], orientationList[indx])
           
        else:
            stableAccelerationVector = phoneAccelList[indx]

        gVector = numpy.array([0.0, 0.0, float(g)])

        # Haal de zwaartekracht van de acceleratie af
        stableAccelerationVector -= gVector

        # Gebruik correctie in vaste ruimte (moet nog aangepast worden)
        stableAccelerationVector += accelCorrectie

        netAccelList = numpy.vstack([netAccelList, stableAccelerationVector])

        velocity += stableAccelerationVector * dT

        velocityList = numpy.vstack([velocityList, velocity])

        absVelocityList = numpy.vstack([absVelocityList, numpy.linalg.norm(velocity)])
        absAccelList = numpy.vstack([absAccelList, numpy.linalg.norm(stableAccelerationVector)])

        pos += velocity * dT + velCorrectie * dT

        posList = numpy.vstack([posList, pos])

    return posList, velocityList, netAccelList, timeList, absVelocityList, absAccelList, orientationList


def correct(useDoubleCorrection=True):
    global accelCorrection
    global velCorrectie

    # Eerst een run doen zonder enige correcties:
    posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)
    # Voert eerst een acceleratiecorrectie uit (vEind = vBegin)
    # Voert daarna een 
    beginVel = velocityList[0]
    beginTime = timeList[0]

    print("Beginsnelheid:", beginVel)
    
    endVel = velocityList[len(velocityList)-1]
    endTime = timeList[len(velocityList)-1]

    print("Eindsnelheid:", endVel)

    deltaVel = endVel - beginVel
    deltaT = endTime - beginTime

    if not useDoubleCorrection:
        accelCorrection = -(1/2) * accelCorrection

    # Correction in speed = delta V / delta T

    accelCorrection = -deltaVel / deltaT

    print("Acceleratiecorrectie:")
    print(accelCorrection)

    # Daarna een run doen met enkel de acceleratiecorrectie

    posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)

    # Voert dezelfde correctie uit als bij snelheid, maar dan voor positie

    beginPos = posList[0]
    beginTime = timeList[0]
    endPos = posList[len(posList)-1]
    endTime = timeList[len(posList)-1]

    deltaT = endTime - beginTime
    deltaPos = endPos - beginPos

    velCorrectie = -deltaPos / deltaT

    print("Beginpositie:", beginPos)
    print("Eindpositie:", endPos)
    print("Snelheidscorrectie:", velCorrectie)

if useCorrection:
    correct()


posList, velocityList, accelList, timeList, absVelocityList, absAccelList, orientationList = parseRawData(accelerometerPath, orientationPath, roundingDecimals, g, accelCorrection, velCorrectie, vel0, pos0, useRotation)

# Je hebt nu data in posList, velocityList, accelList, absVelocityList, absAccelList met als bijbehorende tijdstippen timeList

# Berekening van energieen:

def calcPotentialEnergy(posList):
    # Gezien dat de z-as nu gelijkstaat aan de verticale as, kan worden gesteld dat dit de as is waar de zwaartekracht puur op werkt
    # E_pot = m g h
    return posList.transpose()[2] * g - (min(posList.transpose()[2])) * g# Eindresultaat in Joule / kg, waarbij nul potentiele energie gelijkstaat aan de onderkant van de beweging

def calcKineticEnergy(absVelocityList):
    # Aangenomen wordt dat er geen rotationele energie is, enkel kinetische energie in de (lineaire) beweging van het schip

    # E_kin = 0.5 * m * v^2

    return 0.5 * (absVelocityList.transpose()[0].transpose())**2 # Resultaat in Joule / kg

def EpotMinCurveFit(t, a, b, c, d, e, f, g): # Simpele curvefit om de drift in de minima van de potentiele energie te absorberen
    return (a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5 + g * t ** 6)

def findEpotMinima(timeList, E_pot):
    falling = True
    rising = True
    flipPoints = []
    for indx, energy in enumerate(E_pot):
        try:
            if energy > E_pot[indx-1] and falling:
                rising = True
                falling = False
                flipPoints.append(indx)
            if energy < E_pot[indx-1] and rising:
                falling = True
                rising = False
        except:
            pass
    
    return timeList[flipPoints], E_pot[flipPoints]

def halfCircleFit(x, xCenter, yCenter, radius):
    return -1 * numpy.sqrt(radius**2 - (x-xCenter)**2) + yCenter

def horizontalLineFit(x, y):
    return y

def findCircleParams(printParams=False):
    """Returns circle dCenter, zCenter and radius"""
    xList = numpy.transpose(posList)[0]
    yList = numpy.transpose(posList)[1]
    zList = numpy.transpose(posList)[2]
    dList = numpy.sqrt(yList**2 + xList**2)
    val, cov = curve_fit(halfCircleFit, dList, zList, p0=[10, 5, 12])

    if printParams: print("Radius:", round(val[2], 4), "+-", round(numpy.sqrt(cov[2][2]), 4), "m")
    return val[0], val[1], val[2], numpy.sqrt(cov[2][2])

def updatePlotStyle(textScaling):
    # Code van Boas Bakker voor universele plots
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'xtick.labelsize': 10 * textScaling,
        'ytick.labelsize': 10 * textScaling,
        'axes.labelsize': 12 * textScaling,
        'axes.titlesize': 14 * textScaling,
        'legend.fontsize': 10 * textScaling,
        'axes.grid': True,
        'lines.linewidth': 0.5,
        'lines.markersize': 2,
    })

if plotCalculatedPositions:
    xList = numpy.transpose(posList)[0]
    yList = numpy.transpose(posList)[1]
    zList = numpy.transpose(posList)[2]

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
    updatePlotStyle(1.5)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$v$ [m/s]")
    plt.legend()

if plotAcceleration:
    accelPlot = plt.figure()
    plt.scatter(timeList, numpy.transpose(numpy.transpose(accelList)[0]), color='blue', label='x', marker='.', alpha=0.3, rasterized=True)
    plt.scatter(timeList, numpy.transpose(numpy.transpose(accelList)[1]), color='orange', label='y', marker='.', alpha=0.3, rasterized=True)
    plt.scatter(timeList, numpy.transpose(numpy.transpose(accelList)[2]), color='green', label='z', marker='.', alpha=0.3, rasterized=True)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$a$ [m/s$^2$]")
    plt.legend()
    updatePlotStyle(1.5)
    plt.savefig("Raw Acceleration measured by phone vs Time, alpha 0.3.pdf", format='pdf')

if plot2DPath:

    xList = numpy.transpose(posList)[0]
    yList = numpy.transpose(posList)[1]
    zList = numpy.transpose(posList)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    dList = numpy.sqrt(yList**2 + xList**2)

    dCenter, zCenter, radius, unc = findCircleParams(True)    
    
    dTest = numpy.linspace(min(dList)-1, max(dList)+1, 2000) # dList is cyclisch, maximum moet gebruikt worden
    fitHeights = halfCircleFit(dTest, dCenter, zCenter, radius)

    path2Dplot = plt.figure(figsize=(10, 6))
    plt.scatter(dList, zList, label='Calculated Positions', color='green', alpha=0.3, rasterized=True)
    plt.plot(dTest, fitHeights, color='black', linestyle='--', label=('Circular Fit, r = ' + str(round(radius, 1)) + ' m'), rasterized=True)

    plt.xlabel("$x$ [m]")
    plt.ylabel("$Height$ [m]")
    plt.legend()
    updatePlotStyle(1.5)
    plt.savefig("flattened_path_piraat_fit.pdf", format='pdf')

if plotCalculatedRadii:
    velocities = absVelocityList.transpose()[0] # De snelheid is altijd tangentieel aan het middelpunt, per definitie

    xList = numpy.transpose(posList)[0]
    yList = numpy.transpose(posList)[1]
    zList = numpy.transpose(posList)[2]

    # To correct for the path being in the x- and y-dimension, an arbitrary dimension d is made which uses the pythagorean theorem to unify the x- and y-dimension

    dList = numpy.sqrt(yList**2 + xList**2)

    xAccels = numpy.transpose(numpy.transpose(accelList)[0])
    yAccels = numpy.transpose(numpy.transpose(accelList)[1])
    zAccels = numpy.transpose(numpy.transpose(accelList)[2])

    dAccels = numpy.sqrt(xAccels ** 2 + yAccels ** 2)

    dCenter, zCenter, *args = findCircleParams()

    vectorPointCenter = numpy.array([(dCenter - dList), (zCenter - zList)]).transpose()
    vectorAccel = numpy.array([dAccels, zAccels]).transpose()

    radialAccel = []
    for indx, vec in enumerate(vectorPointCenter):
        radialAccel.append(abs(numpy.dot(vectorAccel[indx], vec) / numpy.linalg.norm(vec)))

    radii = velocities**2 / radialAccel

    s1 = 70
    s2 = 210

    calculatedRadius, *args = curve_fit(horizontalLineFit, timeList[s1:s2], radii[s1:s2])

    print(calculatedRadius)

    radiiPlot = plt.figure(figsize=(10, 6))
    plt.scatter(timeList, radii, label='Calculated Radii', color='blue')
    plt.vlines(timeList[s1], 0, 50)
    plt.vlines(timeList[s2], 0, 50)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$R$ [m]")
    plt.legend()
    updatePlotStyle(1.5)

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

    roll = numpy.atan2(2*(x * w + y * z), 1-2*(x**2 + y**2))
    pitch = numpy.asin(2*(y * w - z * x))
    yaw = numpy.atan2(2*(z * w + x * y), 1-2*(y**2 + z**2))
    #plt.plot(timeList[:len(orientationList.transpose()[0])], orientationList.transpose()[0], marker='.', label='w', color='black', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[1])], orientationList.transpose()[1], marker='.', label='x', color='blue', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[2])], orientationList.transpose()[2], marker='.', label='y', color='orange', rasterized=True)
    #plt.plot(timeList[:len(orientationList.transpose()[3])], orientationList.transpose()[3], marker='.', label='z', color='green', rasterized=True)

    plt.plot(timeList[:len(orientationList.transpose()[1])], roll, marker='.', label='Roll', color='blue', rasterized=True)
    plt.plot(timeList[:len(orientationList.transpose()[2])], pitch, marker='.', label='Pitch', color='orange', rasterized=True)
    plt.plot(timeList[:len(orientationList.transpose()[3])], yaw, marker='.', label='Yaw', color='green', rasterized=True)

    plt.xlabel("$t$ [s]")
    plt.ylabel("Angle [rad]")
   
    plt.legend()
    plt.savefig("Raw Orientation piraat vs time euler.pdf", format='pdf')

if plotEnergies:
    energiesPlot = plt.figure(figsize=(10, 6))

    E_pot = calcPotentialEnergy(posList)
    E_kin = calcKineticEnergy(absVelocityList) * (13/13) # Deze correctie volgt uit het feit dat het massamiddelpunt van het schip een hogere snelheid zal hebben, omdat het een hogere straal heeft dan de meettelefoon
    
    ts, ps = findEpotMinima(timeList, E_pot)

    try:
        val, cov = curve_fit(EpotMinCurveFit, ts, ps)
    except:
        print("Minimums could not be fitted, too little data points.")
        val = [0, 0, 0, 0, 0, 0, 0]

    print("Computer Assisted Taylorpolynomial correction coefficients:", val)

    yTest = EpotMinCurveFit(timeList, val[0], val[1], val[2], val[3], val[4], val[5], val[6])

    
    
    #plt.vlines(ts, 0, 50)
    plt.plot(timeList, numpy.add(E_pot, -0*yTest), color='blue', label='Uncorrected gravitational', rasterized=True)
    plt.plot(ts, ps, 'r.', markersize=15, label="Gravitational minima")
    #plt.plot(timeList, E_kin, color='blue', label='Kinetic', rasterized=True)
    #plt.plot(timeList, numpy.add(numpy.add(E_kin, E_pot), -1 * yTest), color='green', linestyle='--', label='Total', rasterized=True)
    plt.plot(timeList, yTest, 'k--', label='Curve Fit through minima', rasterized=True)
    
    legend = plt.legend() 
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    plt.xlabel("$t$ [s]")
    plt.ylabel("$E/m$ [m$^2$/s$^2$]")
    updatePlotStyle(1.5)
    plt.savefig("correction_plot_piraat_potential.pdf", format='pdf')




plt.show()
