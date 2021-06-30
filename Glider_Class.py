# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:15:05 2020

@author: Garrett
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, pi, exp, cos, sin, atan, acos
from datetime import datetime
import metpy.calc as metcalc
from metpy.units import units
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from DragPolarSeniorDesign_updated import Drag
import xlrd
import pandas as pd
#from mpl_toolkits.basemap import Basemap
from matplotlib import interactive
from BankingFlight import BankingFlight
from scipy.optimize import fsolve, minimize

interactive(True)



class Glider:
    def __init__(self):
        self.CurrentLat = None
        self.CurrentLong = None
        self.CurrentAlt = None
        self.EndLat = None
        self.EndLong = None
        self.mass = None
        self.CurrentDateAndTime = None
        self.airspeed = None
        self.windVelocity = None
        self.windDirection = None
        self.startVelocity_x = None
        self.startVelocity_y = None
        self.startVelocity_z = None
        self.GetFlightTime = None
        self.DistFromPoint = None
        self.EastBound = None
        self.WestBound = None
        self.NorthBound = None
        self.SouthBound = None
        self.FlightTime = None
        self.TotalFlightTime = None
        self.windSpeed = None
        self.windDir = None
        self.HeadTailVector = None
        self.CrossWindVector = None
        self.HeadOrTail = None
        self.GlideVelocity = None
        self.FinalLat = None
        self.FinalLong = None
        self.PathLats = None
        self.PathLongs = None
        self.PathAlts = None
        self.PathTime = None
        self.PathWindSpeed = None
        self.PathWindDirection = None
        self.FinalXPos = None
        self.FinalYPos = None
        self.FinalAlt = None
        self.windDirHeading = None
        self.StartLat = None
        self.StartLong = None
        self.Xdist = None
        self.Ydist = None
        self.NewDist = None
        self.WindFile = None
        self.Filepath = None
        self.FileName = None
        self.FileAlts = None
        self.FileDir = None
        self.FileSpeed = None
        self.plot = None
        self.TotalTime = None
        self.NOAA = None
        self.SteadyState = None
        self.BankingFlight = BankingFlight()
        self.drag = Drag()
        workbook = xlrd.open_workbook(r'stopcrashingexcel.xlsm') # open the workbook
        worksheet = workbook.sheet_by_name('DragBuildUp')
        # get a data w/ selected rows and columns

        # Wing Characteristics---------------------------------------------
# =============================================================================
#         b = worksheet.cell(1,1).value  # Wing Span - in
#         S = worksheet.cell(2,1).value  # Projected Wing Span & Sref in^2
#         AR = worksheet.cell(3,1).value # Wing Aspect Ratio
#         SweepAngle = worksheet.cell(4,1).value # Wing Sweep Angle in Degrees
#         RootChord = worksheet.cell(5,1).value # Wing root chord - in
#         TipChord = worksheet.cell(6,1).value # Wing tip chord - in
#         TipRootRatio = worksheet.cell(7,1).value # tip to root ratio
#         e = worksheet.cell(8,1).value # Oswald Efficiency Factor
#         Weight = worksheet.cell(11,1).value # Aircraft Weight - lb
#         MAC = worksheet.cell(12,1).value # Mean Aerodynamic Chord of the Wing - in
# =============================================================================
        self.drag.ReadExcelFile(worksheet)
        

    def AtmosphericData(self):
        # Function to calculate Density, Temperature, Pressure, Mach Number, Speed of Sound and Dynamic Viscosity
        gamma = 1.4 # Constant Specific Heats Model
        R = 1716  # Ideal Gas Constant
        Atm = np.empty([6,1])

        # just made these return density for this purpose
        if self.CurrentAlt < 36152: # Troposphere
            Atm[0] = T = 59 - 0.00356*self.CurrentAlt # Temperature in Fahrenheit
            Atm[1] = P = 2116 * ((T+459.7)/518.6)**5.256 # Pressure in psf
            Atm[2] = rho = P / (1718*(T+459.7)) # Density in slugs/ft^3
            Atm[3] = mu = 3.62*10**(-7)*((T+459.7)/518.7)**1.5 * ((518.7+198.72)/(T+198.72)) # Dynamic Viscosity lb-s/ft^2
            Atm[4] = SoS = math.sqrt(1.4*1716*(T+459.7)) # Speed of Sound
            Atm[5] = Mach = self.airspeed / SoS # Mach Number
            return rho
        elif self.CurrentAlt > 36152 and self.CurrentAlt < 82345: # Lower Stratosphere
            Atm[0] = T = -70 # Temperature in Fahrenheit
            Atm[1] = P = 473.1*math.exp(1.73-0.000048*self.CurrentAlt) # Pressure in psf
            Atm[2] = rho = P / (1718 * (T + 459.7))  # Density in slugs/ft^3
            Atm[3] = mu = 3.62 * 10 ** (-7) * ((T + 459.7) / 518.7) ** 1.5 * ((518.7 + 198.72) / (T + 198.72))  # Dynamic Viscosity lb-s/ft^2
            Atm[4] = SoS = math.sqrt(1.4 *1716 * (T+459.7))  # Speed of Sound
            Atm[5] = Mach = self.airspeed / SoS  # Mach Number
            return rho

        elif self.CurrentAlt > 82345: # Upper Stratosphere
            Atm[0] = T = -205.05 + 0.00164*self.CurrentAlt # Temperature in Fahrenheit
            Atm[1] = P = 51.97* ((T+459.7)/389.98)**-11.388 # Pressure in psf
            Atm[2] = rho = P / (1718 * (T + 459.7))  # Density in slugs/ft^3
            Atm[3] = mu = 3.62 * 10 ** (-7) * ((T + 459.7) / 518.7) ** 1.5 * ((518.7 + 198.72) / (T + 198.72))  # Dynamic Viscosity lb-s/ft^2
            Atm[4] = SoS = math.sqrt(1.4*1716*(T+459.7))  # Speed of Sound
            Atm[5] = Mach = self.airspeed / SoS  # Mach Number
            return rho
    
    # Gets Wind speed data for a given glider position. If a full date and time is not given it will use present information. 
    
    def GetWindSpeed(self, Year = None, Month = None, Day = None, Hour = None):
        def find_time_var(var, time_basename='time'):
            for coord_name in var.coordinates.split():
                if coord_name.startswith(time_basename):
                    return coord_name
            raise ValueError('No time variable found for ' + var.name)
        
        lat = self.CurrentLat
        long = self.CurrentLong
        alt = self.CurrentAlt 
        self.drag.GetDrag(alt)  # Gathers Drag Characteristics from Drag Class
        self.drag.PlotDrag(alt) # Needed to get glide performance 
        self.drag.GlidingPerformance(alt)   # Allows for code to grabe best range and best endurance glide

 
        alt = alt * units('ft') # Adds units to the altitude
        pressure_alt = metcalc.height_to_pressure_std(alt)  # Converts altitude to pressure altitude
        pressure_alt = pressure_alt * 1/units('hPa')    # Makes pressure altitude unitless
        pressure_alt = float(pressure_alt)  # Turns value back into float for use 
        pressure_alt = pressure_alt*100 # Converts from hPa value to Pa
        best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/'
                              'NCEP/GFS/Global_0p25deg/catalog.xml')
        best_ds = list(best_gfs.datasets.values())[1]
        #print(best_ds.access_urls)

        # Create NCSS object to access the NetcdfSubset
        ncss = NCSS(best_ds.access_urls['NetcdfSubset'])
        #print(best_ds.access_urls['NetcdfSubset'])

        # Create lat/lon box for location you want to get data for
        query = ncss.query()
        if Year == None:
            query.lonlat_point(lat,long).time(datetime.utcnow())    # Data for present Day
        else:
            query.lonlat_point(lat,long).time(datetime(Year, Month, Day, Hour)) # sets date and time to use input 
        query.accept('netcdf4')
        # Request data for winds at altitude
        # First clear the query's variables from previous query for MSLP
        query.var = set()
        query.variables('u-component_of_wind_isobaric', 'v-component_of_wind_isobaric') # Sets desired variables 
        query.vertical_level(pressure_alt) # Gathers desired variables at a given pressure altitude
        data = ncss.get_data(query)
        u_wind_alt = (units.meter / units.second) * data.variables['u-component_of_wind_isobaric'][0,:].squeeze() # GFS gives velocities in m/s, and MetPy requires units for certain functions
        v_wind_alt = (units.meter / units.second) * data.variables['v-component_of_wind_isobaric'][0,:].squeeze() # These lines are establishing an 1x1 array for velocity 
        #time_var = data.variables[find_time_var(u_wind_alt)]
        #lat_var = data.variables['lat'][:]
        #lon_var = data.variables['lon'][:]
        wind_spd_alt = metcalc.wind_speed(u_wind_alt,v_wind_alt) # Calculates Magnitude of wind Vector
        wind_dir_alt = metcalc.wind_direction(u_wind_alt, v_wind_alt, 'to') # Calculates wind direction in heading format 0 degrees in N (where it is heading towards)
        wind_dir_calc = np.rad2deg(np.arctan2(v_wind_alt,u_wind_alt)) # Calculates cartesean angle for wind vector
        wind_dir_calc = float(wind_dir_calc) 
        wind_spd_alt = wind_spd_alt.to(units('ft/s'))
        wind_spd_alt = wind_spd_alt * units('s/ft')
        wind_dir_alt = wind_dir_alt * 1/units('degrees')
        wind_spd_alt = float(wind_spd_alt)
        wind_dir_alt = float(wind_dir_alt)
          
          
        if wind_dir_alt == 360:
            wind_dir_alt = 0
          
      
        self.windSpeed = wind_spd_alt
        self.windDir = wind_dir_calc
        self.windDirHeading = wind_dir_alt


    # If textfile is selected this will read textfile 
    def ReadWindData(self):
        filepath =self.WindFile
        alts = np.loadtxt(filepath, skiprows = 7, usecols=2, unpack=False)
        windDir = np.loadtxt(filepath, skiprows=7, usecols=8, unpack=False)
        windSpeed = np.loadtxt(filepath,skiprows=7, usecols=9, unpack=False)
        
        windSpeed = windSpeed * 1.68781
        alts = alts * 3.281
        alts[0] = 0
        
        alts = np.transpose(alts)
        windDir = np.transpose(windDir)
        windSpeed = np.transpose(windSpeed)
        
        self.FileAlts = alts
        self.FileDir = windDir
        self.FileSpeed = windSpeed
    
    
    # Gets wind data from textfile for given altitude
    def ParseWindSpeed(self):
        Alt = self.CurrentAlt
        Speed = self.FileSpeed
        Dir = self.FileDir
        Speed = np.append(Speed,0)
        Dir = np.append(Dir,0)
        Alts = self.FileAlts
        Alts = np.append(Alts,1e50)
        self.drag.GetDrag(Alt)
        self.drag.PlotDrag(Alt)
        self.drag.GlidingPerformance(Alt)
        for i in range(len(self.FileAlts)-1):
            CheckAlt = self.FileAlts[i]
            CheckAlt2 = self.FileAlts[i+1]
            if CheckAlt < Alt and CheckAlt2 > Alt:
                AltSpeed = Speed[i]
                AltDir = Dir[i]
        
        self.windDirHeading = AltDir
        AltSpeed = AltSpeed * (units.feet / units.second)
        AltDir = np.deg2rad(AltDir)
        WindComp = metcalc.wind_components(AltSpeed,AltDir)
        WindU = WindComp[0]
        WindV = WindComp[1]
        AltSpeed = AltSpeed * (units.second / units.feet)
        WindU = WindU * (units.second / units.feet)
        WindV = WindV * (units.second / units.feet)
        WindCart = np.arctan2(WindV, WindU) * 1/units('radians')
        WindCart = float(WindCart)
        WindCart = np.degrees(WindCart)
        AltSpeed= float(AltSpeed)
        self.windSpeed = AltSpeed
        self.windDir = WindCart
    
    # Calculates the drop distance before gliding flight begins
    def Fall(self):
        # define parameters
        h = 90000 # worst case scenario at maximum height
        # h = V * time
        vals = density(h)
        rho = vals[2]
        a = vals[3]
        
        alpha = 0.174  # 10 deg AOA
        S = 2.52 #ft^2
        Sfall = 1.5 #ft^2
        AR = 6.35   # Aspect Ratio
        e = 0.7     # oswald's efficiency factor
        W = 2.2    # weight (lbs)
        g = 32.2   # gravity (ft/s^2)
        m = W / g   # mass
        Cla = 2 * pi    # lift curve slope
        epsilon = 1 / (3.141592 * e * AR)

        # Determine L/D max with block parameters
        # Drag Class info
        G1 = Drag()
        workbook = xlrd.open_workbook(
            r'stopcrashingexcel.xlsm')  # open the workbook
        worksheet = workbook.sheet_by_name('DragBuildUp')
        G1.ReadExcelFile(worksheet)
        G1.GetDrag(h)
        G1.PlotDrag(h)
        # reference drag class to get max L/D and Cdo, and Vstall
        L_Dmax = 11 # approx
        # CDo = Glider.GivenSpeedGiveCDo(h,Velocity)  # fixed CDo
        #Vstall = 190    # from drag class for 100,000 ft
        Vstall = G1.ReturnStall(h)   # for 90,000 ft
        CDo = G1.GivenSpeedGiveCDo(h)
        x=5


        gam = -atan(1/L_Dmax)   # corresponding minimum gamma
        # print(gam)

        # fsolve to determine velocity required for L = Wcos(gam) - equilibrium glide (design choice)
        def eqns(vars):

            CL, M, V, CLa = vars

            fsolve1 = Cla * alpha - CL   # lift coefficient
            fsolve2 = Cla / (sqrt(1-(V / a)**2) + (Cla / (pi * e * AR))) - CLa    # Lift curve slope corrected for compressibility
            fsolve3 = sqrt((2 * W * cos(gam)) / (rho * CL * S)) - V    # velocity at L = W
            fsolve4 = V / a - M     # Mach number

            return (fsolve1, fsolve2, fsolve3, fsolve4)

        (CL, Mach, Vreq, CLa) = fsolve(eqns, (3, 0.3, 200, 0.1))
        #print('Mach = {:.3f}, Vel Req = {:.3f}ft/s'.format(Mach, Vreq))



        # Use f=ma here to determine a and time to the required velocity
        # Used average velocity to estimate
        Vo = 0  #ft/s
        t = 0
        h = 0

        # make sure glider doesn't stall
        if Vstall > Vreq:
            Vavg = (Vo + Vstall) / 2
            V = np.linspace(Vo, Vstall, 100)
        else:
            Vavg = (Vo + Vreq) / 2
            V = np.linspace(Vo, Vreq, 100)

        Cla = Cla / (
                sqrt(1 - (Vavg / a) ** 2) + (Cla / (pi * e * AR)))  # Lift curve slope corrected for compressibility
        CL = Cla * alpha  # lift coefficient
        CD = CDo + epsilon * CL ** 2
        D = 0.5 * rho * Vavg ** 2 * CD * Sfall

        # loop to step through velocities and get more accurate time
        for i in range(0,len(V)-1, 1):
            Vo = V[i]
            Vnext = V[i+1]
            t = t + (m / (W - D)) * (Vnext - Vo)
            h = h + (Vnext - Vo) * t

    #print('Time to fall = {:.3f}, Altitude loss = {:.3f}ft'.format(t, h))


    # Fastest time calculated without drag
    #hnod = 0; Vo = 0; tnod = 0
    #V = np.linspace(Vo, Vreq, 100)
        self.CurrentAlt = self.CurrentAlt - h

    #for i in range(0,len(V)-1, 1):
        #Vo = V[i]
        #Vnext = V[i+1]
        #tnod = tnod + (m / (W)) * (Vnext - Vo)
        #hnod = hnod + (Vnext - Vo) * tnod


    
    # Sets wind to steady state conditions if selected 
    def SteadyStateGlide(self):
        self.windSpeed = 0
        self.windDir = 0
        self.windDirHeading = 0
        Alt = self.CurrentAlt
        self.drag.GetDrag(Alt)
        self.drag.PlotDrag(Alt)
        self.drag.GlidingPerformance(Alt)
    
    # Gets the descent rate at best range and calculates groundspeed from the glide velocity and headwind/tailwind
    def SinkrateAndGroundspeed(self, course):
        #self.drag.FindMinDrag()
        #self.drag.GlidingPerformance(self.CurrentAlt)
        descentRate = self.drag.BestRangeVelocity[1]
        #self.GetHeadTailCross(course)
        if self.HeadOrTail == 0:
            groundspeed = self.GlideVelocity - self.HeadTailVector
        else:
            groundspeed = self.GlideVelocity + self.HeadTailVector
        
        self.CurrentV_X = np.cos(np.radians(course)) * groundspeed
        self.CurrentV_Y = np.sin(np.radians(course)) * groundspeed
        self.CurrentV_Z = descentRate        
        
    # Uses bounding box and velocity to establish shortest flight time to reach new bounding box
    def FindFlightTime(self):
        #self.GetRegimeFlightDistances()
        V_x = self.CurrentV_X
        V_y = self.CurrentV_Y
        V_z = self.CurrentV_Z
                
        
        EastDist = self.EastBound
        WestDist = self.WestBound
        NorthDist = self.NorthBound
        Southdist = self.SouthBound
        Altdist = self.Altdist
        
        timeChangeNorth = NorthDist / V_y
        timeChangeSouth = Southdist / V_y
        timeChangeEast = EastDist / V_x
        timeChangeWest = WestDist / V_x
        timeChangeAlt = Altdist * -1 / V_z
        
        if timeChangeNorth <= 0:
            timeChangeNorth = 1e20
        if timeChangeSouth <= 0:
            timeChangeSouth = 1e20
        if timeChangeEast <= 0:
            timeChangeEast = 1e20
        if timeChangeWest <= 0:
            timeChangeWest = 1e20
        
        
        shortestTime = timeChangeNorth
        if shortestTime > timeChangeSouth:
            shortestTime = timeChangeSouth
        if shortestTime > timeChangeEast:
            shortestTime = timeChangeEast
        if shortestTime > timeChangeWest:
            shortestTime = timeChangeWest
        if shortestTime > timeChangeNorth:
            shortestTime = timeChangeNorth
        if shortestTime > timeChangeAlt:
            shortestTime = timeChangeAlt
            
        

        self.FlightTime = shortestTime
    
    
    # Uses the minimum time for travel out of the bounding box to calculate distance travel in all axes and new lat/long/altitude
    def GetPosition(self):
        #self.GetFlightTime()
        timeChange = self.FlightTime
        
        currentLat = self.CurrentLat
        currentLong = self.CurrentLong
        currentAlt = self.CurrentAlt
        
        posChangeX = self.CurrentV_X*timeChange
        posChangeY = self.CurrentV_Y*timeChange       
        posChangeZ = self.CurrentV_Z*timeChange
        
        CurrentLatFeet = LattoFeet(currentLat)
        CurrentLongFeet = LongtoFeet(currentLat, currentLong)
        
        FinalXPos = CurrentLongFeet + posChangeX
        FinalYPos = CurrentLatFeet + posChangeY
        FinalAlt = currentAlt + posChangeZ
        
        self.FinalLat = FeettoLat(FinalYPos)
        self.FinalLong = FeettoLong(self.FinalLat,FinalXPos)
        self.FinalAlt = FinalAlt
    
    # establishes range of courses to try. 45 degrees clockwise and counter clockwise from a direct course to the end lat/long 
    def GetCourse(self):
        
        LatChange = np.radians(self.EndLat - self.CurrentLat)
        LongChange = np.radians(self.EndLong - self.CurrentLong)
        #StartLat = np.radians(self.CurrentLat)
        #EndLat = np.radians(self.EndLat)
        
        #y = np.sin(LongChange) * np.cos(EndLat)
        #x = np.cos(StartLat)*np.sin(EndLat)-np.sin(StartLat)*np.cos(EndLat)*np.cos(LongChange) 
        #degree = np.rad2deg(np.arctan2(y,x))
        degree = np.rad2deg(np.arctan2(LatChange,LongChange))
        
        course = []
        
        if degree < 0:
            degree = 360 - np.abs(degree)
            
        degreeLeft = degree - 45
        degreeRight = degree + 45
        
        while(degreeLeft<degreeRight+1):
            course.append(degreeLeft)
            degreeLeft += 1
        for i in range(len(course)):
            if course[i] > 360:
                course[i] = course[i] - 360
            elif course[i] < 0:
                course[i] = 360 - np.abs(course[i])
# =============================================================================
#         if self.SteadyState != None:
#             for i in range(len(course)):
#                 course[i] = np.rad2deg(np.arctan2(LatChange,LongChange))
# =============================================================================
        
        self.course = course
            

    # Finds if wind vector is a head wind or tailwind and gets the head/tailwind component and crosswind component    
    def GetHeadTailCross(self, course):
        
        #if np.abs(heading - self.windDir) == 180:
        #    self.HeadTailWind = self.windSpeed * -1
        #    self.CrossWind = 0
        #if heading - self.windDir == 0:
        #    self.HeadTailWind = self.windSpeed
        #    self.CrossWind = 0
        headwind = 0
        windSpeed = self.windSpeed
        windDir = self.windDir
        
        if np.abs(course - windDir) > 90 and np.abs(course-windDir) < 270:
            dAngleRad = np.radians(course-windDir)
            Headwind = np.cos(dAngleRad)*windSpeed
            CrossWind = np.sin(dAngleRad)*windSpeed
            self.HeadOrTail = 0
            
        elif np.abs(course-windDir) < 90 or np.abs(course-windDir) > 270:
            dAngleRad = np.radians(course-windDir)
            Headwind = np.cos(dAngleRad)*windSpeed
            CrossWind = np.sin(dAngleRad)*windSpeed
            self.HeadOrTail = 1
        elif np.abs(course - windDir) == 90:
            Headwind = 0
            CrossWind = -windSpeed
        elif np.abs(course-windDir) == 270:
            Headwind = 0
            CrossWind = windSpeed
        self.HeadTailVector = Headwind
        self.CrossWindVector = CrossWind
    
    # Gets airspeed for best range, uses that and the headwind to get glide velocity    
    def GetGlideVelocity(self, course):
        SSVelocity = self.drag.BestRangeVelocity[0]
        HeadTailVector = self.HeadTailVector
        
        if self.HeadOrTail == 0:
            GlideVelocity = SSVelocity + .25*HeadTailVector
        elif self.HeadOrTail == 1:
            GlideVelocity = SSVelocity - .1*HeadTailVector
        else:
            GlideVelocity = SSVelocity
        
        self.GlideVelocity = GlideVelocity
        self.GlideVelocityX = GlideVelocity * np.cos(np.radians(course))
        self.GlideVelocityY = GlideVelocity * np.sin(np.radians(course))
        
            
    # Uses Haversine formula to get distance from desired lat/long to ending lat/long
    def GetDistanceFromHome(self, lat, long):
        CurrentLong = long
        CurrentLat = lat

        EndLat = self.EndLat
        EndLong = self.EndLong
        LatChange = np.radians(self.EndLat - self.CurrentLat)
        LongChange = np.radians(self.EndLong - self.CurrentLong)
        degree = np.arctan2(LatChange,LongChange)
        self.Xdist = np.cos(degree)
        self.Ydist = np.sin(degree)
        Radius = 20.902e6 #Radius of Earth in meters
        CurrentLatRadians = np.radians(CurrentLat)
        EndLatRadians = np.radians(EndLat)
        CurrentLongRadians = np.radians(CurrentLong)
        dLatRadians = np.radians(EndLat-CurrentLat)
        dLongRadians = np.radians(EndLong-CurrentLong)
        
        a = (np.sin(dLatRadians/2) * np.sin(dLatRadians/2) + np.cos(CurrentLatRadians) *
             np.cos(EndLatRadians) * np.sin(dLongRadians/2) * np.sin(dLongRadians/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        TotalDist = Radius * c 
        
        #print(TotalDist)
        
        
        return TotalDist
    
    # Sets a bounding box using current location and the 0.25 degree latitude resolution 
    def GetRegimeFlightDistances(self):
        CurrentLat = self.CurrentLat
        CurrentLong = self.CurrentLong
        regimesize = 0.25
        LatDecimal = CurrentLat % 1
        LongDecimal = CurrentLong % 1
        altDist = 200
        if LatDecimal == 0.125 or LatDecimal == 0.375 or LatDecimal == 0.625 or LatDecimal == 0.875:
            SouthLat = 0
            NorthLat = 0
        elif LatDecimal < 0.125:
            SouthLat = .125 + LatDecimal
            NorthLat = .125 - LatDecimal
        elif LatDecimal > 0.125 and LatDecimal <.375:
            SouthLat = LatDecimal - 0.125
            NorthLat = .375 - LatDecimal
        elif LatDecimal > 0.375 and LatDecimal <.625:
            SouthLat = LatDecimal - 0.375
            NorthLat = .625- LatDecimal
        elif LatDecimal > 0.625 and LatDecimal <.875:
            SouthLat = LatDecimal - 0.625
            NorthLat = .875- LatDecimal
        elif LatDecimal > 0.875:
            SouthLat = LatDecimal - 0.875
            NorthLat = 1.125- LatDecimal
        
        if LongDecimal == 0.125 or LongDecimal == 0.375 or LongDecimal == 0.625 or LongDecimal == 0.875:
            WestLong = 0
            EastLong = 0
        elif LongDecimal < 0.125:
            WestLong = .125 + LongDecimal
            EastLong = .125 - LongDecimal
        elif LongDecimal > 0.125 and LongDecimal <.375:
            WestLong = LongDecimal - 0.125
            EastLong = .375 - LatDecimal
        elif LongDecimal > 0.375 and LongDecimal <.625:
            WestLong = LongDecimal - 0.375
            EastLong = .625- LongDecimal
        elif LongDecimal > 0.625 and LongDecimal <.875:
            WestLong = LongDecimal - 0.625
            EastLong = .875- LongDecimal
        elif LongDecimal > 0.875:
            WestLong = LongDecimal - 0.875
            EastLong = 1.125- LongDecimal
        
        #print(EastLong)
        #print(WestLong)
        #print(NorthLat)
        #print(SouthLat)
        
        EastLong = LongtoFeet(CurrentLat,EastLong)
        WestLong = LongtoFeet(CurrentLat,WestLong)
        NorthLat = LattoFeet(NorthLat)
        SouthLat = LattoFeet(SouthLat)
        
        if self.NOAA == None:
            EastLong = 1e20
            WestLong = 1e20
            NorthLat = 1e20
            SouthLat = 1e20
        
        #print(EastLong)
        #print(WestLong)
        #print(NorthLat)
        #print(SouthLat)
        self.EastBound = EastLong
        self.WestBound = -WestLong
        self.NorthBound = NorthLat
        self.SouthBound = -SouthLat
        self.Altdist = altDist
        
    # Gets best path for glider        
    def FindBestPath(self, NOAA = None, Present = None, SteadyState=None):
        
        LoiterRadius = 1000
        self.StartLat = self.CurrentLat
        self.StartLong = self.CurrentLong
        DistFromPoint = self.GetDistanceFromHome(self.CurrentLat, self.CurrentLong)
        alts = []
        lats = []
        longs = []
        times = [0]
        windDir = []
        windVel = []
        alts.append(self.CurrentAlt)
        lats.append(self.CurrentLat)
        longs.append(self.CurrentLong)
        windDir.append(self.windDir)
        windVel.append(self.windSpeed)
        self.Fall()
        while DistFromPoint > LoiterRadius or self.CurrentAlt > 400:
        #for i in range(100):
            if SteadyState != None:
                self.SteadyStateGlide()
                self.SteadyState = 0
            elif NOAA == None:
                self.NOAA = 0
                self.ReadWindData()
                self.ParseWindSpeed()
            elif Present == None:
                self.GetWindSpeed(Year = self.Year, Month = self.Month, Day = self.Day, Hour = self.Hour)
            else:
                self.GetWindSpeed()
            self.GetCourse()
            MinDist = self.GetDistanceFromHome(self.CurrentLat, self.CurrentLong)
            self.NewDist = MinDist
            self.GetRegimeFlightDistances()
            count = 0
            MinDistLoss = -1e20
            for i in range(len(self.course)):
                self.GetHeadTailCross(self.course[i])
                self.GetGlideVelocity(self.course[i])
                self.SinkrateAndGroundspeed(self.course[i])
                self.FindFlightTime()
                self.GetPosition()
                NewDist = self.GetDistanceFromHome(self.FinalLat,self.FinalLong)
                if NewDist < MinDist:
                    count = count + 1
                    MinDist = NewDist
                    NewAlt = self.FinalAlt
                    NewLat = self.FinalLat
                    NewLong = self.FinalLong
                    NewTime = self.FlightTime
                    course = self.course[i]
                elif NewDist >= MinDist and count == 0:
                    DistLoss = MinDist - NewDist
                    if DistLoss > MinDistLoss:
                        MinDistLoss = DistLoss
                        NewAlt = self.FinalAlt
                        NewLat = self.FinalLat
                        NewLong = self.FinalLong
                        NewTime = self.FlightTime
            
                    
            alts.append(NewAlt)
            longs.append(NewLong)
            lats.append(NewLat)
            times.append(NewTime)
            windDir.append(self.windDirHeading)
            windVel.append(self.windSpeed)
            self.CurrentAlt = NewAlt
            self.CurrentLat = NewLat
            self.CurrentLong = NewLong
            print(NewAlt)
            DistFromPoint = self.GetDistanceFromHome(self.CurrentLat, self.CurrentLong)
            if DistFromPoint < LoiterRadius or self.CurrentAlt < 400:
                break
            print(DistFromPoint)
        self.PathAlts = alts
        self.PathLats = lats
        self.PathLongs = longs
        self.PathTime = times
        self.PathWindDirection = windDir
        self.PathWindSpeed = windVel
        self.BankingFlight.GetGlideVals(self.CurrentAlt)
        self.BankingFlight.Banking(self.CurrentAlt)
        self.plotPath()
        TotalTime = 0
        for i in range(len(times)):
            TotalTime = TotalTime + times[i]
        self.TotalTime = TotalTime/60
        self.TotalTime = self.TotalTime + (self.BankingFlight.time / 60)
        #self.plotBank()
        print(self.TotalTime)
        if len(self.Filepath) != 0:
            self.MakeDataFile()
        
    
    # Plots the waypoints in 3D    
    def plotPath(self):
        Alts = self.PathAlts
        Lats = self.PathLats
        Longs = self.PathLongs
        Times = self.PathTime
        TotalTime = 0
        #if self.EndLat > self.StartLat:
        #    bottomLat = self.StartLat
        #    topLat = self.EndLat
        #else:
        #    bottomLat = self.EndLat
        #    topLat = self.StartLat
        #if self.EndLong > self.StartLong:
        #    leftLong = self.StartLong
        #    rightLong = self.EndLong
        #else:
        #    leftLong = self.EndLong
        #    rightLong = self.StartLong
        
        plt.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #bm = Basemap(llcrnrlon = leftLong - 1.25 , llcrnrlat = bottomLat - 1.25 , urcrnrlon = rightLong + 1.25, urcrnrlat = topLat + 1.25 , projection = 'cyl' , resolution = 'l', fix_aspect = False, ax = ax)
        ax.plot(Longs, Lats, Alts)
        ax.scatter(self.EndLong, self.EndLat, 0, c = 'g')
        ax.scatter(Longs[0], Lats[0], Alts[0])
        ax.text(Longs[0], Lats[0], Alts[0], 'Glide Point')
        ax.scatter(self.CurrentLong, self.CurrentLat, self.CurrentAlt, c = 'k')
        ax.text(self.CurrentLong, self.CurrentLat, self.CurrentAlt, 'End Glide')
        ax.text(self.EndLong, self.EndLat, 0, 'Final Point')
        ax.plot(Longs,Lats,0, 'r')
        #ax.add_collection3d(bm.drawcountries(linewidth = 0.2))
        #ax.add_collection3d(bm.drawstates(linewidth = .2))
        #lonstep = 0.75
        #latstep = 0.75
        #meridian = np.arange(leftLong - 1.25, rightLong + 1.25, lonstep)
        #parrallel = np.arange(bottomLat - 1.25, topLat + 1.25, latstep)
        #ax.set_yticks(parrallel)
        #ax.set_xticks(meridian)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude - ft')
        #fig.scatter(p)
        #fig.scatter(a)
        if self.CurrentAlt <= 400:
            self.CurrentAlt = 0
        
        ax.set_title('Glide Path (End Alt = {a:.0f} ft)'.format(a = self.CurrentAlt))
        self.plot = ax
        #print()
        #print()
        #print()
        #print(TotalTime)
        plt.draw()
        
    # Makes data file for XPlane and Mission Planner
    def MakeDataFile(self):
        PathAlts = np.transpose(self.PathAlts)
        PathLats = np.transpose(self.PathLats)
        PathLongs = np.transpose(self.PathLongs)
        col2_8 = np.transpose(np.zeros((len(self.PathAlts))))
        col3 = np.transpose(np.full((len(self.PathAlts)),3))
        col4 = np.transpose(np.full((len(self.PathAlts)),16))
        lastcol = np.transpose(np.ones((len(self.PathAlts))))
        PathTime = np.transpose(self.PathTime)
        d = np.vstack((col2_8, col3, col4, col2_8, col2_8, col2_8, col2_8, PathLats, PathLongs, PathAlts, PathTime, lastcol)).T
        
        PathWind = np.transpose(self.PathWindSpeed)
        PathDir = np.transpose(self.PathWindDirection)
        s = np.vstack((PathAlts, PathWind, PathDir, PathTime)).T
        ds = pd.DataFrame(data=s)
        
        df = pd.DataFrame(data=d)
        filePath = self.Filepath
        fileNameWind  = filePath + '/' + self.Filename + "_windData.csv"
        fileNameWaypoints = filePath +'/' + self.Filename + "_Waypoints.csv"
        ds.to_csv(fileNameWind, header = False)
        df.to_csv(fileNameWaypoints, header = False)
        
# Converts degree latitude into feet
def LattoFeet(lat):
        DegreeToFeetLat = 364320 * lat

        return DegreeToFeetLat
    
# Converts Longitude in degrees to feet    
def LongtoFeet(lat,long):
        DegreesToRadians = np.pi/180
        LengthOfDegreeEQ = 69.172
        DegreeToMilesLong = np.cos((lat*DegreesToRadians))*LengthOfDegreeEQ
        MilesToFeet = 5280
        DegreeOfLong = DegreeToMilesLong*MilesToFeet
        FeetofLong = DegreeOfLong*long
        
        return FeetofLong

# Converts feet from the equator to degree of latitude
def FeettoLat(feet):
        Lat = feet/364320
        return Lat

# Converts feet from prime meridian to degree of longitude
def FeettoLong(lat,feet):
        MilesToFeet = 5280
        LengthOfDegreeEQ = 69.172
        DegreeToMilesLong = np.cos((np.radians(lat)))*LengthOfDegreeEQ
        FeetOfLong = DegreeToMilesLong * MilesToFeet
        long = feet/FeetOfLong
        
        return long     

def QuadraticSolver(a,b,c):
    
    n1 = -b+np.sqrt(b**2 - 4*a*c)
    n2 = -b-np.sqrt(b**2 - 4*a*c)
    d = 2*a
    
    sol1 = n1/d
    sol2 = n2/d
    
    if isinstance(sol1, complex):
        sol1 = -1
    if isinstance(sol2, complex):
        sol2 = -1
        
    if sol1 < 0 and sol2 < 0:
        sol = -1
    elif sol1 < 0 and sol2 >= 0:
        sol = sol2
    elif sol1 >= 0 and sol2 < 0:
        sol = sol1
    
    return[sol]

def density(h):
    if h < 36152:   # temp, press, density relationships to altitude
        T = 59 - 0.00356 * h
        p = 2116 * ((T + 459.7) / 518.6) ** 5.256       # lb/ft^2
        rho = p / (1718 * (T + 459.7))
        a = sqrt(1716 * 1.4 * (T + 459.7))
    elif 36152 <= h < 82345:
        T = -70
        p = 413.1 * exp(1.73 - 0.000048 * h)
        rho = p / (1718 * (T + 459.7))
        a = sqrt(1716 * 1.4 * (T + 459.7))
    elif h >= 82345:
        T = -205.05 + 0.00162 * h
        p = 51.97 * ((T + 459.7) / 389.98) ** -11.388
        rho = p / (1718 * (T + 459.7))
        a = sqrt(1716 * 1.4 * (T + 459.7))
    return [T, p, rho, a]


def Main():
    glider = Glider()
    glider.CurrentLat = 35.4676
    glider.CurrentLong = -97.5164
    glider.EndLat = 34.7767
    glider.EndLong = -98.7970
    glider.CurrentAlt = 100000
    glider.WindFile = r'C:\Users\Garrett\Documents\Senior Design\Senior Design Spring 2020/UpperAirSoundingNorman.txt'
    #glider.ReadWindData()
    #glider.ParseWindSpeed()
    
    #glider.GetWindSpeed()
    #print(glider.windDir)
    #print(glider.windSpeed)
    #glider.GetCourse()
    #print(glider.course)
    #glider.GetHeadTailCross(glider.course[40])
    #print(glider.HeadTailVector)
    #print(glider.CrossWindVector)
    glider.FindBestPath(SteadyState = 1)
#Main()