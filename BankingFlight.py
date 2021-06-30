import math
import matplotlib.pyplot as plt
#import matplotlib.ticker as plticker
#from mpl_toolkits import mplot3d
#from scipy.optimize import fsolve
from DragPolarSeniorDesign_updated import Drag
import numpy as np
#import pandas as pd
import xlrd
#import xlwt

class BankingFlight:
    def __init__(self):
        self.BankRadius = 1000 # bank radius
        self.Weight = 2.2 # weight - lb
        self.WingPFA = 362.88 # in^2
        self.GlideSpeed = None # banking glide speeed - best loiter speed
        self.Maxn = 2.5 # max load factor
        self.DescentRate = None # descent rate of the glider
        self.InitialHeight = None # initial height of the banking maneuver
        self.WindVelocity = None # wind velocity
        self.WindAngle = None # wind angle - cartesian
        self.GlideAngle = None # wind angle - cartesian
        self.omega = None # frequency around the circle
        self.BankAngle = None
        self.dh = 1000 # height decrement update - arbitrary value right now
        self.time = 0 # time in banking flight
        self.Arc = (math.pi/4)*self.BankRadius # arc length for every 90 deg banked
        self.deg = 0 # keeps track of the degrees banked
        self.Height = 0 # Height Tracker
        self.HeightLost = 0
        self.time = 0 # track the time in Loiter
        # arrays to track the gliders position - just initializing to append later
        self.x = [0]*1
        self.y = [0]*1
        self.z = [0]*1
        self.avgDR = 0 # average descent rate - used for plotting the curve
        self.avgGS = 0 # average glide speed - used for plotting the curve
        self.avgBA = 0 # average bank angle
        # access drag spreadsheet
        self.drag = Drag()
        self.workbook = xlrd.open_workbook(
            r'stopcrashingexcel.xlsm')  # open the workbook
        self.worksheet = self.workbook.sheet_by_name('DragBuildUp')

    def GetGlideVals(self,Height):
        # this function updates the glide speeds / DR as we descend
        # call the required methods from the Drag class:
        self.drag.ReadExcelFile(self.worksheet)
        self.drag.GetDrag(Height)
        self.drag.PlotDrag(Height)
        self.drag.GlidingPerformance(Height)
        self.GlideSpeed = self.drag.BestEnduranceVelocity[0]  # banking glide speed
        self.DescentRate = self.drag.BestEnduranceVelocity[1]  # descent rate of the glider
        self.BankAngle = math.atan(self.GlideSpeed ** 2 / (32.2 * self.BankRadius))  # bank angle calc
        self.GlideAngle = self.drag.EnduranceGA # Glide angle
        self.InitialHeight = Height  # initial height of the banking maneuver
        self.omega = self.GlideSpeed / self.BankRadius  # frequency around the circle


    def Banking(self, Height):

        self.InitialHeight = Height # initial height
        self.x = self.BankRadius * np.cos(self.omega * self.time)  # initial x
        self.y = self.BankRadius * np.sin(self.omega * self.time)  # initial y
        self.z = Height  # initial z

        while Height >= 0:
            n = 1 # counter variable used to compute the average glide speed and descent rate
            DRtracker = self.DescentRate
            GStracker = self.GlideSpeed
            BAtracker = self.BankAngle

            HeightLost = DistanceBanked(self.BankRadius, self.GlideSpeed, Height, self.DescentRate)[0] # dont really use this
            CurrentHeight = DistanceBanked(self.BankRadius, self.GlideSpeed, Height, self.DescentRate)[1] # find current height
            Height = CurrentHeight
            self.time = DistanceBanked(self.BankRadius, self.GlideSpeed, Height, self.DescentRate)[2] + self.time # get the time

            # append these new x,y,z vals to their respective arrays
            newx = self.BankRadius * np.cos(self.omega * self.time)
            newy = self.BankRadius * np.sin(self.omega * self.time)
            newz = CurrentHeight
            if newz <0:
                break
            # update the altitude and bank radius arrays
            self.x = np.append(self.x,newx)
            self.y = np.append(self.y,newy)
            self.z = np.append(self.z,newz)
            # get our new descent rate / glide speeds based on altitude
            # reading data from the excel file

            # call the get glide vals function to obtain new speeds and descent rate as we descend
            self.GetGlideVals(newz)
            
            velocity = math.sqrt(self.GlideSpeed ** 2 + (self.DescentRate) ** 2)
            Lift = self.Weight * math.cos(self.GlideAngle) # lift generated while gliding at a specific glide angle
            Liftneeded = self.Weight/32.2 * velocity**2 /(self.BankRadius*math.sin(self.BankAngle)) # lift needed to bank
            Liftdecrement = Liftneeded - Lift
            rho = AtmosphericData(Height, 0, indicator='rho')

            # I used this loop to take into account how banking will cause an increase in our descent rate
            while Lift != Liftneeded:
                # i may be totally wrong here - not sure, but i think its logical?
                CL = Lift / (0.5 * rho * velocity ** 2 * self.WingPFA / 144) # assuming our AoA is constant, glide angle is constant too, since its dependent upon horizontal speed and we're changing descent rate to increase the magnitude of our velocity
                self.DescentRate = self.DescentRate + -0.1 # increment descent rate until our Lift generated equals our lift needed to bank
                newvelocity = math.sqrt(self.GlideSpeed ** 2 + self.DescentRate ** 2) # new magnitude of our velocity
                newLift = CL * (0.5 * rho * newvelocity ** 2 * self.WingPFA / 144) # new lift generated
                self.BankAngle = math.atan(newvelocity**2/(32.2*self.BankRadius))
                delta = Liftneeded - newLift # check the difference
                if delta < 0.01:
                    break
            #print(self.x)

            # tracking average Descent Rate and Glide Speed
            GStracker = GStracker + self.GlideSpeed
            DRtracker = DRtracker + self.DescentRate
            BAtracker = BAtracker + self.BankAngle
            n+=1


        self.avgDR = DRtracker/n
        #print('Average Descent Rate: {:.4f} ft/s'.format(self.avgDR))
        self.avgGS = GStracker/n
        #print('Average Glide Speed: {:.4f} ft/s'.format(self.avgGS))
        self.avgBA = BAtracker/n * 180/math.pi # convert from rad to deg
        #print('Average Bank Angle: {:.4f} deg'.format(self.avgBA))

    def plotBank(self, initialHeight):
        # print(self.x)
        # print(self.y)
        # print(self.z)
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.BankRadius, self.BankRadius,self.InitialHeight)
        # x, y, and z are all arrays of points during the gliders descent
        ax.scatter(self.x, self.y, self.z, c='g')

        #this code just plots the circular path as we descend
        V = self.avgGS  # velocity
        r = self.BankRadius # bank radius
        omega = V / r  # frequency
        t = self.time
        t = np.linspace(0, t, 5000)  # time
        x1 = r * np.cos(omega * t)
        x2 = r * np.sin(omega * t)
        zline = initialHeight + self.avgDR * t
        ax.plot3D(x1, x2, zline, 'gray')
        plt.show()

    def GiveTime(self): # gives the time in flight
        timeMin = self.time / 60
        print('Total time in Loiter: {:.4f} minutes'.format(timeMin))

def AtmosphericData(h,airspeed, indicator=''):
    # Function to calculate Density, Temperature, Pressure, Mach Number, Speed of Sound and Dynamic Viscosity
    gamma = 1.4  # Constant Specific Heats Model
    R = 1716  # Ideal Gas Constant

    if h < 36152:  # Troposphere
        T = 59 - 0.00356 * h + 459.67 # Temperature in Rankine
        P = 2116 * ((T) / 518.6) ** 5.256  # Pressure in psf
        rho = P / (1718 * (T))  # Density in slugs/ft^3
        mu = 3.62 * 10 **(-7) * (T/ 518.7) ** (1.5) * ((518.7 + 198.72) / (T + 198.72))  # Dynamic Viscosity lb-s/ft^2
        SoS = math.sqrt(gamma * R * T)  # Speed of Sound
        Mach = airspeed / math.sqrt(gamma * R * T)  # Mach Number

    elif h > 36152 and h < 82345:  # Lower Stratosphere
        T = -70 + 459.67  # Temperature in Fahrenheit
        P = 473.1 * math.exp(1.73 - 0.000048 * h)  # Pressure in psf
        rho = P / (1718 * (T))  # Density in slugs/ft^3
        mu = 3.62 * 10 ** (-7) * ((T) / 518.7) ** 1.5 * ((518.7 + 198.72) / (T + 198.72))  # Dynamic Viscosity lb-s/ft^2
        SoS = math.sqrt(gamma * R * T)  # Speed of Sound
        Mach = airspeed / math.sqrt(gamma * R * T)  # Mach Number

    elif h > 82345:  # Upper Stratosphere
        T = -205.05 + 0.00164 * h + 459.67  # Temperature in Fahrenheit
        P = 51.97 * ((T) / 389.98) ** -11.388  # Pressure in psf
        rho = P / (1718 * (T))  # Density in slugs/ft^3
        mu = 3.62 * 10 ** (-7) * ((T) / 518.7) ** 1.5 * ((518.7 + 198.72) / (T + 198.72))  # Dynamic Viscosity lb-s/ft^2
        SoS = math.sqrt(gamma * R * T)  # Speed of Sound
        Mach = airspeed / math.sqrt(gamma * R * T)  # Mach Number

    # return whatever value you need
    if indicator == 'T':
        return T
    if indicator == 'P':
        return P
    if indicator == 'rho':
        return rho
    if indicator == 'mu':
        return mu
    if indicator == 'SoS':
        return SoS
    if indicator == 'Mach':
        return Mach


def DistanceBanked(BankRadius, Speed, InitialHeight, DescentRate): # make this a regular function to be called by the program, less maintenance

    # Looking at 1/4th of a projected circle
    Arc = (math.pi / 4) * BankRadius  # arc length for every 90 deg banked
    LengthTraveled = Arc # traveled an arc length over the ground
    deltat = LengthTraveled / Speed # time to cross the length is arcdistance / glidespeed

    # find the total altitude lost
    CurrentHeight = InitialHeight + DescentRate * deltat # descent rate is already negative
    HeightLost = CurrentHeight - InitialHeight # total height lost

    return HeightLost, CurrentHeight, deltat


def main():

    G = BankingFlight()
    G.GetGlideVals(10000)
    G.Banking(10000)
    G.plotBank(10000)
    G.GiveTime()

#main()
