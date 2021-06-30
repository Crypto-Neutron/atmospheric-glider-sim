import math
import numpy as np
import pandas as pd
import xlrd
import xlwt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits import mplot3d
from scipy.optimize import fsolve
#author: Zachary Morrison
# If you are using this in the future and have any questions you can reach me at
# zachmor@okstate.edu
# or (580)730-5274, just text me before you call, since i usually dont answer calls from unknown phone numbers
# also, if there are improvements to be made / somewhere where i went wrong with an assumption, I'd like to learn.


# I havent messed with this class labeled Aircraft yet, may not even use it
class Aircraft():
    def __init__(self):
        # found in the excel file
        self.WingSpan = None
        self.WingArea = None
        self.AspectRatio = None
        self.WingSweepAngle= None
        self.RootChord = None
        self.TipChord = None
        self.e = None
        self.WingMAC = None
        self.Weight = None

    def ReadExcelFile(self, worksheet):
        # python indexes at 0
        self.WingSpan = worksheet.cell(1, 1).value # Wing Span - in
        self.WingArea = worksheet.cell(2, 1).value # Wing Planform Area - in^2
        self.AspectRatio = worksheet.cell(3, 1).value # Wing Aspect Ratio
        self.WingSweepAngle = worksheet.cell(4, 1).value # Wing Sweep Angle - degrees
        self.RootChord = worksheet.cell(5, 1).value # Wing Root Chord - in
        self.TipChord = worksheet.cell(6, 1).value # Wing Tip Chord - in
        self.e = worksheet.cell(8, 1).value # Wing efficiency factor
        self.WingMAC = worksheet.cell(12, 1).value # Wing MAC - in
        self.Weight = worksheet.cell(11, 1).value # Aircraft Weight - lb
        # print(self.Weight, self.RootChord, self.WingMAC, self.e)

# this class does all the drag based / performance calculations
class Drag():
    def __init__(self):

        # found in the excel file
        # Estimated Planform Areas for each component
        self.FuselagePFA = None
        self.WingPFA = None # Wing Planform Area, used as the reference for these calcs
        self.VTailPFA = None
        self.CFPodsPFA = None
        self.CWPodsPFA = None
        self.WTPodsPFA = None
        self.RearPodPFA = None

        # Wetted Areas for each component
        self.FuselageWA = None
        self.WingWA = None
        self.VTailWA = None
        self.CFPodsWA = None
        self.CWPodsWA = None
        self.WTPodsWA = None
        self.RearPodWA = None

        # Reference Lengths for each component
        self.FuselageRL = None
        self.WingRL = None
        self.VTailRL = None
        self.CFPodsRL = None
        self.CWPodsRL = None
        self.WTPodsRL = None
        self.RearPodRL = None

        # flow experienced by the component, flowtype = FT, Reads turbulent %
        self.FuselageFT = None
        self.WingFT = None
        self.VTailFT = None
        self.CFPodsFT = None
        self.CWPodsFT = None
        self.WTPodsFT = None
        self.RearPodFT = None

        # from factors of each component
        self.FuselageFF = None
        self.WingFF = None
        self.VTailFF = None
        self.CFPodsFF = None
        self.CWPodsFF = None
        self.WTPodsFF = None
        self.RearPodFF = None

        # interference factors for each component
        self.FuselageIF = None
        self.WingIF = None
        self.VTailIF = None
        self.CFPodsIF = None
        self.CWPodsIF = None
        self.WTPodsIF = None
        self.RearPodIF = None

        # Zero-Lift Drag Values for each component
        self.FuselageDo = None
        self.WingDo = None
        self.VTailDo = None
        self.CFPodsDo = None
        self.CWPodsDo = None
        self.WTPodsDo = None
        self.RearPodDo = None

        # Induced Drag Value
        self.WingDi = None

        #Total Drag Values
        self.TotalDo = None
        self.TotalDi = None
        self.TotalD = None

        # Aircraft Weight
        self.Weight = 2.2

        #Velocity Array for Drag Calculations
        self.start = 5
        self.stop = 100
        self.n = 1000
        self.Velocity = np.linspace(self.start, self.stop, self.n)

        #Performance Parameter Variables
        self.VminDrag = None # Velocity for at which Minimum Drag occurs
        self.minD = None # minimum drag value
        self.minDi = None # minimum induced drag value
        self.minDo = None # minimum zero-lift drag value

        #Gliding Performance Parameters
        self.DescentRate = None # Glider Descent Rate
        self.glideangle = None
        self.start = -1*math.pi
        self.stop = -3*math.pi/2
        self.n = 1000
        self.glideangle = np.linspace(self.start, self.stop, self.n)
        self.RangeGA = 0
        self.EnduranceGA = 0

        # Stall Velocity
        self.StallV = 0

    def ReadExcelFile(self, worksheet): # Method for parsing the excel spreadsheet
        # look through the worksheet table to get the values
        for k in range(16,23,1):
            # for j in range(1,7,1):
                if k == 16:
                    if worksheet.cell(k,0).value == "Fuselage":
                        self.FuselagePFA = worksheet.cell(k, 1).value
                        self.FuselageWA = worksheet.cell(k, 2).value
                        self.FuselageRL = worksheet.cell(k, 3).value
                        self.FuselageFT = worksheet.cell(k, 6).value
                        self.FuselageFF = worksheet.cell(k, 7).value
                        self.FuselageIF = worksheet.cell(k, 8).value
                if k == 17:
                    if worksheet.cell(k,0).value == "Wing":
                        self.WingPFA = worksheet.cell(k, 1).value
                        self.WingWA = worksheet.cell(k, 2).value
                        self.WingRL = worksheet.cell(k, 3).value
                        self.WingFT = worksheet.cell(k, 6).value
                        self.WingFF = worksheet.cell(k, 7).value
                        self.WingIF = worksheet.cell(k, 8).value
                if k == 18:
                    if worksheet.cell(k,0).value == "Vertical Stabs (2)":
                        self.VTailPFA = worksheet.cell(k, 1).value
                        self.VTailWA = worksheet.cell(k, 2).value
                        self.VTailRL = worksheet.cell(k, 3).value
                        self.VTailFT = worksheet.cell(k, 6).value
                        self.VTailFF = worksheet.cell(k, 7).value
                        self.VTailIF = worksheet.cell(k, 8).value
                if k == 19:
                    if worksheet.cell(k,0).value == "Center Fuselage Pods (2)":
                        self.CFPodsPFA = worksheet.cell(k, 1).value
                        self.CFPodsWA = worksheet.cell(k, 2).value
                        self.CFPodsRL = worksheet.cell(k, 3).value
                        self.CFPodsFT = worksheet.cell(k, 6).value
                        self.CFPodsFF = worksheet.cell(k, 7).value
                        self.CFPodsIF = worksheet.cell(k, 8).value
                if k == 20:
                    if worksheet.cell(k,0).value == "Center Wing Pods (2)":
                        self.CWPodsPFA = worksheet.cell(k, 1).value
                        self.CWPodsWA = worksheet.cell(k, 2).value
                        self.CWPodsRL = worksheet.cell(k, 3).value
                        self.CWPodsFT = worksheet.cell(k, 6).value
                        self.CWPodsFF = worksheet.cell(k, 7).value
                        self.CWPodsIF = worksheet.cell(k, 8).value
                if k == 21:
                    if worksheet.cell(k,0).value == "Wing Tip Pods (2)":
                        self.WTPodsPFA = worksheet.cell(k, 1).value
                        self.WTPodsWA = worksheet.cell(k, 2).value
                        self.WTPodsRL = worksheet.cell(k, 3).value
                        self.WTPodsFT = worksheet.cell(k, 6).value
                        self.WTPodsFF = worksheet.cell(k, 7).value
                        self.WTPodsIF = worksheet.cell(k, 8).value
                if k == 22:
                    if worksheet.cell(k,0).value == "Rear Pod (1)":
                        self.RearPodPFA = worksheet.cell(k, 1).value
                        self.RearPodWA = worksheet.cell(k, 2).value
                        self.RearPodRL = worksheet.cell(k, 3).value
                        self.RearPodFT = worksheet.cell(k, 6).value
                        self.RearPodFF = worksheet.cell(k, 7).value
                        self.RearPodIF = worksheet.cell(k, 8).value

    def GetDrag(self, height):
        # this portion gets us the zero-lift drag contribution of each component
        # Calling the Do function for each component
        Velocity = self.Velocity
        self.FuselageDo = Do(height, self.FuselageRL, self.FuselageFF, self.WingPFA, self.FuselageWA, Velocity, flowtype=self.FuselageFT, multiplier=1, IF = self.FuselageIF)
        self.WingDo = Do(height, self.WingRL, self.WingFF, self.WingPFA, self.WingWA, Velocity, flowtype=self.WingFT, multiplier=1, IF = self.WingIF)
        self.VTailDo = Do(height, self.VTailRL, self.VTailFF, self.WingPFA, self.VTailWA,Velocity, flowtype=self.VTailFT, multiplier=2, IF = self.VTailIF)
        self.CFPodsDo = Do(height, self.CFPodsRL, self.CFPodsFF, self.WingPFA, self.CFPodsWA,Velocity, flowtype=self.CFPodsFT, multiplier=2, IF = self.CFPodsIF)
        self.CWPodsDo = Do(height, self.CWPodsRL, self.CWPodsFF, self.WingPFA, self.CWPodsWA,Velocity, flowtype=self.CWPodsFT, multiplier=2, IF = self.CWPodsIF)
        self.WTPodsDo = Do(height, self.WTPodsRL, self.WTPodsFF, self.WingPFA, self.WTPodsWA,Velocity, flowtype=self.WTPodsFT, multiplier=2, IF = self.WTPodsIF)
        self.RearPodDo = Do(height, self.RearPodRL, self.RearPodFF, self.WingPFA, self.RearPodWA, Velocity, flowtype=self.RearPodFT, multiplier=1, IF = self.RearPodIF)

       # Calling the Di function for each component
        self.WingDi = Di(height, self.WingPFA, Velocity, self.glideangle, Lreq = 2.2)[0]

    def PlotDrag(self, height):
        # the velocity used here is assumed to be the magnitude of the horizontal velocity, since the descent rates are usually very small
        # Code to Plot the drag
        Velocity = self.Velocity
        self.TotalDo = np.array((self.FuselageDo + self.WingDo + self.VTailDo + self.CFPodsDo + self.CWPodsDo + self.WTPodsDo + self.RearPodDo))
        self.TotalDi = self.WingDi
        self.TotalD = (self.TotalDi + self.TotalDo)

        self.StallV = StallSpeed(1.5, self.WingPFA / 144, 2.2, AtmosphericData(height, Velocity[0], indicator='rho'))
        stallV = self.StallV
        #plotting the glider velocity v time
        # fig, ax = plt.subplots()
        # xline = [stallV]
        # for line in xline:
        #     plt.axvline(x=line, color = 'k', label = 'Stall Speed, %.2g ft/s' % stallV)
        # ax.plot(Velocity, self.TotalDo, 'b-', label='Zero-Lift Drag, Do')
        # ax.plot(Velocity, self.TotalDi, 'r-', label='Induced Drag, Di')
        # ax.plot(Velocity, self.TotalD, 'g-', label='Total Drag, D')
        # # ax.plot(glideangle, DragExpected, 'c-', label='Expected Drag, D')
        #
        # plt.ylim(0,4) # sets the y-axis range on the graphs
        # loc = plticker.MultipleLocator(base=0.5) # changes the y-axis increment values
        # ax.yaxis.set_major_locator(loc) # part of this code ^^
        # plt.legend(loc='upper right')
        # plt.xlabel('Velocity, ft/s')
        # plt.ylabel('Drag - lb')
        # plt.title('Glider Drag')
        # plt.show()

    def FindMinDrag(self):
        # dont really use this atm
        self.minDi = min(self.TotalDi)
        self.minDo = min(self.TotalDo)
        self.minD = min(self.TotalD)

        minD = min(self.TotalD)
        IndexminD = np.where(self.TotalD == minD)
        print(self.Velocity[IndexminD], minD)

    def GlidingPerformance(self, height, indicator = ''):
        # this function generates the Rate of Descent vs Equilibrium Glide Velocity graph
        # Stall speed calc
        h = height
        Velocity = self.Velocity
        self.DescentRate = GliderDescent(height, self.WingPFA, 2.2, self.TotalD, self.Velocity) # calling the descent rate function

        self.StallV = StallSpeed(1.5, self.WingPFA / 144, 2.2, AtmosphericData(height, Velocity[0], indicator='rho'))# stall speed
        stallV = self.StallV # stall speed
        rho = AtmosphericData(height,Velocity[0],indicator='rho') # density

        CLarray = self.WingDi = Di(height, self.WingPFA, Velocity, self.glideangle, Lreq = 2.2)[1] # CL
        CDarray = self.TotalD / (0.5*rho*Velocity**2*self.WingPFA/144) # CD
        LD = CLarray/CDarray # CL / CD

        LDmax = max(LD)
        indexLDmax = np.where(LD == LDmax)
        # print(indexLDmax)
        # print(LDmax)
        THLDmax = max(self.DescentRate) # max, since Descent Rate is negative
        indexTHLDmax = np.where(self.DescentRate == THLDmax)
        # print(LD)

        # tangent line stuff - used for validting LDmax
        x=[0,Velocity[indexLDmax],2*Velocity[indexLDmax]]
        m=self.DescentRate[indexLDmax]/Velocity[indexLDmax]
        y = m*x
        # plotting the descent rate, CL/CD curve
        fig, ax = plt.subplots()
        plt.plot(x, y, 'r-')
        plt.plot(Velocity, self.DescentRate)
        # plt.plot(Velocity, LD)
        # plt.plot(Velocity[indexLDmax],LDmax, label="CL/CD", marker = "o", Label=('CL/CD Max: %.2g' % (LDmax)))
        plt.plot(Velocity[indexTHLDmax], self.DescentRate[indexTHLDmax], marker = "o", Label=('Best Endurance: (%.2g ft/s, %.2g ft/s)' %
                                                                                           (Velocity[indexTHLDmax], self.DescentRate[indexTHLDmax])))
        plt.plot(Velocity[indexLDmax], self.DescentRate[indexLDmax], marker="o", Label=('Best Range: (%.2g ft/s, %.2g ft/s)' %
                                                                                           (Velocity[indexLDmax], self.DescentRate[indexLDmax])))

        plt.plot(Velocity[indexTHLDmax],self.DescentRate[indexTHLDmax])
        plt.legend(loc='lower right')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xlabel('Horizontal Velocity, ft/s')
        plt.ylabel('Vertical Velocity, ft/s')
        plt.title('Glider Descent')
        plt.show()
        # made indiactors to return either endurance or range, along with the corresponding horizontal velocity and descent rate
        # also returns CDo at each value if needed
        if indicator == 'Endurance':
             # Best Endurance Horizontal Velocity and Descent Rate
            EnduranceHV = float(Velocity[indexTHLDmax]) # best Endurance Horizontal Velocity
            EnduranceDR = float(self.DescentRate[indexTHLDmax]) # best Endurance Descent Rate
            self.EnduranceGA = float(math.asin(self.TotalD[indexTHLDmax]/self.Weight)) # best endurance glide angle
            # print('Glide Angle for best Endurance: {:.4f} rad'.format(self.EnduranceGA))
            # print('Horizontal Speed for best Endurance: {:.4f} ft/s'.format(EnduranceHV))
            # print('Descent Rate for best Endurance: {:.4f} ft/s'.format(EnduranceDR))
            CDo = float(self.TotalDo[indexTHLDmax] / (0.5*rho*((math.sqrt(EnduranceDR**2 + EnduranceHV**2)))**2*self.WingPFA/144)) # Zero Lift Drag coefficient at this speed
            # print('CDo value at the speed for best endurance {:.4f} '.format(CDo))
            # print('The Stall speed of the aircraft at an altitude of {:.f} ft is {:.4f} ft/s'.format(height, stallV))
            return EnduranceHV, EnduranceDR
        if indicator == 'Range':
            # Best Range Horizontal Velocity and Descent Rate
            RangeHV = float(Velocity[indexLDmax]) # best Range Horizontal Velocity
            RangeDR = float(self.DescentRate[indexLDmax]) # best Range Descent Rate
            self.RangeGA = float(math.asin(self.TotalD[indexLDmax] / self.Weight)) # best range glide angle
            Drag = 2.2*math.sin(self.RangeGA)
            # print('Glide Angle for best Range: {:.4f} rad'.format(self.RangeGA))
            # print('Horizontal Speed for best Range: {:.4f} ft/s'.format(RangeHV))
            # print('Descent Rate for best Range: {:.4f} ft/s'.format(RangeDR))
            CDo = float(self.TotalDo[indexLDmax] / (0.5 * rho * ((math.sqrt(RangeDR**2 + RangeHV**2)))**2 * self.WingPFA / 144)) # Zero Lift Drag coefficient at this speed
            # print('CDo value at the speed for best range {:.4f} '.format(CDo))
            # print('The Stall speed of the aircraft at an altitude of {:.2f} ft is {:.4f} ft/s'.format(height, stallV))
            return RangeHV, RangeDR
        # Returns Horizontal Velocity, Descent Rate, Glide Angle, Zero-Lift Drag Coeff., and Stall Velocity

    def ReturnStall(self, height):
        # assuming CLmax = 1.5 for our glider, change if you need to
        self.StallV = StallSpeed(1.5, self.WingPFA / 144, 2.2, AtmosphericData(height, 0, indicator='rho'))  # stall speed
        return self.StallV

    def GivenSpeedGiveCDo(self, Height):
        rho = AtmosphericData(Height, 0, indicator='rho')
        avgDo = sum(self.TotalDo)/len(self.TotalDo)
        avgV = sum(self.Velocity)/len(self.Velocity) # not super rigorous here
        return avgDo / (0.5*rho*avgV**2*self.WingPFA/144)


# Zero Lift Drag Calculations
def Do(height, referencelength, formfactor, Sref_Wing, Swet_Component, Velocity, flowtype = 0, multiplier=1, IF=1):
    # references Raymer's Aircraft Design: A Conceptual Approach for Equations
    Do = np.zeros_like(Velocity)
    Cf = np.zeros_like(Velocity)
    density = AtmosphericData(height, Velocity[0], indicator='rho')
    mu = AtmosphericData(height, Velocity[0], indicator='mu')
    MachNum = np.zeros_like(Velocity)
    flowtypePercent = flowtype / 100 # flowtype tells use the % of the flow that is turbulent

    for i in range(len(Velocity)):
        MachNum[i] = AtmosphericData(height, Velocity[i], indicator='Mach')
        if flowtypePercent == 1: # use the raymer equation for turbulent skin friction
            Cf[i] = 0.455 / ((math.log10(ReynoldsNumber(density,Velocity[i],referencelength/12,mu)))**2.58 * (1 + 0.144*MachNum[i]**2)**0.65)

        elif flowtypePercent == 0: # use the raymer equation for laminar skin friction
            Cf[i] = 1.328 / math.sqrt(ReynoldsNumber(density,Velocity[i],referencelength/12,mu))

        elif flowtypePercent > 0 and flowtype > 1: # mixed flow, use percentage method
            Cf[i] = (1-flowtypePercent) * (1.328 / math.sqrt(ReynoldsNumber(density,Velocity[i],referencelength/12,mu))) + flowtypePercent* (0.455 / ((math.log10(ReynoldsNumber(density,Velocity[i],referencelength/12,mu)))**2.58 * (1 + 0.144*MachNum[i]**2)**0.65))

        Do[i] = (1/math.sqrt(1-MachNum[i]**2))*multiplier*IF*formfactor*Swet_Component/Sref_Wing* Cf[i] * (0.5 * density * Velocity[i]**2 * Sref_Wing/144)

    return Do

# Induced Drag Calculations
def Di(height, Sref_Wing, Velocity, glideangle, Lreq):
    # induced drag - you know the drill
    Di = np.zeros_like(Velocity)
    CLreq = np.zeros_like(Velocity)
    density= AtmosphericData(height, Velocity[0], indicator='rho')
    MachNum = np.zeros_like(Velocity)
    for i in range(len(Velocity)):
        MachNum[i] = AtmosphericData(height, Velocity[i], indicator='Mach')
        CLreq[i] = (Lreq) / (0.5 * density * Velocity[i]**2 * Sref_Wing/144)
        Di[i] = (1/math.sqrt(1-MachNum[i]**2))*((CLreq[i])**2 / (math.pi*0.7*6.4))* (0.5 * density * Velocity[i]**2 * Sref_Wing/144)
    return Di, CLreq

# Glider Descent Rate Calculations
def GliderDescent(height, Sref_Wing, Weight, Darray, Velocity):
    # glider descent - references Dr. Jacob's Lecture on Steady Performance, slide 18
    DescentRate = np.zeros_like(Darray)
    CLreq = np.zeros_like(Darray)
    CDarray = np.zeros_like(Darray)
    density = AtmosphericData(height,Velocity[0],indicator='rho')
    for i in range(len(DescentRate)):
        CDarray[i] = Darray[i] /(0.5*(density)*Velocity[i]**2 * Sref_Wing)
        CLreq[i] = Weight / (0.5*(density)*Velocity[i]**2 * Sref_Wing)
        DescentRate[i] = -1*math.sqrt( (2*(Weight/Sref_Wing)) / (density * CLreq[i]**3 / CDarray[i]**2))

    return DescentRate

# Atmospheric Data Calculations
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

#Reynolds Number Calculation
def ReynoldsNumber(rho, V, chord, mu):
    Re = rho*V*chord/mu
    return Re

#Stall Speed Calculation
def StallSpeed(CLmax,S, W, rho):
    Stallspeed = math.sqrt((2*W)/(S*rho*CLmax))
    return Stallspeed

def main():
    # reading data from the excel file
    workbook = xlrd.open_workbook(r'C:\Users\Zachary\Desktop\Spring 2020\Senior Design\Glider\The Matrix\Senior Design Sim\Zachs_Code\stopcrashingexcel.xlsm') # open the workbook
    worksheet = workbook.sheet_by_name('DragBuildUp')
    # get a data w/ selected rows and columns

    # Wing Characteristics---------------------------------------------
    b = worksheet.cell(1,1).value  # Wing Span - in
    S = worksheet.cell(2,1).value  # Projected Wing Span & Sref in^2
    AR = worksheet.cell(3,1).value # Wing Aspect Ratio
    SweepAngle = worksheet.cell(4,1).value # Wing Sweep Angle in Degrees
    RootChord = worksheet.cell(5,1).value # Wing root chord - in
    TipChord = worksheet.cell(6,1).value # Wing tip chord - in
    TipRootRatio = worksheet.cell(7,1).value # tip to root ratio
    e = worksheet.cell(8,1).value # Oswald Efficiency Factor
    Weight = worksheet.cell(11,1).value # Aircraft Weight - lb
    MAC = worksheet.cell(12,1).value # Mean Aerodynamic Chord of the Wing - in

    #print(b, S, Weight, MAC) # Self Check

    G1 = Drag()
    G1.ReadExcelFile(worksheet)
    G1.GetDrag(0)
    G1.PlotDrag(0)
    G1.FindMinDrag()
    # G1.GlidingPerformance(0, indicator='Range')
    print(G1.GlidingPerformance(0, indicator= 'Endurance'))
    # G2 = Drag()
    # G2.ReadExcelFile(worksheet)
    # G2.GetDrag(50000)
    # G2.PlotDrag()
    # G2.FindMinDrag()
    # G2.GlidingPerformance(50000)

    # glider = Aircraft()
    # glider.ReadExcelFile(worksheet)

if __name__== "__main__":
    main()



