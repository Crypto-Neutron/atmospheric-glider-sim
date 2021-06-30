import numpy as np

import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Glider_Class import Glider
import random
#******************************************************************



class PlotCanvas(FigureCanvas):

    def __init__(self, parent, width=None, height=None, dpi=100):
        if width == None: width = parent.width()/100
        if height == None: height = parent.height()/100
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)



    def plotit(self, Alts, Lats, Longs, CurrentAlt, EndLat, EndLong, CurrentLat, CurrentLong):
        self.figure.clf()
        TotalTime = 0
        # if self.EndLat > self.StartLat:
        #    bottomLat = self.StartLat
        #    topLat = self.EndLat
        # else:
        #    bottomLat = self.EndLat
        #    topLat = self.StartLat
        # if self.EndLong > self.StartLong:
        #    leftLong = self.StartLong
        #    rightLong = self.EndLong
        # else:
        #    leftLong = self.EndLong
        #    rightLong = self.StartLong
        #plt.clf()
        #fig = plt.figure()
        ax = self.figure.gca(projection='3d')
        # bm = Basemap(llcrnrlon = leftLong - 1.25 , llcrnrlat = bottomLat - 1.25 , urcrnrlon = rightLong + 1.25, urcrnrlat = topLat + 1.25 , projection = 'cyl' , resolution = 'l', fix_aspect = False, ax = ax)
        ax.plot(Longs, Lats, Alts)
        ax.scatter(EndLong, EndLat, 0, c='g')
        ax.scatter(Longs[0], Lats[0], Alts[0])
        ax.text(Longs[0], Lats[0], Alts[0], 'Glide Point')
        ax.scatter(CurrentLong, CurrentLat, CurrentAlt, c='k')
        ax.text(CurrentLong, CurrentLat, CurrentAlt, 'End Glide')
        ax.text(EndLong, EndLat, 0, 'Final Point')
        ax.plot(Longs, Lats, 0, 'r')
        # ax.add_collection3d(bm.drawcountries(linewidth = 0.2))
        # ax.add_collection3d(bm.drawstates(linewidth = .2))
        # lonstep = 0.75
        # latstep = 0.75
        # meridian = np.arange(leftLong - 1.25, rightLong + 1.25, lonstep)
        # parrallel = np.arange(bottomLat - 1.25, topLat + 1.25, latstep)
        # ax.set_yticks(parrallel)
        # ax.set_xticks(meridian)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude - ft')
        # fig.scatter(p)
        # fig.scatter(a)
        if CurrentAlt <= 400:
            CurrentAlt = 0

        ax.set_title('Glide Path (End Alt = {a:.0f} ft)'.format(a=CurrentAlt))
        self.plot = ax
        # print()
        # print()
        # print()
        # print(TotalTime)
        self.draw()


from QT_Simulator import Ui_Dialog


class main_window(QDialog):
    def __init__(self):
        super(main_window,self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.assign_widgets()
        plotwin=self.ui.graphicsView
        self.m = PlotCanvas(plotwin)
        self.show()
        self.Present = None
        self.NOAA = None
        self.steadyState = None
        self.Year = None
        self.Month = None
        self.Day = None
        self. Hour = None
        self.WindFile = None
        global alts, lats, longs, endlat, endlong, endalt

    def assign_widgets(self):
        self.ui.pushButton_Exit.clicked.connect(self.ExitApp)
        self.ui.pushButton_windfilebrowse.clicked.connect(self.GetWindData)
        self.ui.pushButton_outfilebrowse.clicked.connect(self.GetOutFile)
        self.ui.pushButton_run.clicked.connect(self.RunProgram)
        
    def TimeSet(self):
        if self.ui.radioButton_presenttime.isChecked():
            self.Present = 1
        if self.ui.radioButton_customtime.isChecked():
            self.Present = None
            self.Glider.Year = int(self.ui.lineEdit_year.text())
            self.Glider.Month = int(self.ui.lineEdit_month.text())
            self.Glider.Day = int(self.ui.lineEdit_day.text())
            self.Glider.Hour = int(self.ui.lineEdit_hour.text())
            
            
    def WindSet(self):
        if self.ui.radioButton_steadystate.isChecked():
            self.steadyState = 1
            self.NOAA = None
        if self.ui.radioButton_noaaGFS.isChecked():
            self.steadyState = None
            self.NOAA = 1
        if self.ui.radioButton_windprofile.isChecked():
            self.steadyState = None
            self.NOAA = None
            self.WindFile = self.ui.lineEdit_windfilename.text()
            
    def RunProgram(self):
        self.Glider = None
        self.Glider = Glider()
        self.WindSet()
        self.TimeSet()
        self.Glider.Filepath = self.ui.lineEdit_outfiledest.text()
        self.Glider.Filename = self.ui.lineEdit_outfilename.text()
        self.Glider.WindFile = self.WindFile
        self.Glider.CurrentLat = float(self.ui.lineEdit_startlat.text())
        self.Glider.EndLat = float(self.ui.lineEdit_endlat.text())
        self.Glider.CurrentLong = float(self.ui.lineEdit_startlon.text())
        self.Glider.EndLong = float(self.ui.lineEdit_endlon.text())
        self.Glider.CurrentAlt = float(self.ui.lineEdit_startalt.text())
        self.Glider.FindBestPath(NOAA = self.NOAA, Present = self.Present, SteadyState = self.steadyState)
        self.m.plotit(self.Glider.PathAlts, self.Glider.PathLats, self.Glider.PathLongs,
                      self.Glider.CurrentAlt, self.Glider.EndLat, self.Glider.EndLong, self.Glider.CurrentLat,
                      self.Glider.CurrentLong)
        timealoft = '{:.1f}'.format(self.Glider.TotalTime)
        self.ui.lineEdit_timealoft.setText(str(timealoft))
        app.processEvents()
    
    def GetWindData(self):
        filename = QFileDialog.getOpenFileName()[0]
        if len(filename)==0:
            no_file()
            return
        self.ui.lineEdit_windfilename.setText(filename)
        app.processEvents()
        
    def GetOutFile(self):
        filedestination = QFileDialog.getExistingDirectory()
        if len(filedestination)==0:
            no_file()
            return
        self.ui.lineEdit_outfiledest.setText(filedestination)


    def ShowPlot(self):
        self.m.plotit()
        return


    def ExitApp(self):
        app.exit()


def no_file():
    msg = QMessageBox()
    msg.setText('There was no file selected')
    msg.setWindowTitle("No File")
    retval = msg.exec_()
    return None

def bad_file():
    msg = QMessageBox()
    msg.setText('Unable to process the selected file')
    msg.setWindowTitle("Bad File")
    retval = msg.exec_()
    return None


if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    main_win = main_window()
    sys.exit(app.exec_())

    
 