#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: bars.py
# Date: Thu Dec 26 15:08:02 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
from PyQt4 import QtGui
from PyQt4.QtMultimedia import *
#from PyQt4 import QtMoblity
from PyQt4.QtCore import *

class Example(QtGui.QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Main window')
        self.show()
        #dev = QAudioDeviceInfo.defaultInputDevice()
        dev = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)[0]
        fmt = dev.preferredFormat()
        recorder = QAudioInput(dev, fmt)


        ofile = QFile("./out.raw")
        ofile.open(QIODevice.WriteOnly | QIODevice.Truncate)

        def stopfunc():
            print 'stop'
            recorder.stop()
            ofile.close()

        QTimer.singleShot(10000, stopfunc)
        print 'start'
        recorder.start(ofile)
        print 'here'



def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
