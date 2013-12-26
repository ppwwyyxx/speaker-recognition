#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gui.py
# Date: Thu Dec 26 16:28:16 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
import numpy as np
from PyQt4 import uic
from scipy.io import wavfile
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtCore,QtGui
import pyaudio
from VAD import remove_silence
from utils import write_wav


class RecorderThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        while True:
            data = self.main.stream.read(1)
            self.main.recordData.extend([ord(x) for x in data])

class Main(QMainWindow):
    FS = 8000
    TEST_DURATION = 3

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        uic.loadUi("edytor2.ui", self)
        self.statusBar()
        self.recoProgressBar.setValue(0)

        self.enrollRecord.clicked.connect(self.start_record)
        self.stopEnrollRecord.clicked.connect(self.stop_enroll_record)
        self.enrollFile.clicked.connect(self.enroll_file)
        self.enroll.clicked.connect(self.do_enroll)

        self.recoRecord.clicked.connect(self.start_record)
        self.stopRecoRecord.clicked.connect(self.stop_reco_record)
        self.newReco.clicked.connect(self.new_reco)
        self.recoFile.clicked.connect(self.reco_file)
        self.new_reco()


        #UI.init

        self.UploadImage.clicked.connect(self.upload_avatar)
        self.userdata =[]

        #movie test
        self.movie = QMovie(u"image/recording.gif")
        self.Animation.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()

        self.movie2 = QMovie(u"image/recording.gif")
        self.Animation_2.setMovie(self.movie)
        self.movie2.start()
        self.movie2.stop()

        #default user image setting
        defaultimage = QPixmap(u"image/nouser.jpg")
        self.Userimage.setPixmap(defaultimage)
        self.Reco_Userimage.setPixmap(defaultimage)




    ############ RECORD
    def start_record(self):
        self.pyaudio = pyaudio.PyAudio()
        self.status("Recording...")

        self.recordData = []
        self.stream = self.pyaudio.open(format=pyaudio.paUInt8, channels=1, rate=Main.FS,
                        input=True, frames_per_buffer=1)
        self.reco_th = RecorderThread(self)
        self.reco_th.start()
        self.movie.start()

    def stop_record(self):
        self.reco_th.terminate()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.status("Record stopeed")
        self.movie.stop()

    ###### RECOGNIZE
    def new_reco(self):
        self.recoRecordData = np.array((), dtype='uint8')
        self.recoProgressBar.setValue(0)

    def stop_reco_record(self):
        self.stop_record()
        signal = np.array(self.recordData, dtype='uint8')
        self.reco_remove_update(Main.FS, signal)

    def reco_remove_update(self, fs, signal):
        fs, new_signal = remove_silence(fs, signal)
        print "After removed: {0} -> {1}".format(len(signal), len(new_signal))
        self.recoRecordData = np.concatenate((self.recoRecordData, new_signal))
        real_len = float(len(self.recoRecordData)) / Main.FS / Main.TEST_DURATION * 100
        if real_len > 100:
            real_len = 100
        self.recoProgressBar.setValue(real_len)
        write_wav('out.wav', Main.FS, self.recoRecordData)

    def reco_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        self.status(fname)
        fs, signal = wavfile.read(fname)
        self.reco_remove_update(fs, signal)

    ########## ENROLL
    def enroll_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        self.status(fname)
        fs, signal = wavfile.read(fname)
        self.enrollWav = (fs, signal)

    def stop_enroll_record(self):
        self.stop_record()
        signal = np.array(self.recordData, dtype='uint8')
        self.enrollWav = (Main.FS, signal)


    def do_enroll(self):
        name = self.enrollName.text().trimmed()
        if not name:
            self.warn("Please Input Your Name")
            return

        # TODO self.enrollWav

    ####### UI related
    def upload_avatar(self):
        fd = QtGui.QFileDialog(self)
        self.avatarname = fd.getOpenFileName()
        from os.path import isfile
        if isfile(self.avatarname):
            newimage = QtGui.QPixmap(self.avatarname)
            self.Userimage.setPixmap(newimage)
    
    def getWidget(self, splash):
        t = QtCore.QElapsedTimer()
        t.start()
        while (t.elapsed() < 1000):
            str = QtCore.QString("times = ") + QtCore.QString.number(t.elapsed())
            splash.showMessage(str)
            QtCore.QCoreApplication.processEvents()

    def loadUsers(self):
        with open("userlist.txt") as db:
            for line in db:
                tmp = line.split()
                self.userdata.append(tmp)
                self.Userchooser.addItem(tmp[0])
        db.close()

    def showUserInfo(self):
        for user in userdata:
            if user[0] == self.Userchoose.itemData(self.Userchooser.currentIndex()):
                self.enrollName.setText(user[0])
                self.Userage.setValue(user[1])
                if user[2] == 'F':
                    self.Usersex.setcurrentIndex(1)
                else self.Usersex.setcurrentIndex(0)
                if isfile(user[3]):
                    newimage = QtGui.QPixmap(suer[3])
                    self.Userimage.setPixmap(newimage)

    def addUserInfo(self):
        for user in userdata:
            
    ############# UTIL
    def warn(self, s):
        QMessageBox.warning(self, "Warning", s)

    def status(self, s=""):
        self.statusBar().showMessage(s)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pixmap = QtGui.QPixmap(u"image/startup.jpg")
    splash = QtGui.QSplashScreen(pixmap)
    splash.show()
    QtCore.QCoreApplication.processEvents()

    mapp = Main()
    splash.finish(mapp.getWidget(splash))
    mapp.show()
    sys.exit(app.exec_())
