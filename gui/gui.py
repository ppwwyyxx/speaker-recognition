#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gui.py
# Date: Thu Dec 26 20:46:51 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
import time
import numpy as np
import wave
from PyQt4 import uic
from scipy.io import wavfile
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtCore,QtGui

import pyaudio
from VAD import remove_silence
from utils import write_wav, time_str
from interface import ModelInterface

FORMAT=pyaudio.paInt16
NPDtype = 'int16'

class RecorderThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.start_time = time.time()
        while True:
            data = self.main.stream.read(1)
            i = ord(data[0]) + 256 * ord(data[1])
            if i > 32768:
                i -= 65536
            stop = self.main.add_record_data(i)
            if stop:
                break

class Main(QMainWindow):
    CONV_INTERVAL = 500
    FS = 8000
    TEST_DURATION = 3

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        uic.loadUi("edytor2.ui", self)
        self.statusBar()
        self.recoProgressBar.setValue(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_callback)

        self.enrollRecord.clicked.connect(self.start_record)
        self.stopEnrollRecord.clicked.connect(self.stop_enroll_record)
        self.enrollFile.clicked.connect(self.enroll_file)
        self.enroll.clicked.connect(self.do_enroll)
        self.startTrain.clicked.connect(self.start_train)
        self.dumpBtn.clicked.connect(self.dump)
        self.loadBtn.clicked.connect(self.load)

        self.recoRecord.clicked.connect(self.start_record)
        self.stopRecoRecord.clicked.connect(self.stop_reco_record)
        self.newReco.clicked.connect(self.new_reco)
        self.recoFile.clicked.connect(self.reco_file)
        self.new_reco()

<<<<<<< HEAD
        #UI.init

        self.UploadImage.clicked.connect(self.upload_avatar)
        self.userdata =[]
        self.loadUsers()
        self.Userchooser.currentIndexChanged.connect(self.showUserInfo)
        self.ClearInfo.clicked.connect(self.clearUserInfo)
        self.UpdateInfo.clicked.connect(self.updateUserInfo)
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
        self.avatarname = "image/nouser.jpg"
        defaultimage = QPixmap(self.avatarname)
        self.Userimage.setPixmap(defaultimage)
        self.Reco_Userimage.setPixmap(defaultimage)

        self.convRecord.clicked.connect(self.start_conv_record)
        self.convStop.clicked.connect(self.stop_conv)

        self.backend = ModelInterface()


    ############ RECORD
    def start_record(self):
        self.pyaudio = pyaudio.PyAudio()
        self.status("Recording...")
        self.movie.start()

        self.recordData = []
        self.stream = self.pyaudio.open(format=FORMAT, channels=1, rate=Main.FS,
                        input=True, frames_per_buffer=1)
        self.stopped = False
        self.reco_th = RecorderThread(self)
        self.reco_th.start()

        self.timer.start(1000)
        self.record_time = 0
        self.update_all_timer()

    def add_record_data(self, i):
        self.recordData.append(i)
        return self.stopped

    def timer_callback(self):
        self.record_time += 1
        self.status("Recording..." + time_str(self.record_time))
        self.update_all_timer()

    def stop_record(self):
        self.movie.stop()
        self.stopped = True
        self.reco_th.wait()
        self.timer.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.status("Record stopeed")

    ############## conversation
    def start_conv_record(self):
        self.start_record()
        self.conv_now_pos = 0
        self.conv_timer = QTimer(self)
        self.conv_timer.timeout.connect(self.do_conversation)
        self.conv_timer.start(Main.CONV_INTERVAL)

    def stop_conv(self):
        self.stop_record()
        self.conv_timer.stop()

    def do_conversation(self):
        segment_shift = int(Main.CONV_INTERVAL * Main.FS / 1000)
        segment = self.recordData[self.conv_now_pos:
                                  self.conv_now_pos + segment_shift]
        self.conv_now_pos += segment_shift
        signal = np.array(segment, dtype=NPDtype)
        predict = self.backend.predict(Main.FS, signal)
        self.convUsername.setText(predict)

    ###### RECOGNIZE
    def new_reco(self):
        self.recoRecordData = np.array((), dtype=NPDtype)
        self.recoProgressBar.setValue(0)

    def stop_reco_record(self):
        self.stop_record()
        signal = np.array(self.recordData, dtype=NPDtype)
        self.reco_remove_update(Main.FS, signal)

    def reco_remove_update(self, fs, signal):
        fs, new_signal = remove_silence(fs, signal)
        print "After removed: {0} -> {1}".format(len(signal), len(new_signal))
        self.recoRecordData = np.concatenate((self.recoRecordData, new_signal))
        real_len = float(len(self.recoRecordData)) / Main.FS / Main.TEST_DURATION * 100
        if real_len > 100:
            real_len = 100
        self.recoProgressBar.setValue(real_len)
        predict_name = self.backend.predict(Main.FS, self.recoRecordData)
        self.recoUsername.setText(predict_name)
        print predict_name
        # TODO To Delete
        write_wav('out.wav', Main.FS, self.recoRecordData)

    def reco_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        self.status(fname)
        fs, signal = wavfile.read(fname)
        self.reco_remove_update(fs, signal)

    ########## ENROLL
    def enroll_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        self.enrollFileName.setText(fname)
        self.status(fname)
        fs, signal = wavfile.read(fname)
        self.enrollWav = (fs, signal)

    def stop_enroll_record(self):
        self.stop_record()
        print self.recordData[:300]
        signal = np.array(self.recordData, dtype=NPDtype)
        self.enrollWav = (Main.FS, signal)

        write_wav('out.wav', *self.enrollWav)

    def do_enroll(self):
        name = self.Username.text().trimmed()
        if not name:
            self.warn("Please Input Your Name")
            return
        fs, new_signal = remove_silence(*self.enrollWav)
        print "After removed: {0} -> {1}".format(len(self.enrollWav[1]), len(new_signal))
        print "Enroll: {:.4f} seconds".format(float(len(new_signal)) / Main.FS)
        self.backend.enroll(name, fs, new_signal)

    def start_train(self):
        self.backend.train()

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
        while (t.elapsed() < 800):
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
        for user in self.userdata:
            if self.userdata.index(user) == self.Userchooser.currentIndex() - 1:
                self.Username.setText(user[0])
                self.Userage.setValue(int(user[1]))
                if user[2] == 'F':
                    self.Usersex.setCurrentIndex(1)
                else:
                    self.Usersex.setCurrentIndex(0)
                from os.path import isfile
                if isfile(user[3]):
                    self.avatarname = user[3]
                    newimage = QtGui.QPixmap(user[3])
                    self.Userimage.setPixmap(newimage)

    def updateUserInfo(self):
        userindex = self.Userchooser.currentIndex() - 1
        self.userdata[userindex][0] = unicode(self.Username.displayText())
        self.userdata[userindex][1] = self.Userage.value()
        if self.Usersex.currentIndex():
            self.userdata[userindex][2] = 'F'
        else:
            self.userdata[userindex][2] = 'M'  
        if self.avatarname:
            self.userdata[userindex][3] = self.avatarname
        print self.userdata
        db = open("userlist.txt","w")
        for user in self.userdata:
            for i in range (0,4):
                db.write(str(user[i]) + " ")
            db.write("\n")
        db.close()

    def clearUserInfo(self):
        self.Username.setText("")
        self.Userage.setValue(0)
        self.Usersex.setCurrentIndex(0)
        defaultimage = QPixmap(u"image/nouser.jpg")
        self.Userimage.setPixmap(defaultimage)

    #def addUserInfo(self):
    #    for user in self.userdata:
    ############# UTILS
    def warn(self, s):
        QMessageBox.warning(self, "Warning", s)

    def status(self, s=""):
        self.statusBar().showMessage(s)

    def update_all_timer(self):
        s = time_str(self.record_time)
        self.enrollTime.setText(s)
        self.recoTime.setText(s)
        self.convTime.setText(s)

    def dump(self):
        fname = QFileDialog.getSaveFileName(self, "Save Data to:", "", "")
        try:
            self.backend.dump(fname)
        except Exception as e:
            self.warn(str(e))
        else:
            self.status("Dumped to file: " + fname)

    def load(self):
        fname = QFileDialog.getOpenFileName(self, "Open Data File:", "", "")
        try:
            self.backend = ModelInterface.load(fname)
        except Exception as e:
            self.warn(str(e))
        else:
            self.status("Loaded from file: " + fname)



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
