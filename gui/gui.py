#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gui.py
# Date: Thu Dec 26 15:39:04 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
import numpy as np
from PyQt4 import uic
from scipy.io import wavfile
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import pyaudio
CHUNK = 1
from VAD import remove_silence
from utils import write_wav


class RecorderThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        while True:
            data = self.main.stream.read(CHUNK)
            self.main.recordData.extend([ord(x) for x in data])

class Main(QMainWindow):
    FS = 8000

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        uic.loadUi("edytor.ui", self)
        self.statusBar()
        self.enrollRecord.clicked.connect(self.start_record)
        self.stopEnrollRecord.clicked.connect(self.stop_enroll_record)
        self.enrollFile.clicked.connect(self.enroll_file)
        self.enroll.clicked.connect(self.do_enroll)

        self.recoRecord.clicked.connect(self.start_record)
        self.stopRecoRecord.clicked.connect(self.stop_reco_record)


    ############ RECORD
    def start_record(self):
        self.pyaudio = pyaudio.PyAudio()
        self.status("Recording...")

        self.recordData = []
        self.stream = self.pyaudio.open(format=pyaudio.paUInt8, channels=1, rate=Main.FS,
                        input=True, frames_per_buffer=CHUNK)
        self.reco_th = RecorderThread(self)
        self.reco_th.start()

    def stop_record(self):
        self.reco_th.terminate()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.status("Record stopeed")


    ###### RECOGNIZE
    def stop_reco_record(self):
        self.stop_record()
        signal = np.array(self.recordData, dtype='uint8')
        fs, new_signal = remove_silence(Main.FS, signal)
        print "After removed: {0} -> {1}".format(len(signal), len(new_signal))
        write_wav('out.wav', Main.FS, signal)


    ########## ENROLL
    def enroll_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "", "Files (*.wav)")
        self.filename.setText(fname)
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


    ############# UTIL
    def warn(self, s):
        QMessageBox.warning(self, "Warning", s)

    def status(self, s=""):
        self.statusBar().showMessage(s)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mapp = Main()
    mapp.show()
    sys.exit(app.exec_())
