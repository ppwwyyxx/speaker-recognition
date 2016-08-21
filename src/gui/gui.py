#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gui.py
# Date: Sat Feb 21 18:44:12 2015 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
import shutil
import os.path
import glob
import traceback
import time
import numpy as np
from PyQt4 import uic
from scipy.io import wavfile
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtCore,QtGui

import pyaudio
from utils import read_wav, write_wav, time_str, monophonic
from interface import ModelInterface

FORMAT=pyaudio.paInt16
NPDtype = 'int16'
NAMELIST = ['Nobody']

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
    CONV_INTERVAL = 0.4
    CONV_DURATION = 1.5
    CONV_FILTER_DURATION = CONV_DURATION
    FS = 8000
    TEST_DURATION = 3

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        uic.loadUi("edytor.ui", self)
        self.statusBar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_callback)

        self.noiseButton.clicked.connect(self.noise_clicked)
        self.recording_noise = False
        self.loadNoise.clicked.connect(self.load_noise)

        self.enrollRecord.clicked.connect(self.start_enroll_record)
        self.stopEnrollRecord.clicked.connect(self.stop_enroll_record)
        self.enrollFile.clicked.connect(self.enroll_file)
        self.enroll.clicked.connect(self.do_enroll)
        self.startTrain.clicked.connect(self.start_train)
        self.dumpBtn.clicked.connect(self.dump)
        self.loadBtn.clicked.connect(self.load)

        self.recoRecord.clicked.connect(self.start_reco_record)
        self.stopRecoRecord.clicked.connect(self.stop_reco_record)
#        self.newReco.clicked.connect(self.new_reco)
        self.recoFile.clicked.connect(self.reco_file)
        self.recoInputFiles.clicked.connect(self.reco_files)

        #UI.init
        self.userdata =[]
        self.loadUsers()
        self.Userchooser.currentIndexChanged.connect(self.showUserInfo)
        self.ClearInfo.clicked.connect(self.clearUserInfo)
        self.UpdateInfo.clicked.connect(self.updateUserInfo)
        self.UploadImage.clicked.connect(self.upload_avatar)
        #movie test
        self.movie = QMovie(u"image/recording.gif")
        self.movie.start()
        self.movie.stop()
        self.Animation.setMovie(self.movie)
        self.Animation_2.setMovie(self.movie)
        self.Animation_3.setMovie(self.movie)

        self.aladingpic = QPixmap(u"image/a_hello.png")
        self.Alading.setPixmap(self.aladingpic)
        self.Alading_conv.setPixmap(self.aladingpic)

        #default user image setting
        self.avatarname = "image/nouser.jpg"
        self.defaultimage = QPixmap(self.avatarname)
        self.Userimage.setPixmap(self.defaultimage)
        self.recoUserImage.setPixmap(self.defaultimage)
        self.convUserImage.setPixmap(self.defaultimage)
        self.load_avatar('avatar/')

        # Graph Window init
        self.graphwindow = GraphWindow()
        self.newname = ""
        self.lastname = ""
        self.Graph_button.clicked.connect(self.graphwindow.show)
        self.convRecord.clicked.connect(self.start_conv_record)
        self.convStop.clicked.connect(self.stop_conv)

        self.backend = ModelInterface()

        # debug
        QShortcut(QKeySequence("Ctrl+P"), self, self.printDebug)

        #init
        try:
            fs, signal = read_wav("bg.wav")
            self.backend.init_noise(fs, signal)
        except:
            pass


    ############ RECORD
    def start_record(self):
        self.pyaudio = pyaudio.PyAudio()
        self.status("Recording...")
        self.movie.start()
        self.Alading.setPixmap(QPixmap(u"image/a_thinking.png"))


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
        self.conv_result_list = []
        self.start_record()
        self.conv_now_pos = 0
        self.conv_timer = QTimer(self)
        self.conv_timer.timeout.connect(self.do_conversation)
        self.conv_timer.start(Main.CONV_INTERVAL * 1000)
        #reset
        self.graphwindow.wid.reset()

    def stop_conv(self):
        self.stop_record()
        self.conv_timer.stop()

    def do_conversation(self):
        interval_len = int(Main.CONV_INTERVAL * Main.FS)
        segment_len = int(Main.CONV_DURATION * Main.FS)
        self.conv_now_pos += interval_len
        to_filter = self.recordData[max([self.conv_now_pos - segment_len, 0]):
                                   self.conv_now_pos]
        signal = np.array(to_filter, dtype=NPDtype)
        label = None
        try:
            signal = self.backend.filter(Main.FS, signal)
            if len(signal) > 50:
                label = self.backend.predict(Main.FS, signal, True)
        except Exception as e:
            print traceback.format_exc()
            print str(e)

        global last_label_to_show
        label_to_show = label
        if label and self.conv_result_list:
            last_label = self.conv_result_list[-1]
            if last_label and last_label != label:
                label_to_show = last_label_to_show
        self.conv_result_list.append(label)

        print label_to_show, "label to show"
        last_label_to_show = label_to_show

        #ADD FOR GRAPH
        if label_to_show is None:
            label_to_show = 'Nobody'
        if len(NAMELIST) and NAMELIST[-1] != label_to_show:
            NAMELIST.append(label_to_show)
        self.convUsername.setText(label_to_show)
        self.Alading_conv.setPixmap(QPixmap(u"image/a_result.png"))
        self.convUserImage.setPixmap(self.get_avatar(label_to_show))


    ###### RECOGNIZE
    def start_reco_record(self):
        self.Alading.setPixmap(QPixmap(u"image/a_hello"))
        self.recoRecordData = np.array((), dtype=NPDtype)
        self.start_record()

    def stop_reco_record(self):
        self.stop_record()
        signal = np.array(self.recordData, dtype=NPDtype)
        self.reco_remove_update(Main.FS, signal)

    def reco_do_predict(self, fs, signal):
        label = self.backend.predict(fs, signal)
        if not label:
            label = "Nobody"
        print label
        self.recoUsername.setText(label)
        self.Alading.setPixmap(QPixmap(u"image/a_result.png"))
        self.recoUserImage.setPixmap(self.get_avatar(label))

        # TODO To Delete
        write_wav('reco.wav', fs, signal)

    def reco_remove_update(self, fs, signal):
        new_signal = self.backend.filter(fs, signal)
        print "After removed: {0} -> {1}".format(len(signal), len(new_signal))
        self.recoRecordData = np.concatenate((self.recoRecordData, new_signal))
        real_len = float(len(self.recoRecordData)) / Main.FS / Main.TEST_DURATION * 100
        if real_len > 100:
            real_len = 100

        self.reco_do_predict(fs, self.recoRecordData)


    def reco_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        print 'reco_file'
        if not fname:
            return
        self.status(fname)

        fs, signal = read_wav(fname)
        self.reco_do_predict(fs, signal)

    def reco_files(self):
        fnames = QFileDialog.getOpenFileNames(self, "Select Wav Files", "", "Files (*.wav)")
        print 'reco_files'
        for f in fnames:
            fs, sig = read_wav(f)
            newsig = self.backend.filter(fs, sig)
            label = self.backend.predict(fs, newsig)
            print f, label

    ########## ENROLL
    def start_enroll_record(self):
        self.enrollWav = None
        self.enrollFileName.setText("")
        self.start_record()

    def enroll_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open Wav File", "", "Files (*.wav)")
        if not fname:
            return
        self.status(fname)
        self.enrollFileName.setText(fname)
        fs, signal = read_wav(fname)
        signal = monophonic(signal)
        self.enrollWav = (fs, signal)

    def stop_enroll_record(self):
        self.stop_record()
        print self.recordData[:300]
        signal = np.array(self.recordData, dtype=NPDtype)
        self.enrollWav = (Main.FS, signal)

        # TODO To Delete
        write_wav('enroll.wav', *self.enrollWav)

    def do_enroll(self):
        name = self.Username.text().trimmed()
        if not name:
            self.warn("Please Input Your Name")
            return
#        self.addUserInfo()
        new_signal = self.backend.filter(*self.enrollWav)
        print "After removed: {0} -> {1}".format(len(self.enrollWav[1]), len(new_signal))
        print "Enroll: {:.4f} seconds".format(float(len(new_signal)) / Main.FS)
        if len(new_signal) == 0:
            print "Error! Input is silent! Please enroll again"
            return
        self.backend.enroll(name, Main.FS, new_signal)

    def start_train(self):
        self.status("Training...")
        self.backend.train()
        self.status("Training Done.")

    ####### UI related
    def getWidget(self, splash):
        t = QtCore.QElapsedTimer()
        t.start()
        while (t.elapsed() < 800):
            str = QtCore.QString("times = ") + QtCore.QString.number(t.elapsed())
            splash.showMessage(str)
            QtCore.QCoreApplication.processEvents()

    def upload_avatar(self):
        fname = QFileDialog.getOpenFileName(self, "Open JPG File", "", "File (*.jpg)")
        if not fname:
            return
        self.avatarname = fname
        self.Userimage.setPixmap(QPixmap(fname))

    def loadUsers(self):
        with open("avatar/metainfo.txt") as db:
            for line in db:
                tmp = line.split()
                self.userdata.append(tmp)
                self.Userchooser.addItem(tmp[0])

    def showUserInfo(self):
        for user in self.userdata:
            if self.userdata.index(user) == self.Userchooser.currentIndex() - 1:
                self.Username.setText(user[0])
                self.Userage.setValue(int(user[1]))
                if user[2] == 'F':
                    self.Usersex.setCurrentIndex(1)
                else:
                    self.Usersex.setCurrentIndex(0)
                self.Userimage.setPixmap(self.get_avatar(user[0]))

    def updateUserInfo(self):
        userindex = self.Userchooser.currentIndex() - 1
        u = self.userdata[userindex]
        u[0] = unicode(self.Username.displayText())
        u[1] = self.Userage.value()
        if self.Usersex.currentIndex():
            u[2] = 'F'
        else:
            u[2] = 'M'
        with open("avatar/metainfo.txt","w") as db:
            for user in self.userdata:
                for i in range(3):
                    db.write(str(user[i]) + " ")
                db.write("\n")

    def writeuserdata(self):
        with open("avatar/metainfo.txt","w") as db:
            for user in self.userdata:
                for i in range (0,4):
                    db.write(str(user[i]) + " ")
                db.write("\n")

    def clearUserInfo(self):
        self.Username.setText("")
        self.Userage.setValue(0)
        self.Usersex.setCurrentIndex(0)
        self.Userimage.setPixmap(self.defaultimage)

    def addUserInfo(self):
        for user in self.userdata:
            if user[0] == unicode(self.Username.displayText()):
                return
        newuser = []
        newuser.append(unicode(self.Username.displayText()))
        newuser.append(self.Userage.value())
        if self.Usersex.currentIndex():
            newuser.append('F')
        else:
            newuser.append('M')
        if self.avatarname:
            shutil.copy(self.avatarname, 'avatar/' + user[0] + '.jpg')
        self.userdata.append(newuser)
        self.writeuserdata()
        self.Userchooser.addItem(unicode(self.Username.displayText()))


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
        if fname:
            try:
                self.backend.dump(fname)
            except Exception as e:
                self.warn(str(e))
            else:
                self.status("Dumped to file: " + fname)

    def load(self):
        fname = QFileDialog.getOpenFileName(self, "Open Data File:", "", "")
        if fname:
            try:
                self.backend = ModelInterface.load(fname)
            except Exception as e:
                self.warn(str(e))
            else:
                self.status("Loaded from file: " + fname)

    def noise_clicked(self):
        self.recording_noise = not self.recording_noise
        if self.recording_noise:
            self.noiseButton.setText('Stop Recording Noise')
            self.start_record()
        else:
            self.noiseButton.setText('Recording Background Noise')
            self.stop_record()
            signal = np.array(self.recordData, dtype=NPDtype)
            wavfile.write("bg.wav", Main.FS, signal)
            self.backend.init_noise(Main.FS, signal)

    def load_noise(self):
        fname = QFileDialog.getOpenFileName(self, "Open Data File:", "", "Wav File  (*.wav)")
        if fname:
            fs, signal = read_wav(fname)
            self.backend.init_noise(fs, signal)

    def load_avatar(self, dirname):
        self.avatars = {}
        for f in glob.glob(dirname + '/*.jpg'):
            name = os.path.basename(f).split('.')[0]
            print f, name
            self.avatars[name] = QPixmap(f)

    def get_avatar(self, username):
        p = self.avatars.get(str(username), None)
        if p:
            return p
        else:
            return self.defaultimage

    def printDebug(self):
        for name, feat in self.backend.features.iteritems():
            print name, len(feat)
        print "GMMs",
        print len(self.backend.gmmset.gmms)
    '''
    def TempButton(self):
        import random
        randomnamelist = ["ltz","wyx","zxy","Nobody"]
        while self.newname == self.lastname:
            self.newname = randomnamelist[int(random.randrange(0,len(randomnamelist)))]
        NAMELIST.append(self.newname)
        self.lastname = self.newname

    def LogButton(self):
        self.graphwindow.wid.reset()
    '''
class GraphWindow(QWidget):
    def __init__(self,parent = None):
        super(GraphWindow,self).__init__(parent)
        self.setGeometry(300, 100, 920, 510)
        self.setWindowTitle('Conversation Flow Graph')
        self.wid = BurningWidget()
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.wid)
        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)


class BurningWidget(QtGui.QWidget):

    def __init__(self):
        super(BurningWidget, self).__init__()

        self.initUI()
        self.load_avatar('avatar/')
        self.avatarname = "image/nouser.jpg"
        self.defaultimage = QPixmap(self.avatarname)
        self.updateflag = False
        self.updatedelta = 0

    def reset(self):
        self.num = []
        self.nameset = []
        self.nowtime = 0
        self.namelistlen = 0
        self.timer.stop()
        self.timer.start()
        self.updateflag = False
        global NAMELIST
        NAMELIST = ["Nobody"]

    def initUI(self):
        self.setMinimumSize(1, 510)
        self.num = []
        self.Unknowncolor = QColor(204,204,204)
        self.colorlist = [QColor(255,102,102), QColor(255,255,0), QColor(51,153,204), QColor(0,153,51), QColor(0, 255, 255)]
        '''
        with open("timeline.txt") as db:
            for line in db:
                tmp = line.split()
                self.namelist.append(tmp[0])
                self.num.append(int(tmp[1]))

        '''
        self.namelistlen = 0
        self.nameset = []
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.nowtime = 0
        self.timer.start()
        self.timer.timeout.connect(self.timer_out)


    def timer_out(self):
        self.nowtime += 0.1
        if len(NAMELIST) != self.namelistlen:
            self.namelistlen += 1
            self.num.append(self.nowtime)
            if NAMELIST[self.namelistlen - 1] not in self.nameset:
                self.nameset.append(NAMELIST[self.namelistlen - 1])
            self.updateflag = True
            self.updatedelta = 180
            self.updatebigdelta = 120
        if (self.updateflag):
            if (self.updatedelta > 120):
                self.updatedelta -= 15
                self.updatebigdelta += 15
            else:
                self.updateflag = False
        self.timer.start()
        self.repaint()

    def time2string(self,time):
        totalsec = int(time)
        minute = totalsec / 60
        second = totalsec % 60
        if second < 10:
            return str(minute) + ":0" + str(second)
        else:
            return str(minute) + ":" + str(second)

    def paintEvent(self, e):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def load_avatar(self, dirname):
        self.avatars = {}
        for f in glob.glob(dirname + '/*.jpg'):
            name = os.path.basename(f).split('.')[0]
            self.avatars[name] = QPixmap(f)

    def get_avatar(self, username):
        p = self.avatars.get(str(username), None)
        if p:
            return p
        else:
            return self.defaultimage

    def drawWidget(self, qp):

        size = self.size()
        w = size.width()
        h = size.height() / 2 + 100
        barheight = 50

        total = 20
        zoomer = w / total
        laststart = 900 - self.nowtime * 45

        picsize = 180
        margin = -20
        startx = int((900 - len(self.nameset) * 139 - 10) / 2) + 50

        font = QtGui.QFont('Arial', 30, QtGui.QFont.Bold)
        qp.setFont(font)
        qp.drawText(10,40,NAMELIST[-1])

        font = QtGui.QFont('Arial', 20, QtGui.QFont.Bold)
        qp.setFont(font)
        dot = int(self.nowtime) % 4
        qp.drawText(10,80,"Speaking" + "." * dot)
        qp.drawText(10,h - 80,"Timeline")

        for i in range(0,len(self.nameset)):
            if self.nameset[i] != "Nobody":
                if NAMELIST[-1] == self.nameset[i]:
                    qp.drawImage(int (startx + i *(margin + picsize)) + (picsize - self.updatebigdelta) / 2, (picsize - self.updatebigdelta) / 2, QImage(self.get_avatar(self.nameset[i])).scaled(self.updatebigdelta,self.updatebigdelta))
                elif NAMELIST[-2] == self.nameset[i]:
                    qp.drawImage(int (startx + i *(margin + picsize)) + (picsize - self.updatedelta) / 2, (picsize - self.updatedelta) / 2, QImage(self.get_avatar(self.nameset[i])).scaled(self.updatedelta,self.updatedelta))
                else:
                    qp.drawImage(int (startx + i *(margin + picsize) + (picsize - 120) / 2), (picsize - 120) / 2, QImage(self.get_avatar(self.nameset[i])).scaled(120,120))

        font = QtGui.QFont('Arial', 12, QtGui.QFont.Light)
        qp.setFont(font)

        for i in range(0,len(NAMELIST)):
            outside = QPen(QColor(255,255,255))
            inside = QPen(QColor(0,0,0))
            outside.setWidth(4)
            qp.setPen(outside)
            qp.setBrush(self.colorlist[self.nameset.index(NAMELIST[i])])
            if i == len(NAMELIST) - 1:
                if (NAMELIST[i] == "Nobody"):
                    qp.setBrush(self.Unknowncolor)
                qp.drawRoundRect(laststart, h - barheight, 9000, barheight,10,10)
                if (NAMELIST[i] != "Nobody"):
                    qp.drawImage(int(laststart), int(h + 35),QImage(self.get_avatar(NAMELIST[i])).scaled(70,70))
                    qp.setPen(inside)
                    qp.drawText(int(laststart), int (h + 130),NAMELIST[i])
                    qp.drawText(int(laststart), int (h + 150),self.time2string(self.num[i]) + "~" + self.time2string(self.nowtime))

                #qp.drawImage(380,0,QImage(self.get_avatar(NAMELIST[i])))
            else:
                if (NAMELIST[i] == "Nobody"):
                    qp.setBrush(self.Unknowncolor)
                qp.drawRoundRect(laststart, h - barheight, round((self.num[i + 1] - self.num[i]) * zoomer), barheight,30,30)
                if (NAMELIST[i] != "Nobody"):
                    qp.drawImage(int(laststart), int(h + 35),QImage(self.get_avatar(NAMELIST[i])).scaled(70,70))
                    qp.drawEllipse(int(laststart) + 60, h,20,15)
                    qp.drawEllipse(int(laststart) + 30, h + 15,15,10)
                    qp.setPen(inside)
                    qp.drawText(int(laststart), int (h + 130),NAMELIST[i])
                    qp.drawText(int(laststart), int (h + 150),self.time2string(self.num[i]) + "~" + self.time2string(self.num[i + 1]))
                laststart = 900 - (self.nowtime - self.num[i + 1]) * zoomer
                #laststart + round((self.num[i + 1] - self.num[i]) * zoomer)

        pen = QtGui.QPen(QtGui.QColor(20, 20, 20), 1,
            QtCore.Qt.SolidLine)

        qp.setPen(pen)
        qp.setBrush(QtCore.Qt.NoBrush)

        '''
        for i in range(0,len(NAMELIST)):
            qp.drawLine(i, 0, i, 5)
            metrics = qp.fontMetrics()
            fw = metrics.width(str(self.num[i]))
            if not i:
                qp.drawText(0, 150, str(NAMELIST[i]))
                qp.drawText(0, 150 - 15, str(self.num[i]))
            else :
                qp.drawText((int(self.num[i - 1]) * zoomer)-fw/2, 150, str(NAMELIST[i]))
                qp.drawText((int(self.num[i - 1]) * zoomer)-fw/2, 150 - 15, str(self.num[i]))
        '''




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
