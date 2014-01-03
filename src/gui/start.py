import sys
from PyQt4 import QtCore, QtGui
from edytor import Ui_notepad
class StartQt4(QtGui.QMainWindow):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self, parent)
		self.ui = Ui_notepad()
		self.ui.setupUi(self)
		self.createProgressBar()
		self.active = 0;
		QtCore.QObject.connect(self.ui.Choose_file,QtCore.SIGNAL("clicked()"),self.file_dialog)
		QtCore.QObject.connect(self.ui.UploadImage,QtCore.SIGNAL("clicked()"),self.upload_avatar)
		QtCore.QObject.connect(self.ui.Startrecord,QtCore.SIGNAL("clicked()"),self.toggle_record)
		#self.ui.horizontalLayout.addWidget(self.progressBar)
		
		#movie test
		self.movie = QtGui.QMovie(u"image/recording.gif")
		self.ui.Animation.setMovie(self.movie)
		self.movie.start()
		self.movie.stop()

		self.movie2 = QtGui.QMovie(u"image/recording.gif")
		self.ui.Animation_2.setMovie(self.movie)
		self.movie2.start()
		self.movie2.stop()

		#default user image setting
		defaultimage = QtGui.QPixmap(u"image/nouser.jpg")
		self.ui.Userimage.setPixmap(defaultimage)
		self.ui.Reco_Userimage.setPixmap(defaultimage)


	def file_dialog(self):
		fd = QtGui.QFileDialog(self)
		self.filename = fd.getOpenFileName()
		from os.path import isfile
		if isfile(self.filename):
			self.ui.Filename.setText(self.filename)
			text = open(self.filename).read()
			#here text is what we got from user
			#self.ui.editor_window.setText(text)
	def upload_avatar(self):
		fd = QtGui.QFileDialog(self)
		self.avatarname = fd.getOpenFileName()
		from os.path import isfile
		if isfile(self.avatarname):
			newimage = QtGui.QPixmap(self.avatarname)
			self.ui.Userimage.setPixmap(newimage)
	def toggle_record(self):
		if (not self.active):
			self.ui.Startrecord.setText("Stop")
			self.movie.start()
		else:
			self.ui.Startrecord.setText("Record")
			self.movie.stop()
		self.active = not self.active

	def getWidget(self, splash):
		t = QtCore.QElapsedTimer()
		t.start()
		while (t.elapsed() < 1000):
			str = QtCore.QString("times = ") + QtCore.QString.number(t.elapsed())
			splash.showMessage(str)
			QtCore.QCoreApplication.processEvents()
	def advanceProgressBar(self):
		curVal = self.progressBar.value()
		maxVal = self.progressBar.maximum()
		self.progressBar.setValue(curVal + (maxVal - curVal) / 100)
	def createProgressBar(self):
		self.progressBar = QtGui.QProgressBar()
		self.progressBar.setRange(0, 10000)
		self.progressBar.setValue(0)

		timer = QtCore.QTimer(self)
		timer.timeout.connect(self.advanceProgressBar)
		timer.start(1000)
if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)

	pixmap = QtGui.QPixmap(u"image/startup.jpg")
	splash = QtGui.QSplashScreen(pixmap)
	splash.show()
	QtCore.QCoreApplication.processEvents()

	myapp = StartQt4()
	splash.finish(myapp.getWidget(splash))
	myapp.show()
	sys.exit(app.exec_())