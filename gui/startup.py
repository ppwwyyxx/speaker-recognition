from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
import sys  
  
class MainWindow(QMainWindow):  
    def __init__(self,parent=None):  
        super(MainWindow,self).__init__(parent)  
        self.setWindowTitle("Splash Example")   
        edit=QTextEdit()  
        edit.setText("Splash Example")  
        self.setCentralWidget(edit)  
  
        self.resize(600,450)  
  
        QThread.sleep(3)  
          
app=QApplication(sys.argv)  
  
splash=QSplashScreen(QPixmap("image/startup.jpg"))  
splash.show()  
app.processEvents()  
window=MainWindow()  
window.show()  
splash.finish(window)  
  
app.exec_()  