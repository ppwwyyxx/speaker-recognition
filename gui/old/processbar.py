# -*- coding: utf-8 -*-   
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
import sys  
  
QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))  
   
class Progess(QDialog):  
    def __init__(self,parent=None):  
        super(Progess,self).__init__(parent)  
        self.setWindowTitle(self.tr("使用进度条"))  
        numLabel=QLabel(self.tr("文件数目"))  
        self.numLineEdit=QLineEdit("10")  
        typeLabel=QLabel(self.tr("显示类型"))  
        self.typeComboBox=QComboBox()  
        self.typeComboBox.addItem(self.tr("进度条"))  
        self.typeComboBox.addItem(self.tr("进度对话框"))  
  
        self.progressBar=QProgressBar()  
  
        startPushButton=QPushButton(self.tr("开始"))  
  
        layout=QGridLayout()  
        layout.addWidget(numLabel,0,0)  
        layout.addWidget(self.numLineEdit,0,1)  
        layout.addWidget(typeLabel,1,0)  
        layout.addWidget(self.typeComboBox,1,1)  
        layout.addWidget(self.progressBar,2,0,1,2)  
        layout.addWidget(startPushButton,3,1)  
        layout.setMargin(15)  
        layout.setSpacing(10)  
  
        self.setLayout(layout)  
          
        self.connect(startPushButton,SIGNAL("clicked()"),self.slotStart)  
  
    def slotStart(self):  
        num=int(self.numLineEdit.text())  
  
        if self.typeComboBox.currentIndex()==0:  
            self.progressBar.setMinimum(0)  
            self.progressBar.setMaximum(num)  
  
            for i in range(num):  
                self.progressBar.setValue(i)  
                QThread.msleep(100)  
  
        elif self.typeComboBox.currentIndex()==1:  
            progressDialog=QProgressDialog(self)  
            progressDialog.setWindowModality(Qt.WindowModal)  
            progressDialog.setMinimumDuration(5)  
            progressDialog.setWindowTitle(self.tr("请等待"))  
            progressDialog.setLabelText(self.tr("拷贝..."))  
            progressDialog.setCancelButtonText(self.tr("取消"))  
            progressDialog.setRange(0,num)  
  
            for i in range(num):  
                progressDialog.setValue(i)  
                QThread.msleep(100)  
                if progressDialog.wasCanceled():  
                    return  
                  
app=QApplication(sys.argv)  
progess=Progess()  
progess.show()  
app.exec_()  