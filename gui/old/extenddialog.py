# -*- coding: utf-8 -*-   
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
import sys  
  
QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))  
  
class Extension(QDialog):  
    def __init__(self,parent=None):  
        super(Extension,self).__init__(parent)  
        self.setWindowTitle(self.tr("可扩展对话框"))  
  
        nameLabel=QLabel(self.tr("姓名:"))  
        nameLineEdit=QLineEdit()  
        sexLabel=QLabel(self.tr("性别:"))  
        sexComboBox=QComboBox()  
        sexComboBox.addItem(self.tr("男"))  
        sexComboBox.addItem(self.tr("女"))  
  
        okButton=QPushButton(self.tr("确定"))  
        detailButton=QPushButton(self.tr("详细"))  
  
        self.connect(detailButton,SIGNAL("clicked()"),self.slotExtension)  
  
        btnBox=QDialogButtonBox(Qt.Vertical)  
        btnBox.addButton(okButton,QDialogButtonBox.ActionRole)  
        btnBox.addButton(detailButton,QDialogButtonBox.ActionRole)  
  
        baseLayout=QGridLayout()  
        baseLayout.addWidget(nameLabel,0,0)  
        baseLayout.addWidget(nameLineEdit,0,1)  
        baseLayout.addWidget(okButton,0,2)  
        baseLayout.addWidget(sexLabel,1,0)  
        baseLayout.addWidget(sexComboBox,1,1)  
        baseLayout.addWidget(detailButton,1,2)  
  
        ageLabel=QLabel(self.tr("年龄:"))  
        ageLineEdit=QLineEdit("30")  
        departmentLabel=QLabel(self.tr("部门:"))  
        departmentComboBox=QComboBox()  
        departmentComboBox.addItem(self.tr("部门一"))  
        departmentComboBox.addItem(self.tr("部门二"))  
        departmentComboBox.addItem(self.tr("部门三"))  
        emailLabel=QLabel("email:")  
        emailLineEdit=QLineEdit()  
  
        self.detailWidget=QWidget()  
        detailLayout=QGridLayout(self.detailWidget)  
        detailLayout.addWidget(ageLabel,0,0)  
        detailLayout.addWidget(ageLineEdit,0,1)  
        detailLayout.addWidget(departmentLabel,1,0)  
        detailLayout.addWidget(departmentComboBox,1,1)  
        detailLayout.addWidget(emailLabel,2,0)  
        detailLayout.addWidget(emailLineEdit,2,1)  
        self.detailWidget.hide()  
          
        mainLayout=QVBoxLayout()  
        mainLayout.addLayout(baseLayout)  
        mainLayout.addWidget(self.detailWidget)  
        mainLayout.setSizeConstraint(QLayout.SetFixedSize)  
        mainLayout.setSpacing(10)  
  
        self.setLayout(mainLayout)  
          
    def slotExtension(self):  
        if self.detailWidget.isHidden():  
            self.detailWidget.show()  
        else:  
            self.detailWidget.hide()  
  
app=QApplication(sys.argv)  
main=Extension()  
main.show()  
app.exec_()  