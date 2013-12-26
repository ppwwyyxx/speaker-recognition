# -*- coding: utf-8 -*-   
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
import sys  
  
QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))  
  
class StockDialog(QDialog):  
    def __init__(self,parent=None):  
        super(StockDialog,self).__init__(parent)  
        self.setWindowTitle(self.tr("综合布局实例"))  
  
        mainSplitter=QSplitter(Qt.Horizontal)  
        mainSplitter.setOpaqueResize(True)  
   
        self.listWidget=QListWidget(mainSplitter)  
        self.listWidget.insertItem(0,self.tr("个人基本资料"))  
        self.listWidget.insertItem(1,self.tr("联系方式"))  
        self.listWidget.insertItem(2,self.tr("详细信息"))  
  
        frame=QFrame(mainSplitter)  
        self.stack=QStackedWidget()  
        self.stack.setFrameStyle(QFrame.Panel|QFrame.Raised)  
          
        baseInfo=BaseInfo()  
        contact=Contact()  
        detail=Detail()  
        self.stack.addWidget(baseInfo)  
        self.stack.addWidget(contact)  
        self.stack.addWidget(detail)  
  
        amendPushButton=QPushButton(self.tr("修改"))  
        closePushButton=QPushButton(self.tr("关闭"))  
  
        buttonLayout=QHBoxLayout()  
        buttonLayout.addStretch(1)  
        buttonLayout.addWidget(amendPushButton)  
        buttonLayout.addWidget(closePushButton)  
          
        mainLayout=QVBoxLayout(frame)  
        mainLayout.setMargin(10)  
        mainLayout.setSpacing(6)  
        mainLayout.addWidget(self.stack)  
        mainLayout.addLayout(buttonLayout)  
          
        self.connect(self.listWidget,SIGNAL("currentRowChanged(int)"),self.stack,SLOT("setCurrentIndex(int)"))  
        self.connect(closePushButton,SIGNAL("clicked()"),self,SLOT("close()"))  
  
        layout=QHBoxLayout(self)  
        layout.addWidget(mainSplitter)  
        self.setLayout(layout)  
        #例20代码开始-----------------------   
        self.faderWidget=None  
        self.connect(self.listWidget,SIGNAL("currentItemChanged(QListWidgetItem,QListWidgetItem"),  
                     self.changePage)  
        self.connect(self.stack,SIGNAL("currentChanged(int)"),self.fadeInWidget)  
          
    def changePage(self,current,previous):  
        if not current:  
            current=previous  
        self.stack.setCurrentWidget(current)  
  
    def fadeInWidget(self,index):  
        self.faderWidget=FaderWidget(self.stack.widget(index))  
        self.faderWidget.start()  
  
class FaderWidget(QWidget):  
    def __init__(self,parent=None):  
        super(FaderWidget,self).__init__(parent)  
  
        if parent:  
            self.startColor=parent.palette().window().color()  
        else:  
            self.startColor=Qt.White  
  
        self.currentAlpha=0  
        self.duration=1000  
  
        self.timer=QTimer(self)  
        self.connect(self.timer,SIGNAL("timeout()"),self.update)  
        self.setAttribute(Qt.WA_DeleteOnClose)  
        self.resize(parent.size())  
  
    def start(self):  
        self.currentAlpha=255  
        self.timer.start(100)  
        self.show()  
  
    def paintEvent(self,event):  
        semiTransparentColor=self.startColor  
        semiTransparentColor.setAlpha(self.currentAlpha)  
        painter=QPainter(self)  
        painter.fillRect(self.rect(),semiTransparentColor)  
        self.currentAlpha-=(255*self.timer.interval()/self.duration)  
  
        if self.currentAlpha<=0:  
            self.timer.stop()  
            self.close()  
      
    #例20代码结束-----------------------   
class BaseInfo(QWidget):  
    def __init__(self,parent=None):  
        super(BaseInfo,self).__init__(parent)  
                 
        label1=QLabel(self.tr("用户名:"))  
        label2=QLabel(self.tr("姓名："))  
        label3=QLabel(self.tr("性别:"))  
        label4=QLabel(self.tr("部门:"))  
        label5=QLabel(self.tr("年龄:"))  
        otherLabel=QLabel(self.tr("备注:"))  
        otherLabel.setFrameStyle(QFrame.Panel|QFrame.Sunken)  
        userLineEdit=QLineEdit()  
        nameLineEdit=QLineEdit()  
        sexComboBox=QComboBox()  
        sexComboBox.insertItem(0,self.tr("男"))  
        sexComboBox.insertItem(1,self.tr("女"))  
        departmentTextEdit=QTextEdit()  
        ageLineEdit=QLineEdit()  
  
        labelCol=0  
        contentCol=1  
  
        leftLayout=QGridLayout()  
        leftLayout.addWidget(label1,0,labelCol)  
        leftLayout.addWidget(userLineEdit,0,contentCol)  
        leftLayout.addWidget(label2,1,labelCol)  
        leftLayout.addWidget(nameLineEdit,1,contentCol)  
        leftLayout.addWidget(label3,2,labelCol)  
        leftLayout.addWidget(sexComboBox,2,contentCol)  
        leftLayout.addWidget(label4,3,labelCol)  
        leftLayout.addWidget(departmentTextEdit,3,contentCol)  
        leftLayout.addWidget(label5,4,labelCol)  
        leftLayout.addWidget(ageLineEdit,4,contentCol)  
        leftLayout.addWidget(otherLabel,5,labelCol,1,2)  
        leftLayout.setColumnStretch(0,1)  
        leftLayout.setColumnStretch(1,3)  
  
        label6=QLabel(self.tr("头像:"))  
        iconLabel=QLabel()  
        icon=QPixmap("image/2.jpg")  
        iconLabel.setPixmap(icon)  
        iconLabel.resize(icon.width(),icon.height())  
        iconPushButton=QPushButton(self.tr("改变"))  
        hLayout=QHBoxLayout()  
        hLayout.setSpacing(20)  
        hLayout.addWidget(label6)  
        hLayout.addWidget(iconLabel)  
        hLayout.addWidget(iconPushButton)  
  
        label7=QLabel(self.tr("个人说明:"))  
        descTextEdit=QTextEdit()  
  
        rightLayout=QVBoxLayout()  
        rightLayout.setMargin(10)  
        rightLayout.addLayout(hLayout)  
        rightLayout.addWidget(label7)  
        rightLayout.addWidget(descTextEdit)  
        mainLayout=QGridLayout(self)  
        mainLayout.setMargin(15)  
        mainLayout.setSpacing(10)  
        mainLayout.addLayout(leftLayout,0,0)  
        mainLayout.addLayout(rightLayout,0,1)  
        mainLayout.setSizeConstraint(QLayout.SetFixedSize)  
  
class Contact(QWidget):  
    def __init__(self,parent=None):  
        super(Contact,self).__init__(parent)  
        label1=QLabel(self.tr("电子邮件:"))  
        label2=QLabel(self.tr("联系地址:"))  
        label3=QLabel(self.tr("邮政编码:"))  
        label4=QLabel(self.tr("移动电话:"))  
        label5=QLabel(self.tr("办公电话:"))  
  
        mailLineEdit=QLineEdit()  
        addressLineEdit=QLineEdit()  
        codeLineEdit=QLineEdit()  
        mpLineEdit=QLineEdit()  
        phoneLineEdit=QLineEdit()  
        receiveCheckBox=QCheckBox(self.tr("接收留言"))  
  
        layout=QGridLayout(self)  
        layout.addWidget(label1,0,0)  
        layout.addWidget(mailLineEdit,0,1)  
        layout.addWidget(label2,1,0)  
        layout.addWidget(addressLineEdit,1,1)  
        layout.addWidget(label3,2,0)  
        layout.addWidget(codeLineEdit,2,1)  
        layout.addWidget(label4,3,0)  
        layout.addWidget(mpLineEdit,3,1)  
        layout.addWidget(receiveCheckBox,3,2)  
        layout.addWidget(label5,4,0)  
        layout.addWidget(phoneLineEdit,4,1)  
  
class Detail(QWidget):  
    def __init__(self,parent=None):  
        super(Detail,self).__init__(parent)  
        label1=QLabel(self.tr("国家/地区:"))  
        label2=QLabel(self.tr("省份:"))  
        label3=QLabel(self.tr("城市:"))  
        label4=QLabel(self.tr("个人说明:"))  
  
        countryComboBox=QComboBox()  
        countryComboBox.addItem(self.tr("中华人民共和国"))  
        countryComboBox.addItem(self.tr("香港"))  
        countryComboBox.addItem(self.tr("台北"))  
        countryComboBox.addItem(self.tr("澳门"))  
        provinceComboBox=QComboBox()  
        provinceComboBox.addItem(self.tr("安徽省"))  
        provinceComboBox.addItem(self.tr("北京市"))  
        provinceComboBox.addItem(self.tr("江苏省"))  
        cityLineEdit=QLineEdit()  
        remarkTextEdit=QTextEdit()  
  
        layout=QGridLayout(self)  
        layout.addWidget(label1,0,0)  
        layout.addWidget(countryComboBox,0,1)  
        layout.addWidget(label2,1,0)  
        layout.addWidget(provinceComboBox,1,1)  
        layout.addWidget(label3,2,0)  
        layout.addWidget(cityLineEdit,2,1)  
        layout.addWidget(label4,3,0)  
        layout.addWidget(remarkTextEdit,3,1)  
          
app=QApplication(sys.argv)  
main=StockDialog()  
main.show()  
app.exec_()  