#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test.py
# Date: Thu Dec 26 12:20:01 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys
from PyQt4 import QtCore, QtGui, uic

class MyForm(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        uic.loadUi("edytor.ui", self)
        #self.ui = Ui_notepad()
        #self.ui.setupUi(self)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
