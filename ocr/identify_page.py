# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'identify_page.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(926, 623)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 520, 641, 63))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.rc_label = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.rc_label.setFont(font)
        self.rc_label.setText("")
        self.rc_label.setObjectName("rc_label")
        self.gridLayout_3.addWidget(self.rc_label, 0, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 2, 1, 1)
        self.rc_name = QtWidgets.QLabel(self.groupBox_2)
        self.rc_name.setEnabled(True)
        self.rc_name.setMinimumSize(QtCore.QSize(180, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.rc_name.setFont(font)
        self.rc_name.setText("")
        self.rc_name.setObjectName("rc_name")
        self.gridLayout_3.addWidget(self.rc_name, 0, 1, 1, 1)
        self.label_2.raise_()
        self.rc_label.raise_()
        self.label.raise_()
        self.rc_name.raise_()
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(690, 20, 211, 481))
        self.groupBox_3.setMinimumSize(QtCore.QSize(211, 321))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout.setObjectName("gridLayout")
        self.rc_last = QtWidgets.QPushButton(self.groupBox_3)
        self.rc_last.setMinimumSize(QtCore.QSize(181, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rc_last.setFont(font)
        self.rc_last.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rc_last.setAutoDefault(False)
        self.rc_last.setObjectName("rc_last")
        self.gridLayout.addWidget(self.rc_last, 1, 0, 1, 1)
        self.rc_del = QtWidgets.QPushButton(self.groupBox_3)
        self.rc_del.setMinimumSize(QtCore.QSize(181, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rc_del.setFont(font)
        self.rc_del.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rc_del.setAutoDefault(False)
        self.rc_del.setObjectName("rc_del")
        self.gridLayout.addWidget(self.rc_del, 3, 0, 1, 1)
        self.rc_next = QtWidgets.QPushButton(self.groupBox_3)
        self.rc_next.setMinimumSize(QtCore.QSize(181, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rc_next.setFont(font)
        self.rc_next.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rc_next.setAutoDefault(False)
        self.rc_next.setObjectName("rc_next")
        self.gridLayout.addWidget(self.rc_next, 2, 0, 1, 1)
        self.return_main = QtWidgets.QPushButton(self.groupBox_3)
        self.return_main.setMinimumSize(QtCore.QSize(181, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.return_main.setFont(font)
        self.return_main.setFocusPolicy(QtCore.Qt.NoFocus)
        self.return_main.setAutoDefault(False)
        self.return_main.setObjectName("return_main")
        self.gridLayout.addWidget(self.return_main, 4, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 640, 480))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pic_label = QtWidgets.QLabel(self.groupBox)
        self.pic_label.setText("")
        self.pic_label.setObjectName("pic_label")
        self.gridLayout_2.addWidget(self.pic_label, 1, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(680, 520, 221, 61))
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.rc_jump = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rc_jump.setFont(font)
        self.rc_jump.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rc_jump.setAutoDefault(False)
        self.rc_jump.setObjectName("rc_jump")
        self.gridLayout_4.addWidget(self.rc_jump, 0, 1, 1, 1)
        self.rc_index = QtWidgets.QLineEdit(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.rc_index.setFont(font)
        self.rc_index.setText("")
        self.rc_index.setObjectName("rc_index")
        self.gridLayout_4.addWidget(self.rc_index, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_2.setTitle(_translate("MainWindow", "识别结果"))
        self.label.setText(_translate("MainWindow", "文件名："))
        self.label_2.setText(_translate("MainWindow", "标签："))
        self.groupBox_3.setTitle(_translate("MainWindow", "GroupBox"))
        self.rc_last.setText(_translate("MainWindow", "上一张"))
        self.rc_del.setText(_translate("MainWindow", "删除"))
        self.rc_next.setText(_translate("MainWindow", "下一张"))
        self.return_main.setText(_translate("MainWindow", "返回主页面"))
        self.groupBox.setTitle(_translate("MainWindow", "识别的图像"))
        self.groupBox_4.setTitle(_translate("MainWindow", "定位"))
        self.rc_jump.setText(_translate("MainWindow", "跳转"))