# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(666, 311)
        MainWindow.setMinimumSize(QtCore.QSize(666, 311))
        MainWindow.setMaximumSize(QtCore.QSize(666, 311))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 10, 631, 276))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.output_name = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.output_name.setFont(font)
        self.output_name.setObjectName("output_name")
        self.gridLayout.addWidget(self.output_name, 1, 0, 1, 1)
        self.label_name = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.label_name.setFont(font)
        self.label_name.setObjectName("label_name")
        self.gridLayout.addWidget(self.label_name, 6, 0, 1, 1)
        self.label_content = QtWidgets.QTextEdit(self.layoutWidget)
        self.label_content.setObjectName("label_content")
        self.gridLayout.addWidget(self.label_content, 6, 1, 1, 3)
        self.voc = QtWidgets.QRadioButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(20)
        self.voc.setFont(font)
        self.voc.setObjectName("voc")
        self.gridLayout.addWidget(self.voc, 2, 1, 1, 1)
        self.output_path = QtWidgets.QLineEdit(self.layoutWidget)
        self.output_path.setObjectName("output_path")
        self.gridLayout.addWidget(self.output_path, 1, 1, 1, 3)
        self.type_name = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.type_name.setFont(font)
        self.type_name.setObjectName("type_name")
        self.gridLayout.addWidget(self.type_name, 2, 0, 1, 1)
        self.input_path = QtWidgets.QLineEdit(self.layoutWidget)
        self.input_path.setObjectName("input_path")
        self.gridLayout.addWidget(self.input_path, 0, 1, 1, 3)
        self.add_input = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.add_input.setFont(font)
        self.add_input.setObjectName("add_input")
        self.gridLayout.addWidget(self.add_input, 0, 4, 1, 1)
        self.add_output = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.add_output.setFont(font)
        self.add_output.setObjectName("add_output")
        self.gridLayout.addWidget(self.add_output, 1, 4, 1, 1)
        self.coco = QtWidgets.QRadioButton(self.layoutWidget)
        self.coco.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(20)
        self.coco.setFont(font)
        self.coco.setObjectName("coco")
        self.gridLayout.addWidget(self.coco, 2, 2, 1, 1)
        self.cancel = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(20)
        self.cancel.setFont(font)
        self.cancel.setObjectName("cancel")
        self.gridLayout.addWidget(self.cancel, 7, 4, 1, 1)
        self.execute = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(20)
        self.execute.setFont(font)
        self.execute.setObjectName("execute")
        self.gridLayout.addWidget(self.execute, 7, 3, 1, 1)
        self.input_name = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.input_name.setFont(font)
        self.input_name.setObjectName("input_name")
        self.gridLayout.addWidget(self.input_name, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.status_message = QtWidgets.QStatusBar(MainWindow)
        self.status_message.setObjectName("status_message")
        MainWindow.setStatusBar(self.status_message)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.output_name.setText(_translate("MainWindow", "输出文件夹地址："))
        self.label_name.setText(_translate("MainWindow", "标签列表："))
        self.voc.setText(_translate("MainWindow", "VOC"))
        self.type_name.setText(_translate("MainWindow", "转换类型："))
        self.add_input.setText(_translate("MainWindow", "添加"))
        self.add_output.setText(_translate("MainWindow", "修改"))
        self.coco.setText(_translate("MainWindow", "COCO"))
        self.cancel.setText(_translate("MainWindow", "Cancel"))
        self.execute.setText(_translate("MainWindow", "Transform"))
        self.input_name.setText(_translate("MainWindow", "输入文件夹地址："))
