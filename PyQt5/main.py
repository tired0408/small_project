# coding=gbk

import sys,os
import time
# ��ӿ��ļ����ӳ���py�����ͬ�ļ��У�ֱ����Ӿͺ�
# sys.path.append("./Program_Child/")
# import Child_1 as Progress_Vision
# import Child_2

from PyQt5 import QtCore, QtGui, uic, QtWidgets

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PIL import Image, ImageGrab, ImageFilter

# ������ļ���ȫ�ֱ���
import Global_Var as globalvar

import cgitb
cgitb.enable(format="text")  # ���pyqt�쳣������ͱ�������û���κ���ʾ������

# һ�������桢�����滭���桢һ������������
Ui_MainWindow, _ = uic.loadUiType(".\GUI\MainWindow.ui")
Ui_Draw_0, QtBaseClass_1 = uic.loadUiType(".\GUI\Draw.ui")
Ui_Draw_1, QtBaseClass_2 = uic.loadUiType(".\GUI\Draw_Num.ui")
Ui_Progress, QtBaseClass_3 = uic.loadUiType(".\GUI\Progress.ui")

# ���ļ�ȫ�ֱ�����ʼ��,�ò�����ɾ��
globalvar._init()

# *******************************
# **********   ������
# *******************************
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # ���ñ��⣨��ʹ����QT Designer ��ƺ��Ժ������ôʹ�ã�
        self.setWindowTitle('Main')
        # ����ť����������
        self.Single_Button.clicked.connect(self.closeWindow)
        self.Num_Button.clicked.connect(self.Num_Vision)
        self.Draw_0_Button.clicked.connect(self.OpenDraw0)
        self.Draw_1_Button.clicked.connect(self.OpenDraw1)
        # �����ӽ���ʵ��
        self.single_thread = SingleVisionThread()
        self.NumThread = Num_Vision_Thread()
        self.Draw0_window = Draw0_Dialog()
        self.Draw1_window = Draw1_Dialog()
        # ���̵߳��źŲ۰󶨴�����
        self.single_thread.finish_signal.connect(self.finish_single)
        self.NumThread.Finish_Signal.connect(self.finish_single)
        self.Draw0_window.Finish_Signal.connect(self.Draw0_Finish_Single)
        self.Draw1_window.finish_signal.connect(self.Draw1_Finish_Single)

    #������ʾͼƬ
    def showpic(self,filename):
        #����QtGui.QPixmap��������һ��ͼƬ������ڱ���png��
        png=QPixmap(filename)
        Position = self.pic_label.geometry()
        scaredPixmap = png.scaled(Position.width()+Position.left(),
                                  Position.height()+ Position.top(),
                                  aspectRatioMode=Qt.KeepAspectRatio)
        self.pic_label.setPixmap(scaredPixmap)

    # �ر������ڲ���
    def closeWindow(self):
        self.close()

    # �رմ��ڵ���ʾ
    def closeEvent(self, event):
        box = QtWidgets.QMessageBox()
        Messages_S = "ȷ���˳���"
        Messages_S = Messages_S
        reply = box.warning(self, '��ʾ', Messages_S, box.Yes | box.No, box.Yes)
        if reply == box.Yes:
            event.accept()
        else:
            event.ignore()

    # �򿪾��浯��
    def OpenWarningBox(self, Messages_S):
        box = QtWidgets.QMessageBox()
        Messages_S = Messages_S
        box.warning(self, '��ʾ', Messages_S, box.Ok)

    # ��ť���¶�Ӧ�Ĳ���
    def single_vision(self):
        self.textEdit.setText('')
        self.Num_Button.setDisabled(True)
        self.Draw_0_Button.setDisabled(True)
        self.Draw_1_Button.setDisabled(True)
        self.single_thread.start()
        Progress_window.show()

    # �����Ժ���Ҫ�رս���������ʾ������������ð�ť
    def finish_single(self, Result):
        Progress_window.close()
        self.textEdit.setText(Result)
        self.Num_Button.setDisabled(False)
        self.Draw_0_Button.setDisabled(False)
        self.Draw_1_Button.setDisabled(False)

    # ��Ӧ�ڶ�����ť�Ĺ���
    def Num_Vision(self):
        self.textEdit.setText('')
        filenames = QFileDialog.getOpenFileName(self, "ѡ��ͼƬ", "", "jpg files(*.jpg)")
        if len(filenames[0]) > 0:
            filename = filenames[0]
            # ����ȫ�ֱ�����������ļ�����
            globalvar.Set_value('filename', filename)
            # print(Visions.knn_Single(filename))
            self.NumThread.start()
            Progress_window.show()
            self.showpic(filename)

    # ��Ӧ��дʶ�����������
    def OpenDraw0(self):
        self.textEdit.setText('')
        self.Draw0_window.pos_xy = []
        self.Draw0_window.show()
        # self.hide()
    def OpenDraw1(self):
        self.textEdit.setText('')
        self.Draw1_window.pos_xy = []
        self.Draw1_window.show()
        # self.hide()

    # ��дʶ�𴰿ڹرս��еĶ���
    def Draw0_Finish_Single(self):
        filename = './cache.jpg'
        globalvar.Set_value('filename', filename)
        # print(Visions.knn_Single(filename))
        self.single_thread.start()
        Progress_window.show()
        self.showpic(filename)
    def Draw1_Finish_Single(self):
        filename = './cache.jpg'
        # ���ÿ��ļ�ȫ�ֱ������������������
        globalvar.Set_value('filename', filename)
        # �������߳�
        self.NumThread.start()
        # �򿪽��ȴ���
        Progress_window.show()
        # ��ʾ��ͼƬ
        self.showpic(filename)



# *******************************
# **********   �����ַ���д����
# *******************************
class Draw0_Dialog(QtWidgets.QDialog, Ui_Draw_0):
    Finish_Signal = pyqtSignal()
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        Ui_Draw_0.__init__(self)
        self.setupUi(self)

        # ���ñ��⣨��ʹ����QT Designer ��ƺ��Ժ������ôʹ�ã�
        self.setWindowTitle('�����ַ���дʶ��')

        # setMouseTracking����ΪFalse�����򲻰������ʱҲ���������¼�
        self.setMouseTracking(False)

        self.pos_xy = []

    # �رմ��ڵ���ʾ
    def closeDialog(self):
        self.close()

    # **********   �رմ��ڵ���ʾ
    def closeEvent(self, event):
        Position = self.geometry()
        img = ImageGrab.grab((Position.left(),
                              Position.top(),
                              Position.width() + Position.left(),
                              Position.height() + Position.top()))
        img = img.filter(ImageFilter.BLUR)
        img.save("./cache.jpg")
        # ����ȫ�ֱ�����������ļ�����
        event.accept()
        self.Finish_Signal.emit()


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 10, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end

        painter.end()

    def mouseMoveEvent(self, event):
        #�м����pos_tmp��ȡ��ǰ��
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp��ӵ�self.pos_xy��
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

# *******************************
# **********   ����ַ���д����
# *******************************
class Draw1_Dialog(QtWidgets.QDialog, Ui_Draw_1):
    finish_signal = pyqtSignal()
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        Ui_Draw_1.__init__(self)
        self.setupUi(self)

        # ���ñ��⣨��ʹ����QT Designer ��ƺ��Ժ������ôʹ�ã�
        self.setWindowTitle('����ַ���дʶ��')

        # setMouseTracking����ΪFalse�����򲻰������ʱҲ���������¼�
        self.setMouseTracking(False)

        self.pos_xy = []

    # �رմ��ڵ���ʾ
    def closeDialog(self):
        self.close()

    # **********   �رմ��ڵ���ʾ
    def closeEvent(self, event):
        Position = self.geometry()
        img = ImageGrab.grab((Position.left(),
                              Position.top(),
                              Position.width() + Position.left(),
                              Position.height() + Position.top()))
        img = img.filter(ImageFilter.BLUR)
        img.save("./cache.jpg")
        # ����ȫ�ֱ�����������ļ�����
        event.accept()
        self.finish_signal.emit()


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 10, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end

        painter.end()

    def mouseMoveEvent(self, event):
        #�м����pos_tmp��ȡ��ǰ��
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp��ӵ�self.pos_xy��
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

# *******************************
# **********   ����������
# *******************************
class Progress_Dialog(QtWidgets.QDialog, Ui_Progress):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        Ui_Progress.__init__(self)
        self.setupUi(self)
        # ���ñ��⣨��ʹ����QT Designer ��ƺ��Ժ������ôʹ�ã�
        self.setWindowTitle('����ʶ��')

    # **********   �رմ��ڵ���ʾ
    def closeEvent(self, event):
        # window.show()
        event.accept()


# *************************************
# ********** ���߳�����ʶ�𵥸��ַ���
# *************************************
class SingleVisionThread(QThread):
    # ɨ����ź��һ�θ�����ʾ
    finish_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super(SingleVisionThread, self).__init__(parent)

    def run(self):
        # �������д������
        # �������д������
        # �������д������
        # �������д������
        # �������д������
        Result = 'ʶ�����'
        time.sleep(1)
        self.finish_signal.emit(str(Result))

# *************************************
# ********** ���߳�����ʶ�����ַ���
# *************************************
class Num_Vision_Thread(QThread):
    # ɨ����ź��һ�θ�����ʾ
    Finish_Signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super(Num_Vision_Thread, self).__init__(parent)

    def run(self):
        # �������д������
        # �������д������
        # �������д������
        # �������д������
        # �������д������
        Result = 'ʶ�����'
        time.sleep(5)
        self.Finish_Signal.emit(str(Result))




# ��������򴴽���һ���µ� Qt Gui Ӧ�á���ÿ�� QT Ӧ�ö�����ͨ�������н������ã����Ա��봫��sys.argv ������
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    # Draw0_Window = Draw0_Dialog()
    # Draw1_Window = Draw1_Dialog()
    Progress_window = Progress_Dialog()
    window = MyApp()
    window.show()

    sys.exit(app.exec_())
