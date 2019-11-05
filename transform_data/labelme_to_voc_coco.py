import sys
import os
# 解决pyqt打包成exe报ImportError: unable to find Qt5Core.dll on PATH。原因：pyqt5库对系统变量的加载存在bug
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
import traceback
import cgitb
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from labelme2coco import main as coco_main
from labelme2voc import main as voc_main
from page import Ui_MainWindow

cgitb.enable(format='text')  # 解决pyqt5异常只要进入事件循环,程序就崩溃,而没有任何提示


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # 设置标题（在使用了QT Designer 设计好以后可以这么使用）
        self.setWindowTitle("Labelme_to_voc")
        self.cancel.clicked.connect(self.closeWindow)  # 绑定关闭窗口函数
        self.add_input.clicked.connect(self.add_input_path)  # 设置输入文件夹地址
        self.add_output.clicked.connect(self.add_output_path)  # 设置输出文件夹地址
        self.coco.setChecked(True)  # 设置默认转化为COCO数据集
        self.execute.clicked.connect(self.transform_data)  # 转换数据集

        # 连接子进程的信号和槽函数
        self.transform_thread = transform_thread()
        self.transform_thread.transform_signal.connect(self.transform_thread_deal)


    # 关闭窗口的提示
    def closeEvent(self, event):
        box = QtWidgets.QMessageBox()
        Messages_S = "确认退出？"
        Messages_S = Messages_S
        reply = box.warning(self, '提示', Messages_S, box.Yes | box.No, box.Yes)
        if reply == box.Yes:
            event.accept()
        else:
            event.ignore()

    # 按钮绑定的关闭窗口操作
    def closeWindow(self):
        self.close()

    def add_input_path(self):
        input_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹地址")
        self.input_path.setText(input_path)
        output_path = self.output_path.text()
        if not output_path:
            self.output_path.setText(input_path+"_output")

    def add_output_path(self):
        output_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹地址")
        self.output_path.setText(output_path)

    # 开始转换数据
    def transform_data(self):
        self.execute.setDisabled(True)
        self.status_message.showMessage("正在转换中，请稍等...")
        self.transform_thread.start()


    def transform_thread_deal(self, result, msg):
        if result:
            self.execute.setDisabled(False)
            self.status_message.showMessage("转换完成", 3000)  # 毫秒级别
        else:
            self.execute.setDisabled(False)
            self.status_message.showMessage(msg)


class transform_thread(QThread):
    """
    多线程任务
    """
    transform_signal = pyqtSignal(bool, str)
    def __init__(self, parent=None):
        super(transform_thread, self).__init__(parent)

    def run(self):
        try:
            # 从主页面获取需要的输入地址，输出地址，标签内容
            input_path = window.input_path.text()
            output_path = window.output_path.text()
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            label_content = window.label_content.toPlainText()
            label_list = label_content.split("\n")
            label_list = list(set(label_list))  # 去除重复标签
            if "" in label_list:
                label_list.remove("")  # 去除空标签
            if len(label_list) == 0:
                raise Exception("标签列表不能为空。")
            if window.coco.isChecked():
                coco_main(input_path, output_path, label_list)
            elif window.voc.isChecked():
                voc_main(input_path, output_path, label_list)
            print("转换完成")
            self.transform_signal.emit(True, None)
        except Exception as e:
            self.transform_signal.emit(False, str(e))
            print(traceback.format_exc())




# 这段主程序创建了一个新的 Qt Gui 应用。每个 QT 应用都可以通过命令行进行配置，所以必须传入sys.argv参数。
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
