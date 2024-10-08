from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt


class Ui_Show:
    def chat_show(self, ico, text, dir):  # 头像，文本，方向
        self.widget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget.setLayoutDirection(dir)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMaximumSize(QtCore.QSize(50, 50))
        self.label.setText("")
        self.label.setPixmap(ico)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(self.widget)
        self.textBrowser.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textBrowser.setStyleSheet("padding:10px;\n"
                                       "background-color: rgba(71,121,214,20);\n"
                                       "font: 10pt \"黑体\";")
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(text)
        self.textBrowser.setMinimumSize(QtCore.QSize(0, 0))
        #
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout.addWidget(self.textBrowser)
        self.verticalLayout_2.addWidget(self.widget)

    def video_show(self, rgb_frame):
        # 创建QImage，用于在QLabel中显示
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        # 创建QImage，用于在QLabel中显示
        show_video = QtGui.QImage(
            rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        self.label_video.setPixmap(QtGui.QPixmap(show_video))

    def statement_show(self, res):
        # res = res[0]
        state = '' + f"emotion:{res}"
        # state = state + f"emotion:{res['dominant_emotion']}"
        self.textBrowser_res.clear()
        self.textBrowser_res.setText(state)

    def features_show(self, img):
        """
        show features image in QLabel
        :param img:
        :return:
        """
        scaled_pixmap = img.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.show_feature.setPixmap(scaled_pixmap)
        # layout = QVBoxLayout()
        # layout.addWidget(self.show_feature)
