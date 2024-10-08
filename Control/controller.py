import sys
import cv2
import time
import datetime
import numpy as np

from PyQt5.QtWidgets import QApplication

from Interface.ui.vis import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QFontMetrics
from Client.listener.detector import Detector
from Models.llm import ModelThread
from db_tool import Mysql
from Models.run_msa import MSA
from Models.data_process.get_Vfeatures import GetFeatures
from Models.data_process.get_ATfeatures import GetATFeatures
from Models.features_vis import Thread_Feature_Visualization

from ui_show import Ui_Show

import pyttsx3

import config
import threading

"""
vdetector() used for vision detection and process
adetector() used for audio detection and check the speaker sound
"""


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # 创建tts对象
        self.engine = pyttsx3.init()

        self.setupUi(self)
        # 创建一个事件对象
        self.sync_event = threading.Event()

        # 创建全局变量保存模态特征
        # self.features = {
        #     'feature_t': [],
        #     'feature_a': [],
        #     'feature_v': [],
        #     'feature_m': []
        # }
        self.features = []
        self.features_len = 1

        # for detect
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(config.VIDEO_File_NAME, self.fourcc, 20.0, (640, 480))
        self.detector = Detector()

        ### acquire video from camera and show it
        self.delay_frames = 20  # restoring frame gap 1s for analysis
        self.frames = []
        ##### sound energy detect and judge if the video need to process
        self.filename = ''
        self.data_process_flag = 0
        ### for acontrollor
        self.silence_start = None
        ### signals and slots
        self.pushButton_start.clicked.connect(self.start_button)

        #####################################################
        # for chat
        self.speak_text = ''
        self.sum = 0  # 气泡数量
        self.widgetlist = []  # 记录气泡
        self.text = ""  # 存储信息
        self.icon = QtGui.QPixmap("1.jpg")  # 头像
        # 设置聊天窗口样式 隐藏滚动条
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # 信号与槽
        self.Button_emit.clicked.connect(self.create_user_widget)  # 创建气泡
        # self.Button_emit.clicked.connect(self.set_widget)  # 修改气泡长宽
        self.plainTextEdit.undoAvailable.connect(self.Event)  # 监听输入框状态
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.rangeChanged.connect(self.adjustScrollToMaxValue)  # 监听窗口滚动条范围
        # 数据库存储
        self.mysql = Mysql()

    def start_button(self):
        # begin to detect sound in the background
        self.get_at_features = GetATFeatures()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.time_functions)
        self.timer.start(30)

    def time_functions(self):
        # show video all time
        self.vcontroller()
        # process video data
        self.data_process()
        # check if existing sounds
        self.sound_detect()
        # check the msa model select
        self.msa_model = self.select_MsaModels.currentText()
        # 可视化数据特征
        self.feature_visualization(self.features)

    def vcontroller(self):
        """
        1.capture the video through local camera according to if existing sound and show it
        """
        frame = self.detector.vdetector()
        rgb_frame = cv2.resize(
            frame, (self.label_video.width(), self.label_video.height())
        )  # 调整尺寸
        rgb_frame = cv2.cvtColor(
            rgb_frame, cv2.COLOR_BGR2RGB
        )  # transfer from BGR to RGB
        Ui_Show.video_show(self, rgb_frame)

        if config.WITH_SOUND:  # begin or stop record with or without the audio
            if config.CAPTURE_FRAME is None:  # prevent initial again
                config.CAPTURE_FRAME = 1
                video_name = f"{config.VIDEO_File_NAME}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.filename = f"{video_name}.avi"
                self.video_writer = cv2.VideoWriter(self.filename, self.fourcc, 20.0, (640, 480))
                print("keep frame detection")
            # if self.delay_frames % 20 == 0:
            #     self.frames.append(rgb_frame)
            #     self.delay_frames += 1

            self.video_writer.write(frame)
            #

        else:
            if config.CAPTURE_FRAME is not None:  # means it has turned into silence from loud. prevent print again
                print("speech end or not keep frame detection")
                config.CAPTURE_FRAME = None
                self.delay_frames = 20
                self.data_process_flag = 1

    def data_process(self):
        '''
        1. analysis video data with deepface
        2. get features from original data with get_features
        3. use msa model to analysis sentiment according to input features
        4. show the sentiment res in the ui
        '''
        if self.data_process_flag and config.WITH_AUDIO:
            print('begin process data')
            self.video_writer.release()  # 关闭视频写入器，保存视频
            # if len(self.frames) != 0:
            #     # process data
            #     res = MSA().msa2(self.frames[-1])
            #     Ui_Show.statement_show(self, res)
            #     print("process data once")
            #     print("the data len is", len(self.frames))
            #     # data = np.concatenate(self.frames, axis=0)
            # self.frames = []
            audio_feature, text_feature, self.speak_text = self.get_at_features.getAudioEmbedding()
            video_feature = GetFeatures(self.filename).getVideoEmbedding()
            # msa_res = MSA().msa(self.sync_event, self.msa_model, self.filename,
            #                     (text_feature, audio_feature, video_feature))

            msa_res = MSA().msa(self.sync_event, 'lf_dnn', self.filename,
                                (text_feature, audio_feature, video_feature))

            sentiment = ''

            if msa_res['M'] > 0:
                sentiment = 'happy'
            elif -0.6 < msa_res['M'] < -0:
                sentiment = 'sad'
            elif msa_res['M'] < -0.6:
                sentiment = 'angry'
            # elif -0.1 < msa_res['M'] < 0.1:
            #     sentiment = 'neutral'

            Ui_Show.statement_show(self, sentiment)

            # 语音对话
            self.create_user_widget(text=self.speak_text, sentiment=sentiment)
            self.data_process_flag = 0
            config.WITH_AUDIO = 0
            config.CAPTURE_FRAME = None

            # 保存特征用于可视化
            # self.features['feature_t'].append(np.array(msa_res['Feature_t']))
            # self.features['feature_a'].append(np.array(msa_res['Feature_a']))
            # self.features['feature_v'].append(np.array(msa_res['Feature_v']))
            # self.features['feature_m'].append(np.array(msa_res['Feature_f']))
            self.features.append(np.array(msa_res['Feature_f']))

    # ############-check if there are sounds-###################
    def sound_detect(self):
        """
        the system begin to grab video frames and store it when detect sound
        :return:
        """
        data = self.detector.energy_detector()  # audio data

        threshold = 10000
        silence_time = 2
        # Calculate audio energy
        energy = np.sum(data.astype(float) ** 2)

        # Check if energy exceeds the threshold
        if energy < threshold:
            if self.silence_start is None:
                self.silence_start = time.time()  # starting to timing
        else:
            config.WITH_SOUND = 1
            self.silence_start = None

        # Check if the silence time exceed 1s
        if self.silence_start and time.time() - self.silence_start > silence_time:
            config.WITH_SOUND = 0
            print("silence coming")
            self.silence_start = None
        # <<<<< ############-check if there are sounds-###################

    ########### 人机交互模块feature visualization    #######################

    def feature_visualization(self, features):
        if len(features) > self.features_len:  # 判断是否有新数据
            # self.thread_feature = Thread_Feature_Visualization(features)
            # self.thread_feature.features_signal.connect(self.feature_visualization)  # 处理线程返回的结果
            # self.thread_feature.start()
            self.features_len = len(features)
            img = Thread_Feature_Visualization(features).run()
            Ui_Show.features_show(self, img)

    ############### for constructing the chat framework#########>>>>>>>>>#####
    # 创建气泡
    def create_user_widget(self, text='', sentiment='正常'):
        if text == '':
            self.text = self.plainTextEdit.toPlainText()
            self.plainTextEdit.setPlainText("")
        else:
            self.text = text

        # 显示己方对话
        Ui_Show.chat_show(self, self.icon, self.text, QtCore.Qt.LeftToRight)  # 调用new_widget.py中方法生成左气泡

        # 调用额外线程进行回话，不然ui无法及时刷新
        # 显示机器回话
        self.thread = ModelThread(self.text, sentiment, self.mysql)
        self.thread.response_signal.connect(self.create_machine_widget)  # 处理线程返回的结果
        self.thread.start()

    def create_machine_widget(self, response):

        Ui_Show.chat_show(self, self.icon, response, QtCore.Qt.RightToLeft)  # 调用new_widget.py中方法生成右气泡

        # self.engine.say(response)  # TTS
        # self.engine.runAndWait()

    # 修改气泡长宽
    def set_widget(self):
        font = QFont()
        font.setPointSize(4)
        fm = QFontMetrics(font)
        text_width = fm.width(self.text)  # + 115  # 根据字体大小生成适合的气泡宽度
        # if self.sum != 0:
        if text_width > 632:  # 宽度上限
            text_width = int(self.textBrowser.document().size().width()) + 100  # 固定宽度
        self.widget.setMinimumSize(text_width, int(self.textBrowser.document().size().height()) + 40)  # 规定气泡大小
        self.widget.setMaximumSize(text_width, int(self.textBrowser.document().size().height()) + 40)  # 规定气泡大小
        self.scrollArea.verticalScrollBar().setValue(10)

    # 回车绑定发送
    def Event(self):
        if not self.plainTextEdit.isEnabled():  # 这里通过文本框的是否可输入
            self.plainTextEdit.setEnabled(True)
            self.pushButton.click()
            self.plainTextEdit.setFocus()

    # 窗口滚动到最底部
    def adjustScrollToMaxValue(self):
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    ##############<<<<<<<<<<<<<<<< for constructing the chat framework##############################

    def closeEvent(self, event):
        self.timer_showvideo.stop()
        self.detector.cap.release()
        super().closeEvent(event)
        self.mysql.clear_table()
        self.mysql.close_connection()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())
